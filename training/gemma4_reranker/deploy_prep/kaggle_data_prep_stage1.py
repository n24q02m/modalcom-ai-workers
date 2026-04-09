import os
import json
import logging
import numpy as np
import concurrent.futures
from typing import Union, List
from tqdm.auto import tqdm
from tenacity import retry, wait_exponential, stop_after_attempt
from google import genai
from google.genai import types
from pydantic import BaseModel
from datasets import load_dataset
from huggingface_hub import login, HfApi

# Initialize logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("KaggleDataPrep")

# =======================================================
# 1. KAGGLE SECRETS (HF_TOKEN & GEMINI_API_KEY)
# =======================================================
try:
    from kaggle_secrets import UserSecretsClient
    user_secrets = UserSecretsClient()
    
    # 1.a HuggingFace Authentication
    hf_token = user_secrets.get_secret("HF_TOKEN")
    if hf_token:
        login(token=hf_token)
        logger.info("Successfully authenticated with HuggingFace Hub.")
    else:
        logger.warning("HF_TOKEN not found. Dataset push/pull might fail if not public.")
        
    # 1.b Google AI API Key Authentication
    google_key = None
    for key_name in ["GEMINI_API_KEY", "GOOGLE_API_KEY"]:
        try:
            google_key = user_secrets.get_secret(key_name)
            if google_key: 
                break
        except Exception:
            pass
            
    if google_key:
        os.environ["GEMINI_API_KEY"] = google_key
        logger.info(f"Successfully loaded {key_name} from Kaggle Secrets.")
except ImportError:
    logger.warning("Not running in a Kaggle environment (No kaggle_secrets found).")
except Exception as e:
    logger.warning(f"Error accessing Kaggle secrets: {e}")

api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
if not api_key:
    raise ValueError("CRITICAL: Missing GOOGLE_API_KEY or GEMINI_API_KEY. Please add to Kaggle Add-ons -> Secrets.")


# =======================================================
# 2. CONFIGURATION (Gemini Models & Repo)
# =======================================================
EMBEDDING_MODEL = "models/text-embedding-004" 
MULTIMODAL_EMBEDDING_MODEL = "models/gemini-embedding-2-preview" 
TEACHER_MODEL = "models/gemini-3-flash-preview"
HF_REPO = "n24q02m/gemma4-stage1-data-jsonl"
OUTPUT_PATH = "/kaggle/working/stage1_train.jsonl"


class RelevanceOutput(BaseModel):
    relevance_score: float

class GeminiMiner:
    def __init__(self):
        self.client = genai.Client()
            
    @retry(wait=wait_exponential(multiplier=1, min=2, max=30), stop=stop_after_attempt(5))
    def embed_content(self, contents: Union[str, List[str]], modality: str = "text") -> Union[np.ndarray, List[np.ndarray]]:
        model_to_use = MULTIMODAL_EMBEDDING_MODEL if modality in ["image", "video", "audio"] else EMBEDDING_MODEL
        response = self.client.models.embed_content(
            model=model_to_use,
            contents=contents,
            config=types.EmbedContentConfig(
                task_type="RETRIEVAL_DOCUMENT"
            )
        )
        if isinstance(contents, list):
            return [np.array(emb.values) for emb in response.embeddings]
        return np.array(response.embeddings[0].values)

    @retry(wait=wait_exponential(multiplier=1, min=2, max=15), stop=stop_after_attempt(3))
    def get_teacher_score(self, query: str, document: str, modality: str = "text") -> float:
        prompt = f"""You are an expert search and relevance ranking system.
Evaluate the relevance of the following Document to the provided Query.
Output EXACTLY a JSON payload with a single key 'relevance_score' containing a float between 0.0 (completely irrelevant) and 1.0 (highly relevant).
Query: {query}
Document: {document}"""
        response = self.client.models.generate_content(
            model=TEACHER_MODEL,
            contents=prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=RelevanceOutput,
            )
        )
        try:
            result = json.loads(response.text)
            return float(result.get("relevance_score", 0.0))
        except Exception as e:
            logger.warning(f"Failed to parse teacher score, returning 0.0. Error: {e}")
            return 0.0

if __name__ == "__main__":
    
    # Install dependencies on the fly if running inside Python script on Kaggle
    os.system("pip install -q datasets huggingface_hub sentence-transformers pydantic tenacity google-genai")
    
    miner = GeminiMiner()

    # =======================================================
    # 3. REAL DATASET LOADING (NO MORE MOCKS/TODO)
    # =======================================================
    def get_processed_count(filepath: str) -> int:
        if not os.path.exists(filepath): return 0
        with open(filepath, 'r', encoding='utf-8') as f:
            return sum(1 for _ in f)

    processed_count = get_processed_count(OUTPUT_PATH)
    logger.info(f"Existing processed records in {OUTPUT_PATH}: {processed_count}")

    raw_data = []

    # 3.a Load MS MARCO (Text Base)
    try:
        logger.info("Loading exact MS MARCO dataset (100k split)...")
        ds_marco = load_dataset("microsoft/ms_marco", "v2.1", split="train[:100000]", trust_remote_code=True)
        for row in ds_marco:
            passages = row.get("passages", {})
            if not passages or not passages.get("passage_text"): continue
            
            # Find positive document
            positives = [text for text, is_sel in zip(passages["passage_text"], passages.get("is_selected", [])) if is_sel]
            if positives:
                raw_data.append({
                    "query": row["query"], 
                    "positive": positives[0], 
                    "modality": "text", 
                    "corpus": passages["passage_text"]
                })
    except Exception as e:
        logger.error(f"Failed to load MS Marco: {e}")

    # 3.b Load VisualNews for text-image correlation (Placeholder standard set)
    try:
        logger.info("Loading sample Visual dataset...")
        ds_vnews = load_dataset("FuxiaoLiu/VisualNews-Rerank", split="train[:20000]", trust_remote_code=True)
        for row in ds_vnews:
            raw_data.append({
                "query": row["caption"], 
                "positive": row["article"], 
                "modality": "image", 
                "corpus": [row["article"]]
            })
    except Exception as e:
        logger.warning(f"Could not load visual dataset stream. Note: {e}")

    logger.info(f"Total raw data prepared: {len(raw_data)} queries.")
    
    # =======================================================
    # 4. EXECUTE HARD NEGATIVE MINING 
    # =======================================================
    if processed_count >= len(raw_data) and len(raw_data) > 0:
        logger.info("All data already processed.")
    else:
        with open(OUTPUT_PATH, 'a', encoding='utf-8') as out_file:
            for i, item in enumerate(tqdm(raw_data[processed_count:], desc="Processing Data")):
                query = item['query']
                positive = item['positive']
                modality = item.get('modality', 'text')
                
                # Fetch random negatives across dataset if internal corpus isn't broad enough
                if len(item['corpus']) < 15:
                    import random
                    random_pool = [x['positive'] for x in random.sample(raw_data, min(50, len(raw_data))) if x['modality'] == modality]
                    corpus = list(set(item['corpus'] + random_pool))
                else:
                    corpus = item['corpus']
                
                try:
                    # BATCHING EMBEDDING CALL: 1 single API request for the query + entire corpus
                    all_texts_to_embed = [query] + corpus
                    all_embs = miner.embed_content(contents=all_texts_to_embed, modality=modality)
                    q_emb = all_embs[0]
                    corpus_embs = np.array(all_embs[1:])
                    
                    scores = np.dot(corpus_embs, q_emb)
                    ranked_indices = np.argsort(scores)[::-1]
                    
                    # Hard negative logic: skip top 2, pick 7 from 2..50
                    start_idx = min(2, len(ranked_indices))
                    end_idx = min(50, len(ranked_indices))
                    pool_indices = ranked_indices[start_idx:end_idx] if start_idx != end_idx else ranked_indices
                    
                    num_neg = min(7, len(pool_indices))
                    selected_neg_indices = np.random.choice(pool_indices, size=num_neg, replace=False)
                    negatives = [corpus[idx] for idx in selected_neg_indices]
                    
                    # THREAD POOL FOR TEACHER SCORES: Evaluating 8 scores in parallel
                    scoring_tasks = [(query, positive, modality)] + [(query, n, modality) for n in negatives]
                    
                    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
                        # Map runs concurrently and preserves order
                        results = list(executor.map(lambda args: miner.get_teacher_score(*args), scoring_tasks))
                        
                    t_pos_score = results[0]
                    t_neg_scores = results[1:]
                    
                    processed_record = {
                        "query": query,
                        "positive": positive,
                        "negatives": negatives,
                        "modality": modality,
                        "teacher_pos_score": t_pos_score,
                        "teacher_neg_scores": t_neg_scores
                    }
                    
                    out_file.write(json.dumps(processed_record) + "\n")
                    if i % 100 == 0: 
                        out_file.flush()
                    
                except Exception as e:
                    logger.error(f"Failed query '{query}': {e}")
                    continue
        
    logger.info(f"Data preparation complete. Output saved to {OUTPUT_PATH}")

    # =======================================================
    # 5. UPLOAD TO HUGGINGFACE
    # =======================================================
    logger.info(f"Uploading output to HuggingFace repository: {HF_REPO}...")
    try:
        api = HfApi()
        api.create_repo(repo_id=HF_REPO, repo_type="dataset", exist_ok=True)
        api.upload_file(
            path_or_fileobj=OUTPUT_PATH,
            path_in_repo="stage1_train.jsonl",
            repo_id=HF_REPO,
            repo_type="dataset"
        )
        logger.info(f"Dataset successfully pushed up: https://huggingface.co/datasets/{HF_REPO}")
    except Exception as e:
        logger.error(f"HF Upload completely failed (Missing Token or network error): {e}")
