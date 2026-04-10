import nbformat as nbf

nb = nbf.v4.new_notebook()

code1 = "!pip install -q -U google-generativeai tenacity pydantic"

code2 = """import os
import json
import logging
import numpy as np
from tqdm.auto import tqdm
from tenacity import retry, wait_exponential, stop_after_attempt
import google.generativeai as genai
from pydantic import BaseModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("KaggleDataPrep")

# Try to load secrets inside kaggle
try:
    from kaggle_secrets import UserSecretsClient
    user_secrets = UserSecretsClient()
    # Pull secrets from kaggle
    kaggle_api = user_secrets.get_secret("GEMINI_API_KEY")
    if not kaggle_api:
        kaggle_api = user_secrets.get_secret("GOOGLE_API_KEY")
        
    if kaggle_api:
        os.environ["GEMINI_API_KEY"] = kaggle_api
        logger.info("Keys injected from Kaggle Secrets")
except ImportError:
    logger.warning("Not in Kaggle environment.")
except Exception as e:
    logger.warning(f"Error accessing Kaggle secrets: {e}")

api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
if not api_key:
    # A big warning for the user so they can clearly paste it in or add secret
    raise ValueError("Missing API Key! Please add a secret named GEMINI_API_KEY or GOOGLE_API_KEY in Kaggle Add-ons -> Secrets.")

genai.configure(api_key=api_key)
"""

code3 = """# --- Configuration ---
EMBEDDING_MODEL = "models/text-embedding-004" 
MULTIMODAL_EMBEDDING_MODEL = "models/gemini-embedding-2-preview" 
TEACHER_MODEL = "models/gemini-3-flash-preview"

class RelevanceOutput(BaseModel):
    relevance_score: float

class GeminiMiner:
    def __init__(self):
        self.teacher_client = genai.GenerativeModel(TEACHER_MODEL)
            
    @retry(wait=wait_exponential(multiplier=1, min=2, max=30), stop=stop_after_attempt(5))
    def embed_content(self, content: str, modality: str = "text") -> np.ndarray:
        model_to_use = MULTIMODAL_EMBEDDING_MODEL if modality != "text" else EMBEDDING_MODEL
        response = genai.embed_content(
            model=model_to_use,
            content=content,
            task_type="retrieval_document"
        )
        return np.array(response['embedding'])

    @retry(wait=wait_exponential(multiplier=1, min=2, max=15), stop=stop_after_attempt(3))
    def get_teacher_score(self, query: str, document: str, modality: str = "text") -> float:
        prompt = f\"\"\"You are an expert search and relevance ranking system.
Evaluate the relevance of the following Document to the provided Query.
Output EXACTLY a JSON payload with a single key 'relevance_score' containing a float between 0.0 (completely irrelevant) and 1.0 (highly relevant).
Query: {query}
Document: {document}\"\"\"
        response = self.teacher_client.generate_content(
            prompt,
            generation_config=genai.GenerationConfig(
                response_mime_type="application/json",
                response_schema=RelevanceOutput,
            )
        )
        try:
            result = json.loads(response.text)
            return float(result.get("relevance_score", 0.0))
        except Exception as e:
            logger.warning(f"Failed to parse teacher score, return 0.0. Error: {e}")
            return 0.0

miner = GeminiMiner()
"""

code4 = """# --- Checkpointing & Dataset Loading ---
RAW_DATASET_PATH = "/kaggle/input/raw-gemma4-dataset/raw_dataset.jsonl" 
OUTPUT_PATH = "/kaggle/working/stage1_train.jsonl"

def get_processed_count(filepath: str) -> int:
    if not os.path.exists(filepath):
        return 0
    with open(filepath, 'r', encoding='utf-8') as f:
        return sum(1 for _ in f)

def load_raw_data(filepath: str):
    if not os.path.exists(filepath):
        logger.info(f"Using mock data as {filepath} not found.")
        return [{"query": f"Query {i}", "positive": f"Pos {i}", "corpus": [f"Neg {j}" for j in range(15)], "modality": "text"} for i in range(10)]
    with open(filepath, 'r', encoding='utf-8') as f:
        return [json.loads(line) for line in f]

raw_data = load_raw_data(RAW_DATASET_PATH)
processed_count = get_processed_count(OUTPUT_PATH)
logger.info(f"Loaded {len(raw_data)} records. Resuming from {processed_count}...")
"""

code5 = """# --- Main Execution Loop (Append Mode) ---
with open(OUTPUT_PATH, 'a', encoding='utf-8') as out_file:
    for item in tqdm(raw_data[processed_count:], desc="Processing Data"):
        query = item['query']
        positive = item['positive']
        corpus = item['corpus']
        modality = item.get('modality', 'text')
        
        try:
            q_emb = miner.embed_content(query, modality)
            
            corpus_embs = np.array([miner.embed_content(doc, modality) for doc in corpus])
            scores = np.dot(corpus_embs, q_emb)
            ranked_indices = np.argsort(scores)[::-1]
            
            start_idx = min(10, len(ranked_indices))
            end_idx = min(50, len(ranked_indices))
            pool_indices = ranked_indices[start_idx:end_idx] if start_idx != end_idx else ranked_indices
            
            num_neg= min(7, len(pool_indices))
            selected_neg_indices = np.random.choice(pool_indices, size=num_neg, replace=False)
            negatives = [corpus[i] for i in selected_neg_indices]
            
            t_pos_score = miner.get_teacher_score(query, positive, modality)
            t_neg_scores = [miner.get_teacher_score(query, n, modality) for n in negatives]
            
            processed_record = {
                "query": query,
                "positive": positive,
                "negatives": negatives,
                "modality": modality,
                "teacher_pos_score": t_pos_score,
                "teacher_neg_scores": t_neg_scores
            }
            
            out_file.write(json.dumps(processed_record) + "\\n")
            out_file.flush() 
            
        except Exception as e:
            logger.error(f"Failed query '{query}': {e}")
            continue

logger.info(f"Data preparation complete. Output saved to {OUTPUT_PATH}")
"""
nb["cells"] = [
    nbf.v4.new_markdown_cell("# Kaggle Stage 1: Data Preparation"),
    nbf.v4.new_code_cell(code1),
    nbf.v4.new_code_cell(code2),
    nbf.v4.new_code_cell(code3),
    nbf.v4.new_code_cell(code4),
    nbf.v4.new_code_cell(code5),
]

with open(
    "/home/nqminh/projects/modalcom-ai-workers/training/gemma4_reranker/deploy_prep/kaggle_data_prep_stage1.ipynb",
    "w",
) as f:
    nbf.write(nb, f)
