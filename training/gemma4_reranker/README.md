# Gemma-4 Multilingual Multimodal Reranker (Training Package)

This package is the **training/fine-tuning pipeline only**.

- Training artifacts are produced under `training/gemma4_reranker/`
- Serving is handled by Modal worker: `src/ai_workers/workers/mm_reranker.py`
- `modalcom-ai-workers` should load merged checkpoints from HF Hub and serve inference APIs only.

## Production data flow

1. **Hard-negative mining output (grouped JSONL)**
   - one row = query + positive + negative list + teacher scores
2. **Dataset preparation** (`prepare_dataset.py`)
   - strict validation
   - grouped -> pointwise expansion
   - deterministic train/val split (query-level anti-leakage)
   - dedup + data quality report
3. **Training** (`run_train.py`)
   - stage-based LoRA/QLoRA training
   - optional merge/push after training
4. **Serving**
   - Modal worker loads merged model from HF Hub

## Canonical schema layers

### Grouped schema (`TrainSample`)
Used for hard-negative mining output.

### Pointwise schema (`PointwiseSample`)
Used by trainer loss:
- `label=1` => relevant
- `label=0` => irrelevant
- trainer internally maps to class ids (`yes`=0, `no`=1)

## Stage datasets

- Stage 1: `stage1_train.jsonl`, `stage1_val.jsonl` (text + image)
- Stage 2: `stage2_train.jsonl`, `stage2_val.jsonl` (audio + replay)
- Stage 3: `stage3_train.jsonl`, `stage3_val.jsonl` (video + replay)

## Minimal execution

1) Prepare pointwise train/val:

```bash
python -m training.gemma4_reranker.prepare_dataset \
  --input-grouped-jsonl /kaggle/working/stage1_train.jsonl \
  --output-dir /kaggle/working/data \
  --stage stage1
```

2) Train stage 1:

```bash
python -m training.gemma4_reranker.run_train \
  --stage stage1 \
  --train-jsonl /kaggle/working/data/stage1_train.jsonl \
  --val-jsonl /kaggle/working/data/stage1_val.jsonl \
  --output-dir /kaggle/working/checkpoints
```

3) Optional merge + push:

```bash
python -m training.gemma4_reranker.run_train \
  --stage stage1 \
  --train-jsonl /kaggle/working/data/stage1_train.jsonl \
  --val-jsonl /kaggle/working/data/stage1_val.jsonl \
  --merge-after-train \
  --push-merged
```

## Quality gates (recommended)

- Validation split is deterministic and query-group aware.
- Data quality report (`stage*_report.json`) must be reviewed before training:
  - class balance
  - modality/language/source distribution
  - teacher score coverage
- Keep `per_device_train_batch_size=1` for mixed multimodal batches unless all modality tensors are shape-compatible.
