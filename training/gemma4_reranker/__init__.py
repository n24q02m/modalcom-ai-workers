"""Gemma-4-E4B Multimodal Reranker training package.

3-stage incremental fine-tuning pipeline:
  Stage 1: Text + Image (base capability)
  Stage 2: + Audio (with experience replay)
  Stage 3: + Video (with experience replay)

Target: Kaggle T4 16GB, QLoRA NF4, FP16 precision.
"""
