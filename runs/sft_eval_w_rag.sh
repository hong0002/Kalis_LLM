CUDA_VISIBLE_DEVICES=1 python scripts/eval_generation_metrics.py \
  --run_dir outputs/kanana_law_sft/20251203_203519_kanana-1.5-8b-instruct-2505 \
  --use_rag \
  --rag_top_k 5 \
  --rag_max_context_chars 1500