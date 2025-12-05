
CUDA_VISIBLE_DEVICES=0 python scripts/eval_generation_metrics.py \
  --run_dir outputs/kanana_law_sft/20251204_193908_kanana-1.5-8b-instruct-2505 \
  --use_rag \
  --rag_top_k 5 \
  --rag_max_context_chars 1500 \
  --use_bertscore \
  --use_meteor