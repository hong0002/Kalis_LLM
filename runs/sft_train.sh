export PYTHONPATH=$(pwd)   # 현재 디렉터리를 파이썬 모듈 경로에 추가

CUDA_VISIBLE_DEVICES=0 python3 scripts/train_sft.py --config configs/law_sft_kanana.yaml