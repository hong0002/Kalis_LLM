import torch
from typing import Tuple, Dict, Any, List
from unsloth import FastLanguageModel


def build_kanana_lora_model_and_tokenizer(
    model_cfg: Dict[str, Any],
    lora_cfg: Dict[str, Any],
    training_cfg: Dict[str, Any],
):
    """
    Kanana 1.5 8B instruct + Unsloth + LoRA 초기화 함수
    """
    model_name = model_cfg["name"]
    max_seq_length = int(model_cfg.get("max_seq_length", 2048))
    load_in_4bit = bool(model_cfg.get("load_in_4bit", True))

    dtype_str = model_cfg.get("dtype", None)
    if dtype_str is None:
        dtype = None
    else:
        dtype_str = str(dtype_str).lower()
        if "bf16" in dtype_str:
            dtype = torch.bfloat16
        elif "16" in dtype_str:
            dtype = torch.float16
        else:
            dtype = None

    print(f"[Model] Loading base model: {model_name}")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=max_seq_length,
        dtype=dtype,
        load_in_4bit=load_in_4bit,
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    target_modules: List[str] = lora_cfg["target_modules"]

    print("[Model] Applying LoRA adapters with Unsloth...")
    model = FastLanguageModel.get_peft_model(
        model,
        r=int(lora_cfg.get("r", 64)),
        lora_alpha=int(lora_cfg.get("alpha", 128)),
        lora_dropout=float(lora_cfg.get("dropout", 0.05)),
        target_modules=target_modules,
        use_gradient_checkpointing=bool(
            training_cfg.get("gradient_checkpointing", True)
        ),
        random_state=training_cfg.get("seed", 42),
    )

    # 학습 모드 설정 (generate patch 등)
    FastLanguageModel.for_training(model)

    return model, tokenizer
