from typing import Dict, Any, Optional
from trl import SFTTrainer, SFTConfig


def create_sft_trainer(
    model,
    tokenizer,
    train_dataset,
    eval_dataset: Optional[object],
    model_cfg: Dict[str, Any],
    training_cfg: Dict[str, Any],
):
    max_seq_length = int(model_cfg.get("max_seq_length", 2048))
    packing = bool(training_cfg.get("packing", False))

    def formatting_prompts_func(examples):
        """
        examples["messages"] : 배치(list) 형태
        각 메시지 리스트에 대해 chat template 적용 → 문자열
        반환값: list[str]  (Unsloth 요구사항)
        """
        texts = []
        for messages in examples["messages"]:
            text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=False,
            )
            texts.append(text)
        return texts

    # TensorBoard / 평가 / best model 관련 설정들
    report_to = training_cfg.get("report_to", ["tensorboard"])
    logging_dir = training_cfg.get("logging_dir", None)
    evaluation_strategy = training_cfg.get("evaluation_strategy", "no")
    eval_steps = training_cfg.get("eval_steps", None)
    save_strategy = training_cfg.get("save_strategy", "steps")
    save_steps = training_cfg.get("save_steps", 500)
    save_total_limit = training_cfg.get("save_total_limit", 3)
    load_best_model_at_end = bool(training_cfg.get("load_best_model_at_end", True))
    metric_for_best_model = training_cfg.get("metric_for_best_model", "eval_loss")
    greater_is_better = bool(training_cfg.get("greater_is_better", False))

    sft_config = SFTConfig(
        output_dir=training_cfg["output_dir"],
        num_train_epochs=float(training_cfg.get("num_train_epochs", 3)),
        max_steps=int(training_cfg.get("max_steps", -1)),

        per_device_train_batch_size=int(
            training_cfg.get("per_device_train_batch_size", 1)
        ),
        gradient_accumulation_steps=int(
            training_cfg.get("gradient_accumulation_steps", 8)
        ),
        learning_rate=float(training_cfg.get("learning_rate", 1e-4)),
        lr_scheduler_type=training_cfg.get("lr_scheduler_type", "cosine"),
        warmup_ratio=float(training_cfg.get("warmup_ratio", 0.03)),
        weight_decay=float(training_cfg.get("weight_decay", 0.01)),

        logging_steps=int(training_cfg.get("logging_steps", 10)),
        logging_dir=logging_dir,

        eval_strategy=evaluation_strategy,
        eval_steps=eval_steps,

        save_strategy=save_strategy,
        save_steps=save_steps,
        save_total_limit=save_total_limit,

        load_best_model_at_end=load_best_model_at_end,
        metric_for_best_model=metric_for_best_model,
        greater_is_better=greater_is_better,

        max_seq_length=max_seq_length,
        packing=packing,
        bf16=bool(training_cfg.get("bf16", True)),
        gradient_checkpointing=bool(
            training_cfg.get("gradient_checkpointing", True)
        ),
        report_to=report_to,

        dataset_text_field=None,  # formatting_func 사용
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        formatting_func=formatting_prompts_func,
        args=sft_config,
    )

    return trainer
