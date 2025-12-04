from unsloth import FastLanguageModel
from transformers import TextStreamer

run_dir = "outputs/kanana_law_sft/20251203_203519_kanana-1.5-8b-instruct-2505"

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=run_dir,
    max_seq_length=2048,
    load_in_4bit=True,
    dtype=None,
)

FastLanguageModel.for_inference(model)  # kv cache 설정

messages = [
    {"role": "system", "content": "당신은 국토안전관리 관련 법령 상담을 도와주는 AI입니다."},
    {"role": "user", "content": "공중화장실 신축할 때 토지보상법상 공익사업에 해당하는지 알려줘."},
]

prompt = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
)

inputs = tokenizer(
    [prompt],
    return_tensors="pt",
).to(model.device)

streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

_ = model.generate(
    **inputs,
    max_new_tokens=512,
    do_sample=True,
    top_p=0.9,
    temperature=0.7,
    streamer=streamer,
)
