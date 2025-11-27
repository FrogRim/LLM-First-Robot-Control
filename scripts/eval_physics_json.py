import json
import time
import sys
from pathlib import Path
from typing import List, Dict, Any
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
import torch

# Import JSON sanitizer
sys.path.insert(0, str(Path(__file__).parent))
from json_sanitizer import extract_json_robust


def load_test_data(path: str) -> List[Dict[str, Any]]:
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def evaluate_json_parse_rate(model, tokenizer, samples: List[Dict[str, Any]], max_new_tokens: int = 256, debug: bool = False) -> Dict[str, Any]:
    json_success = 0
    inference_times: List[float] = []
    confidences: List[float] = []

    device = next(model.parameters()).device

    # 강화된 시스템 프롬프트: 스키마 + 예시 포함
    system_prompt = """You are a physics-aware robot control system. Output ONLY valid JSON without code fences, explanations, or extra text.

Required JSON schema:
{
  "physical_analysis": {
    "material_inference": "string",
    "mass_category": "string",
    "friction_coefficient": "string",
    "fragility": "string",
    "stiffness": "string",
    "confidence": 0.85
  },
  "control_parameters": {
    "grip_force": 0.5,
    "lift_speed": 0.5,
    "approach_angle": 0.0,
    "contact_force": 0.3,
    "safety_margin": 0.8
  },
  "reasoning": "string",
  "affordance_assessment": {
    "success_probability": 0.9,
    "risk_factors": [],
    "recommended_approach": "string"
  }
}

Example output:
{"physical_analysis": {"material_inference": "plastic", "mass_category": "light", "friction_coefficient": "0.4-0.6 estimated", "fragility": "normal", "stiffness": "medium", "confidence": 0.85}, "control_parameters": {"grip_force": 0.5, "lift_speed": 0.6, "approach_angle": 0.0, "contact_force": 0.3, "safety_margin": 0.8}, "reasoning": "Lightweight plastic requires moderate grip.", "affordance_assessment": {"success_probability": 0.9, "risk_factors": [], "recommended_approach": "standard_confident_approach"}}"""

    for idx, sample in enumerate(samples):
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"{sample['instruction']}\n\nInput: {sample['input']}\n\nOutput JSON:"},
        ]
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(prompt, return_tensors="pt").to(device)

        start = time.time()
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                eos_token_id=tokenizer.eos_token_id,
            )
        dt_ms = (time.time() - start) * 1000.0
        inference_times.append(dt_ms)

        # 생성된 토큰만 디코드 (입력 프롬프트 제외)
        generated_ids = outputs[0][len(inputs.input_ids[0]):]
        text = tokenizer.decode(generated_ids, skip_special_tokens=True)
        
        # 견고한 JSON 추출 사용
        parsed = extract_json_robust(text, debug=debug)
        
        if parsed:
            json_success += 1
            # confidence가 있으면 수집
            phys = parsed.get("physical_analysis", {})
            if isinstance(phys, dict) and "confidence" in phys:
                confidences.append(float(phys["confidence"]))
        elif debug:
            print(f"\n[FAILED] Sample {idx}: {sample['input'][:50]}...")
            print(f"[FAILED] Response preview: {text[:300]}...")

    n = max(1, len(samples))
    return {
        "count": n,
        "json_parse_rate": json_success / n * 100.0,
        "avg_inference_ms": sum(inference_times) / len(inference_times) if inference_times else 0.0,
        "avg_confidence": sum(confidences) / len(confidences) if confidences else 0.0,
    }


def load_model(adapter_dir: str, base_model: str = "Qwen/Qwen2.5-14B-Instruct"):
    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    base = AutoModelForCausalLM.from_pretrained(
        base_model,
        device_map="auto",
        trust_remote_code=True,
        quantization_config=bnb,
        dtype=torch.bfloat16,
    )
    model = PeftModel.from_pretrained(base, adapter_dir)
    tok = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    tok.pad_token = tok.eos_token
    return model, tok


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter_dir", default="./droid-physics-qwen14b-qlora")
    parser.add_argument("--test_path", default="./droid_physics_llm_test_alpaca_v2.json")
    parser.add_argument("--limit", type=int, default=30)
    parser.add_argument("--debug", action="store_true", help="Enable debug output for failed parses")
    parser.add_argument("--max_new_tokens", type=int, default=256, help="Max tokens to generate")
    args = parser.parse_args()

    model, tokenizer = load_model(args.adapter_dir)
    model.eval()

    data = load_test_data(args.test_path)
    samples = data[: args.limit]

    metrics = evaluate_json_parse_rate(model, tokenizer, samples, max_new_tokens=args.max_new_tokens, debug=args.debug)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()



