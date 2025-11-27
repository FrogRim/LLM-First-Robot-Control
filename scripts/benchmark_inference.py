import json
import time
from typing import List
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
import torch


PROMPTS: List[str] = [
    "Pick up the plastic bottle and place it in the container",
    "Carefully grasp the glass cup and put it on the tray",
    "Grab the metal tool and set it in the box without tilting",
    "Lift the wooden block quickly and position it in the holder",
    "Gently take the ceramic mug and move it to the rack",
]


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


def run_benchmark(model, tokenizer, prompts: List[str], max_new_tokens: int = 96):
    device = next(model.parameters()).device
    times_ms: List[float] = []

    # 동일한 강화 프롬프트 사용
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
}"""

    for p in prompts:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": p + "\n\nOutput JSON:"},
        ]
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        start = time.time()
        with torch.no_grad():
            _ = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                eos_token_id=tokenizer.eos_token_id,
            )
        times_ms.append((time.time() - start) * 1000.0)

    return {
        "num_prompts": len(prompts),
        "avg_inference_ms": sum(times_ms) / len(times_ms) if times_ms else 0.0,
        "p50_ms": sorted(times_ms)[len(times_ms)//2] if times_ms else 0.0,
        "p90_ms": sorted(times_ms)[int(len(times_ms)*0.9)-1] if times_ms else 0.0,
    }


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter_dir", default="./droid-physics-qwen14b-qlora")
    args = parser.parse_args()

    model, tok = load_model(args.adapter_dir)
    model.eval()
    metrics = run_benchmark(model, tok, PROMPTS)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()



