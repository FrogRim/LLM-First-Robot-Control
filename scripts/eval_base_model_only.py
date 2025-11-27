#!/usr/bin/env python3
"""
Base Model만 평가 (메모리 효율적)
"""

import json
import time
import sys
from pathlib import Path
from typing import List, Dict, Any
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch

sys.path.insert(0, str(Path(__file__).parent))
from json_sanitizer import extract_json_robust


def load_test_data(path: str, limit: int = None) -> List[Dict[str, Any]]:
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data[:limit] if limit else data


def evaluate_base_model(
    model,
    tokenizer,
    samples: List[Dict[str, Any]],
    max_new_tokens: int = 256
) -> Dict[str, Any]:
    print(f"\n{'='*60}")
    print(f"Base Model 평가 중 (Fine-tuning 없음)")
    print(f"{'='*60}")
    
    json_success = 0
    inference_times: List[float] = []
    confidences: List[float] = []
    results_detail = []
    
    device = next(model.parameters()).device
    
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

    for idx, sample in enumerate(samples):
        print(f"  [{idx+1}/{len(samples)}] 평가 중...", end='\r')
        
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
        
        generated_ids = outputs[0][len(inputs.input_ids[0]):]
        text = tokenizer.decode(generated_ids, skip_special_tokens=True)
        
        parsed = extract_json_robust(text, debug=False)
        
        is_json_valid = parsed is not None
        predicted_material = None
        
        if parsed:
            json_success += 1
            phys = parsed.get("physical_analysis", {})
            if isinstance(phys, dict):
                if "confidence" in phys:
                    confidences.append(float(phys["confidence"]))
                predicted_material = phys.get("material_inference", "")
        
        results_detail.append({
            "sample_idx": idx,
            "json_valid": is_json_valid,
            "predicted_material": predicted_material,
            "inference_time_ms": dt_ms,
            "raw_text_preview": text[:200] if not parsed else "JSON valid"
        })
    
    print()
    
    n = len(samples)
    return {
        "model_name": "Base Model (No Fine-tuning)",
        "total_samples": n,
        "json_parse_success": json_success,
        "json_parse_rate_percent": (json_success / n) * 100.0,
        "avg_inference_time_ms": sum(inference_times) / n,
        "min_inference_time_ms": min(inference_times),
        "max_inference_time_ms": max(inference_times),
        "avg_confidence": sum(confidences) / len(confidences) if confidences else 0.0,
        "details": results_detail
    }


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Base Model 평가")
    parser.add_argument("--base_model", default="Qwen/Qwen2.5-14B-Instruct")
    parser.add_argument("--test_data", default="./droid_physics_llm_test_alpaca_v2.json")
    parser.add_argument("--limit", type=int, default=30)
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--output", default="./base_model_results.json")
    
    args = parser.parse_args()
    
    print(f"📂 테스트 데이터 로딩: {args.test_data}")
    test_samples = load_test_data(args.test_data, limit=args.limit)
    print(f"✅ {len(test_samples)}개 샘플 로드 완료\n")
    
    print(f"🔧 Base Model 로딩 중: {args.base_model}")
    
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        device_map="auto",
        trust_remote_code=True,
        quantization_config=bnb_config,
        torch_dtype=torch.bfloat16,
    )
    
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    
    print("✅ Base Model 로딩 완료\n")
    
    results = evaluate_base_model(
        model=model,
        tokenizer=tokenizer,
        samples=test_samples,
        max_new_tokens=args.max_new_tokens
    )
    
    print(f"\n{'='*60}")
    print(f"📊 Base Model 결과")
    print(f"{'='*60}")
    print(f"  총 샘플: {results['total_samples']}개")
    print(f"  JSON 파싱 성공: {results['json_parse_success']}개")
    print(f"  JSON 파싱률: {results['json_parse_rate_percent']:.1f}%")
    print(f"  평균 추론 시간: {results['avg_inference_time_ms']:.1f}ms")
    print(f"  평균 신뢰도: {results['avg_confidence']:.3f}")
    print(f"{'='*60}\n")
    
    output_path = Path(args.output)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"💾 결과 저장 완료: {output_path}\n")


if __name__ == "__main__":
    main()

