#!/usr/bin/env python3
"""
일반화 테스트 - Base Model 전용
재료명 없는 명령어로 암묵적 추론 능력 평가
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


# 재료명 없는 테스트 케이스
IMPLICIT_TEST_CASES = [
    {
        "instruction": "Pick up the transparent cup and place it on the tray",
        "description": "투명한 컵 (glass 추론 필요)",
        "expected_material": "glass",
        "keywords": ["transparent", "cup"]
    },
    {
        "instruction": "Grab the heavy shiny tool and put it in the box",
        "description": "무거운 반짝이는 도구 (metal 추론 필요)",
        "expected_material": "metal",
        "keywords": ["heavy", "shiny", "tool"]
    },
    {
        "instruction": "Carefully lift the fragile dish and set it on the shelf",
        "description": "깨지기 쉬운 접시 (ceramic 추론 필요)",
        "expected_material": "ceramic",
        "keywords": ["fragile", "dish"]
    },
    {
        "instruction": "Pick up the lightweight container quickly",
        "description": "가벼운 용기 (plastic 추론 필요)",
        "expected_material": "plastic",
        "keywords": ["lightweight", "container"]
    },
    {
        "instruction": "Gently grasp the soft cloth and move it aside",
        "description": "부드러운 천 (fabric 추론 필요)",
        "expected_material": "fabric",
        "keywords": ["soft", "cloth"]
    },
    {
        "instruction": "Grab the solid block and position it in the holder",
        "description": "단단한 블록 (wood 추론 필요)",
        "expected_material": "wood",
        "keywords": ["solid", "block"]
    },
    {
        "instruction": "Pick up the flexible tube carefully",
        "description": "유연한 튜브 (rubber 추론 필요)",
        "expected_material": "rubber",
        "keywords": ["flexible", "tube"]
    },
]


def evaluate_implicit_inference(model, tokenizer, test_cases: List[Dict], max_new_tokens: int = 128):
    """재료명 없는 암묵적 추론 평가"""
    
    print("\n" + "=" * 70)
    print("🔬 암묵적 재료 추론 테스트 (재료명 없음)")
    print("=" * 70)
    
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

    results = []
    correct_materials = 0
    
    for idx, test_case in enumerate(test_cases):
        print(f"\n[{idx+1}/{len(test_cases)}] {test_case['description']}")
        print(f"  명령어: \"{test_case['instruction']}\"")
        print(f"  키워드: {', '.join(test_case['keywords'])}")
        print(f"  예상 재료: {test_case['expected_material']}")
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"{test_case['instruction']}\n\nOutput JSON:"},
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
        inference_time = (time.time() - start) * 1000.0
        
        generated_ids = outputs[0][len(inputs.input_ids[0]):]
        text = tokenizer.decode(generated_ids, skip_special_tokens=True)
        
        parsed = extract_json_robust(text, debug=False)
        
        is_correct = False
        predicted_material = None
        confidence = 0.0
        
        if parsed:
            phys = parsed.get("physical_analysis", {})
            if isinstance(phys, dict):
                predicted_material = phys.get("material_inference", "").lower()
                confidence = phys.get("confidence", 0.0)
                expected_material = test_case["expected_material"].lower()
                
                # 유연한 매칭 (예: "glass" in "borosilicate glass")
                if expected_material in predicted_material or predicted_material in expected_material:
                    is_correct = True
                    correct_materials += 1
                    print(f"  ✅ 정답! 예측: '{predicted_material}' (신뢰도: {confidence:.2f})")
                else:
                    print(f"  ❌ 오답. 예측: '{predicted_material}', 정답: '{expected_material}' (신뢰도: {confidence:.2f})")
                
                # 물리 속성
                print(f"     질량: {phys.get('mass_category', 'N/A')}")
                print(f"     깨지기 쉬움: {phys.get('fragility', 'N/A')}")
        else:
            print(f"  ❌ JSON 파싱 실패")
        
        results.append({
            "test_case": test_case['description'],
            "instruction": test_case['instruction'],
            "keywords": test_case['keywords'],
            "expected_material": test_case['expected_material'],
            "predicted_material": predicted_material,
            "confidence": confidence,
            "is_correct": is_correct,
            "json_valid": parsed is not None,
            "inference_time_ms": inference_time,
        })
    
    accuracy = (correct_materials / len(test_cases)) * 100.0
    
    print("\n" + "=" * 70)
    print(f"📊 암묵적 추론 정확도: {correct_materials}/{len(test_cases)} = {accuracy:.1f}%")
    print("=" * 70)
    
    return {
        "test_type": "implicit_material_inference",
        "total_cases": len(test_cases),
        "correct_predictions": correct_materials,
        "accuracy_percent": accuracy,
        "avg_confidence": sum(r['confidence'] for r in results) / len(results),
        "results": results
    }


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Base Model 일반화 테스트")
    parser.add_argument("--base_model", default="Qwen/Qwen2.5-14B-Instruct")
    parser.add_argument("--max_new_tokens", type=int, default=128)
    parser.add_argument("--output", default="./base_generalization_results.json")
    
    args = parser.parse_args()
    
    print("🔧 Base Model 로딩 중...")
    print(f"   모델: {args.base_model}")
    
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
    
    print("✅ 모델 로딩 완료\n")
    
    # 테스트 실행
    results = evaluate_implicit_inference(
        model, tokenizer, IMPLICIT_TEST_CASES, max_new_tokens=args.max_new_tokens
    )
    
    # 결과 저장
    output_path = Path(args.output)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"\n💾 결과 저장 완료: {output_path}\n")


if __name__ == "__main__":
    main()

