#!/usr/bin/env python3
"""
일반화 테스트: 재료명 없는 명령어로 암묵적 추론 능력 평가
"""

import json
import time
import sys
from pathlib import Path
from typing import List, Dict, Any, Tuple
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
import torch

# Import JSON sanitizer
sys.path.insert(0, str(Path(__file__).parent))
from json_sanitizer import extract_json_robust


# 재료명 없는 테스트 케이스
IMPLICIT_TEST_CASES = [
    {
        "instruction": "Pick up the transparent cup and place it on the tray",
        "description": "투명한 컵 (glass 추론 필요)",
        "expected_material": "glass",
        "expected_properties": {
            "fragility": "high",
            "mass": "light",
        }
    },
    {
        "instruction": "Grab the heavy shiny tool and put it in the box",
        "description": "무거운 반짝이는 도구 (metal 추론 필요)",
        "expected_material": "metal",
        "expected_properties": {
            "mass": "heavy",
            "fragility": "low",
        }
    },
    {
        "instruction": "Carefully lift the fragile dish and set it on the shelf",
        "description": "깨지기 쉬운 접시 (ceramic 추론 필요)",
        "expected_material": "ceramic",
        "expected_properties": {
            "fragility": "high",
        }
    },
    {
        "instruction": "Pick up the lightweight container quickly",
        "description": "가벼운 용기 (plastic 추론 필요)",
        "expected_material": "plastic",
        "expected_properties": {
            "mass": "light",
        }
    },
    {
        "instruction": "Gently grasp the soft cloth and move it aside",
        "description": "부드러운 천 (fabric 추론 필요)",
        "expected_material": "fabric",
        "expected_properties": {
            "mass": "very_light",
            "stiffness": "soft",
        }
    },
    {
        "instruction": "Grab the solid block and position it in the holder",
        "description": "단단한 블록 (wood 추론 필요)",
        "expected_material": "wood",
        "expected_properties": {
            "stiffness": "hard",
        }
    },
    {
        "instruction": "Pick up the flexible tube carefully",
        "description": "유연한 튜브 (rubber 추론 필요)",
        "expected_material": "rubber",
        "expected_properties": {
            "stiffness": "flexible",
        }
    },
]


# 모호한 명령어 테스트
AMBIGUOUS_TEST_CASES = [
    {
        "instruction": "Pick up the object very carefully and slowly",
        "input": "Object: unknown material",
        "description": "'조심스럽게'와 '천천히' → 낮은 grip force, 느린 속도",
        "expected_behavior": {
            "grip_force": "low",
            "lift_speed": "slow",
            "safety_margin": "high",
        }
    },
    {
        "instruction": "Grab the item quickly and move it fast",
        "input": "Object: unknown material",
        "description": "'빠르게' → 높은 속도",
        "expected_behavior": {
            "lift_speed": "fast",
        }
    },
    {
        "instruction": "Handle the object with extra care as it might break",
        "input": "Object: unknown material",
        "description": "'깨질 수 있다' → fragile 인식, 높은 safety margin",
        "expected_behavior": {
            "fragility": "high",
            "safety_margin": "high",
            "grip_force": "low",
        }
    },
]


def evaluate_implicit_inference(
    model,
    tokenizer,
    test_cases: List[Dict[str, Any]],
    max_new_tokens: int = 256
) -> Dict[str, Any]:
    """
    재료명 없는 명령어로 암묵적 추론 능력 평가
    """
    print("\n" + "=" * 60)
    print("암묵적 추론 테스트: 재료명 없는 명령어")
    print("=" * 60)
    
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
        print(f"\n[{idx+1}/{len(test_cases)}] 테스트: {test_case['description']}")
        print(f"  명령어: \"{test_case['instruction']}\"")
        
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
        
        # 생성된 토큰만 디코드
        generated_ids = outputs[0][len(inputs.input_ids[0]):]
        text = tokenizer.decode(generated_ids, skip_special_tokens=True)
        
        # JSON 파싱
        parsed = extract_json_robust(text, debug=False)
        
        is_correct = False
        predicted_material = None
        
        if parsed:
            phys = parsed.get("physical_analysis", {})
            if isinstance(phys, dict):
                predicted_material = phys.get("material_inference", "").lower()
                expected_material = test_case["expected_material"].lower()
                
                if predicted_material == expected_material:
                    is_correct = True
                    correct_materials += 1
                    print(f"  ✅ 정답! 예측: {predicted_material}")
                else:
                    print(f"  ❌ 오답. 예측: {predicted_material}, 정답: {expected_material}")
                
                # 물리 속성도 체크
                print(f"     - 질량: {phys.get('mass_category', 'N/A')}")
                print(f"     - 깨지기 쉬움: {phys.get('fragility', 'N/A')}")
                print(f"     - 강성: {phys.get('stiffness', 'N/A')}")
                print(f"     - 신뢰도: {phys.get('confidence', 0.0):.2f}")
        else:
            print(f"  ❌ JSON 파싱 실패")
        
        results.append({
            "test_case": test_case['description'],
            "instruction": test_case['instruction'],
            "expected_material": test_case['expected_material'],
            "predicted_material": predicted_material,
            "is_correct": is_correct,
            "json_valid": parsed is not None,
            "inference_time_ms": inference_time,
            "full_response": parsed if parsed else text[:200]
        })
    
    accuracy = (correct_materials / len(test_cases)) * 100.0
    
    print("\n" + "=" * 60)
    print(f"📊 암묵적 추론 정확도: {correct_materials}/{len(test_cases)} = {accuracy:.1f}%")
    print("=" * 60)
    
    return {
        "test_type": "implicit_material_inference",
        "total_cases": len(test_cases),
        "correct_predictions": correct_materials,
        "accuracy_percent": accuracy,
        "results": results
    }


def evaluate_ambiguous_commands(
    model,
    tokenizer,
    test_cases: List[Dict[str, Any]],
    max_new_tokens: int = 256
) -> Dict[str, Any]:
    """
    모호한 명령어로 맥락 이해 능력 평가
    """
    print("\n" + "=" * 60)
    print("맥락 이해 테스트: 모호한 명령어")
    print("=" * 60)
    
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
    behavior_matches = 0
    
    for idx, test_case in enumerate(test_cases):
        print(f"\n[{idx+1}/{len(test_cases)}] 테스트: {test_case['description']}")
        print(f"  명령어: \"{test_case['instruction']}\"")
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"{test_case['instruction']}\n\nInput: {test_case['input']}\n\nOutput JSON:"},
        ]
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                eos_token_id=tokenizer.eos_token_id,
            )
        
        generated_ids = outputs[0][len(inputs.input_ids[0]):]
        text = tokenizer.decode(generated_ids, skip_special_tokens=True)
        parsed = extract_json_robust(text, debug=False)
        
        behavior_correct = False
        
        if parsed:
            ctrl = parsed.get("control_parameters", {})
            phys = parsed.get("physical_analysis", {})
            
            # 예상 행동 체크
            expected = test_case["expected_behavior"]
            checks = []
            
            if "grip_force" in expected:
                grip = ctrl.get("grip_force", 0.5)
                if expected["grip_force"] == "low" and grip < 0.3:
                    checks.append(True)
                    print(f"  ✅ Grip force 낮음: {grip:.2f}")
                else:
                    checks.append(False)
                    print(f"  ❌ Grip force: {grip:.2f}")
            
            if "lift_speed" in expected:
                speed = ctrl.get("lift_speed", 0.5)
                if expected["lift_speed"] == "slow" and speed < 0.3:
                    checks.append(True)
                    print(f"  ✅ Lift speed 느림: {speed:.2f}")
                elif expected["lift_speed"] == "fast" and speed > 0.7:
                    checks.append(True)
                    print(f"  ✅ Lift speed 빠름: {speed:.2f}")
                else:
                    checks.append(False)
                    print(f"  ❌ Lift speed: {speed:.2f}")
            
            if "safety_margin" in expected:
                margin = ctrl.get("safety_margin", 0.8)
                if expected["safety_margin"] == "high" and margin > 1.2:
                    checks.append(True)
                    print(f"  ✅ Safety margin 높음: {margin:.2f}")
                else:
                    checks.append(False)
                    print(f"  ❌ Safety margin: {margin:.2f}")
            
            if "fragility" in expected:
                frag = phys.get("fragility", "").lower()
                if "high" in expected["fragility"].lower() and ("high" in frag or "fragile" in frag):
                    checks.append(True)
                    print(f"  ✅ Fragility 인식: {frag}")
                else:
                    checks.append(False)
                    print(f"  ❌ Fragility: {frag}")
            
            # 모든 체크가 통과하면 정답
            if checks and all(checks):
                behavior_correct = True
                behavior_matches += 1
        else:
            print(f"  ❌ JSON 파싱 실패")
        
        results.append({
            "test_case": test_case['description'],
            "instruction": test_case['instruction'],
            "expected_behavior": test_case['expected_behavior'],
            "behavior_correct": behavior_correct,
            "json_valid": parsed is not None,
            "full_response": parsed if parsed else text[:200]
        })
    
    accuracy = (behavior_matches / len(test_cases)) * 100.0
    
    print("\n" + "=" * 60)
    print(f"📊 맥락 이해 정확도: {behavior_matches}/{len(test_cases)} = {accuracy:.1f}%")
    print("=" * 60)
    
    return {
        "test_type": "ambiguous_command_understanding",
        "total_cases": len(test_cases),
        "correct_behaviors": behavior_matches,
        "accuracy_percent": accuracy,
        "results": results
    }


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="일반화 테스트")
    parser.add_argument("--adapter_dir", default="./droid-physics-qwen14b-qlora", help="LoRA 어댑터")
    parser.add_argument("--base_model", default="Qwen/Qwen2.5-14B-Instruct", help="베이스 모델")
    parser.add_argument("--max_new_tokens", type=int, default=256, help="생성 토큰 수")
    parser.add_argument("--test_type", choices=["implicit", "ambiguous", "all"], default="all", help="테스트 타입")
    parser.add_argument("--output", default="./generalization_test_results.json", help="결과 저장")
    
    args = parser.parse_args()
    
    # 모델 로드
    print("🔧 모델 로딩 중...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    
    base_model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        device_map="auto",
        trust_remote_code=True,
        quantization_config=bnb_config,
        torch_dtype=torch.bfloat16,
    )
    
    model = PeftModel.from_pretrained(base_model, args.adapter_dir)
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    
    print("✅ 모델 로딩 완료\n")
    
    all_results = {}
    
    # 암묵적 추론 테스트
    if args.test_type in ["implicit", "all"]:
        implicit_results = evaluate_implicit_inference(
            model, tokenizer, IMPLICIT_TEST_CASES, max_new_tokens=args.max_new_tokens
        )
        all_results["implicit_inference"] = implicit_results
    
    # 모호한 명령어 테스트
    if args.test_type in ["ambiguous", "all"]:
        ambiguous_results = evaluate_ambiguous_commands(
            model, tokenizer, AMBIGUOUS_TEST_CASES, max_new_tokens=args.max_new_tokens
        )
        all_results["ambiguous_commands"] = ambiguous_results
    
    # 결과 저장
    output_path = Path(args.output)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    
    print(f"\n💾 결과 저장 완료: {output_path}\n")


if __name__ == "__main__":
    main()

