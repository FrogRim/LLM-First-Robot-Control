#!/usr/bin/env python3
"""
Baseline 비교 실험: Fine-tuned vs Base Model
"""

import json
import time
import sys
from pathlib import Path
from typing import List, Dict, Any, Tuple
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
import torch

# Import evaluation functions
sys.path.insert(0, str(Path(__file__).parent))
from json_sanitizer import extract_json_robust


def load_test_data(path: str, limit: int = None) -> List[Dict[str, Any]]:
    """테스트 데이터 로드"""
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data[:limit] if limit else data


def evaluate_model(
    model,
    tokenizer,
    samples: List[Dict[str, Any]],
    max_new_tokens: int = 256,
    model_name: str = "Model"
) -> Dict[str, Any]:
    """
    모델 평가 함수
    """
    print(f"\n{'='*60}")
    print(f"평가 중: {model_name}")
    print(f"{'='*60}")
    
    json_success = 0
    material_correct = 0
    inference_times: List[float] = []
    confidences: List[float] = []
    results_detail = []
    
    device = next(model.parameters()).device
    
    # 동일한 시스템 프롬프트 사용
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
        
        # 생성된 토큰만 디코드
        generated_ids = outputs[0][len(inputs.input_ids[0]):]
        text = tokenizer.decode(generated_ids, skip_special_tokens=True)
        
        # JSON 파싱
        parsed = extract_json_robust(text, debug=False)
        
        # Ground truth 재료 추출 (출력에서)
        gt_material = None
        if 'output' in sample:
            try:
                gt_json = json.loads(sample['output'])
                gt_material = gt_json.get('physical_analysis', {}).get('material_inference', '')
            except:
                pass
        
        # 결과 기록
        is_json_valid = parsed is not None
        is_material_correct = False
        predicted_material = None
        
        if parsed:
            json_success += 1
            phys = parsed.get("physical_analysis", {})
            if isinstance(phys, dict):
                if "confidence" in phys:
                    confidences.append(float(phys["confidence"]))
                predicted_material = phys.get("material_inference", "")
                
                # 재료 정확도 체크
                if gt_material and predicted_material:
                    # 대소문자 구분 없이 비교
                    if predicted_material.lower() == gt_material.lower():
                        is_material_correct = True
                        material_correct += 1
        
        results_detail.append({
            "sample_idx": idx,
            "instruction": sample['instruction'][:60] + "...",
            "input": sample['input'][:40] + "...",
            "json_valid": is_json_valid,
            "predicted_material": predicted_material,
            "gt_material": gt_material,
            "material_correct": is_material_correct,
            "inference_time_ms": dt_ms
        })
    
    print()  # 줄바꿈
    
    n = len(samples)
    return {
        "model_name": model_name,
        "total_samples": n,
        "json_parse_success": json_success,
        "json_parse_rate_percent": (json_success / n) * 100.0,
        "material_accuracy_percent": (material_correct / n) * 100.0,
        "avg_inference_time_ms": sum(inference_times) / n,
        "min_inference_time_ms": min(inference_times),
        "max_inference_time_ms": max(inference_times),
        "avg_confidence": sum(confidences) / len(confidences) if confidences else 0.0,
        "details": results_detail
    }


def load_base_model(base_model_name: str) -> Tuple[Any, Any]:
    """Base model 로드 (Fine-tuning 없음)"""
    print(f"🔧 Base Model 로딩 중: {base_model_name}")
    
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    
    model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        device_map="auto",
        trust_remote_code=True,
        quantization_config=bnb_config,
        torch_dtype=torch.bfloat16,
    )
    
    tokenizer = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    
    print("✅ Base Model 로딩 완료\n")
    return model, tokenizer


def load_finetuned_model(base_model_name: str, adapter_dir: str) -> Tuple[Any, Any]:
    """Fine-tuned model 로드 (LoRA 어댑터 포함)"""
    print(f"🔧 Fine-tuned Model 로딩 중: {adapter_dir}")
    
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        device_map="auto",
        trust_remote_code=True,
        quantization_config=bnb_config,
        torch_dtype=torch.bfloat16,
    )
    
    model = PeftModel.from_pretrained(base_model, adapter_dir)
    tokenizer = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    
    print("✅ Fine-tuned Model 로딩 완료\n")
    return model, tokenizer


def compare_results(base_results: Dict, finetuned_results: Dict) -> Dict:
    """두 모델 결과 비교"""
    print("\n" + "=" * 60)
    print("📊 Baseline 비교 결과")
    print("=" * 60)
    
    comparison = {
        "base_model": base_results,
        "finetuned_model": finetuned_results,
        "improvements": {}
    }
    
    # 개선율 계산
    json_improvement = finetuned_results["json_parse_rate_percent"] - base_results["json_parse_rate_percent"]
    material_improvement = finetuned_results["material_accuracy_percent"] - base_results["material_accuracy_percent"]
    time_ratio = finetuned_results["avg_inference_time_ms"] / base_results["avg_inference_time_ms"]
    
    comparison["improvements"] = {
        "json_parse_rate_improvement_pp": json_improvement,
        "material_accuracy_improvement_pp": material_improvement,
        "inference_time_ratio": time_ratio,
        "inference_time_change_percent": (time_ratio - 1) * 100
    }
    
    # 출력
    print("\n【Base Model (Fine-tuning 없음)】")
    print(f"  JSON 파싱률: {base_results['json_parse_rate_percent']:.1f}%")
    print(f"  재료 인식 정확도: {base_results['material_accuracy_percent']:.1f}%")
    print(f"  평균 추론 시간: {base_results['avg_inference_time_ms']:.1f}ms")
    print(f"  평균 신뢰도: {base_results['avg_confidence']:.3f}")
    
    print("\n【Fine-tuned Model (QLoRA)】")
    print(f"  JSON 파싱률: {finetuned_results['json_parse_rate_percent']:.1f}%")
    print(f"  재료 인식 정확도: {finetuned_results['material_accuracy_percent']:.1f}%")
    print(f"  평균 추론 시간: {finetuned_results['avg_inference_time_ms']:.1f}ms")
    print(f"  평균 신뢰도: {finetuned_results['avg_confidence']:.3f}")
    
    print("\n【개선율】")
    print(f"  JSON 파싱률: {json_improvement:+.1f}%p")
    print(f"  재료 인식 정확도: {material_improvement:+.1f}%p")
    print(f"  추론 시간: {(time_ratio-1)*100:+.1f}% (비율: {time_ratio:.2f}x)")
    
    if json_improvement > 0:
        print(f"\n✅ Fine-tuning으로 JSON 파싱률 {json_improvement:.1f}%p 향상!")
    if material_improvement > 0:
        print(f"✅ Fine-tuning으로 재료 인식 정확도 {material_improvement:.1f}%p 향상!")
    
    print("\n" + "=" * 60)
    
    return comparison


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Baseline 비교 실험")
    parser.add_argument("--base_model", default="Qwen/Qwen2.5-14B-Instruct", help="베이스 모델")
    parser.add_argument("--adapter_dir", default="./droid-physics-qwen14b-qlora", help="LoRA 어댑터")
    parser.add_argument("--test_data", default="./droid_physics_llm_test_alpaca_v2.json", help="테스트 데이터")
    parser.add_argument("--limit", type=int, default=30, help="평가 샘플 수")
    parser.add_argument("--max_new_tokens", type=int, default=256, help="생성 토큰 수")
    parser.add_argument("--skip_base", action="store_true", help="Base model 평가 건너뛰기")
    parser.add_argument("--output", default="./baseline_comparison_results.json", help="결과 저장")
    
    args = parser.parse_args()
    
    # 테스트 데이터 로드
    print(f"📂 테스트 데이터 로딩: {args.test_data}")
    test_samples = load_test_data(args.test_data, limit=args.limit)
    print(f"✅ {len(test_samples)}개 샘플 로드 완료\n")
    
    results = {}
    
    # Base Model 평가
    if not args.skip_base:
        base_model, base_tokenizer = load_base_model(args.base_model)
        base_results = evaluate_model(
            base_model,
            base_tokenizer,
            test_samples,
            max_new_tokens=args.max_new_tokens,
            model_name="Base Model (No Fine-tuning)"
        )
        results["base"] = base_results
        
        # 메모리 정리
        del base_model
        del base_tokenizer
        torch.cuda.empty_cache()
    else:
        print("⏭️  Base model 평가 건너뛰기 (--skip_base)\n")
        results["base"] = None
    
    # Fine-tuned Model 평가
    finetuned_model, finetuned_tokenizer = load_finetuned_model(args.base_model, args.adapter_dir)
    finetuned_results = evaluate_model(
        finetuned_model,
        finetuned_tokenizer,
        test_samples,
        max_new_tokens=args.max_new_tokens,
        model_name="Fine-tuned Model (QLoRA)"
    )
    results["finetuned"] = finetuned_results
    
    # 비교 (Base model 평가를 수행한 경우만)
    if not args.skip_base:
        comparison = compare_results(base_results, finetuned_results)
        results["comparison"] = comparison["improvements"]
    
    # 결과 저장
    output_path = Path(args.output)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"\n💾 결과 저장 완료: {output_path}\n")


if __name__ == "__main__":
    main()

