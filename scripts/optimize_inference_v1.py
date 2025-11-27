#!/usr/bin/env python3
"""
추론 시간 최적화 1단계: max_new_tokens 조정

목표: 30초 → 15-20초

방법:
- max_new_tokens: 256 → 128
- temperature 최적화
- JSON 파싱률 유지 확인

실행:
python scripts/optimize_inference_v1.py
"""

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


def load_model(adapter_dir: str, base_model: str = "Qwen/Qwen2.5-14B-Instruct"):
    """모델 로드 (기존과 동일)"""
    print(f"📥 Loading model from {adapter_dir}...")
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
        torch_dtype=torch.bfloat16,
    )
    model = PeftModel.from_pretrained(base, adapter_dir)
    tok = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    tok.pad_token = tok.eos_token
    print("✅ Model loaded!")
    return model, tok


def run_benchmark(
    model,
    tokenizer,
    samples: List[Dict[str, Any]],
    max_new_tokens: int = 256,
    temperature: float = 0.0,
    top_p: float = 1.0,
    debug: bool = False
) -> Dict[str, Any]:
    """
    벤치마크 실행

    Args:
        max_new_tokens: 생성할 최대 토큰 수 (256 → 128로 줄이기)
        temperature: 0.0 (결정적), 0.7 (약간 랜덤)
        top_p: nucleus sampling (1.0 = 비활성화)
    """
    device = next(model.parameters()).device

    # 강화된 시스템 프롬프트
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

    results = []
    json_success = 0
    inference_times = []
    confidences = []

    print(f"\n🔬 Running benchmark with max_new_tokens={max_new_tokens}, temperature={temperature}, top_p={top_p}")
    print(f"📊 Testing {len(samples)} samples...\n")

    for idx, sample in enumerate(samples):
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"{sample['instruction']}\n\nInput: {sample['input']}\n\nOutput JSON:"},
        ]
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(prompt, return_tensors="pt").to(device)

        # 추론 시작
        start = time.time()
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature if temperature > 0 else None,
                top_p=top_p if temperature > 0 else None,
                do_sample=(temperature > 0),
                eos_token_id=tokenizer.eos_token_id,
            )
        dt_ms = (time.time() - start) * 1000.0
        inference_times.append(dt_ms)

        # 생성된 토큰만 디코드
        generated_ids = outputs[0][len(inputs.input_ids[0]):]
        text = tokenizer.decode(generated_ids, skip_special_tokens=True)

        # JSON 파싱
        parsed = extract_json_robust(text, debug=debug)

        success = parsed is not None
        if success:
            json_success += 1
            # confidence 수집
            phys = parsed.get("physical_analysis", {})
            if isinstance(phys, dict) and "confidence" in phys:
                confidences.append(float(phys["confidence"]))

        results.append({
            "sample_idx": idx,
            "input": sample["input"][:50] + "...",
            "inference_time_ms": dt_ms,
            "json_parsed": success,
        })

        # 진행 상황 출력
        status = "✅" if success else "❌"
        print(f"  [{idx+1}/{len(samples)}] {status} {dt_ms:>7.0f}ms - {sample['input'][:40]}...")

        if not success and debug:
            print(f"      [DEBUG] Response: {text[:200]}...")

    # 통계 계산
    n = max(1, len(samples))
    avg_time = sum(inference_times) / len(inference_times) if inference_times else 0.0
    min_time = min(inference_times) if inference_times else 0.0
    max_time = max(inference_times) if inference_times else 0.0

    return {
        "config": {
            "max_new_tokens": max_new_tokens,
            "temperature": temperature,
            "top_p": top_p,
        },
        "metrics": {
            "total_samples": n,
            "json_parse_success": json_success,
            "json_parse_rate_%": round(json_success / n * 100.0, 2),
            "avg_inference_ms": round(avg_time, 0),
            "min_inference_ms": round(min_time, 0),
            "max_inference_ms": round(max_time, 0),
            "avg_confidence": round(sum(confidences) / len(confidences), 3) if confidences else 0.0,
        },
        "details": results,
    }


def main():
    import argparse

    parser = argparse.ArgumentParser(description="추론 시간 최적화 벤치마크")
    parser.add_argument("--adapter_dir", default="./droid-physics-qwen14b-qlora", help="LoRA adapter 디렉토리")
    parser.add_argument("--test_path", default="./droid_physics_llm_test_alpaca_v2.json", help="테스트 데이터 경로")
    parser.add_argument("--limit", type=int, default=10, help="테스트 샘플 수 (기본: 10)")
    parser.add_argument("--debug", action="store_true", help="디버그 모드")
    parser.add_argument("--output", default="optimization_results_v1.json", help="결과 저장 경로")
    args = parser.parse_args()

    # 모델 로드
    model, tokenizer = load_model(args.adapter_dir)
    model.eval()

    # 테스트 데이터 로드
    print(f"📂 Loading test data from {args.test_path}...")
    with open(args.test_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    samples = data[:args.limit]
    print(f"✅ Loaded {len(samples)} test samples\n")

    # ============================================
    # Baseline: max_new_tokens=256 (현재)
    # ============================================
    print("=" * 70)
    print("📊 BASELINE: max_new_tokens=256 (현재 설정)")
    print("=" * 70)
    baseline_results = run_benchmark(
        model, tokenizer, samples,
        max_new_tokens=256,
        temperature=0.0,
        top_p=1.0,
        debug=args.debug
    )

    print("\n" + "=" * 70)
    print("📈 BASELINE 결과:")
    print("=" * 70)
    print(json.dumps(baseline_results["metrics"], indent=2, ensure_ascii=False))

    # ============================================
    # 최적화 1: max_new_tokens=128
    # ============================================
    print("\n\n" + "=" * 70)
    print("🚀 OPTIMIZED: max_new_tokens=128 (50% 감소)")
    print("=" * 70)
    optimized_results = run_benchmark(
        model, tokenizer, samples,
        max_new_tokens=128,
        temperature=0.0,
        top_p=1.0,
        debug=args.debug
    )

    print("\n" + "=" * 70)
    print("📈 OPTIMIZED 결과:")
    print("=" * 70)
    print(json.dumps(optimized_results["metrics"], indent=2, ensure_ascii=False))

    # ============================================
    # 비교 분석
    # ============================================
    print("\n\n" + "=" * 70)
    print("📊 비교 분석")
    print("=" * 70)

    baseline_time = baseline_results["metrics"]["avg_inference_ms"]
    optimized_time = optimized_results["metrics"]["avg_inference_ms"]
    speedup = baseline_time / optimized_time if optimized_time > 0 else 0
    time_saved = baseline_time - optimized_time

    baseline_parse = baseline_results["metrics"]["json_parse_rate_%"]
    optimized_parse = optimized_results["metrics"]["json_parse_rate_%"]
    parse_diff = optimized_parse - baseline_parse

    print(f"""
추론 시간:
  Baseline:   {baseline_time:>7.0f}ms
  Optimized:  {optimized_time:>7.0f}ms
  개선:       {time_saved:>7.0f}ms ({speedup:.2f}x faster) {'✅' if speedup >= 1.5 else '⚠️'}

JSON 파싱률:
  Baseline:   {baseline_parse:>6.1f}%
  Optimized:  {optimized_parse:>6.1f}%
  차이:       {parse_diff:>+6.1f}%p {'✅' if parse_diff >= -5 else '❌'}

신뢰도:
  Baseline:   {baseline_results["metrics"]["avg_confidence"]:.3f}
  Optimized:  {optimized_results["metrics"]["avg_confidence"]:.3f}

목표 달성:
  목표 추론 시간: <200ms
  달성 여부:      {'✅ 달성!' if optimized_time < 200 else f'❌ 미달 ({optimized_time:.0f}ms)'}

  목표 파싱률: ≥95%
  달성 여부:   {'✅ 달성!' if optimized_parse >= 95 else f'⚠️ {optimized_parse:.1f}%'}
    """)

    # 결과 저장
    full_results = {
        "baseline": baseline_results,
        "optimized": optimized_results,
        "comparison": {
            "speedup_factor": round(speedup, 2),
            "time_saved_ms": round(time_saved, 0),
            "parse_rate_diff_%p": round(parse_diff, 2),
        }
    }

    output_path = args.output
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(full_results, f, indent=2, ensure_ascii=False)

    print(f"\n💾 전체 결과 저장: {output_path}")
    print("=" * 70)

    # 최종 추천
    print("\n🎯 최종 추천:")
    if speedup >= 1.5 and parse_diff >= -5:
        print("  ✅ max_new_tokens=128 사용 권장!")
        print(f"  - 추론 시간 {speedup:.1f}배 빨라짐")
        print(f"  - JSON 파싱률 {optimized_parse:.1f}% 유지")
    elif speedup >= 1.3:
        print("  ⚠️  max_new_tokens=128 사용 고려")
        print(f"  - 추론 시간 개선: {speedup:.1f}배")
        print(f"  - 파싱률 저하: {parse_diff:+.1f}%p")
    else:
        print("  ❌ max_new_tokens=256 유지 권장")
        print("  - 개선 효과 미미")

    print("\n✅ 최적화 벤치마크 완료!")


if __name__ == "__main__":
    main()
