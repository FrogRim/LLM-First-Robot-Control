#!/usr/bin/env python3
"""
Genesis AI 안정성 테스트

개선 사항:
1. show_viewer=False (다중 씬 충돌 방지)
2. 재료별 최적 형상 선택 (Box/Sphere)

목표: 객체 생성 성공률 25% → 70-80%
"""

import json
import sys
from pathlib import Path

# 프로젝트 루트를 Python path에 추가
sys.path.insert(0, str(Path(__file__).parent.parent))

from llm_genesis_integration import LLMGenesisIntegration


def test_genesis_stability():
    """Genesis AI 안정성 테스트"""

    print("="*70)
    print("🧪 Genesis AI 안정성 테스트")
    print("="*70)
    print()

    # 테스트 시나리오 (visual_demo.py와 동일)
    scenarios = [
        {
            "name": "플라스틱 병 - 부드럽게",
            "command": "Pick up the plastic bottle gently and place it in the container",
            "expected_material": "plastic",
            "expected_shape": "box",
        },
        {
            "name": "무거운 금속 도구 - 단단하게",
            "command": "Grab the heavy metal tool firmly and set it down carefully",
            "expected_material": "metal",
            "expected_shape": "box",
        },
        {
            "name": "깨지기 쉬운 유리컵 - 매우 조심스럽게",
            "command": "Lift the glass cup very slowly and steadily without tilting",
            "expected_material": "glass",
            "expected_shape": "sphere",
        },
        {
            "name": "나무 블록 - 빠르게",
            "command": "Grab the wooden block quickly and position it in the holder",
            "expected_material": "wood",
            "expected_shape": "box",
        },
    ]

    # Genesis AI는 비활성화 (LLM 추론만 테스트)
    # 실제 Genesis 테스트는 GPU가 필요하므로 --with-genesis 옵션으로 별도 실행
    enable_genesis = "--with-genesis" in sys.argv

    print(f"모드: {'Genesis AI 포함' if enable_genesis else 'LLM 추론만 (빠름)'}")
    print()

    # LLM + Genesis 통합 클래스 초기화
    print("🚀 LLM + Genesis AI 통합 초기화 중...")
    try:
        llm_genesis = LLMGenesisIntegration(
            adapter_dir="./droid-physics-qwen14b-qlora",
            enable_genesis=enable_genesis
        )
        print("✅ 초기화 완료!\n")
    except Exception as e:
        print(f"❌ 초기화 실패: {e}")
        return

    # 결과 저장
    results = {
        "total_scenarios": len(scenarios),
        "successful_inferences": 0,
        "successful_simulations": 0,
        "genesis_enabled": enable_genesis,
        "scenarios": []
    }

    # 각 시나리오 테스트
    for idx, scenario in enumerate(scenarios):
        print("="*70)
        print(f"📝 시나리오 {idx+1}/{len(scenarios)}: {scenario['name']}")
        print(f"명령: {scenario['command']}")
        print("="*70)

        # LLM 추론
        print("\n🧠 LLM 추론 중...")
        try:
            params = llm_genesis.generate_physics_params(
                scenario['command'],
                max_tokens=128,  # 최적화된 토큰 수
                temperature=0.0
            )

            inference_success = params.get('_metadata', {}).get('success', False)
            inference_time = params.get('_metadata', {}).get('inference_time_ms', 0)

            if inference_success:
                results['successful_inferences'] += 1
                print(f"✅ LLM 추론 성공 ({inference_time:.0f}ms)")

                # 물리 분석 출력
                physical = params.get('physical_analysis', {})
                material = physical.get('material_inference', 'unknown')
                confidence = physical.get('confidence', 0.0)

                print(f"\n📊 물리 분석:")
                print(f"  재료: {material} (예상: {scenario['expected_material']})")
                print(f"  신뢰도: {confidence:.2f}")
                print(f"  질량: {physical.get('mass_category', 'unknown')}")
                print(f"  마찰: {physical.get('friction_coefficient', 'unknown')}")
                print(f"  깨지기 쉬움: {physical.get('fragility', 'unknown')}")

                # 제어 파라미터 출력
                control = params.get('control_parameters', {})
                print(f"\n🎮 제어 파라미터:")
                print(f"  Grip Force: {control.get('grip_force', 0):.2f} N")
                print(f"  Lift Speed: {control.get('lift_speed', 0):.2f} m/s")
                print(f"  Safety Margin: {control.get('safety_margin', 0):.2f}")

                # Genesis AI 시뮬레이션 (활성화된 경우)
                if enable_genesis:
                    print(f"\n🌍 Genesis AI 객체 생성 중...")
                    obj = llm_genesis.create_object_from_params(params, f"scenario_{idx}")

                    if obj is not None:
                        results['successful_simulations'] += 1
                        print("✅ 객체 생성 성공!")

                        # 시뮬레이션 실행
                        print(f"\n▶️  시뮬레이션 실행 중...")
                        sim_result = llm_genesis.run_simulation(duration_sec=2.0)

                        if sim_result.get('success'):
                            print("✅ 시뮬레이션 완료!")
                        else:
                            print(f"⚠️  시뮬레이션 오류: {sim_result.get('error')}")
                    else:
                        print("❌ 객체 생성 실패")

                # 결과 저장
                results['scenarios'].append({
                    "name": scenario['name'],
                    "inference_success": True,
                    "inference_time_ms": inference_time,
                    "material": material,
                    "expected_material": scenario['expected_material'],
                    "material_match": (material.lower() == scenario['expected_material'].lower()),
                    "confidence": confidence,
                    "simulation_success": enable_genesis and (obj is not None),
                })

            else:
                print(f"❌ LLM 추론 실패")
                results['scenarios'].append({
                    "name": scenario['name'],
                    "inference_success": False,
                })

        except Exception as e:
            print(f"❌ 오류 발생: {e}")
            import traceback
            traceback.print_exc()
            results['scenarios'].append({
                "name": scenario['name'],
                "error": str(e),
            })

        print()

    # 최종 결과
    print("="*70)
    print("📊 최종 결과")
    print("="*70)

    inference_rate = results['successful_inferences'] / results['total_scenarios'] * 100
    print(f"\nLLM 추론 성공률: {results['successful_inferences']}/{results['total_scenarios']} ({inference_rate:.0f}%)")

    if enable_genesis:
        sim_rate = results['successful_simulations'] / results['total_scenarios'] * 100
        print(f"객체 생성 성공률: {results['successful_simulations']}/{results['total_scenarios']} ({sim_rate:.0f}%)")

        print(f"\n🎯 개선 목표 달성:")
        if sim_rate >= 70:
            print(f"  ✅ 목표 달성! ({sim_rate:.0f}% ≥ 70%)")
        else:
            print(f"  ⚠️  목표 미달 ({sim_rate:.0f}% < 70%)")

    # 재료 인식 정확도
    material_matches = sum(1 for s in results['scenarios'] if s.get('material_match', False))
    material_rate = material_matches / results['total_scenarios'] * 100
    print(f"재료 인식 정확도: {material_matches}/{results['total_scenarios']} ({material_rate:.0f}%)")

    # 결과 저장
    output_file = "genesis_stability_test_results.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n💾 결과 저장: {output_file}")

    print("\n" + "="*70)
    print("✅ 테스트 완료!")
    print("="*70)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Genesis AI 안정성 테스트")
    parser.add_argument("--with-genesis", action="store_true", help="Genesis AI 시뮬레이션 포함 (GPU 필요)")

    # argparse를 사용하지만 sys.argv는 그대로 유지 (코드 내부에서 체크)
    args = parser.parse_args()

    test_genesis_stability()
