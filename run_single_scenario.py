#!/usr/bin/env python3
"""
단일 시나리오 시각화 테스트
각 시나리오를 독립적으로 실행하여 Genesis AI로 시각화
"""

import sys
from llm_genesis_integration import LLMGenesisIntegration


def main():
    if len(sys.argv) < 2:
        print("사용법: python run_single_scenario.py <명령>")
        print("\n예시:")
        print('  python run_single_scenario.py "Pick up the plastic bottle gently"')
        print('  python run_single_scenario.py "Grab the heavy metal tool firmly"')
        print('  python run_single_scenario.py "Lift the glass cup very slowly"')
        print('  python run_single_scenario.py "Grab the wooden block quickly"')
        sys.exit(1)

    command = sys.argv[1]

    print("="*80)
    print("단일 시나리오 시각화 테스트")
    print("="*80)

    # 통합 클래스 초기화
    integration = LLMGenesisIntegration(
        adapter_dir="./droid-physics-qwen14b-qlora",
        enable_genesis=True
    )

    try:
        # 명령 실행
        result = integration.execute_command(
            command=command,
            simulate=True,
            duration_sec=3.0  # 3초 시뮬레이션
        )

        print("\n" + "="*80)
        print("✅ 테스트 완료!")
        print("="*80)

    finally:
        integration.cleanup()


if __name__ == "__main__":
    main()
