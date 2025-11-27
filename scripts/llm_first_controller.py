#!/usr/bin/env python3
"""
LLM-First 제어 시스템 래퍼
비교 실험을 위한 LLM 기반 컨트롤러
"""

import json
import time
from typing import Dict, Any, Optional
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from llm_genesis_integration import LLMGenesisIntegration


class LLMFirstController:
    """LLM-First 제어 시스템 클래스 (비교 실험용 래퍼)"""

    def __init__(self, enable_genesis: bool = True):
        """
        Args:
            enable_genesis: Genesis AI 시뮬레이션 활성화
        """
        print("🤖 LLM-First 제어 시스템 초기화...")

        # LLM + Genesis 통합 클래스 초기화
        self.llm_integration = LLMGenesisIntegration(
            adapter_dir="./droid-physics-qwen14b-qlora",
            enable_genesis=enable_genesis
        )

        self.genesis_enabled = enable_genesis

    def generate_physics_params(self, command: str) -> Dict[str, Any]:
        """
        명령어에서 물리 파라미터 생성 (LLM 기반)

        Args:
            command: 자연어 명령

        Returns:
            물리 파라미터 JSON 딕셔너리
        """
        # 기존 LLM 통합 클래스의 메소드 사용
        params = self.llm_integration.generate_physics_params(command)

        # 메타데이터 추가
        if '_metadata' not in params:
            params['_metadata'] = {}

        params['_metadata'].update({
            'method': 'llm_first',
            'model': 'Qwen2.5-14B-QLoRA',
            'success': True
        })

        return params

    def create_object_from_params(self, params: Dict[str, Any], object_name: str = "target_object") -> Optional[Any]:
        """물리 파라미터로 Genesis AI 객체 생성"""
        return self.llm_integration.create_object_from_params(params, object_name)

    def run_simulation(self, duration_sec: float = 2.0, real_time: bool = True) -> Dict[str, Any]:
        """Genesis AI 시뮬레이션 실행"""
        return self.llm_integration.run_simulation(duration_sec, real_time)

    def execute_command(self, command: str, simulate: bool = True, duration_sec: float = 2.0) -> Dict[str, Any]:
        """전체 파이프라인 실행: 자연어 → LLM → 시뮬레이션"""
        print(f"\n{'='*70}")
        print(f"📝 LLM-First 명령: {command}")
        print(f"{'='*70}")

        # LLM 추론
        print("\n[1/3] LLM 추론 중...")
        start_time = time.time()
        params = self.generate_physics_params(command)
        llm_time = (time.time() - start_time) * 1000

        if not params.get('_metadata', {}).get('success', False):
            print("✗ LLM 추론 실패")
            return {
                'command': command,
                'success': False,
                'params': params
            }

        print(f"✓ LLM 추론 완료 ({llm_time:.1f}ms)")
        print("\n생성된 물리 파라미터:")
        print(json.dumps(params.get('physical_analysis', {}), indent=2, ensure_ascii=False))
        print(json.dumps(params.get('control_parameters', {}), indent=2, ensure_ascii=False))

        # 시뮬레이션 실행
        if simulate and self.genesis_enabled:
            print("\n[2/3] Genesis AI 객체 생성 중...")
            obj = self.create_object_from_params(params, "llm_first_object")

            if obj is not None:
                print("\n[3/3] 시뮬레이션 실행 중...")
                sim_result = self.run_simulation(duration_sec=duration_sec)
            else:
                sim_result = {'success': False, 'error': 'Object creation failed'}
        else:
            sim_result = {'success': False, 'reason': 'Simulation disabled'}

        result = {
            'command': command,
            'success': True,
            'params': params,
            'llm_time_ms': llm_time,
            'simulation': sim_result
        }

        print(f"\n{'='*70}")
        print("✅ LLM-First 실행 완료")
        print(f"{'='*70}\n")

        return result

    def reset_scene(self):
        """씬 리셋"""
        self.llm_integration.reset_scene()

    def cleanup(self):
        """리소스 정리"""
        self.llm_integration.cleanup()


def main():
    """LLM-First 제어 시스템 테스트"""
    print("🧪 LLM-First 제어 시스템 테스트\n")

    controller = LLMFirstController(enable_genesis=True)

    test_commands = [
        "Pick up the plastic bottle gently",
        "Grab the heavy metal tool firmly",
        "Lift the glass cup very slowly",
        "Sort the rubber ball to the left side"
    ]

    try:
        for i, cmd in enumerate(test_commands, 1):
            print(f"\n\n{'#'*70}")
            print(f"테스트 케이스 {i}/{len(test_commands)}")
            print(f"{'#'*70}")

            result = controller.execute_command(cmd, simulate=True, duration_sec=2.0)

            if i < len(test_commands):
                controller.reset_scene()
                time.sleep(1)

    finally:
        controller.cleanup()


if __name__ == "__main__":
    main()
