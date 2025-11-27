#!/usr/bin/env python3
"""
Genesis AI 시각적 데모 스크립트
다양한 테스트 시나리오로 LLM + Genesis AI 통합을 시연합니다.
"""

import json
import time
import argparse
from typing import List, Dict, Any
from pathlib import Path
from llm_genesis_integration import LLMGenesisIntegration


class VisualDemo:
    """시각적 데모 실행 클래스"""

    def __init__(
        self,
        adapter_dir: str = "./droid-physics-qwen14b-qlora",
        enable_genesis: bool = True,
        output_dir: str = "./demo_results"
    ):
        """
        Args:
            adapter_dir: QLora 어댑터 디렉토리
            enable_genesis: Genesis AI 시각화 활성화
            output_dir: 결과 저장 디렉토리
        """
        self.integration = LLMGenesisIntegration(
            adapter_dir=adapter_dir,
            enable_genesis=enable_genesis
        )

        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.results = []

    def run_scenario(
        self,
        name: str,
        command: str,
        duration_sec: float = 2.0,
        pause_after: float = 1.0
    ) -> Dict[str, Any]:
        """
        단일 시나리오 실행

        Args:
            name: 시나리오 이름
            command: 자연어 명령
            duration_sec: 시뮬레이션 시간
            pause_after: 다음 시나리오 전 대기 시간

        Returns:
            시나리오 실행 결과
        """
        print(f"\n{'='*80}")
        print(f"🎬 시나리오: {name}")
        print(f"{'='*80}")

        result = self.integration.execute_command(
            command=command,
            simulate=True,
            duration_sec=duration_sec
        )

        result['scenario_name'] = name

        # 결과 저장
        self.results.append(result)

        # 다음 시나리오를 위한 대기 및 리셋
        if pause_after > 0:
            print(f"\n⏸️  {pause_after}초 대기...")
            time.sleep(pause_after)
            self.integration.reset_scene()

        return result

    def run_basic_scenarios(self):
        """기본 테스트 시나리오 실행"""
        print("\n" + "="*80)
        print("📋 기본 테스트 시나리오 시작")
        print("="*80)

        scenarios = [
            {
                "name": "가벼운 플라스틱 물체 - 부드럽게",
                "command": "Pick up the plastic bottle gently and place it in the container",
                "duration": 2.0
            },
            {
                "name": "무거운 금속 물체 - 단단하게",
                "command": "Grab the heavy metal tool firmly and set it down carefully",
                "duration": 2.5
            },
            {
                "name": "깨지기 쉬운 유리컵 - 매우 조심스럽게",
                "command": "Lift the glass cup very slowly and steadily without tilting",
                "duration": 3.0
            },
            {
                "name": "나무 블록 - 빠르게",
                "command": "Grab the wooden block quickly and position it in the holder",
                "duration": 1.5
            },
        ]

        for scenario in scenarios:
            self.run_scenario(
                name=scenario["name"],
                command=scenario["command"],
                duration_sec=scenario["duration"],
                pause_after=1.0
            )

    def run_advanced_scenarios(self):
        """고급 테스트 시나리오 실행"""
        print("\n" + "="*80)
        print("🎯 고급 테스트 시나리오 시작")
        print("="*80)

        scenarios = [
            {
                "name": "복합 재질 - 부드러운 고무공",
                "command": "Pick up the soft rubber ball and toss it into the basket",
                "duration": 2.0
            },
            {
                "name": "정밀 제어 - 세라믹 머그",
                "command": "Carefully grasp the ceramic mug by the handle and move it to the tray",
                "duration": 2.5
            },
            {
                "name": "불안정한 물체 - 긴 금속 막대",
                "command": "Grab the long metal rod from the middle to maintain balance",
                "duration": 2.0
            },
            {
                "name": "한국어 명령 - 유리병",
                "command": "유리병을 천천히 들어 올려서 선반에 놓으세요",
                "duration": 2.5
            },
        ]

        for scenario in scenarios:
            self.run_scenario(
                name=scenario["name"],
                command=scenario["command"],
                duration_sec=scenario["duration"],
                pause_after=1.0
            )

    def run_stress_test(self):
        """스트레스 테스트 - 다양한 조건"""
        print("\n" + "="*80)
        print("💪 스트레스 테스트 시작")
        print("="*80)

        scenarios = [
            {
                "name": "최소 정보 - 단순 명령",
                "command": "Pick up the object",
                "duration": 2.0
            },
            {
                "name": "복잡한 명령 - 다단계",
                "command": "Pick up the plastic bottle, rotate it 90 degrees, and gently place it upside down in the metal container while avoiding contact with other objects",
                "duration": 3.0
            },
            {
                "name": "모호한 명령",
                "command": "Move the thing over there carefully",
                "duration": 2.0
            },
        ]

        for scenario in scenarios:
            self.run_scenario(
                name=scenario["name"],
                command=scenario["command"],
                duration_sec=scenario["duration"],
                pause_after=1.0
            )

    def analyze_results(self) -> Dict[str, Any]:
        """결과 분석"""
        print("\n" + "="*80)
        print("📊 결과 분석")
        print("="*80)

        total = len(self.results)
        successful_inferences = sum(
            1 for r in self.results
            if r.get('params', {}).get('_metadata', {}).get('success', False)
        )
        successful_simulations = sum(
            1 for r in self.results
            if r.get('simulation', {}).get('success', False)
        )

        inference_times = [
            r['params']['_metadata']['inference_time_ms']
            for r in self.results
            if r.get('params', {}).get('_metadata', {}).get('success', False)
        ]

        avg_inference_time = sum(inference_times) / len(inference_times) if inference_times else 0

        analysis = {
            "total_scenarios": total,
            "successful_inferences": successful_inferences,
            "successful_simulations": successful_simulations,
            "inference_success_rate": successful_inferences / total if total > 0 else 0,
            "simulation_success_rate": successful_simulations / total if total > 0 else 0,
            "avg_inference_time_ms": avg_inference_time,
            "min_inference_time_ms": min(inference_times) if inference_times else 0,
            "max_inference_time_ms": max(inference_times) if inference_times else 0,
        }

        print(f"\n총 시나리오 수: {total}")
        print(f"LLM 추론 성공: {successful_inferences}/{total} ({analysis['inference_success_rate']:.1%})")
        print(f"시뮬레이션 성공: {successful_simulations}/{total} ({analysis['simulation_success_rate']:.1%})")
        print(f"\n평균 추론 시간: {avg_inference_time:.1f}ms")
        print(f"최소 추론 시간: {analysis['min_inference_time_ms']:.1f}ms")
        print(f"최대 추론 시간: {analysis['max_inference_time_ms']:.1f}ms")

        return analysis

    def save_results(self, filename: str = "demo_results.json"):
        """결과 저장"""
        output_path = self.output_dir / filename

        report = {
            "metadata": {
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "total_scenarios": len(self.results),
            },
            "analysis": self.analyze_results(),
            "scenarios": self.results
        }

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2)

        print(f"\n💾 결과 저장: {output_path}")

    def cleanup(self):
        """리소스 정리"""
        self.integration.cleanup()


def main():
    parser = argparse.ArgumentParser(description="Genesis AI 시각적 데모")
    parser.add_argument(
        "--adapter-dir",
        default="./droid-physics-qwen14b-qlora",
        help="QLora 어댑터 디렉토리"
    )
    parser.add_argument(
        "--no-genesis",
        action="store_true",
        help="Genesis AI 시각화 비활성화 (LLM 추론만)"
    )
    parser.add_argument(
        "--output-dir",
        default="./demo_results",
        help="결과 저장 디렉토리"
    )
    parser.add_argument(
        "--mode",
        choices=["basic", "advanced", "stress", "all"],
        default="basic",
        help="실행 모드"
    )

    args = parser.parse_args()

    print("="*80)
    print("🎬 Genesis AI 시각적 데모 시작")
    print("="*80)
    print(f"모드: {args.mode}")
    print(f"Genesis AI 시각화: {'비활성화' if args.no_genesis else '활성화'}")
    print("="*80)

    demo = VisualDemo(
        adapter_dir=args.adapter_dir,
        enable_genesis=not args.no_genesis,
        output_dir=args.output_dir
    )

    try:
        if args.mode == "basic":
            demo.run_basic_scenarios()
        elif args.mode == "advanced":
            demo.run_advanced_scenarios()
        elif args.mode == "stress":
            demo.run_stress_test()
        elif args.mode == "all":
            demo.run_basic_scenarios()
            demo.run_advanced_scenarios()
            demo.run_stress_test()

        # 결과 분석 및 저장
        demo.analyze_results()
        demo.save_results(f"demo_results_{args.mode}_{time.strftime('%Y%m%d_%H%M%S')}.json")

    finally:
        demo.cleanup()

    print("\n" + "="*80)
    print("✅ 데모 완료!")
    print("="*80)


if __name__ == "__main__":
    main()
