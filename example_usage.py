#!/usr/bin/env python3
"""
PhysicalAI → Genesis+Franka 변환 파이프라인 사용 예제

이 스크립트는 파이프라인의 기본적인 사용법을 보여줍니다.
"""

import asyncio
import json
import logging
from pathlib import Path

# 파이프라인 모듈 임포트
from src.config_monitoring_system import ConfigurationManager, MetricsCollector, MonitoringDashboard
from src.pipeline_orchestrator import PipelineOrchestrator, PipelineConfig, ProcessingMode

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

async def basic_example():
    """기본 사용 예제"""
    print("=== 기본 파이프라인 실행 예제 ===")

    # 1. 설정 관리자 초기화
    config_manager = ConfigurationManager("./config")

    # 2. 메트릭 수집기 초기화
    metrics_collector = MetricsCollector(config_manager)
    metrics_collector.start_collection()

    # 3. 파이프라인 오케스트레이터 초기화
    orchestrator = PipelineOrchestrator(config_manager, metrics_collector)

    try:
        # 4. 파이프라인 설정
        config = PipelineConfig(
            processing_mode=ProcessingMode.PIPELINE,
            max_workers=4,
            batch_size=16,
            quality_threshold=0.7,
            enable_quality_check=True,
            enable_llm_enhancement=False  # 개발 중에는 비활성화
        )

        # 5. 입력 데이터 경로 (예제 데이터)
        input_path = "data/robot_dataset/"

        # 6. 파이프라인 실행
        print(f"입력 데이터: {input_path}")
        print("파이프라인 실행 중...")

        result = await orchestrator.run_pipeline(input_path, config)

        # 7. 결과 출력
        print("\n=== 실행 결과 ===")
        print(f"상태: {result['status']}")
        print(f"총 항목: {result.get('total_items', 0)}")
        print(f"처리 완료: {result.get('processed_items', 0)}")
        print(f"실패: {result.get('failed_items', 0)}")
        print(f"처리 시간: {result.get('processing_time', 0):.2f}초")
        print(f"평균 품질: {result.get('average_quality', 0):.3f}")

        # 8. 결과 저장
        if result.get('results'):
            output_path = Path("results/example_output.json")
            output_path.parent.mkdir(exist_ok=True)

            with open(output_path, 'w') as f:
                json.dump(result, f, indent=2)

            print(f"결과 저장: {output_path}")

    except Exception as e:
        print(f"오류 발생: {e}")
        import traceback
        traceback.print_exc()

    finally:
        # 9. 정리
        metrics_collector.stop_collection()
        await orchestrator.stop_pipeline()


async def monitoring_example():
    """모니터링 시스템 사용 예제"""
    print("\n=== 모니터링 시스템 예제 ===")

    # 설정 관리자 및 메트릭 수집기 초기화
    config_manager = ConfigurationManager()
    metrics_collector = MetricsCollector(config_manager)
    dashboard = MonitoringDashboard(config_manager, metrics_collector)

    # 모니터링 시작
    metrics_collector.start_collection()

    try:
        # 몇 초간 메트릭 수집
        print("메트릭 수집 중...")
        await asyncio.sleep(5)

        # 시스템 상태 조회
        system_status = dashboard.get_system_status()
        print("\n=== 시스템 상태 ===")
        print(json.dumps(system_status, indent=2))

        # 성능 리포트 생성
        from datetime import timedelta
        performance_report = dashboard.get_performance_report(duration=timedelta(minutes=1))
        print("\n=== 성능 리포트 ===")
        print(json.dumps(performance_report, indent=2))

    finally:
        metrics_collector.stop_collection()


async def configuration_example():
    """설정 관리 예제"""
    print("\n=== 설정 관리 예제 ===")

    # 설정 관리자 초기화
    config_manager = ConfigurationManager()

    # 설정값 조회
    print("현재 설정값:")
    print(f"  배치 크기: {config_manager.get_config('data.batch_size', 32)}")
    print(f"  최대 워커 수: {config_manager.get_config('data.max_workers', 4)}")
    print(f"  품질 임계값: {config_manager.get_config('language.quality_threshold', 0.7)}")

    # 런타임 설정 변경
    config_manager.set_config('data.batch_size', 64, source='runtime_example')
    print(f"변경된 배치 크기: {config_manager.get_config('data.batch_size')}")

    # 설정 변경 콜백 등록
    def on_quality_change(key, old_value, new_value):
        print(f"품질 임계값 변경: {old_value} → {new_value}")

    config_manager.register_callback('language.quality_threshold', on_quality_change)

    # 설정 변경 (콜백 트리거됨)
    config_manager.set_config('language.quality_threshold', 0.8)


async def custom_task_example():
    """커스텀 작업 예제"""
    print("\n=== 커스텀 작업 예제 ===")

    from src.pipeline_orchestrator import ProcessingTask, TaskResult, TaskStatus
    import time

    class ExampleTask(ProcessingTask):
        """예제 커스텀 작업"""

        async def execute(self, input_data):
            start_time = time.time()

            try:
                # 간단한 처리 시뮬레이션
                await asyncio.sleep(0.1)
                processed_data = f"Processed: {input_data}"

                return TaskResult(
                    task_id=self.task_id,
                    status=TaskStatus.COMPLETED,
                    input_data=input_data,
                    output_data=processed_data,
                    processing_time=time.time() - start_time,
                    quality_metrics={'custom_score': 0.95}
                )

            except Exception as e:
                return TaskResult(
                    task_id=self.task_id,
                    status=TaskStatus.FAILED,
                    input_data=input_data,
                    output_data=None,
                    processing_time=time.time() - start_time,
                    error_message=str(e)
                )

        def validate_input(self, input_data):
            return isinstance(input_data, str)

    # 커스텀 작업 실행
    task = ExampleTask("custom_task_1", {})
    result = await task.execute("test_input")

    print(f"작업 결과: {result.status.value}")
    print(f"출력 데이터: {result.output_data}")
    print(f"처리 시간: {result.processing_time:.3f}초")
    print(f"품질 메트릭: {result.quality_metrics}")


async def main():
    """메인 실행 함수"""
    print("PhysicalAI → Genesis+Franka 변환 파이프라인 예제")
    print("=" * 60)

    # 각 예제 실행
    await basic_example()
    await monitoring_example()
    await configuration_example()
    await custom_task_example()

    print("\n예제 실행 완료!")


if __name__ == "__main__":
    # 예제 실행
    asyncio.run(main())