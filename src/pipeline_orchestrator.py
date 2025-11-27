"""
통합 파이프라인 오케스트레이터 (Pipeline Orchestrator)

PhysicalAI → Genesis+Franka 변환 파이프라인의 모든 컴포넌트를 통합하고
워크플로우를 관리하는 중앙 오케스트레이션 시스템
"""

import asyncio
import logging
import time
import traceback
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any, Union, Callable, Iterator
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
import concurrent.futures
from datetime import datetime, timedelta
import uuid
import json

from data_abstraction_layer import (
    StandardDataSchema, AdapterFactory, DataValidator, PhysicalAIAdapter
)
from physics_mapping_layer import (
    UniversalPhysicsMapper, TrajectoryProcessor, UnitSystemNormalizer
)
from language_generation_layer import (
    TemplateEngine, LLMEnhancer, QualityAssuranceSystem, GeneratedAnnotation
)
from config_monitoring_system import (
    ConfigurationManager, MetricsCollector, MonitoringDashboard, timer
)


# ============================================================================
# 파이프라인 상태 및 제어 구조
# ============================================================================

class PipelineStatus(Enum):
    """파이프라인 상태"""
    IDLE = "idle"
    RUNNING = "running"
    PAUSED = "paused"
    STOPPING = "stopping"
    FAILED = "failed"
    COMPLETED = "completed"


class TaskStatus(Enum):
    """작업 상태"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"
    RETRYING = "retrying"


class ProcessingMode(Enum):
    """처리 모드"""
    SEQUENTIAL = "sequential"      # 순차 처리
    PARALLEL = "parallel"         # 병렬 처리
    PIPELINE = "pipeline"         # 파이프라인 방식
    ADAPTIVE = "adaptive"         # 적응적 처리


@dataclass
class TaskResult:
    """작업 결과"""
    task_id: str
    status: TaskStatus
    input_data: Any
    output_data: Any
    processing_time: float
    error_message: Optional[str] = None
    quality_metrics: Optional[Dict[str, float]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PipelineConfig:
    """파이프라인 설정"""
    # 처리 모드 설정
    processing_mode: ProcessingMode = ProcessingMode.PIPELINE
    max_workers: int = 4
    batch_size: int = 32
    retry_attempts: int = 3
    retry_delay: float = 1.0

    # 품질 관리 설정
    quality_threshold: float = 0.7
    enable_quality_check: bool = True
    enable_llm_enhancement: bool = True

    # 성능 최적화 설정
    enable_caching: bool = True
    checkpoint_interval: int = 100
    memory_limit_mb: float = 4096
    timeout_seconds: float = 300

    # 출력 설정
    output_format: str = "json"  # json, yaml, hdf5
    compression_enabled: bool = True
    include_metadata: bool = True


@dataclass
class PipelineStats:
    """파이프라인 통계"""
    start_time: datetime
    total_items: int = 0
    processed_items: int = 0
    failed_items: int = 0
    skipped_items: int = 0
    current_throughput: float = 0.0
    average_quality: float = 0.0
    estimated_completion: Optional[datetime] = None


# ============================================================================
# 작업 단위 정의
# ============================================================================

class ProcessingTask(ABC):
    """처리 작업 추상 클래스"""

    def __init__(self, task_id: str, config: Dict[str, Any]):
        self.task_id = task_id
        self.config = config
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

    @abstractmethod
    async def execute(self, input_data: Any) -> TaskResult:
        """작업 실행"""
        pass

    @abstractmethod
    def validate_input(self, input_data: Any) -> bool:
        """입력 데이터 검증"""
        pass

    def get_estimated_time(self, input_data: Any) -> float:
        """예상 처리 시간 반환 (초)"""
        return 1.0  # 기본값


class DataExtractionTask(ProcessingTask):
    """데이터 추출 작업"""

    def __init__(self, task_id: str, config: Dict[str, Any]):
        super().__init__(task_id, config)
        self.adapter = AdapterFactory.create_adapter(
            config.get('adapter_type', 'physicalai'),
            config.get('adapter_config', {})
        )

    async def execute(self, input_data: Any) -> TaskResult:
        start_time = time.time()

        try:
            if isinstance(input_data, (str, Path)):
                # 파일 경로인 경우
                data_iterator = self.adapter.load_dataset(input_data)
                schemas = list(data_iterator)
                output_data = schemas[0] if schemas else None
            else:
                # 이미 처리된 데이터인 경우
                output_data = input_data

            processing_time = time.time() - start_time

            return TaskResult(
                task_id=self.task_id,
                status=TaskStatus.COMPLETED,
                input_data=input_data,
                output_data=output_data,
                processing_time=processing_time
            )

        except Exception as e:
            self.logger.error(f"Data extraction failed: {e}")
            return TaskResult(
                task_id=self.task_id,
                status=TaskStatus.FAILED,
                input_data=input_data,
                output_data=None,
                processing_time=time.time() - start_time,
                error_message=str(e)
            )

    def validate_input(self, input_data: Any) -> bool:
        if isinstance(input_data, (str, Path)):
            return Path(input_data).exists()
        return isinstance(input_data, StandardDataSchema)


class PhysicsConversionTask(ProcessingTask):
    """물리 변환 작업"""

    def __init__(self, task_id: str, config: Dict[str, Any]):
        super().__init__(task_id, config)
        self.physics_mapper = UniversalPhysicsMapper(config.get('physics_config', {}))
        self.trajectory_processor = TrajectoryProcessor(config.get('trajectory_config', {}))
        self.unit_normalizer = UnitSystemNormalizer()

    async def execute(self, input_data: StandardDataSchema) -> TaskResult:
        start_time = time.time()

        try:
            # 물리 속성 변환
            converted_props = self.physics_mapper.map_properties(
                input_data.physical_properties,
                input_data.metadata.source_engine,
                input_data.metadata.target_engine
            )

            # 로봇 동역학 변환
            converted_robot = self.physics_mapper.map_robot_dynamics(
                input_data.robot_config,
                input_data.metadata.source_engine,
                input_data.metadata.target_engine
            )

            # 궤적 처리
            processed_trajectory = self.trajectory_processor.resample_trajectory(
                input_data.trajectory_data,
                self.config.get('target_frequency', 100.0)
            )

            # 변환 검증
            validation_result = self.physics_mapper.validate_conversion(
                input_data.physical_properties,
                converted_props
            )

            # 출력 스키마 생성
            output_schema = StandardDataSchema(
                robot_config=converted_robot,
                physical_properties=converted_props,
                trajectory_data=processed_trajectory,
                scene_description=input_data.scene_description,
                metadata=input_data.metadata
            )

            processing_time = time.time() - start_time

            return TaskResult(
                task_id=self.task_id,
                status=TaskStatus.COMPLETED,
                input_data=input_data,
                output_data=output_schema,
                processing_time=processing_time,
                quality_metrics={'physics_accuracy': validation_result.get('accuracy_score', 0.0)}
            )

        except Exception as e:
            self.logger.error(f"Physics conversion failed: {e}")
            return TaskResult(
                task_id=self.task_id,
                status=TaskStatus.FAILED,
                input_data=input_data,
                output_data=None,
                processing_time=time.time() - start_time,
                error_message=str(e)
            )

    def validate_input(self, input_data: Any) -> bool:
        return isinstance(input_data, StandardDataSchema)


class LanguageGenerationTask(ProcessingTask):
    """언어 생성 작업"""

    def __init__(self, task_id: str, config: Dict[str, Any]):
        super().__init__(task_id, config)
        self.template_engine = TemplateEngine(config.get('template_config', {}))
        self.llm_enhancer = LLMEnhancer(config.get('llm_config', {}))
        self.qa_system = QualityAssuranceSystem(config.get('qa_config', {}))

    async def execute(self, input_data: StandardDataSchema) -> TaskResult:
        start_time = time.time()

        try:
            annotations = []

            # 실행 명령 생성
            instruction = self.template_engine.generate_instruction(input_data)
            annotations.append(instruction)

            # 행동 설명 생성
            description = self.template_engine.generate_description(input_data)
            annotations.append(description)

            # 물리적 설명 생성
            explanation = self.template_engine.generate_explanation(input_data)
            annotations.append(explanation)

            # LLM 기반 품질 향상 (설정에 따라)
            if self.config.get('enable_llm_enhancement', False):
                for i, annotation in enumerate(annotations):
                    if annotation.confidence < 0.8:
                        enhanced = self.llm_enhancer.enhance_annotation(annotation)
                        annotations[i] = enhanced

            # 품질 평가
            quality_scores = []
            for annotation in annotations:
                metrics = self.qa_system.assess_quality(annotation)
                validation = self.qa_system.validate_annotation(annotation)

                quality_scores.append(metrics.overall_score)

                if not validation['is_valid']:
                    self.logger.warning(f"Annotation validation failed: {validation['errors']}")

            # 출력 스키마 업데이트
            output_schema = input_data
            output_schema.natural_language_annotations = [ann.text for ann in annotations]

            avg_quality = sum(quality_scores) / len(quality_scores) if quality_scores else 0.0
            processing_time = time.time() - start_time

            return TaskResult(
                task_id=self.task_id,
                status=TaskStatus.COMPLETED,
                input_data=input_data,
                output_data=output_schema,
                processing_time=processing_time,
                quality_metrics={'language_quality': avg_quality}
            )

        except Exception as e:
            self.logger.error(f"Language generation failed: {e}")
            return TaskResult(
                task_id=self.task_id,
                status=TaskStatus.FAILED,
                input_data=input_data,
                output_data=None,
                processing_time=time.time() - start_time,
                error_message=str(e)
            )

    def validate_input(self, input_data: Any) -> bool:
        return isinstance(input_data, StandardDataSchema)


class QualityAssuranceTask(ProcessingTask):
    """품질 보증 작업"""

    def __init__(self, task_id: str, config: Dict[str, Any]):
        super().__init__(task_id, config)
        self.qa_system = QualityAssuranceSystem(config.get('qa_config', {}))
        self.quality_threshold = config.get('quality_threshold', 0.7)

    async def execute(self, input_data: StandardDataSchema) -> TaskResult:
        start_time = time.time()

        try:
            # 전체 데이터 품질 검증
            quality_passed = True
            quality_scores = []
            validation_issues = []

            # 언어 품질 검사
            for annotation_text in input_data.natural_language_annotations:
                # 임시 어노테이션 객체 생성
                annotation = GeneratedAnnotation(
                    annotation_id=str(uuid.uuid4())[:8],
                    annotation_type="description",  # 임시 타입
                    text=annotation_text,
                    confidence=1.0
                )

                metrics = self.qa_system.assess_quality(annotation)
                validation = self.qa_system.validate_annotation(annotation)

                quality_scores.append(metrics.overall_score)

                if not validation['is_valid']:
                    validation_issues.extend(validation['errors'])
                    quality_passed = False

                if metrics.overall_score < self.quality_threshold:
                    quality_passed = False

            # 물리적 일관성 검사
            physics_score = self._check_physics_consistency(input_data)
            quality_scores.append(physics_score)

            if physics_score < self.quality_threshold:
                quality_passed = False

            avg_quality = sum(quality_scores) / len(quality_scores) if quality_scores else 0.0
            processing_time = time.time() - start_time

            # 품질 검사 통과 여부에 따른 상태 결정
            status = TaskStatus.COMPLETED if quality_passed else TaskStatus.FAILED

            return TaskResult(
                task_id=self.task_id,
                status=status,
                input_data=input_data,
                output_data=input_data if quality_passed else None,
                processing_time=processing_time,
                quality_metrics={
                    'overall_quality': avg_quality,
                    'language_quality': np.mean(quality_scores[:-1]) if len(quality_scores) > 1 else 0.0,
                    'physics_quality': physics_score,
                    'quality_passed': quality_passed
                },
                error_message='; '.join(validation_issues) if validation_issues else None
            )

        except Exception as e:
            self.logger.error(f"Quality assurance failed: {e}")
            return TaskResult(
                task_id=self.task_id,
                status=TaskStatus.FAILED,
                input_data=input_data,
                output_data=None,
                processing_time=time.time() - start_time,
                error_message=str(e)
            )

    def validate_input(self, input_data: Any) -> bool:
        return isinstance(input_data, StandardDataSchema)

    def _check_physics_consistency(self, data: StandardDataSchema) -> float:
        """물리적 일관성 검사"""
        # 간단한 물리적 일관성 검사
        score = 1.0

        # 질량 유효성
        if data.physical_properties.mass <= 0:
            score -= 0.3

        # 마찰계수 범위
        if not (0 <= data.physical_properties.friction_coefficient <= 2.0):
            score -= 0.2

        # 탄성계수 범위
        if not (0 <= data.physical_properties.restitution <= 1.0):
            score -= 0.3

        # 궤적 데이터 일관성
        if data.trajectory_data.timestamps.size > 0:
            if not np.all(np.diff(data.trajectory_data.timestamps) > 0):
                score -= 0.2

        return max(score, 0.0)


# ============================================================================
# 파이프라인 오케스트레이터
# ============================================================================

class PipelineOrchestrator:
    """파이프라인 오케스트레이터"""

    def __init__(self, config_manager: ConfigurationManager,
                 metrics_collector: MetricsCollector):
        self.config_manager = config_manager
        self.metrics_collector = metrics_collector
        self.logger = logging.getLogger(__name__)

        # 파이프라인 상태
        self.status = PipelineStatus.IDLE
        self.stats = None
        self.current_config: Optional[PipelineConfig] = None

        # 작업 관리
        self.task_queue: asyncio.Queue = asyncio.Queue()
        self.workers: List[asyncio.Task] = []
        self.executor = concurrent.futures.ThreadPoolExecutor()

        # 결과 저장
        self.results: List[TaskResult] = []
        self.checkpoints: Dict[int, List[TaskResult]] = {}

        # 에러 처리
        self.error_count = 0
        self.max_errors = 10

    async def run_pipeline(self, input_data: Union[str, Path, List[Any]],
                          config: Optional[PipelineConfig] = None) -> Dict[str, Any]:
        """파이프라인 실행"""

        # 설정 초기화
        self.current_config = config or self._create_default_config()
        self.status = PipelineStatus.RUNNING
        self.stats = PipelineStats(start_time=datetime.now())
        self.results.clear()
        self.error_count = 0

        self.logger.info("Starting pipeline execution")
        self.metrics_collector.record_metric("pipeline.status", 1, labels={"status": "running"})

        try:
            # 입력 데이터 준비
            input_items = self._prepare_input_data(input_data)
            self.stats.total_items = len(input_items)

            # 워커 시작
            await self._start_workers()

            # 작업 생성 및 큐에 추가
            await self._create_and_queue_tasks(input_items)

            # 모든 작업 완료 대기
            await self._wait_for_completion()

            # 결과 수집 및 정리
            final_results = self._collect_results()

            self.status = PipelineStatus.COMPLETED
            completion_time = datetime.now()
            total_time = (completion_time - self.stats.start_time).total_seconds()

            self.logger.info(f"Pipeline completed in {total_time:.2f} seconds")
            self.metrics_collector.record_metric("pipeline.completion_time", total_time)

            return {
                'status': 'completed',
                'total_items': self.stats.total_items,
                'processed_items': self.stats.processed_items,
                'failed_items': self.stats.failed_items,
                'processing_time': total_time,
                'average_quality': self.stats.average_quality,
                'results': final_results
            }

        except Exception as e:
            self.status = PipelineStatus.FAILED
            self.logger.error(f"Pipeline failed: {e}\n{traceback.format_exc()}")
            self.metrics_collector.record_metric("pipeline.status", 1, labels={"status": "failed"})

            return {
                'status': 'failed',
                'error': str(e),
                'processed_items': self.stats.processed_items,
                'failed_items': self.stats.failed_items
            }

        finally:
            # 정리 작업
            await self._cleanup()

    async def pause_pipeline(self):
        """파이프라인 일시정지"""
        if self.status == PipelineStatus.RUNNING:
            self.status = PipelineStatus.PAUSED
            self.logger.info("Pipeline paused")

    async def resume_pipeline(self):
        """파이프라인 재시작"""
        if self.status == PipelineStatus.PAUSED:
            self.status = PipelineStatus.RUNNING
            self.logger.info("Pipeline resumed")

    async def stop_pipeline(self):
        """파이프라인 중지"""
        self.status = PipelineStatus.STOPPING
        self.logger.info("Stopping pipeline...")

        # 모든 워커 취소
        for worker in self.workers:
            worker.cancel()

        await asyncio.gather(*self.workers, return_exceptions=True)
        self.status = PipelineStatus.IDLE
        self.logger.info("Pipeline stopped")

    def get_status(self) -> Dict[str, Any]:
        """현재 상태 조회"""
        if not self.stats:
            return {'status': self.status.value}

        current_time = datetime.now()
        elapsed_time = (current_time - self.stats.start_time).total_seconds()

        # 처리율 계산
        if elapsed_time > 0:
            self.stats.current_throughput = self.stats.processed_items / elapsed_time

        # 완료 예상 시간 계산
        if self.stats.current_throughput > 0:
            remaining_items = self.stats.total_items - self.stats.processed_items
            remaining_time = remaining_items / self.stats.current_throughput
            self.stats.estimated_completion = current_time + timedelta(seconds=remaining_time)

        return {
            'status': self.status.value,
            'stats': {
                'start_time': self.stats.start_time.isoformat(),
                'elapsed_time': elapsed_time,
                'total_items': self.stats.total_items,
                'processed_items': self.stats.processed_items,
                'failed_items': self.stats.failed_items,
                'skipped_items': self.stats.skipped_items,
                'current_throughput': self.stats.current_throughput,
                'average_quality': self.stats.average_quality,
                'estimated_completion': self.stats.estimated_completion.isoformat() if self.stats.estimated_completion else None,
                'progress_percentage': (self.stats.processed_items / max(self.stats.total_items, 1)) * 100
            },
            'error_count': self.error_count,
            'config': self.current_config.__dict__ if self.current_config else None
        }

    @timer
    async def _start_workers(self):
        """워커 시작"""
        num_workers = self.current_config.max_workers
        self.workers = []

        for i in range(num_workers):
            worker = asyncio.create_task(self._worker_loop(f"worker_{i}"))
            self.workers.append(worker)

        self.logger.info(f"Started {num_workers} workers")

    async def _worker_loop(self, worker_id: str):
        """워커 루프"""
        self.logger.debug(f"Worker {worker_id} started")

        while self.status not in [PipelineStatus.STOPPING, PipelineStatus.FAILED, PipelineStatus.COMPLETED]:
            try:
                if self.status == PipelineStatus.PAUSED:
                    await asyncio.sleep(1)
                    continue

                # 작업 가져오기 (타임아웃 설정)
                try:
                    task_data = await asyncio.wait_for(
                        self.task_queue.get(),
                        timeout=1.0
                    )
                except asyncio.TimeoutError:
                    continue

                if task_data is None:  # 종료 신호
                    break

                # 작업 실행
                result = await self._execute_task_pipeline(task_data)
                self.results.append(result)

                # 통계 업데이트
                self._update_stats(result)

                # 체크포인트 확인
                if len(self.results) % self.current_config.checkpoint_interval == 0:
                    await self._create_checkpoint()

                # 작업 완료 표시
                self.task_queue.task_done()

            except Exception as e:
                self.logger.error(f"Worker {worker_id} error: {e}")
                self.error_count += 1

                if self.error_count >= self.max_errors:
                    self.logger.critical("Maximum error count reached, stopping pipeline")
                    self.status = PipelineStatus.FAILED
                    break

        self.logger.debug(f"Worker {worker_id} stopped")

    async def _execute_task_pipeline(self, input_data: Any) -> TaskResult:
        """작업 파이프라인 실행"""
        task_id = str(uuid.uuid4())[:8]

        # 작업 체인 정의
        tasks = [
            DataExtractionTask(f"{task_id}_extract", self.config_manager.get_config("data", {})),
            PhysicsConversionTask(f"{task_id}_physics", self.config_manager.get_config("physics", {})),
            LanguageGenerationTask(f"{task_id}_language", self.config_manager.get_config("language", {})),
        ]

        # 품질 검사 활성화 시 추가
        if self.current_config.enable_quality_check:
            tasks.append(QualityAssuranceTask(f"{task_id}_qa", self.config_manager.get_config("quality", {})))

        # 순차적으로 작업 실행
        current_data = input_data
        processing_times = []
        quality_metrics = {}

        for task in tasks:
            if self.status in [PipelineStatus.STOPPING, PipelineStatus.FAILED]:
                break

            start_time = time.time()

            try:
                # 재시도 로직
                result = await self._execute_with_retry(task, current_data)

                if result.status == TaskStatus.FAILED:
                    return result

                current_data = result.output_data
                processing_times.append(result.processing_time)

                if result.quality_metrics:
                    quality_metrics.update(result.quality_metrics)

            except Exception as e:
                self.logger.error(f"Task {task.task_id} failed: {e}")
                return TaskResult(
                    task_id=task_id,
                    status=TaskStatus.FAILED,
                    input_data=input_data,
                    output_data=None,
                    processing_time=time.time() - start_time,
                    error_message=str(e)
                )

        # 최종 결과
        total_time = sum(processing_times)
        overall_quality = np.mean(list(quality_metrics.values())) if quality_metrics else 0.0

        return TaskResult(
            task_id=task_id,
            status=TaskStatus.COMPLETED,
            input_data=input_data,
            output_data=current_data,
            processing_time=total_time,
            quality_metrics={'overall_quality': overall_quality, **quality_metrics}
        )

    async def _execute_with_retry(self, task: ProcessingTask, input_data: Any) -> TaskResult:
        """재시도 로직을 포함한 작업 실행"""
        last_error = None

        for attempt in range(self.current_config.retry_attempts):
            try:
                result = await task.execute(input_data)

                if result.status != TaskStatus.FAILED:
                    return result

                last_error = result.error_message

            except Exception as e:
                last_error = str(e)
                self.logger.warning(f"Task {task.task_id} attempt {attempt + 1} failed: {e}")

            if attempt < self.current_config.retry_attempts - 1:
                await asyncio.sleep(self.current_config.retry_delay * (2 ** attempt))

        # 모든 재시도 실패
        return TaskResult(
            task_id=task.task_id,
            status=TaskStatus.FAILED,
            input_data=input_data,
            output_data=None,
            processing_time=0.0,
            error_message=f"Failed after {self.current_config.retry_attempts} attempts: {last_error}"
        )

    def _prepare_input_data(self, input_data: Union[str, Path, List[Any]]) -> List[Any]:
        """입력 데이터 준비"""
        if isinstance(input_data, (str, Path)):
            # 파일 경로인 경우
            path = Path(input_data)
            if path.is_file():
                return [path]
            elif path.is_dir():
                # 디렉토리 내 모든 지원 파일 찾기
                supported_extensions = ['.h5', '.hdf5', '.json']
                files = []
                for ext in supported_extensions:
                    files.extend(path.glob(f"*{ext}"))
                return list(files)
        elif isinstance(input_data, list):
            return input_data
        else:
            return [input_data]

    async def _create_and_queue_tasks(self, input_items: List[Any]):
        """작업 생성 및 큐에 추가"""
        for item in input_items:
            await self.task_queue.put(item)

        self.logger.info(f"Queued {len(input_items)} tasks")

    async def _wait_for_completion(self):
        """모든 작업 완료 대기"""
        # 큐의 모든 작업이 완료될 때까지 대기
        await self.task_queue.join()

        # 워커들에게 종료 신호 전송
        for _ in self.workers:
            await self.task_queue.put(None)

        # 모든 워커 완료 대기
        await asyncio.gather(*self.workers)

    def _update_stats(self, result: TaskResult):
        """통계 업데이트"""
        if result.status == TaskStatus.COMPLETED:
            self.stats.processed_items += 1
        elif result.status == TaskStatus.FAILED:
            self.stats.failed_items += 1
        elif result.status == TaskStatus.SKIPPED:
            self.stats.skipped_items += 1

        # 품질 점수 업데이트
        if result.quality_metrics and 'overall_quality' in result.quality_metrics:
            current_quality = result.quality_metrics['overall_quality']
            total_quality_items = self.stats.processed_items

            if total_quality_items == 1:
                self.stats.average_quality = current_quality
            else:
                # 이동 평균 계산
                self.stats.average_quality = (
                    (self.stats.average_quality * (total_quality_items - 1) + current_quality) /
                    total_quality_items
                )

        # 메트릭 기록
        self.metrics_collector.record_metric("pipeline.processed_items", self.stats.processed_items)
        self.metrics_collector.record_metric("pipeline.failed_items", self.stats.failed_items)
        if result.processing_time:
            self.metrics_collector.record_timer("pipeline.processing_time", result.processing_time)

    async def _create_checkpoint(self):
        """체크포인트 생성"""
        checkpoint_id = len(self.results)
        self.checkpoints[checkpoint_id] = self.results.copy()
        self.logger.info(f"Checkpoint created at {checkpoint_id} items")

    def _collect_results(self) -> List[Dict[str, Any]]:
        """결과 수집"""
        collected_results = []

        for result in self.results:
            if result.status == TaskStatus.COMPLETED and result.output_data:
                # StandardDataSchema를 직렬화 가능한 형태로 변환
                if isinstance(result.output_data, StandardDataSchema):
                    result_dict = {
                        'task_id': result.task_id,
                        'processing_time': result.processing_time,
                        'quality_metrics': result.quality_metrics,
                        'robot_config': {
                            'robot_type': result.output_data.robot_config.robot_type.value,
                            'joint_count': result.output_data.robot_config.joint_count,
                            'joint_names': result.output_data.robot_config.joint_names
                        },
                        'physical_properties': {
                            'mass': result.output_data.physical_properties.mass,
                            'friction_coefficient': result.output_data.physical_properties.friction_coefficient,
                            'restitution': result.output_data.physical_properties.restitution
                        },
                        'natural_language_annotations': result.output_data.natural_language_annotations,
                        'metadata': {
                            'source_engine': result.output_data.metadata.source_engine.value,
                            'target_engine': result.output_data.metadata.target_engine.value,
                            'conversion_timestamp': result.output_data.metadata.conversion_timestamp
                        }
                    }
                    collected_results.append(result_dict)

        return collected_results

    async def _cleanup(self):
        """정리 작업"""
        # 워커 정리
        for worker in self.workers:
            if not worker.done():
                worker.cancel()

        self.workers.clear()

        # 실행자 종료
        self.executor.shutdown(wait=True)

        self.logger.info("Pipeline cleanup completed")

    def _create_default_config(self) -> PipelineConfig:
        """기본 설정 생성"""
        return PipelineConfig(
            processing_mode=ProcessingMode.PIPELINE,
            max_workers=self.config_manager.get_config("data.max_workers", 4),
            batch_size=self.config_manager.get_config("data.batch_size", 32),
            quality_threshold=self.config_manager.get_config("language.quality_threshold", 0.7),
            enable_quality_check=self.config_manager.get_config("quality.enabled", True),
            enable_llm_enhancement=self.config_manager.get_config("language.llm_enhancement", False)
        )


# ============================================================================
# CLI 인터페이스
# ============================================================================

class PipelineCLI:
    """파이프라인 명령줄 인터페이스"""

    def __init__(self):
        self.config_manager = ConfigurationManager()
        self.metrics_collector = MetricsCollector(self.config_manager)
        self.dashboard = MonitoringDashboard(self.config_manager, self.metrics_collector)
        self.orchestrator = PipelineOrchestrator(self.config_manager, self.metrics_collector)

        # 모니터링 시작
        self.metrics_collector.start_collection()

    async def run(self, input_path: str, output_path: str = None, config_file: str = None):
        """파이프라인 실행"""
        # 설정 파일 로딩
        if config_file:
            self.config_manager.load_from_file(config_file)

        # 파이프라인 실행
        result = await self.orchestrator.run_pipeline(input_path)

        # 결과 저장
        if output_path and result.get('results'):
            output_file = Path(output_path)
            output_file.parent.mkdir(parents=True, exist_ok=True)

            if output_file.suffix.lower() == '.json':
                with open(output_file, 'w') as f:
                    json.dump(result, f, indent=2)
            elif output_file.suffix.lower() == '.yaml':
                import yaml
                with open(output_file, 'w') as f:
                    yaml.dump(result, f, default_flow_style=False)

            print(f"Results saved to {output_path}")

        # 상태 출력
        print(f"Pipeline completed: {result['status']}")
        print(f"Processed items: {result.get('processed_items', 0)}")
        print(f"Failed items: {result.get('failed_items', 0)}")
        if 'processing_time' in result:
            print(f"Total time: {result['processing_time']:.2f} seconds")

    def get_status(self):
        """상태 조회"""
        status = self.orchestrator.get_status()
        system_status = self.dashboard.get_system_status()

        print(json.dumps({
            'pipeline': status,
            'system': system_status
        }, indent=2))

    def stop(self):
        """정리 작업"""
        self.metrics_collector.stop_collection()


# ============================================================================
# 메인 실행 함수
# ============================================================================

async def main():
    """메인 실행 함수"""
    import argparse

    parser = argparse.ArgumentParser(description="PhysicalAI to Genesis+Franka Conversion Pipeline")
    parser.add_argument("command", choices=["run", "status"], help="Command to execute")
    parser.add_argument("--input", "-i", help="Input data path")
    parser.add_argument("--output", "-o", help="Output path")
    parser.add_argument("--config", "-c", help="Configuration file")

    args = parser.parse_args()

    cli = PipelineCLI()

    try:
        if args.command == "run":
            if not args.input:
                print("Error: --input is required for run command")
                return

            await cli.run(args.input, args.output, args.config)

        elif args.command == "status":
            cli.get_status()

    finally:
        cli.stop()


if __name__ == "__main__":
    import numpy as np  # numpy import for stats calculations
    asyncio.run(main())