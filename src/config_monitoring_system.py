"""
설정 관리 및 모니터링 시스템 (Configuration & Monitoring System)

파이프라인의 모든 설정을 중앙에서 관리하고
실시간 성능 및 품질 모니터링을 제공하는 시스템
"""

import yaml
import json
import logging
import time
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any, Union, Callable
from dataclasses import dataclass, field, asdict
from enum import Enum
from pathlib import Path
import threading
from collections import defaultdict, deque
import numpy as np
from datetime import datetime, timedelta
import psutil
import os


# ============================================================================
# 설정 관리 데이터 구조
# ============================================================================

class ConfigScope(Enum):
    """설정 범위"""
    GLOBAL = "global"          # 전역 설정
    PIPELINE = "pipeline"      # 파이프라인별 설정
    MODULE = "module"          # 모듈별 설정
    RUNTIME = "runtime"        # 런타임 설정
    FILE = "file"              # 파일 기반 설정


class ConfigPriority(Enum):
    """설정 우선순위"""
    DEFAULT = 1      # 기본값
    FILE = 2         # 파일 설정
    ENV = 3          # 환경변수
    RUNTIME = 4      # 런타임 설정
    OVERRIDE = 5     # 강제 오버라이드


@dataclass
class ConfigValue:
    """설정값 래퍼"""
    value: Any
    scope: ConfigScope
    priority: ConfigPriority
    timestamp: str
    source: str  # 설정 출처
    validation_rule: Optional[str] = None
    description: Optional[str] = None


@dataclass
class ConfigSchema:
    """설정 스키마 정의"""
    key: str
    data_type: type
    default_value: Any
    required: bool = False
    validation_func: Optional[Callable] = None
    description: str = ""
    allowed_values: Optional[List[Any]] = None
    min_value: Optional[Union[int, float]] = None
    max_value: Optional[Union[int, float]] = None


# ============================================================================
# 모니터링 관련 데이터 구조
# ============================================================================

class MetricType(Enum):
    """메트릭 타입"""
    COUNTER = "counter"           # 누적 카운터
    GAUGE = "gauge"              # 현재 값
    HISTOGRAM = "histogram"       # 분포
    TIMER = "timer"              # 실행 시간


class AlertSeverity(Enum):
    """알림 심각도"""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class Metric:
    """메트릭 데이터"""
    name: str
    metric_type: MetricType
    value: Union[float, int]
    timestamp: str
    labels: Dict[str, str] = field(default_factory=dict)
    unit: str = ""
    description: str = ""


@dataclass
class Alert:
    """알림 데이터"""
    alert_id: str
    name: str
    severity: AlertSeverity
    message: str
    timestamp: str
    metric_name: str
    threshold_value: float
    actual_value: float
    resolved: bool = False
    resolution_timestamp: Optional[str] = None


@dataclass
class PerformanceSnapshot:
    """성능 스냅샷"""
    timestamp: str
    cpu_usage: float
    memory_usage: float
    disk_usage: float
    network_io: Dict[str, float]
    gpu_usage: Optional[float] = None
    pipeline_throughput: float = 0.0
    error_rate: float = 0.0
    quality_score: float = 0.0


# ============================================================================
# 설정 관리 시스템
# ============================================================================

class ConfigurationManager:
    """중앙화된 설정 관리자"""

    def __init__(self, config_dir: Union[str, Path] = "./config"):
        self.config_dir = Path(config_dir)
        self.config_dir.mkdir(exist_ok=True)

        self.logger = logging.getLogger(__name__)

        # 설정 저장소
        self.configs: Dict[str, ConfigValue] = {}
        self.schemas: Dict[str, ConfigSchema] = {}

        # 설정 변경 콜백
        self.change_callbacks: Dict[str, List[Callable]] = defaultdict(list)

        # 환경별 설정
        self.environments = ["development", "testing", "staging", "production"]
        self.current_environment = os.getenv("PIPELINE_ENV", "development")

        # 기본 설정 로딩
        self._load_default_schemas()
        self._load_configurations()

    def register_schema(self, schema: ConfigSchema):
        """설정 스키마 등록"""
        self.schemas[schema.key] = schema
        self.logger.info(f"Registered config schema: {schema.key}")

        # 기본값 설정
        if schema.key not in self.configs:
            self.set_config(
                schema.key,
                schema.default_value,
                ConfigScope.GLOBAL,
                ConfigPriority.DEFAULT,
                "default_schema"
            )

    def set_config(self, key: str, value: Any,
                   scope: ConfigScope = ConfigScope.GLOBAL,
                   priority: ConfigPriority = ConfigPriority.RUNTIME,
                   source: str = "manual") -> bool:
        """설정값 설정"""

        # 스키마 검증
        if key in self.schemas:
            if not self._validate_config(key, value):
                self.logger.error(f"Config validation failed for {key}: {value}")
                return False

        # 우선순위 확인
        existing_config = self.configs.get(key)
        if existing_config and existing_config.priority.value > priority.value:
            self.logger.warning(f"Config {key} not updated due to lower priority")
            return False

        # 설정값 저장
        config_value = ConfigValue(
            value=value,
            scope=scope,
            priority=priority,
            timestamp=datetime.now().isoformat(),
            source=source
        )

        old_value = self.configs.get(key)
        self.configs[key] = config_value

        self.logger.info(f"Config updated: {key} = {value} (source: {source})")

        # 콜백 실행
        self._trigger_callbacks(key, old_value.value if old_value else None, value)

        return True

    def get_config(self, key: str, default: Any = None) -> Any:
        """설정값 조회"""
        config = self.configs.get(key)
        if config:
            return config.value
        return default

    def get_config_with_metadata(self, key: str) -> Optional[ConfigValue]:
        """메타데이터를 포함한 설정값 조회"""
        return self.configs.get(key)

    def register_callback(self, key: str, callback: Callable[[str, Any, Any], None]):
        """설정 변경 콜백 등록"""
        self.change_callbacks[key].append(callback)

    def load_from_file(self, file_path: Union[str, Path], scope: ConfigScope = ConfigScope.FILE):
        """파일에서 설정 로딩"""
        file_path = Path(file_path)

        if not file_path.exists():
            self.logger.warning(f"Config file not found: {file_path}")
            return

        try:
            if file_path.suffix.lower() == '.yaml' or file_path.suffix.lower() == '.yml':
                with open(file_path, 'r') as f:
                    config_data = yaml.safe_load(f)
            elif file_path.suffix.lower() == '.json':
                with open(file_path, 'r') as f:
                    config_data = json.load(f)
            else:
                self.logger.error(f"Unsupported config file format: {file_path.suffix}")
                return

            self._load_config_dict(config_data, scope, f"file:{file_path}")

        except Exception as e:
            self.logger.error(f"Failed to load config file {file_path}: {e}")

    def load_from_env(self, prefix: str = "PIPELINE_"):
        """환경변수에서 설정 로딩"""
        for key, value in os.environ.items():
            if key.startswith(prefix):
                config_key = key[len(prefix):].lower().replace('_', '.')

                # 타입 추론 및 변환
                converted_value = self._convert_env_value(value)

                self.set_config(
                    config_key,
                    converted_value,
                    ConfigScope.GLOBAL,
                    ConfigPriority.ENV,
                    f"env:{key}"
                )

    def save_to_file(self, file_path: Union[str, Path], scope: Optional[ConfigScope] = None):
        """설정을 파일에 저장"""
        file_path = Path(file_path)

        # 저장할 설정 필터링
        configs_to_save = {}
        for key, config in self.configs.items():
            if scope is None or config.scope == scope:
                configs_to_save[key] = config.value

        try:
            if file_path.suffix.lower() in ['.yaml', '.yml']:
                with open(file_path, 'w') as f:
                    yaml.dump(configs_to_save, f, default_flow_style=False)
            elif file_path.suffix.lower() == '.json':
                with open(file_path, 'w') as f:
                    json.dump(configs_to_save, f, indent=2)

            self.logger.info(f"Config saved to {file_path}")

        except Exception as e:
            self.logger.error(f"Failed to save config to {file_path}: {e}")

    def get_environment_config(self, environment: str = None) -> Dict[str, Any]:
        """환경별 설정 조회"""
        if environment is None:
            environment = self.current_environment

        env_config_file = self.config_dir / f"{environment}.yaml"
        if env_config_file.exists():
            with open(env_config_file, 'r') as f:
                return yaml.safe_load(f)
        return {}

    def reload_configurations(self):
        """설정 재로딩"""
        self.logger.info("Reloading configurations...")

        # 기본 설정 파일들 로딩
        config_files = [
            self.config_dir / "default.yaml",
            self.config_dir / f"{self.current_environment}.yaml",
            self.config_dir / "local.yaml"  # 로컬 오버라이드
        ]

        for config_file in config_files:
            if config_file.exists():
                self.load_from_file(config_file)

        # 환경변수 재로딩
        self.load_from_env()

    def _validate_config(self, key: str, value: Any) -> bool:
        """설정값 검증"""
        schema = self.schemas.get(key)
        if not schema:
            return True  # 스키마가 없으면 통과

        # 타입 검사
        if not isinstance(value, schema.data_type):
            if schema.data_type == float and isinstance(value, int):
                value = float(value)  # int to float 허용
            else:
                return False

        # 허용값 검사
        if schema.allowed_values and value not in schema.allowed_values:
            return False

        # 범위 검사
        if schema.min_value is not None and value < schema.min_value:
            return False
        if schema.max_value is not None and value > schema.max_value:
            return False

        # 커스텀 검증 함수
        if schema.validation_func and not schema.validation_func(value):
            return False

        return True

    def _load_config_dict(self, config_dict: Dict[str, Any],
                         scope: ConfigScope, source: str):
        """딕셔너리에서 설정 로딩"""
        def _flatten_dict(d: Dict[str, Any], prefix: str = "") -> Dict[str, Any]:
            result = {}
            for key, value in d.items():
                full_key = f"{prefix}.{key}" if prefix else key
                if isinstance(value, dict):
                    result.update(_flatten_dict(value, full_key))
                else:
                    result[full_key] = value
            return result

        flattened = _flatten_dict(config_dict)
        for key, value in flattened.items():
            self.set_config(key, value, scope, ConfigPriority.FILE, source)

    def _convert_env_value(self, value: str) -> Any:
        """환경변수 값 타입 변환"""
        # 불린 변환
        if value.lower() in ('true', 'false'):
            return value.lower() == 'true'

        # 숫자 변환
        try:
            if '.' in value:
                return float(value)
            else:
                return int(value)
        except ValueError:
            pass

        # JSON 파싱 시도
        try:
            return json.loads(value)
        except (json.JSONDecodeError, ValueError):
            pass

        # 문자열 그대로 반환
        return value

    def _trigger_callbacks(self, key: str, old_value: Any, new_value: Any):
        """설정 변경 콜백 실행"""
        for callback in self.change_callbacks[key]:
            try:
                callback(key, old_value, new_value)
            except Exception as e:
                self.logger.error(f"Config callback error for {key}: {e}")

    def _load_configurations(self):
        """기본 설정 로딩"""
        # 기본 설정 파일들을 순서대로 로딩
        config_files = [
            "default.yaml",
            f"{self.current_environment}.yaml",
            "local.yaml"
        ]

        for filename in config_files:
            config_path = self.config_dir / filename
            if config_path.exists():
                self.load_from_file(config_path)

        # 환경변수 로딩
        self.load_from_env()

    def _load_default_schemas(self):
        """기본 설정 스키마 로딩"""
        default_schemas = [
            # 데이터 처리 설정
            ConfigSchema("data.batch_size", int, 32, description="Batch size for data processing"),
            ConfigSchema("data.max_workers", int, 4, min_value=1, max_value=32, description="Number of worker processes"),
            ConfigSchema("data.cache_enabled", bool, True, description="Enable data caching"),

            # 물리 변환 설정
            ConfigSchema("physics.accuracy_threshold", float, 0.95, min_value=0.0, max_value=1.0, description="Physics conversion accuracy threshold"),
            ConfigSchema("physics.validation_enabled", bool, True, description="Enable physics validation"),

            # 언어 생성 설정
            ConfigSchema("language.complexity_level", str, "intermediate", allowed_values=["simple", "intermediate", "advanced", "technical"], description="Default language complexity"),
            ConfigSchema("language.quality_threshold", float, 0.7, min_value=0.0, max_value=1.0, description="Minimum quality threshold"),

            # 시스템 설정
            ConfigSchema("system.log_level", str, "INFO", allowed_values=["DEBUG", "INFO", "WARNING", "ERROR"], description="Logging level"),
            ConfigSchema("system.max_memory_usage", float, 0.8, min_value=0.1, max_value=0.95, description="Maximum memory usage ratio"),

            # 모니터링 설정
            ConfigSchema("monitoring.enabled", bool, True, description="Enable monitoring"),
            ConfigSchema("monitoring.collection_interval", int, 10, min_value=1, max_value=3600, description="Metrics collection interval (seconds)"),
            ConfigSchema("monitoring.retention_days", int, 30, min_value=1, max_value=365, description="Metrics retention period")
        ]

        for schema in default_schemas:
            self.register_schema(schema)


# ============================================================================
# 모니터링 시스템
# ============================================================================

class MetricsCollector:
    """메트릭 수집기"""

    def __init__(self, config_manager: ConfigurationManager):
        self.config_manager = config_manager
        self.logger = logging.getLogger(__name__)

        # 메트릭 저장소
        self.metrics: Dict[str, deque] = defaultdict(lambda: deque(maxlen=10000))
        self.alerts: Dict[str, Alert] = {}

        # 알림 규칙
        self.alert_rules: Dict[str, Dict[str, Any]] = {}

        # 수집 스레드
        self.collection_thread = None
        self.is_collecting = False

        # 성능 추적
        self.performance_history: deque = deque(maxlen=1000)

        self._setup_default_alerts()

    def start_collection(self):
        """메트릭 수집 시작"""
        if self.is_collecting:
            return

        self.is_collecting = True
        self.collection_thread = threading.Thread(target=self._collection_loop, daemon=True)
        self.collection_thread.start()
        self.logger.info("Metrics collection started")

    def stop_collection(self):
        """메트릭 수집 중지"""
        self.is_collecting = False
        if self.collection_thread:
            self.collection_thread.join(timeout=5)
        self.logger.info("Metrics collection stopped")

    def record_metric(self, name: str, value: Union[float, int],
                     metric_type: MetricType = MetricType.GAUGE,
                     labels: Dict[str, str] = None,
                     unit: str = "",
                     description: str = ""):
        """메트릭 기록"""
        metric = Metric(
            name=name,
            metric_type=metric_type,
            value=value,
            timestamp=datetime.now().isoformat(),
            labels=labels or {},
            unit=unit,
            description=description
        )

        self.metrics[name].append(metric)

        # 알림 규칙 확인
        self._check_alert_rules(metric)

    def increment_counter(self, name: str, amount: Union[float, int] = 1,
                         labels: Dict[str, str] = None):
        """카운터 증가"""
        current_value = 0
        if self.metrics[name]:
            current_value = self.metrics[name][-1].value

        self.record_metric(
            name,
            current_value + amount,
            MetricType.COUNTER,
            labels
        )

    def record_timer(self, name: str, duration: float,
                    labels: Dict[str, str] = None):
        """실행 시간 기록"""
        self.record_metric(
            name,
            duration,
            MetricType.TIMER,
            labels,
            unit="seconds"
        )

    def get_metrics(self, name: str, since: Optional[datetime] = None) -> List[Metric]:
        """메트릭 조회"""
        metrics = list(self.metrics[name])

        if since:
            since_iso = since.isoformat()
            metrics = [m for m in metrics if m.timestamp >= since_iso]

        return metrics

    def get_latest_metric(self, name: str) -> Optional[Metric]:
        """최신 메트릭 조회"""
        if self.metrics[name]:
            return self.metrics[name][-1]
        return None

    def get_metric_summary(self, name: str, duration: timedelta = timedelta(hours=1)) -> Dict[str, float]:
        """메트릭 요약 통계"""
        since = datetime.now() - duration
        metrics = self.get_metrics(name, since)

        if not metrics:
            return {}

        values = [m.value for m in metrics]

        return {
            'count': len(values),
            'min': min(values),
            'max': max(values),
            'mean': np.mean(values),
            'median': np.median(values),
            'std': np.std(values),
            'p95': np.percentile(values, 95),
            'p99': np.percentile(values, 99)
        }

    def add_alert_rule(self, name: str, metric_name: str,
                      condition: str, threshold: float,
                      severity: AlertSeverity = AlertSeverity.WARNING,
                      message_template: str = None):
        """알림 규칙 추가"""
        self.alert_rules[name] = {
            'metric_name': metric_name,
            'condition': condition,  # 'gt', 'lt', 'eq', 'ne'
            'threshold': threshold,
            'severity': severity,
            'message_template': message_template or f"{metric_name} {condition} {threshold}"
        }

    def get_active_alerts(self) -> List[Alert]:
        """활성 알림 조회"""
        return [alert for alert in self.alerts.values() if not alert.resolved]

    def resolve_alert(self, alert_id: str):
        """알림 해결"""
        if alert_id in self.alerts:
            self.alerts[alert_id].resolved = True
            self.alerts[alert_id].resolution_timestamp = datetime.now().isoformat()

    def _collection_loop(self):
        """메트릭 수집 루프"""
        interval = self.config_manager.get_config("monitoring.collection_interval", 10)

        while self.is_collecting:
            try:
                self._collect_system_metrics()
                time.sleep(interval)
            except Exception as e:
                self.logger.error(f"Metrics collection error: {e}")
                time.sleep(interval)

    def _collect_system_metrics(self):
        """시스템 메트릭 수집"""
        # CPU 사용률
        cpu_usage = psutil.cpu_percent(interval=1)
        self.record_metric("system.cpu_usage", cpu_usage, MetricType.GAUGE, unit="percent")

        # 메모리 사용률
        memory = psutil.virtual_memory()
        self.record_metric("system.memory_usage", memory.percent, MetricType.GAUGE, unit="percent")
        self.record_metric("system.memory_available", memory.available / (1024**3), MetricType.GAUGE, unit="GB")

        # 디스크 사용률
        disk = psutil.disk_usage('/')
        self.record_metric("system.disk_usage", disk.percent, MetricType.GAUGE, unit="percent")

        # 네트워크 I/O
        net_io = psutil.net_io_counters()
        self.record_metric("system.network_bytes_sent", net_io.bytes_sent, MetricType.COUNTER, unit="bytes")
        self.record_metric("system.network_bytes_recv", net_io.bytes_recv, MetricType.COUNTER, unit="bytes")

        # 성능 스냅샷 생성
        snapshot = PerformanceSnapshot(
            timestamp=datetime.now().isoformat(),
            cpu_usage=cpu_usage,
            memory_usage=memory.percent,
            disk_usage=disk.percent,
            network_io={
                'bytes_sent': net_io.bytes_sent,
                'bytes_recv': net_io.bytes_recv
            }
        )
        self.performance_history.append(snapshot)

    def _check_alert_rules(self, metric: Metric):
        """알림 규칙 확인"""
        for rule_name, rule in self.alert_rules.items():
            if rule['metric_name'] == metric.name:
                if self._evaluate_condition(metric.value, rule['condition'], rule['threshold']):
                    self._trigger_alert(rule_name, rule, metric)

    def _evaluate_condition(self, value: float, condition: str, threshold: float) -> bool:
        """조건 평가"""
        if condition == 'gt':
            return value > threshold
        elif condition == 'lt':
            return value < threshold
        elif condition == 'eq':
            return abs(value - threshold) < 1e-9
        elif condition == 'ne':
            return abs(value - threshold) >= 1e-9
        elif condition == 'gte':
            return value >= threshold
        elif condition == 'lte':
            return value <= threshold
        return False

    def _trigger_alert(self, rule_name: str, rule: Dict[str, Any], metric: Metric):
        """알림 발생"""
        alert_id = f"{rule_name}_{metric.timestamp}"

        # 이미 동일한 알림이 활성 상태인지 확인
        existing_alerts = [a for a in self.alerts.values()
                         if a.name == rule_name and not a.resolved]
        if existing_alerts:
            return  # 중복 알림 방지

        alert = Alert(
            alert_id=alert_id,
            name=rule_name,
            severity=rule['severity'],
            message=rule['message_template'].format(
                value=metric.value,
                threshold=rule['threshold']
            ),
            timestamp=metric.timestamp,
            metric_name=metric.name,
            threshold_value=rule['threshold'],
            actual_value=metric.value
        )

        self.alerts[alert_id] = alert
        self.logger.warning(f"Alert triggered: {alert.name} - {alert.message}")

    def _setup_default_alerts(self):
        """기본 알림 규칙 설정"""
        # 시스템 리소스 알림
        self.add_alert_rule(
            "high_cpu_usage",
            "system.cpu_usage",
            "gt",
            80.0,
            AlertSeverity.WARNING,
            "High CPU usage: {value:.1f}% > {threshold}%"
        )

        self.add_alert_rule(
            "high_memory_usage",
            "system.memory_usage",
            "gt",
            85.0,
            AlertSeverity.WARNING,
            "High memory usage: {value:.1f}% > {threshold}%"
        )

        self.add_alert_rule(
            "low_disk_space",
            "system.disk_usage",
            "gt",
            90.0,
            AlertSeverity.ERROR,
            "Low disk space: {value:.1f}% > {threshold}%"
        )


# ============================================================================
# 통합 모니터링 대시보드
# ============================================================================

class MonitoringDashboard:
    """모니터링 대시보드"""

    def __init__(self, config_manager: ConfigurationManager,
                 metrics_collector: MetricsCollector):
        self.config_manager = config_manager
        self.metrics_collector = metrics_collector
        self.logger = logging.getLogger(__name__)

    def get_system_status(self) -> Dict[str, Any]:
        """시스템 상태 조회"""
        # 최신 시스템 메트릭
        cpu_metric = self.metrics_collector.get_latest_metric("system.cpu_usage")
        memory_metric = self.metrics_collector.get_latest_metric("system.memory_usage")
        disk_metric = self.metrics_collector.get_latest_metric("system.disk_usage")

        # 활성 알림
        active_alerts = self.metrics_collector.get_active_alerts()

        # 파이프라인 통계
        pipeline_stats = self._get_pipeline_statistics()

        return {
            'timestamp': datetime.now().isoformat(),
            'system': {
                'cpu_usage': cpu_metric.value if cpu_metric else 0.0,
                'memory_usage': memory_metric.value if memory_metric else 0.0,
                'disk_usage': disk_metric.value if disk_metric else 0.0,
            },
            'alerts': {
                'active_count': len(active_alerts),
                'critical_count': len([a for a in active_alerts if a.severity == AlertSeverity.CRITICAL]),
                'alerts': [asdict(alert) for alert in active_alerts[:10]]  # 최근 10개
            },
            'pipeline': pipeline_stats,
            'configuration': {
                'environment': self.config_manager.current_environment,
                'config_count': len(self.config_manager.configs)
            }
        }

    def get_performance_report(self, duration: timedelta = timedelta(hours=1)) -> Dict[str, Any]:
        """성능 리포트 생성"""
        # 시스템 메트릭 요약
        cpu_summary = self.metrics_collector.get_metric_summary("system.cpu_usage", duration)
        memory_summary = self.metrics_collector.get_metric_summary("system.memory_usage", duration)

        # 처리 시간 통계
        processing_times = self.metrics_collector.get_metrics("pipeline.processing_time", datetime.now() - duration)

        # 품질 메트릭
        quality_scores = self.metrics_collector.get_metrics("pipeline.quality_score", datetime.now() - duration)

        return {
            'period': {
                'start': (datetime.now() - duration).isoformat(),
                'end': datetime.now().isoformat(),
                'duration_hours': duration.total_seconds() / 3600
            },
            'system_performance': {
                'cpu': cpu_summary,
                'memory': memory_summary
            },
            'processing_performance': {
                'total_items': len(processing_times),
                'avg_processing_time': np.mean([m.value for m in processing_times]) if processing_times else 0,
                'throughput_items_per_hour': len(processing_times) / (duration.total_seconds() / 3600) if processing_times else 0
            },
            'quality_metrics': {
                'total_assessments': len(quality_scores),
                'avg_quality_score': np.mean([m.value for m in quality_scores]) if quality_scores else 0,
                'quality_distribution': self._calculate_quality_distribution(quality_scores)
            }
        }

    def export_metrics(self, format: str = "json", output_path: Optional[Path] = None) -> Dict[str, Any]:
        """메트릭 내보내기"""
        export_data = {
            'export_timestamp': datetime.now().isoformat(),
            'system_status': self.get_system_status(),
            'performance_report': self.get_performance_report(),
            'configuration_snapshot': self._get_configuration_snapshot()
        }

        if output_path:
            if format.lower() == "json":
                with open(output_path, 'w') as f:
                    json.dump(export_data, f, indent=2)
            elif format.lower() == "yaml":
                with open(output_path, 'w') as f:
                    yaml.dump(export_data, f, default_flow_style=False)

            self.logger.info(f"Metrics exported to {output_path}")

        return export_data

    def _get_pipeline_statistics(self) -> Dict[str, Any]:
        """파이프라인 통계 수집"""
        # 처리 통계
        processed_items = self.metrics_collector.get_latest_metric("pipeline.processed_items")
        failed_items = self.metrics_collector.get_latest_metric("pipeline.failed_items")

        total_processed = processed_items.value if processed_items else 0
        total_failed = failed_items.value if failed_items else 0

        success_rate = (total_processed - total_failed) / max(total_processed, 1) * 100

        return {
            'total_processed': total_processed,
            'total_failed': total_failed,
            'success_rate': success_rate,
            'avg_quality_score': self._get_avg_quality_score(),
            'uptime_hours': self._calculate_uptime()
        }

    def _get_avg_quality_score(self) -> float:
        """평균 품질 점수 계산"""
        quality_metrics = self.metrics_collector.get_metrics(
            "pipeline.quality_score",
            datetime.now() - timedelta(hours=24)
        )
        if quality_metrics:
            return np.mean([m.value for m in quality_metrics])
        return 0.0

    def _calculate_uptime(self) -> float:
        """가동 시간 계산"""
        if self.metrics_collector.performance_history:
            start_time = datetime.fromisoformat(self.metrics_collector.performance_history[0].timestamp)
            uptime = datetime.now() - start_time
            return uptime.total_seconds() / 3600
        return 0.0

    def _calculate_quality_distribution(self, quality_scores: List[Metric]) -> Dict[str, int]:
        """품질 점수 분포 계산"""
        if not quality_scores:
            return {}

        scores = [m.value for m in quality_scores]
        distribution = {
            'excellent': len([s for s in scores if s >= 0.9]),
            'good': len([s for s in scores if 0.7 <= s < 0.9]),
            'fair': len([s for s in scores if 0.5 <= s < 0.7]),
            'poor': len([s for s in scores if s < 0.5])
        }

        return distribution

    def _get_configuration_snapshot(self) -> Dict[str, Any]:
        """설정 스냅샷 생성"""
        return {
            'environment': self.config_manager.current_environment,
            'total_configs': len(self.config_manager.configs),
            'config_by_scope': {
                scope.value: len([c for c in self.config_manager.configs.values() if c.scope == scope])
                for scope in ConfigScope
            },
            'recent_changes': self._get_recent_config_changes()
        }

    def _get_recent_config_changes(self, hours: int = 24) -> List[Dict[str, Any]]:
        """최근 설정 변경 내역"""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        recent_changes = []

        for key, config in self.config_manager.configs.items():
            config_time = datetime.fromisoformat(config.timestamp)
            if config_time > cutoff_time:
                recent_changes.append({
                    'key': key,
                    'value': config.value,
                    'timestamp': config.timestamp,
                    'source': config.source
                })

        return sorted(recent_changes, key=lambda x: x['timestamp'], reverse=True)


# ============================================================================
# 타이머 데코레이터
# ============================================================================

class Timer:
    """실행 시간 측정 데코레이터"""

    def __init__(self, metrics_collector: MetricsCollector, metric_name: str):
        self.metrics_collector = metrics_collector
        self.metric_name = metric_name

    def __call__(self, func):
        def wrapper(*args, **kwargs):
            start_time = time.time()
            try:
                result = func(*args, **kwargs)
                return result
            finally:
                duration = time.time() - start_time
                self.metrics_collector.record_timer(self.metric_name, duration)
        return wrapper


def timer(metrics_collector: MetricsCollector, metric_name: str = None):
    """타이머 데코레이터 팩토리"""
    def decorator(func):
        name = metric_name or f"timer.{func.__module__}.{func.__name__}"
        return Timer(metrics_collector, name)(func)
    return decorator