#!/usr/bin/env python3
"""
ROS2 메시지 전달 시스템 (ROS2 Message Interface System)

졸업논문: "LLM-First 기반 물리 속성 추출 로봇 제어"
LLM 추론 결과를 ROS2 메시지로 변환하여 실시간 제어 레이어에 전달

ROS2 메시지 타입:
- geometry_msgs/Twist: 기본 움직임 제어
- sensor_msgs/JointState: 관절 상태 제어
- trajectory_msgs/JointTrajectory: 궤적 제어
- std_msgs/Float64MultiArray: 사용자 정의 제어 파라미터
"""

import time
import json
import threading
import asyncio
import logging
from typing import Dict, List, Optional, Any, Callable, Union
from dataclasses import dataclass, asdict
from enum import Enum
from queue import Queue, Empty, Full
from collections import deque
import numpy as np

# 기존 모듈 연동
from control_parameter_mapper import ControlParameters_Extended, ROS2MessageGenerator
from llm_first_layer import ActionIntent, LLMParsingResult

logger = logging.getLogger(__name__)

# ============================================================================
# ROS2 메시지 데이터 구조
# ============================================================================

class MessageType(Enum):
    """ROS2 메시지 타입"""
    TWIST = "geometry_msgs/Twist"
    JOINT_STATE = "sensor_msgs/JointState"
    JOINT_TRAJECTORY = "trajectory_msgs/JointTrajectory"
    CONTROL_COMMAND = "franka_msgs/ControlCommand"
    GRIPPER_COMMAND = "franka_gripper/GripperCommand"
    ERROR_RECOVERY = "franka_msgs/ErrorRecovery"

class MessagePriority(Enum):
    """메시지 우선순위"""
    LOW = 1
    NORMAL = 2
    HIGH = 3
    CRITICAL = 4
    EMERGENCY = 5

@dataclass
class ROS2Message:
    """ROS2 메시지 래퍼"""
    message_type: MessageType
    payload: Dict[str, Any]
    priority: MessagePriority = MessagePriority.NORMAL
    timestamp: float = 0.0
    sequence_id: int = 0
    timeout: float = 5.0
    retry_count: int = 0
    max_retries: int = 3
    callback: Optional[Callable] = None

@dataclass
class PublisherConfig:
    """퍼블리셔 설정"""
    topic_name: str
    message_type: MessageType
    queue_size: int = 10
    qos_profile: str = "default"
    publish_rate: float = 100.0  # Hz

@dataclass
class SubscriberConfig:
    """서브스크라이버 설정"""
    topic_name: str
    message_type: MessageType
    callback: Callable
    queue_size: int = 10
    qos_profile: str = "default"

# ============================================================================
# 메시지 큐 관리자
# ============================================================================

class MessageQueueManager:
    """메시지 큐 관리자"""

    def __init__(self, max_queue_size: int = 1000):
        self.max_queue_size = max_queue_size
        self.message_queues = {
            MessagePriority.EMERGENCY: Queue(maxsize=10),
            MessagePriority.CRITICAL: Queue(maxsize=50),
            MessagePriority.HIGH: Queue(maxsize=100),
            MessagePriority.NORMAL: Queue(maxsize=500),
            MessagePriority.LOW: Queue(maxsize=1000)
        }
        self.sequence_counter = 0
        self.lock = threading.Lock()

    def enqueue_message(self, message: ROS2Message) -> bool:
        """메시지 큐에 추가"""
        try:
            with self.lock:
                message.sequence_id = self.sequence_counter
                self.sequence_counter += 1

            queue = self.message_queues[message.priority]
            queue.put(message, block=False)

            logger.debug(f"메시지 큐 추가: {message.message_type.value} (우선순위: {message.priority.name})")
            return True

        except Full:
            logger.warning(f"메시지 큐 포화: {message.priority.name}")
            return False

    def dequeue_message(self, timeout: float = 0.1) -> Optional[ROS2Message]:
        """우선순위에 따라 메시지 추출"""

        # 우선순위 순서로 확인
        for priority in [MessagePriority.EMERGENCY, MessagePriority.CRITICAL,
                        MessagePriority.HIGH, MessagePriority.NORMAL, MessagePriority.LOW]:
            queue = self.message_queues[priority]
            try:
                message = queue.get(block=False)
                return message
            except Empty:
                continue

        return None

    def get_queue_status(self) -> Dict[str, int]:
        """큐 상태 조회"""
        status = {}
        for priority, queue in self.message_queues.items():
            status[priority.name] = queue.qsize()
        return status

    def clear_queues(self, priority: Optional[MessagePriority] = None):
        """큐 초기화"""
        if priority:
            while not self.message_queues[priority].empty():
                try:
                    self.message_queues[priority].get(block=False)
                except Empty:
                    break
        else:
            for queue in self.message_queues.values():
                while not queue.empty():
                    try:
                        queue.get(block=False)
                    except Empty:
                        break

# ============================================================================
# ROS2 메시지 변환기
# ============================================================================

class ROS2MessageConverter:
    """ROS2 메시지 변환기"""

    def __init__(self):
        self.message_generator = ROS2MessageGenerator()

    def convert_control_parameters(self,
                                 params: ControlParameters_Extended,
                                 action: ActionIntent) -> List[ROS2Message]:
        """제어 파라미터를 ROS2 메시지로 변환"""

        messages = []

        # 1. Twist 메시지 (기본 움직임)
        twist_payload = {
            "linear": {
                "x": 0.0,
                "y": 0.0,
                "z": params.lift_speed * 0.1  # m/s
            },
            "angular": {
                "x": 0.0,
                "y": 0.0,
                "z": np.radians(params.approach_angle)
            }
        }

        twist_msg = ROS2Message(
            message_type=MessageType.TWIST,
            payload=twist_payload,
            priority=MessagePriority.NORMAL,
            timestamp=time.time()
        )
        messages.append(twist_msg)

        # 2. 그립퍼 명령
        gripper_payload = {
            "position": params.grip_force,
            "max_effort": params.contact_force * 50.0,  # N
            "speed": 0.1  # m/s
        }

        gripper_msg = ROS2Message(
            message_type=MessageType.GRIPPER_COMMAND,
            payload=gripper_payload,
            priority=MessagePriority.HIGH,
            timestamp=time.time()
        )
        messages.append(gripper_msg)

        # 3. 제어 명령 (사용자 정의)
        control_payload = {
            "header": {
                "stamp": time.time(),
                "frame_id": "robot_base"
            },
            "action": action.value,
            "parameters": {
                "grip_force": params.grip_force,
                "lift_speed": params.lift_speed,
                "approach_angle": params.approach_angle,
                "contact_force": params.contact_force,
                "safety_margin": params.safety_margin
            },
            "safety_limits": {
                "force_limit": params.force_limit,
                "acceleration_limit": params.acceleration_limit,
                "jerk_limit": params.jerk_limit
            },
            "timing": {
                "pre_grasp_delay": params.pre_grasp_delay,
                "post_grasp_delay": params.post_grasp_delay
            }
        }

        control_msg = ROS2Message(
            message_type=MessageType.CONTROL_COMMAND,
            payload=control_payload,
            priority=MessagePriority.HIGH,
            timestamp=time.time()
        )
        messages.append(control_msg)

        return messages

    def convert_trajectory(self,
                          joint_positions: List[List[float]],
                          timestamps: List[float],
                          joint_names: List[str] = None) -> ROS2Message:
        """궤적을 ROS2 메시지로 변환"""

        if not joint_names:
            joint_names = [f"panda_joint{i+1}" for i in range(7)]

        points = []
        for i, (positions, timestamp) in enumerate(zip(joint_positions, timestamps)):
            point = {
                "positions": positions,
                "velocities": [0.0] * len(positions),
                "accelerations": [0.0] * len(positions),
                "effort": [0.0] * len(positions),
                "time_from_start": timestamp
            }
            points.append(point)

        trajectory_payload = {
            "header": {
                "stamp": time.time(),
                "frame_id": "robot_base"
            },
            "joint_names": joint_names,
            "points": points
        }

        return ROS2Message(
            message_type=MessageType.JOINT_TRAJECTORY,
            payload=trajectory_payload,
            priority=MessagePriority.NORMAL,
            timestamp=time.time()
        )

    def convert_emergency_stop(self) -> ROS2Message:
        """비상정지 메시지 생성"""

        emergency_payload = {
            "header": {
                "stamp": time.time(),
                "frame_id": "robot_base"
            },
            "command": "EMERGENCY_STOP",
            "reason": "Safety triggered emergency stop"
        }

        return ROS2Message(
            message_type=MessageType.ERROR_RECOVERY,
            payload=emergency_payload,
            priority=MessagePriority.EMERGENCY,
            timestamp=time.time()
        )

# ============================================================================
# ROS2 통신 인터페이스 (목업)
# ============================================================================

class MockROS2Interface:
    """목업 ROS2 인터페이스 (실제 ROS2 없이 테스트용)"""

    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.publishers = {}
        self.subscribers = {}
        self.is_connected = False
        self.message_history = deque(maxlen=1000)
        self.callback_registry = {}

    def initialize(self) -> bool:
        """ROS2 노드 초기화 (목업)"""
        logger.info("목업 ROS2 인터페이스 초기화")
        self.is_connected = True
        return True

    def create_publisher(self, config: PublisherConfig) -> bool:
        """퍼블리셔 생성"""
        if not self.is_connected:
            return False

        self.publishers[config.topic_name] = {
            'config': config,
            'message_count': 0,
            'last_publish_time': 0.0
        }

        logger.info(f"퍼블리셔 생성: {config.topic_name} ({config.message_type.value})")
        return True

    def create_subscriber(self, config: SubscriberConfig) -> bool:
        """서브스크라이버 생성"""
        if not self.is_connected:
            return False

        self.subscribers[config.topic_name] = {
            'config': config,
            'message_count': 0
        }

        self.callback_registry[config.topic_name] = config.callback
        logger.info(f"서브스크라이버 생성: {config.topic_name} ({config.message_type.value})")
        return True

    def publish_message(self, topic_name: str, message: ROS2Message) -> bool:
        """메시지 발행"""
        if topic_name not in self.publishers:
            logger.error(f"퍼블리셔 없음: {topic_name}")
            return False

        publisher_info = self.publishers[topic_name]
        publisher_info['message_count'] += 1
        publisher_info['last_publish_time'] = time.time()

        # 메시지 히스토리에 추가
        self.message_history.append({
            'topic': topic_name,
            'message_type': message.message_type.value,
            'priority': message.priority.name,
            'timestamp': message.timestamp,
            'sequence_id': message.sequence_id,
            'payload_size': len(str(message.payload))
        })

        logger.debug(f"메시지 발행: {topic_name} (타입: {message.message_type.value})")
        return True

    def simulate_message_receive(self, topic_name: str, payload: Dict[str, Any]):
        """메시지 수신 시뮬레이션"""
        if topic_name in self.callback_registry:
            callback = self.callback_registry[topic_name]
            try:
                callback(payload)
            except Exception as e:
                logger.error(f"콜백 실행 실패: {e}")

    def get_statistics(self) -> Dict[str, Any]:
        """통신 통계 조회"""
        stats = {
            'connected': self.is_connected,
            'publishers': len(self.publishers),
            'subscribers': len(self.subscribers),
            'total_messages': len(self.message_history),
            'publisher_stats': {}
        }

        for topic, info in self.publishers.items():
            stats['publisher_stats'][topic] = {
                'message_count': info['message_count'],
                'last_publish_time': info['last_publish_time']
            }

        return stats

    def shutdown(self):
        """ROS2 인터페이스 종료"""
        self.is_connected = False
        self.publishers.clear()
        self.subscribers.clear()
        logger.info("목업 ROS2 인터페이스 종료")

# ============================================================================
# ROS2 메시지 전달 시스템
# ============================================================================

class ROS2MessageInterface:
    """ROS2 메시지 전달 시스템"""

    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.queue_manager = MessageQueueManager()
        self.message_converter = ROS2MessageConverter()
        self.ros2_interface = MockROS2Interface(config)  # 실제 환경에서는 실제 ROS2 인터페이스 사용

        # 메시지 처리 스레드
        self.message_thread = None
        self.is_running = False

        # 성능 모니터링
        self.message_stats = {
            'sent': 0,
            'failed': 0,
            'retried': 0,
            'average_latency': 0.0
        }
        self.latency_history = deque(maxlen=100)

    def initialize(self) -> bool:
        """시스템 초기화"""
        logger.info("ROS2 메시지 인터페이스 초기화")

        # ROS2 인터페이스 초기화
        if not self.ros2_interface.initialize():
            logger.error("ROS2 인터페이스 초기화 실패")
            return False

        # 기본 퍼블리셔 생성
        default_publishers = [
            PublisherConfig("/franka/twist_command", MessageType.TWIST, queue_size=10),
            PublisherConfig("/franka/gripper_command", MessageType.GRIPPER_COMMAND, queue_size=5),
            PublisherConfig("/franka/control_command", MessageType.CONTROL_COMMAND, queue_size=10),
            PublisherConfig("/franka/joint_trajectory", MessageType.JOINT_TRAJECTORY, queue_size=5),
            PublisherConfig("/franka/emergency_stop", MessageType.ERROR_RECOVERY, queue_size=1)
        ]

        for pub_config in default_publishers:
            if not self.ros2_interface.create_publisher(pub_config):
                logger.warning(f"퍼블리셔 생성 실패: {pub_config.topic_name}")

        # 메시지 처리 스레드 시작
        self.is_running = True
        self.message_thread = threading.Thread(target=self._message_processing_loop, daemon=True)
        self.message_thread.start()

        logger.info("ROS2 메시지 인터페이스 초기화 완료")
        return True

    def send_control_command(self,
                           params: ControlParameters_Extended,
                           action: ActionIntent,
                           priority: MessagePriority = MessagePriority.NORMAL) -> bool:
        """제어 명령 전송"""

        try:
            # 제어 파라미터를 ROS2 메시지로 변환
            messages = self.message_converter.convert_control_parameters(params, action)

            # 우선순위 설정
            for message in messages:
                message.priority = priority

            # 메시지 큐에 추가
            success_count = 0
            for message in messages:
                if self.queue_manager.enqueue_message(message):
                    success_count += 1

            logger.info(f"제어 명령 큐 추가: {success_count}/{len(messages)} 성공")
            return success_count == len(messages)

        except Exception as e:
            logger.error(f"제어 명령 전송 실패: {e}")
            return False

    def send_trajectory_command(self,
                              joint_positions: List[List[float]],
                              timestamps: List[float],
                              priority: MessagePriority = MessagePriority.NORMAL) -> bool:
        """궤적 명령 전송"""

        try:
            message = self.message_converter.convert_trajectory(joint_positions, timestamps)
            message.priority = priority

            return self.queue_manager.enqueue_message(message)

        except Exception as e:
            logger.error(f"궤적 명령 전송 실패: {e}")
            return False

    def send_emergency_stop(self) -> bool:
        """비상정지 명령 전송"""

        try:
            message = self.message_converter.convert_emergency_stop()
            return self.queue_manager.enqueue_message(message)

        except Exception as e:
            logger.error(f"비상정지 명령 전송 실패: {e}")
            return False

    def _message_processing_loop(self):
        """메시지 처리 루프"""
        logger.info("메시지 처리 루프 시작")

        while self.is_running:
            try:
                # 큐에서 메시지 추출
                message = self.queue_manager.dequeue_message(timeout=0.1)
                if message is None:
                    continue

                # 메시지 전송
                success = self._send_message(message)

                if success:
                    self.message_stats['sent'] += 1
                    if message.callback:
                        try:
                            message.callback(True, message)
                        except Exception as e:
                            logger.error(f"콜백 실행 실패: {e}")
                else:
                    self._handle_send_failure(message)

            except Exception as e:
                logger.error(f"메시지 처리 루프 오류: {e}")

        logger.info("메시지 처리 루프 종료")

    def _send_message(self, message: ROS2Message) -> bool:
        """개별 메시지 전송"""

        send_start_time = time.time()

        try:
            # 메시지 타입에 따른 토픽 선택
            topic_mapping = {
                MessageType.TWIST: "/franka/twist_command",
                MessageType.GRIPPER_COMMAND: "/franka/gripper_command",
                MessageType.CONTROL_COMMAND: "/franka/control_command",
                MessageType.JOINT_TRAJECTORY: "/franka/joint_trajectory",
                MessageType.ERROR_RECOVERY: "/franka/emergency_stop"
            }

            topic_name = topic_mapping.get(message.message_type)
            if not topic_name:
                logger.error(f"알 수 없는 메시지 타입: {message.message_type}")
                return False

            # ROS2 인터페이스를 통해 메시지 발행
            success = self.ros2_interface.publish_message(topic_name, message)

            # 지연시간 기록
            latency = time.time() - send_start_time
            self.latency_history.append(latency)
            self.message_stats['average_latency'] = np.mean(self.latency_history)

            return success

        except Exception as e:
            logger.error(f"메시지 전송 오류: {e}")
            return False

    def _handle_send_failure(self, message: ROS2Message):
        """전송 실패 처리"""

        message.retry_count += 1
        self.message_stats['failed'] += 1

        if message.retry_count <= message.max_retries:
            # 재시도
            self.queue_manager.enqueue_message(message)
            self.message_stats['retried'] += 1
            logger.warning(f"메시지 재시도: {message.sequence_id} ({message.retry_count}/{message.max_retries})")
        else:
            # 최대 재시도 횟수 초과
            logger.error(f"메시지 전송 포기: {message.sequence_id}")
            if message.callback:
                try:
                    message.callback(False, message)
                except Exception as e:
                    logger.error(f"실패 콜백 실행 오류: {e}")

    def get_system_status(self) -> Dict[str, Any]:
        """시스템 상태 조회"""

        queue_status = self.queue_manager.get_queue_status()
        ros2_stats = self.ros2_interface.get_statistics()

        status = {
            'is_running': self.is_running,
            'queue_status': queue_status,
            'message_stats': self.message_stats.copy(),
            'ros2_interface': ros2_stats,
            'average_latency_ms': self.message_stats['average_latency'] * 1000,
            'success_rate': (
                self.message_stats['sent'] /
                (self.message_stats['sent'] + self.message_stats['failed'])
                if (self.message_stats['sent'] + self.message_stats['failed']) > 0 else 0.0
            )
        }

        return status

    def shutdown(self):
        """시스템 종료"""
        logger.info("ROS2 메시지 인터페이스 종료 시작")

        self.is_running = False

        if self.message_thread and self.message_thread.is_alive():
            self.message_thread.join(timeout=2.0)

        self.queue_manager.clear_queues()
        self.ros2_interface.shutdown()

        logger.info("ROS2 메시지 인터페이스 종료 완료")

# ============================================================================
# 통합 인터페이스 및 팩토리
# ============================================================================

def create_ros2_interface(config: Dict[str, Any] = None) -> ROS2MessageInterface:
    """ROS2 메시지 인터페이스 팩토리"""
    return ROS2MessageInterface(config)

def create_message_callback(name: str) -> Callable:
    """메시지 콜백 팩토리"""
    def callback(success: bool, message: ROS2Message):
        if success:
            logger.info(f"{name} 메시지 전송 성공: {message.sequence_id}")
        else:
            logger.error(f"{name} 메시지 전송 실패: {message.sequence_id}")

    return callback

if __name__ == "__main__":
    # 테스트 실행
    import time
    from control_parameter_mapper import ControlParameters_Extended
    from llm_first_layer import ActionIntent

    # ROS2 인터페이스 초기화
    ros2_interface = create_ros2_interface({"debug": True})

    if not ros2_interface.initialize():
        print("ROS2 인터페이스 초기화 실패")
        exit(1)

    print("ROS2 메시지 전달 시스템 테스트 시작")

    # 테스트 제어 파라미터
    test_params = ControlParameters_Extended(
        grip_force=0.7,
        lift_speed=0.4,
        approach_angle=10.0,
        contact_force=0.5,
        safety_margin=0.9,
        force_limit=25.0
    )

    # 제어 명령 전송 테스트
    print("\n1. 제어 명령 전송 테스트")
    success = ros2_interface.send_control_command(
        test_params,
        ActionIntent.PICK,
        MessagePriority.HIGH
    )
    print(f"제어 명령 전송: {'성공' if success else '실패'}")

    # 궤적 명령 전송 테스트
    print("\n2. 궤적 명령 전송 테스트")
    joint_positions = [
        [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
        [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7],
        [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
    ]
    timestamps = [0.0, 1.0, 2.0]

    success = ros2_interface.send_trajectory_command(
        joint_positions,
        timestamps,
        MessagePriority.NORMAL
    )
    print(f"궤적 명령 전송: {'성공' if success else '실패'}")

    # 비상정지 테스트
    print("\n3. 비상정지 테스트")
    success = ros2_interface.send_emergency_stop()
    print(f"비상정지 명령: {'성공' if success else '실패'}")

    # 상태 모니터링
    print("\n4. 시스템 상태 모니터링")
    time.sleep(1.0)  # 메시지 처리 대기

    status = ros2_interface.get_system_status()
    print(f"실행 상태: {status['is_running']}")
    print(f"전송된 메시지: {status['message_stats']['sent']}")
    print(f"실패한 메시지: {status['message_stats']['failed']}")
    print(f"평균 지연시간: {status['average_latency_ms']:.1f}ms")
    print(f"성공률: {status['success_rate']:.1%}")

    print("\n큐 상태:")
    for priority, count in status['queue_status'].items():
        if count > 0:
            print(f"  {priority}: {count}개")

    # 시스템 종료
    print("\n5. 시스템 종료")
    ros2_interface.shutdown()
    print("ROS2 메시지 전달 시스템 테스트 완료")