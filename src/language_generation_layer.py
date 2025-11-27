"""
언어 생성 계층 (Language Generation Layer) 및 품질 보증 시스템

물리적 속성과 로봇 행동을 자연어로 변환하고
생성된 언어의 품질을 체계적으로 관리하는 시스템
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple, Union, Any, Set
from dataclasses import dataclass, field
from enum import Enum
import re
import json
import logging
from pathlib import Path
import numpy as np
from collections import defaultdict

from data_abstraction_layer import StandardDataSchema, PhysicalProperties, RobotConfiguration
from physics_mapping_layer import TrajectoryData


# ============================================================================
# 언어 생성 관련 데이터 구조
# ============================================================================

class LanguageComplexity(Enum):
    """언어 복잡성 레벨"""
    SIMPLE = "simple"          # 간단한 명령어
    INTERMEDIATE = "intermediate"  # 중간 복잡도
    ADVANCED = "advanced"      # 고급 설명
    TECHNICAL = "technical"    # 기술적 상세 설명


class AnnotationType(Enum):
    """어노테이션 타입"""
    INSTRUCTION = "instruction"      # 실행 명령
    DESCRIPTION = "description"      # 행동 설명
    EXPLANATION = "explanation"      # 물리적 설명
    FEEDBACK = "feedback"           # 피드백
    QUESTION = "question"           # 질문


@dataclass
class LanguageTemplate:
    """언어 생성 템플릿"""
    template_id: str
    template_type: AnnotationType
    complexity_level: LanguageComplexity
    template_text: str
    required_variables: List[str]
    optional_variables: List[str] = field(default_factory=list)

    # 물리적 컨텍스트
    applicable_robot_types: List[str] = field(default_factory=list)
    applicable_scenarios: List[str] = field(default_factory=list)

    # 품질 메타데이터
    usage_count: int = 0
    success_rate: float = 1.0
    average_quality_score: float = 0.0


@dataclass
class GeneratedAnnotation:
    """생성된 어노테이션"""
    annotation_id: str
    annotation_type: AnnotationType
    text: str
    confidence: float

    # 생성 메타데이터
    template_used: Optional[str] = None
    generation_method: str = "template"  # template, llm, hybrid
    generation_timestamp: str = ""

    # 품질 메트릭
    readability_score: float = 0.0
    technical_accuracy: float = 0.0
    semantic_coherence: float = 0.0

    # 검증 결과
    validation_passed: bool = False
    validation_errors: List[str] = field(default_factory=list)
    validation_warnings: List[str] = field(default_factory=list)


@dataclass
class QualityMetrics:
    """품질 평가 메트릭"""
    # 언어학적 품질
    grammar_score: float = 0.0      # 문법 정확성
    readability_score: float = 0.0  # 가독성
    fluency_score: float = 0.0      # 유창성

    # 의미적 품질
    semantic_accuracy: float = 0.0  # 의미적 정확성
    relevance_score: float = 0.0    # 관련성
    coherence_score: float = 0.0    # 일관성

    # 기술적 품질
    technical_accuracy: float = 0.0  # 기술적 정확성
    completeness_score: float = 0.0  # 완전성
    specificity_score: float = 0.0   # 구체성

    # 종합 점수
    overall_score: float = 0.0
    confidence_interval: Tuple[float, float] = (0.0, 0.0)


# ============================================================================
# 템플릿 기반 언어 생성 시스템
# ============================================================================

class TemplateEngine:
    """템플릿 기반 자연어 생성 엔진"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(__name__)

        # 템플릿 저장소
        self.templates: Dict[str, LanguageTemplate] = {}
        self.template_categories: Dict[AnnotationType, List[str]] = defaultdict(list)

        # 언어 규칙
        self.grammar_rules = self._load_grammar_rules()
        self.vocabulary = self._load_vocabulary()

        # 물리적 속성 매핑
        self.physics_to_language_mapping = self._initialize_physics_mapping()

        self._load_default_templates()

    def generate_instruction(self, data: StandardDataSchema,
                           complexity: LanguageComplexity = LanguageComplexity.INTERMEDIATE,
                           context: Optional[Dict[str, Any]] = None) -> GeneratedAnnotation:
        """실행 명령 생성"""

        # 적절한 템플릿 선택
        template = self._select_template(
            AnnotationType.INSTRUCTION,
            complexity,
            data.robot_config.robot_type,
            context
        )

        if not template:
            return self._fallback_generation(data, AnnotationType.INSTRUCTION)

        # 변수 추출 및 바인딩
        variables = self._extract_variables(data, context)

        # 텍스트 생성
        generated_text = self._populate_template(template, variables)

        # 후처리
        processed_text = self._post_process_text(generated_text)

        annotation = GeneratedAnnotation(
            annotation_id=self._generate_id(),
            annotation_type=AnnotationType.INSTRUCTION,
            text=processed_text,
            confidence=self._calculate_confidence(template, variables),
            template_used=template.template_id,
            generation_method="template"
        )

        # 기본 품질 평가
        self._assess_basic_quality(annotation)

        return annotation

    def generate_description(self, data: StandardDataSchema,
                           focus_aspect: str = "general",
                           complexity: LanguageComplexity = LanguageComplexity.INTERMEDIATE) -> GeneratedAnnotation:
        """행동 설명 생성"""

        template = self._select_template(
            AnnotationType.DESCRIPTION,
            complexity,
            data.robot_config.robot_type,
            {"focus": focus_aspect}
        )

        variables = self._extract_variables(data, {"focus": focus_aspect})
        generated_text = self._populate_template(template, variables)
        processed_text = self._post_process_text(generated_text)

        annotation = GeneratedAnnotation(
            annotation_id=self._generate_id(),
            annotation_type=AnnotationType.DESCRIPTION,
            text=processed_text,
            confidence=self._calculate_confidence(template, variables),
            template_used=template.template_id,
            generation_method="template"
        )

        self._assess_basic_quality(annotation)
        return annotation

    def generate_explanation(self, data: StandardDataSchema,
                           physics_focus: List[str] = None) -> GeneratedAnnotation:
        """물리적 설명 생성"""

        if physics_focus is None:
            physics_focus = ["mass", "friction", "stiffness"]

        context = {"physics_focus": physics_focus}
        template = self._select_template(
            AnnotationType.EXPLANATION,
            LanguageComplexity.TECHNICAL,
            data.robot_config.robot_type,
            context
        )

        variables = self._extract_variables(data, context)
        generated_text = self._populate_template(template, variables)
        processed_text = self._post_process_text(generated_text)

        annotation = GeneratedAnnotation(
            annotation_id=self._generate_id(),
            annotation_type=AnnotationType.EXPLANATION,
            text=processed_text,
            confidence=self._calculate_confidence(template, variables),
            template_used=template.template_id,
            generation_method="template"
        )

        self._assess_basic_quality(annotation)
        return annotation

    def _select_template(self, annotation_type: AnnotationType,
                        complexity: LanguageComplexity,
                        robot_type: str,
                        context: Optional[Dict[str, Any]] = None) -> Optional[LanguageTemplate]:
        """적절한 템플릿 선택"""

        candidates = []

        for template_id in self.template_categories[annotation_type]:
            template = self.templates[template_id]

            # 복잡성 레벨 매칭
            if template.complexity_level != complexity:
                continue

            # 로봇 타입 호환성 검사
            if (template.applicable_robot_types and
                robot_type not in template.applicable_robot_types):
                continue

            # 컨텍스트 기반 필터링
            if context and not self._check_context_compatibility(template, context):
                continue

            candidates.append(template)

        if not candidates:
            return None

        # 성공률 기반 선택 (가중 랜덤)
        weights = [t.success_rate * (1.0 + t.average_quality_score) for t in candidates]
        selected_idx = np.random.choice(len(candidates), p=np.array(weights)/sum(weights))

        return candidates[selected_idx]

    def _extract_variables(self, data: StandardDataSchema,
                          context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """데이터에서 변수 추출"""

        variables = {}

        # 로봇 정보
        variables.update({
            'robot_type': data.robot_config.robot_type.value,
            'joint_count': data.robot_config.joint_count,
            'robot_name': self._get_robot_friendly_name(data.robot_config.robot_type)
        })

        # 물리적 속성
        physics = data.physical_properties
        variables.update({
            'mass': self._format_mass(physics.mass),
            'mass_value': physics.mass,
            'friction': self._format_friction(physics.friction_coefficient),
            'friction_value': physics.friction_coefficient,
            'material': physics.material_type or "unknown material",
            'stiffness_description': self._describe_stiffness(physics.contact_stiffness),
        })

        # 궤적 정보
        if data.trajectory_data.timestamps.size > 0:
            duration = data.trajectory_data.timestamps[-1] - data.trajectory_data.timestamps[0]
            variables.update({
                'trajectory_duration': f"{duration:.1f} seconds",
                'movement_type': self._classify_movement(data.trajectory_data)
            })

        # 작업 정보
        variables.update({
            'task_type': data.scene_description.task_type,
            'task_description': data.scene_description.task_description,
            'environment': data.scene_description.environment_type
        })

        # 컨텍스트 변수
        if context:
            variables.update(context)

        return variables

    def _populate_template(self, template: LanguageTemplate,
                          variables: Dict[str, Any]) -> str:
        """템플릿에 변수 채워넣기"""

        text = template.template_text

        # 필수 변수 검사
        missing_vars = []
        for var in template.required_variables:
            if var not in variables:
                missing_vars.append(var)

        if missing_vars:
            self.logger.warning(f"Missing required variables: {missing_vars}")
            # 기본값으로 대체
            for var in missing_vars:
                variables[var] = f"[{var}]"

        # 변수 치환
        try:
            formatted_text = text.format(**variables)
        except KeyError as e:
            self.logger.error(f"Template formatting error: {e}")
            return f"Error: Failed to format template {template.template_id}"

        return formatted_text

    def _post_process_text(self, text: str) -> str:
        """텍스트 후처리"""

        # 기본 정리
        text = text.strip()
        text = re.sub(r'\s+', ' ', text)  # 여러 공백을 하나로
        text = re.sub(r'\n+', '\n', text)  # 여러 줄바꿈을 하나로

        # 문장 부호 정리
        text = re.sub(r'\.+', '.', text)  # 여러 마침표를 하나로
        text = re.sub(r'\s+\.', '.', text)  # 마침표 앞 공백 제거

        # 첫 글자 대문자
        if text and text[0].islower():
            text = text[0].upper() + text[1:]

        # 마침표로 끝나도록
        if text and not text.endswith('.'):
            text += '.'

        return text

    def _load_default_templates(self):
        """기본 템플릿 로딩"""

        # 실행 명령 템플릿
        instruction_templates = [
            LanguageTemplate(
                template_id="franka_simple_move",
                template_type=AnnotationType.INSTRUCTION,
                complexity_level=LanguageComplexity.SIMPLE,
                template_text="Move the {robot_name} robot to perform {task_type}",
                required_variables=["robot_name", "task_type"],
                applicable_robot_types=["franka_panda"]
            ),
            LanguageTemplate(
                template_id="franka_detailed_manipulation",
                template_type=AnnotationType.INSTRUCTION,
                complexity_level=LanguageComplexity.INTERMEDIATE,
                template_text="Use the {robot_name} to carefully {task_description} while maintaining a grip force suitable for the {material} object",
                required_variables=["robot_name", "task_description", "material"],
                applicable_robot_types=["franka_panda"]
            ),
            LanguageTemplate(
                template_id="physics_aware_instruction",
                template_type=AnnotationType.INSTRUCTION,
                complexity_level=LanguageComplexity.ADVANCED,
                template_text="Execute {task_type} with {robot_name}, considering the object's {mass} and {friction} properties. Adjust gripper force and movement speed accordingly",
                required_variables=["task_type", "robot_name", "mass", "friction"],
                applicable_robot_types=["franka_panda"]
            )
        ]

        # 설명 템플릿
        description_templates = [
            LanguageTemplate(
                template_id="movement_description",
                template_type=AnnotationType.DESCRIPTION,
                complexity_level=LanguageComplexity.INTERMEDIATE,
                template_text="The {robot_name} performs {movement_type} motion over {trajectory_duration} in a {environment} environment",
                required_variables=["robot_name", "movement_type", "trajectory_duration", "environment"]
            ),
            LanguageTemplate(
                template_id="task_description",
                template_type=AnnotationType.DESCRIPTION,
                complexity_level=LanguageComplexity.ADVANCED,
                template_text="During this {task_type} task, the robot demonstrates {movement_type} with precise control, handling a {material} object weighing {mass}",
                required_variables=["task_type", "movement_type", "material", "mass"]
            )
        ]

        # 설명 템플릿
        explanation_templates = [
            LanguageTemplate(
                template_id="physics_explanation",
                template_type=AnnotationType.EXPLANATION,
                complexity_level=LanguageComplexity.TECHNICAL,
                template_text="The object's physical properties include {mass} mass and {friction} friction coefficient. {stiffness_description} These parameters affect the robot's interaction dynamics",
                required_variables=["mass", "friction", "stiffness_description"]
            )
        ]

        # 템플릿 등록
        all_templates = instruction_templates + description_templates + explanation_templates

        for template in all_templates:
            self.templates[template.template_id] = template
            self.template_categories[template.template_type].append(template.template_id)

    def _get_robot_friendly_name(self, robot_type) -> str:
        """로봇 타입의 친숙한 이름 반환"""
        name_mapping = {
            "franka_panda": "Franka Panda",
            "ur5": "UR5",
            "kuka_iiwa": "KUKA iiwa",
            "generic_6dof": "6-DOF robot"
        }
        return name_mapping.get(robot_type.value if hasattr(robot_type, 'value') else str(robot_type), "robot")

    def _format_mass(self, mass: float) -> str:
        """질량을 자연어로 포맷팅"""
        if mass < 0.1:
            return f"{mass*1000:.0f}g (lightweight)"
        elif mass < 1.0:
            return f"{mass:.1f}kg (light)"
        elif mass < 5.0:
            return f"{mass:.1f}kg (moderate weight)"
        else:
            return f"{mass:.1f}kg (heavy)"

    def _format_friction(self, friction: float) -> str:
        """마찰계수를 자연어로 포맷팅"""
        if friction < 0.3:
            return f"low friction (μ={friction:.2f})"
        elif friction < 0.7:
            return f"moderate friction (μ={friction:.2f})"
        else:
            return f"high friction (μ={friction:.2f})"

    def _describe_stiffness(self, stiffness: Optional[float]) -> str:
        """강성을 자연어로 설명"""
        if stiffness is None:
            return "The material stiffness is not specified"

        if stiffness < 1000:
            return "The material is relatively flexible"
        elif stiffness < 100000:
            return "The material has moderate stiffness"
        else:
            return "The material is very rigid"

    def _classify_movement(self, trajectory: TrajectoryData) -> str:
        """움직임 분류"""
        if trajectory.joint_positions.size == 0:
            return "stationary"

        # 속도 분석
        if trajectory.joint_velocities is not None:
            max_velocity = np.max(np.abs(trajectory.joint_velocities))
            if max_velocity < 0.1:
                return "slow and precise"
            elif max_velocity < 1.0:
                return "moderate speed"
            else:
                return "fast"

        return "controlled"

    def _calculate_confidence(self, template: LanguageTemplate,
                            variables: Dict[str, Any]) -> float:
        """생성 신뢰도 계산"""
        base_confidence = 0.8

        # 템플릿 성공률 반영
        template_factor = template.success_rate

        # 변수 완전성 반영
        required_vars_present = sum(1 for var in template.required_variables if var in variables)
        completeness_factor = required_vars_present / len(template.required_variables)

        # 템플릿 사용 경험 반영
        experience_factor = min(1.0, template.usage_count / 100.0)

        confidence = base_confidence * template_factor * completeness_factor * (0.5 + 0.5 * experience_factor)

        return min(confidence, 1.0)

    def _assess_basic_quality(self, annotation: GeneratedAnnotation):
        """기본 품질 평가"""
        text = annotation.text

        # 가독성 점수 (길이 기반 간단 추정)
        word_count = len(text.split())
        if 5 <= word_count <= 30:
            annotation.readability_score = 0.9
        elif word_count < 5:
            annotation.readability_score = 0.6
        else:
            annotation.readability_score = max(0.3, 1.0 - (word_count - 30) * 0.02)

        # 기술적 정확성 (키워드 기반 간단 추정)
        technical_keywords = ['mass', 'friction', 'stiffness', 'force', 'velocity', 'acceleration']
        technical_score = sum(1 for keyword in technical_keywords if keyword in text.lower())
        annotation.technical_accuracy = min(1.0, technical_score / 3.0)

        # 의미적 일관성 (문장 구조 기반 간단 추정)
        sentences = text.split('.')
        complete_sentences = [s.strip() for s in sentences if len(s.strip()) > 3]
        annotation.semantic_coherence = min(1.0, len(complete_sentences) / max(1, len(sentences) - 1))

    def _fallback_generation(self, data: StandardDataSchema,
                           annotation_type: AnnotationType) -> GeneratedAnnotation:
        """폴백 생성 (템플릿이 없을 때)"""

        fallback_texts = {
            AnnotationType.INSTRUCTION: f"Operate the {data.robot_config.robot_type.value} robot for {data.scene_description.task_type}",
            AnnotationType.DESCRIPTION: f"The robot performs {data.scene_description.task_type} in {data.scene_description.environment_type} environment",
            AnnotationType.EXPLANATION: f"Physical interaction with mass {data.physical_properties.mass}kg and friction {data.physical_properties.friction_coefficient}"
        }

        return GeneratedAnnotation(
            annotation_id=self._generate_id(),
            annotation_type=annotation_type,
            text=fallback_texts.get(annotation_type, "Robot operation"),
            confidence=0.3,  # 낮은 신뢰도
            generation_method="fallback"
        )

    def _generate_id(self) -> str:
        """고유 ID 생성"""
        import uuid
        return str(uuid.uuid4())[:8]

    def _check_context_compatibility(self, template: LanguageTemplate,
                                   context: Dict[str, Any]) -> bool:
        """컨텍스트 호환성 검사"""
        # 간단한 구현: 추후 확장 가능
        return True

    def _load_grammar_rules(self) -> Dict[str, Any]:
        """문법 규칙 로딩"""
        return {
            'sentence_patterns': ['SVO', 'SVA', 'SVOA'],
            'tense_preference': 'present',
            'voice_preference': 'active'
        }

    def _load_vocabulary(self) -> Dict[str, List[str]]:
        """어휘 로딩"""
        return {
            'motion_verbs': ['move', 'reach', 'grasp', 'place', 'manipulate', 'control'],
            'precision_adjectives': ['precise', 'accurate', 'careful', 'controlled', 'smooth'],
            'force_adjectives': ['gentle', 'firm', 'light', 'strong', 'appropriate']
        }

    def _initialize_physics_mapping(self) -> Dict[str, Dict[str, str]]:
        """물리 속성 → 언어 매핑 초기화"""
        return {
            'mass_categories': {
                'very_light': 'under 100g',
                'light': '100g to 1kg',
                'moderate': '1kg to 5kg',
                'heavy': 'over 5kg'
            },
            'friction_categories': {
                'slippery': 'low friction surface',
                'normal': 'moderate friction',
                'grippy': 'high friction surface'
            }
        }


# ============================================================================
# LLM 기반 언어 품질 향상 시스템
# ============================================================================

class LLMEnhancer:
    """LLM 기반 언어 품질 향상 시스템"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(__name__)

        # LLM 클라이언트 초기화 (OpenAI API 가정)
        self.model_name = config.get('model_name', 'gpt-4')
        self.max_tokens = config.get('max_tokens', 150)
        self.temperature = config.get('temperature', 0.3)

        # 프롬프트 템플릿
        self.enhancement_prompts = self._load_enhancement_prompts()

    def enhance_annotation(self, annotation: GeneratedAnnotation,
                          enhancement_type: str = "general") -> GeneratedAnnotation:
        """어노테이션 품질 향상"""

        if annotation.confidence > 0.9:
            return annotation  # 이미 고품질이면 스킵

        try:
            prompt = self._build_enhancement_prompt(annotation, enhancement_type)
            enhanced_text = self._call_llm(prompt)

            enhanced_annotation = GeneratedAnnotation(
                annotation_id=self._generate_id(),
                annotation_type=annotation.annotation_type,
                text=enhanced_text,
                confidence=min(annotation.confidence + 0.2, 1.0),
                template_used=annotation.template_used,
                generation_method="llm_enhanced"
            )

            return enhanced_annotation

        except Exception as e:
            self.logger.error(f"LLM enhancement failed: {e}")
            return annotation  # 원본 반환

    def _build_enhancement_prompt(self, annotation: GeneratedAnnotation,
                                enhancement_type: str) -> str:
        """향상 프롬프트 구성"""

        base_prompt = self.enhancement_prompts.get(enhancement_type, self.enhancement_prompts['general'])

        prompt = f"""
{base_prompt}

Original text: "{annotation.text}"
Annotation type: {annotation.annotation_type.value}

Please provide an enhanced version that is:
1. More natural and fluent
2. Technically accurate
3. Clear and concise
4. Appropriate for robotics context

Enhanced text:"""

        return prompt

    def _call_llm(self, prompt: str) -> str:
        """LLM API 호출"""
        # 실제 구현에서는 OpenAI API 또는 다른 LLM API 사용
        # 여기서는 시뮬레이션

        # import openai
        # response = openai.ChatCompletion.create(
        #     model=self.model_name,
        #     messages=[{"role": "user", "content": prompt}],
        #     max_tokens=self.max_tokens,
        #     temperature=self.temperature
        # )
        # return response.choices[0].message.content.strip()

        # 시뮬레이션 응답
        return "Enhanced version of the original text with improved fluency and technical accuracy."

    def _load_enhancement_prompts(self) -> Dict[str, str]:
        """향상 프롬프트 로딩"""
        return {
            'general': "You are an expert in robotics and natural language. Improve the given text to make it more natural and technically accurate.",
            'technical': "You are a robotics engineer. Enhance the technical accuracy and precision of the given robotics instruction.",
            'simplification': "You are writing for non-experts. Simplify the given robotics text while maintaining accuracy.",
            'elaboration': "You are providing detailed explanations. Expand the given text with relevant technical details."
        }

    def _generate_id(self) -> str:
        """고유 ID 생성"""
        import uuid
        return str(uuid.uuid4())[:8]


# ============================================================================
# 품질 보증 시스템
# ============================================================================

class QualityAssuranceSystem:
    """언어 품질 보증 시스템"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(__name__)

        # 품질 임계값
        self.quality_thresholds = {
            'minimum_readability': 0.6,
            'minimum_technical_accuracy': 0.5,
            'minimum_coherence': 0.7,
            'minimum_overall': 0.6
        }

        # 검증 규칙
        self.validation_rules = self._load_validation_rules()

        # 품질 평가 모델 (간단한 규칙 기반으로 시작)
        self.quality_assessor = RuleBasedQualityAssessor()

    def assess_quality(self, annotation: GeneratedAnnotation) -> QualityMetrics:
        """품질 종합 평가"""

        metrics = QualityMetrics()

        # 각 차원별 평가
        metrics.grammar_score = self._assess_grammar(annotation.text)
        metrics.readability_score = self._assess_readability(annotation.text)
        metrics.fluency_score = self._assess_fluency(annotation.text)

        metrics.semantic_accuracy = self._assess_semantic_accuracy(annotation)
        metrics.relevance_score = self._assess_relevance(annotation)
        metrics.coherence_score = self._assess_coherence(annotation.text)

        metrics.technical_accuracy = self._assess_technical_accuracy(annotation)
        metrics.completeness_score = self._assess_completeness(annotation)
        metrics.specificity_score = self._assess_specificity(annotation.text)

        # 종합 점수 계산
        metrics.overall_score = self._calculate_overall_score(metrics)
        metrics.confidence_interval = self._estimate_confidence_interval(metrics)

        return metrics

    def validate_annotation(self, annotation: GeneratedAnnotation) -> Dict[str, Any]:
        """어노테이션 검증"""

        validation_result = {
            'is_valid': True,
            'errors': [],
            'warnings': [],
            'suggestions': []
        }

        # 구문적 검증
        syntax_errors = self._check_syntax(annotation.text)
        validation_result['errors'].extend(syntax_errors)

        # 의미적 검증
        semantic_warnings = self._check_semantics(annotation)
        validation_result['warnings'].extend(semantic_warnings)

        # 기술적 정확성 검증
        technical_issues = self._check_technical_accuracy(annotation)
        validation_result['warnings'].extend(technical_issues)

        # 개선 제안
        suggestions = self._generate_suggestions(annotation)
        validation_result['suggestions'].extend(suggestions)

        # 전체 유효성 판단
        if validation_result['errors']:
            validation_result['is_valid'] = False

        # 어노테이션 객체 업데이트
        annotation.validation_passed = validation_result['is_valid']
        annotation.validation_errors = validation_result['errors']
        annotation.validation_warnings = validation_result['warnings']

        return validation_result

    def _assess_grammar(self, text: str) -> float:
        """문법 평가"""
        # 간단한 규칙 기반 평가
        score = 1.0

        # 기본 문법 검사
        if not text.strip():
            return 0.0

        # 문장 부호 검사
        if not text.endswith('.'):
            score -= 0.1

        # 대문자 시작 검사
        if text and text[0].islower():
            score -= 0.1

        # 중복 공백 검사
        if '  ' in text:
            score -= 0.05

        return max(score, 0.0)

    def _assess_readability(self, text: str) -> float:
        """가독성 평가"""
        words = text.split()
        word_count = len(words)

        # 적절한 길이인지 확인
        if 5 <= word_count <= 25:
            length_score = 1.0
        elif word_count < 5:
            length_score = word_count / 5.0
        else:
            length_score = max(0.3, 1.0 - (word_count - 25) * 0.02)

        # 복잡한 단어 비율
        complex_words = [w for w in words if len(w) > 10]
        complexity_ratio = len(complex_words) / max(1, word_count)
        complexity_score = max(0.0, 1.0 - complexity_ratio * 2)

        return (length_score + complexity_score) / 2

    def _assess_fluency(self, text: str) -> float:
        """유창성 평가"""
        # 간단한 휴리스틱 기반 평가
        score = 1.0

        # 반복 단어 검사
        words = text.lower().split()
        unique_words = set(words)
        repetition_ratio = len(words) / len(unique_words)

        if repetition_ratio > 1.5:
            score -= 0.2

        # 자연스러운 연결어 사용
        connectors = ['and', 'with', 'while', 'during', 'for', 'to']
        connector_count = sum(1 for word in words if word in connectors)

        if connector_count > 0:
            score += 0.1

        return min(score, 1.0)

    def _assess_semantic_accuracy(self, annotation: GeneratedAnnotation) -> float:
        """의미적 정확성 평가"""
        # 어노테이션 타입에 따른 기대 내용 확인
        text = annotation.text.lower()

        if annotation.annotation_type == AnnotationType.INSTRUCTION:
            # 실행 명령에 적절한 동사가 있는지 확인
            action_verbs = ['move', 'reach', 'grasp', 'place', 'manipulate', 'control', 'operate']
            has_action = any(verb in text for verb in action_verbs)
            return 0.8 if has_action else 0.4

        elif annotation.annotation_type == AnnotationType.DESCRIPTION:
            # 설명에 적절한 설명 요소가 있는지 확인
            descriptive_elements = ['robot', 'motion', 'task', 'environment']
            element_count = sum(1 for element in descriptive_elements if element in text)
            return min(1.0, element_count / 2.0)

        elif annotation.annotation_type == AnnotationType.EXPLANATION:
            # 설명에 기술적 용어가 있는지 확인
            technical_terms = ['mass', 'friction', 'force', 'velocity', 'acceleration', 'stiffness']
            term_count = sum(1 for term in technical_terms if term in text)
            return min(1.0, term_count / 2.0)

        return 0.7  # 기본 점수

    def _assess_relevance(self, annotation: GeneratedAnnotation) -> float:
        """관련성 평가"""
        # 간단한 키워드 기반 관련성 평가
        robotics_keywords = ['robot', 'joint', 'arm', 'gripper', 'motion', 'control', 'manipulation']
        text = annotation.text.lower()

        keyword_count = sum(1 for keyword in robotics_keywords if keyword in text)
        return min(1.0, keyword_count / 3.0)

    def _assess_coherence(self, text: str) -> float:
        """일관성 평가"""
        sentences = [s.strip() for s in text.split('.') if s.strip()]

        if len(sentences) <= 1:
            return 1.0  # 단일 문장은 일관성 문제 없음

        # 간단한 일관성 체크: 주어의 일관성
        subjects = []
        for sentence in sentences:
            words = sentence.split()
            if words:
                # 첫 번째 명사를 주어로 가정
                for word in words[:3]:
                    if word.lower() in ['robot', 'arm', 'gripper', 'system']:
                        subjects.append(word.lower())
                        break

        if len(set(subjects)) <= 1:
            return 1.0  # 일관된 주어
        else:
            return 0.7  # 약간의 비일관성

    def _assess_technical_accuracy(self, annotation: GeneratedAnnotation) -> float:
        """기술적 정확성 평가"""
        return annotation.technical_accuracy  # 이미 계산된 값 사용

    def _assess_completeness(self, annotation: GeneratedAnnotation) -> float:
        """완전성 평가"""
        text = annotation.text.lower()

        # 어노테이션 타입별 기대 요소
        if annotation.annotation_type == AnnotationType.INSTRUCTION:
            required_elements = ['action', 'object', 'goal']
            action_words = ['move', 'reach', 'grasp', 'place', 'control']
            object_words = ['robot', 'arm', 'gripper', 'object']
            goal_words = ['to', 'for', 'goal', 'task']

            has_action = any(word in text for word in action_words)
            has_object = any(word in text for word in object_words)
            has_goal = any(word in text for word in goal_words)

            completeness = (has_action + has_object + has_goal) / 3.0
            return completeness

        return 0.8  # 기본 완전성 점수

    def _assess_specificity(self, text: str) -> float:
        """구체성 평가"""
        # 구체적인 수치나 세부사항의 존재 확인
        import re

        # 숫자 패턴
        number_pattern = r'\d+\.?\d*'
        numbers = re.findall(number_pattern, text)

        # 구체적인 용어
        specific_terms = ['kg', 'cm', 'mm', 'degrees', 'seconds', 'newton', 'force']
        term_count = sum(1 for term in specific_terms if term in text.lower())

        specificity = min(1.0, (len(numbers) + term_count) / 3.0)
        return specificity

    def _calculate_overall_score(self, metrics: QualityMetrics) -> float:
        """종합 점수 계산"""
        # 가중 평균
        weights = {
            'language': 0.3,    # 언어학적 품질
            'semantic': 0.4,    # 의미적 품질
            'technical': 0.3    # 기술적 품질
        }

        language_score = (metrics.grammar_score + metrics.readability_score + metrics.fluency_score) / 3
        semantic_score = (metrics.semantic_accuracy + metrics.relevance_score + metrics.coherence_score) / 3
        technical_score = (metrics.technical_accuracy + metrics.completeness_score + metrics.specificity_score) / 3

        overall = (language_score * weights['language'] +
                  semantic_score * weights['semantic'] +
                  technical_score * weights['technical'])

        return overall

    def _estimate_confidence_interval(self, metrics: QualityMetrics) -> Tuple[float, float]:
        """신뢰 구간 추정"""
        # 간단한 추정: 표준편차 기반
        scores = [
            metrics.grammar_score, metrics.readability_score, metrics.fluency_score,
            metrics.semantic_accuracy, metrics.relevance_score, metrics.coherence_score,
            metrics.technical_accuracy, metrics.completeness_score, metrics.specificity_score
        ]

        std_dev = np.std(scores)
        margin = 1.96 * std_dev / np.sqrt(len(scores))  # 95% 신뢰구간

        lower = max(0.0, metrics.overall_score - margin)
        upper = min(1.0, metrics.overall_score + margin)

        return (lower, upper)

    def _check_syntax(self, text: str) -> List[str]:
        """구문 검사"""
        errors = []

        if not text.strip():
            errors.append("Empty text")
            return errors

        # 기본 구문 검사
        if not text.endswith('.'):
            errors.append("Missing period at the end")

        # 괄호 매칭 검사
        if text.count('(') != text.count(')'):
            errors.append("Unmatched parentheses")

        return errors

    def _check_semantics(self, annotation: GeneratedAnnotation) -> List[str]:
        """의미적 검사"""
        warnings = []
        text = annotation.text.lower()

        # 모순된 표현 검사
        contradictions = [
            (['fast', 'quick'], ['slow', 'careful']),
            (['heavy'], ['light', 'lightweight']),
            (['rigid', 'stiff'], ['flexible', 'soft'])
        ]

        for positive_terms, negative_terms in contradictions:
            has_positive = any(term in text for term in positive_terms)
            has_negative = any(term in text for term in negative_terms)

            if has_positive and has_negative:
                warnings.append(f"Potential contradiction: {positive_terms} vs {negative_terms}")

        return warnings

    def _check_technical_accuracy(self, annotation: GeneratedAnnotation) -> List[str]:
        """기술적 정확성 검사"""
        warnings = []
        text = annotation.text.lower()

        # 물리적으로 불가능한 표현 검사
        impossible_phrases = [
            'negative mass',
            'friction greater than 2',
            'infinite stiffness',
            'zero gravity and stable motion'
        ]

        for phrase in impossible_phrases:
            if phrase in text:
                warnings.append(f"Physically impossible statement: {phrase}")

        return warnings

    def _generate_suggestions(self, annotation: GeneratedAnnotation) -> List[str]:
        """개선 제안 생성"""
        suggestions = []
        text = annotation.text

        # 길이 기반 제안
        word_count = len(text.split())
        if word_count < 5:
            suggestions.append("Consider adding more detail to make the instruction clearer")
        elif word_count > 30:
            suggestions.append("Consider shortening the text for better readability")

        # 구체성 개선 제안
        if not any(char.isdigit() for char in text):
            suggestions.append("Consider adding specific values (e.g., force, time, distance)")

        # 기술적 용어 제안
        if annotation.annotation_type == AnnotationType.EXPLANATION:
            technical_terms = ['mass', 'friction', 'stiffness', 'velocity']
            missing_terms = [term for term in technical_terms if term not in text.lower()]
            if missing_terms:
                suggestions.append(f"Consider mentioning: {', '.join(missing_terms)}")

        return suggestions

    def _load_validation_rules(self) -> Dict[str, Any]:
        """검증 규칙 로딩"""
        return {
            'max_length': 200,  # 최대 문자 수
            'min_length': 10,   # 최소 문자 수
            'required_punctuation': ['.'],
            'forbidden_phrases': ['TODO', 'FIXME', '[placeholder]']
        }


class RuleBasedQualityAssessor:
    """규칙 기반 품질 평가기"""

    def __init__(self):
        self.logger = logging.getLogger(__name__)

    def assess(self, text: str) -> Dict[str, float]:
        """텍스트 품질 평가"""
        return {
            'grammar': self._assess_grammar(text),
            'readability': self._assess_readability(text),
            'technical': self._assess_technical_content(text)
        }

    def _assess_grammar(self, text: str) -> float:
        """문법 평가"""
        # 기본 문법 규칙 기반 평가
        score = 1.0

        if not text.endswith('.'):
            score -= 0.2
        if text and text[0].islower():
            score -= 0.1
        if '  ' in text:
            score -= 0.1

        return max(score, 0.0)

    def _assess_readability(self, text: str) -> float:
        """가독성 평가"""
        word_count = len(text.split())

        if 5 <= word_count <= 25:
            return 1.0
        elif word_count < 5:
            return word_count / 5.0
        else:
            return max(0.3, 1.0 - (word_count - 25) * 0.02)

    def _assess_technical_content(self, text: str) -> float:
        """기술적 내용 평가"""
        technical_keywords = ['robot', 'joint', 'mass', 'friction', 'force', 'control']
        text_lower = text.lower()

        keyword_count = sum(1 for keyword in technical_keywords if keyword in text_lower)
        return min(1.0, keyword_count / 3.0)