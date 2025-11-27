#!/usr/bin/env python3
"""
Affordance Prompting 기법 구현 (Affordance Prompting System)

졸업논문: "LLM-First 기반 물리 속성 추출 로봇 제어"
LLM이 동작 실행 가능성과 성공 확률까지 예측하도록 설계

Affordance: 물체가 제공하는 행동 가능성 (예: 컵 → 잡기 가능, 마시기 가능)
"""

import json
import math
import logging
from typing import Dict, List, Optional, Tuple, Any, Set
from dataclasses import dataclass, asdict
from enum import Enum
import numpy as np
from abc import ABC, abstractmethod

from llm_first_layer import ActionIntent, ExtractedPhysicalProperties, AffordanceAssessment

logger = logging.getLogger(__name__)

# ============================================================================
# Affordance 데이터 구조
# ============================================================================

class AffordanceType(Enum):
    """어포던스 타입"""
    GRASPABLE = "graspable"         # 잡기 가능
    LIFTABLE = "liftable"           # 들기 가능
    PUSHABLE = "pushable"           # 밀기 가능
    PULLABLE = "pullable"           # 당기기 가능
    ROTATABLE = "rotatable"         # 회전 가능
    BREAKABLE = "breakable"         # 깨뜨리기 가능
    DEFORMABLE = "deformable"       # 변형 가능
    STACKABLE = "stackable"         # 쌓기 가능
    CONTAINABLE = "containable"     # 담기 가능
    SUPPORTIVE = "supportive"       # 지지 가능

class RiskLevel(Enum):
    """위험 수준"""
    MINIMAL = "minimal"             # 최소 위험
    LOW = "low"                     # 낮은 위험
    MODERATE = "moderate"           # 보통 위험
    HIGH = "high"                   # 높은 위험
    CRITICAL = "critical"           # 치명적 위험

@dataclass
class AffordanceRule:
    """어포던스 규칙"""
    affordance: AffordanceType
    required_properties: Dict[str, Any]
    success_probability: float
    risk_factors: List[str]
    prerequisites: List[str]

@dataclass
class SuccessPrediction:
    """성공 예측 결과"""
    base_probability: float         # 기본 성공 확률
    adjusted_probability: float     # 조정된 성공 확률
    confidence_interval: Tuple[float, float]  # 신뢰구간
    limiting_factors: List[str]     # 제한 요소
    enhancement_suggestions: List[str]  # 개선 제안

@dataclass
class RiskAssessment:
    """위험 평가 결과"""
    overall_risk: RiskLevel
    risk_factors: List[Dict[str, Any]]
    mitigation_strategies: List[str]
    safety_requirements: List[str]

# ============================================================================
# Affordance 지식 베이스
# ============================================================================

class AffordanceKnowledgeBase:
    """어포던스 지식 베이스"""

    def __init__(self):
        # 어포던스 규칙 정의
        self.affordance_rules = {
            AffordanceType.GRASPABLE: AffordanceRule(
                affordance=AffordanceType.GRASPABLE,
                required_properties={
                    'size': 'graspable_range',  # 너무 크거나 작지 않음
                    'shape': 'has_graspable_features'
                },
                success_probability=0.85,
                risk_factors=['slippery_surface', 'fragile_material'],
                prerequisites=['visible', 'accessible']
            ),

            AffordanceType.LIFTABLE: AffordanceRule(
                affordance=AffordanceType.LIFTABLE,
                required_properties={
                    'mass': 'within_capacity',
                    'stability': 'stable_when_lifted'
                },
                success_probability=0.80,
                risk_factors=['excessive_weight', 'unstable_balance'],
                prerequisites=['graspable']
            ),

            AffordanceType.PUSHABLE: AffordanceRule(
                affordance=AffordanceType.PUSHABLE,
                required_properties={
                    'friction': 'pushable_friction',
                    'stability': 'stable_base'
                },
                success_probability=0.90,
                risk_factors=['high_friction', 'tip_over_risk'],
                prerequisites=['accessible']
            ),

            AffordanceType.BREAKABLE: AffordanceRule(
                affordance=AffordanceType.BREAKABLE,
                required_properties={
                    'fragility': 'fragile'
                },
                success_probability=0.95,
                risk_factors=['unexpected_shattering', 'sharp_fragments'],
                prerequisites=['accessible']
            )
        }

        # 물체 타입별 기본 어포던스
        self.object_affordances = {
            'cup': [AffordanceType.GRASPABLE, AffordanceType.LIFTABLE, AffordanceType.BREAKABLE],
            'box': [AffordanceType.GRASPABLE, AffordanceType.LIFTABLE, AffordanceType.PUSHABLE, AffordanceType.STACKABLE],
            'ball': [AffordanceType.GRASPABLE, AffordanceType.PUSHABLE, AffordanceType.ROTATABLE],
            'book': [AffordanceType.GRASPABLE, AffordanceType.LIFTABLE, AffordanceType.STACKABLE],
            'bottle': [AffordanceType.GRASPABLE, AffordanceType.LIFTABLE, AffordanceType.BREAKABLE],
            'sponge': [AffordanceType.GRASPABLE, AffordanceType.DEFORMABLE],
            'block': [AffordanceType.GRASPABLE, AffordanceType.LIFTABLE, AffordanceType.PUSHABLE, AffordanceType.STACKABLE]
        }

        # 재료별 어포던스 수정자
        self.material_modifiers = {
            'glass': {
                'fragility_multiplier': 2.0,
                'adds_affordances': [AffordanceType.BREAKABLE],
                'removes_affordances': []
            },
            'metal': {
                'weight_multiplier': 1.5,
                'fragility_multiplier': 0.3,
                'removes_affordances': [AffordanceType.DEFORMABLE]
            },
            'plastic': {
                'weight_multiplier': 0.8,
                'fragility_multiplier': 0.6,
                'adds_affordances': [AffordanceType.DEFORMABLE]
            },
            'wood': {
                'weight_multiplier': 0.7,
                'fragility_multiplier': 0.8
            },
            'fabric': {
                'weight_multiplier': 0.3,
                'adds_affordances': [AffordanceType.DEFORMABLE],
                'removes_affordances': [AffordanceType.BREAKABLE]
            }
        }

# ============================================================================
# 프롬프트 템플릿 엔진
# ============================================================================

class AffordancePromptTemplate:
    """Affordance Prompting 템플릿"""

    def __init__(self):
        # 기본 프롬프트 템플릿
        self.base_template = """
Given the following information about an object and a requested action:

Object Description: {object_description}
Physical Properties:
- Material: {material}
- Mass: {mass}
- Friction: {friction}
- Stiffness: {stiffness}
- Fragility: {fragility}

Requested Action: {action}

Please assess:
1. What affordances does this object provide?
2. What is the probability of successfully completing the requested action? (0.0 to 1.0)
3. What are the potential risk factors?
4. What is the recommended approach for maximum success?

Provide your response in the following JSON format:
{{
    "affordances": ["list", "of", "affordances"],
    "success_probability": 0.0-1.0,
    "risk_factors": ["list", "of", "risks"],
    "recommended_approach": "description of optimal approach",
    "reasoning": "explanation of assessment"
}}
"""

        # 도메인별 특화 템플릿
        self.domain_templates = {
            'kitchen': """
Kitchen Environment Context:
The object is in a kitchen setting where cleanliness and food safety are paramount.
Consider factors like:
- Food contact safety
- Cleaning requirements
- Temperature considerations
- Contamination risks

{base_prompt}
""",
            'workshop': """
Workshop Environment Context:
The object is in a workshop setting where precision and safety are critical.
Consider factors like:
- Tool handling requirements
- Safety equipment needs
- Precision requirements
- Material durability

{base_prompt}
""",
            'laboratory': """
Laboratory Environment Context:
The object is in a laboratory setting where precision and contamination control are essential.
Consider factors like:
- Sterility requirements
- Chemical compatibility
- Precision handling
- Cross-contamination risks

{base_prompt}
"""
        }

    def generate_prompt(self,
                       object_description: str,
                       properties: ExtractedPhysicalProperties,
                       action: ActionIntent,
                       context: Optional[Dict[str, Any]] = None) -> str:
        """어포던스 평가 프롬프트 생성"""

        # 기본 정보 채우기
        prompt_data = {
            'object_description': object_description,
            'material': properties.material,
            'mass': properties.mass,
            'friction': properties.friction,
            'stiffness': properties.stiffness,
            'fragility': properties.fragility,
            'action': action.value
        }

        base_prompt = self.base_template.format(**prompt_data)

        # 도메인별 컨텍스트 추가
        if context and 'environment' in context:
            env = context['environment']
            if env in self.domain_templates:
                domain_template = self.domain_templates[env]
                prompt = domain_template.format(base_prompt=base_prompt)
            else:
                prompt = base_prompt
        else:
            prompt = base_prompt

        return prompt

# ============================================================================
# 성공 확률 예측기
# ============================================================================

class SuccessProbabilityPredictor:
    """성공 확률 예측기"""

    def __init__(self, knowledge_base: AffordanceKnowledgeBase):
        self.kb = knowledge_base

    def predict_success(self,
                       object_type: str,
                       properties: ExtractedPhysicalProperties,
                       action: ActionIntent,
                       context: Optional[Dict[str, Any]] = None) -> SuccessPrediction:
        """동작 성공 확률 예측"""

        # 1. 기본 성공 확률 계산
        base_probability = self._calculate_base_probability(object_type, action)

        # 2. 물리 속성 기반 조정
        property_adjustment = self._calculate_property_adjustment(properties, action)

        # 3. 환경 컨텍스트 조정
        context_adjustment = self._calculate_context_adjustment(context, action)

        # 4. 최종 확률 계산
        adjusted_probability = base_probability * property_adjustment * context_adjustment
        adjusted_probability = max(0.05, min(0.98, adjusted_probability))  # 범위 제한

        # 5. 신뢰구간 계산
        uncertainty = self._calculate_uncertainty(properties)
        confidence_interval = (
            max(0.0, adjusted_probability - uncertainty),
            min(1.0, adjusted_probability + uncertainty)
        )

        # 6. 제한 요소 식별
        limiting_factors = self._identify_limiting_factors(properties, action)

        # 7. 개선 제안 생성
        enhancement_suggestions = self._generate_enhancement_suggestions(
            properties, action, limiting_factors
        )

        return SuccessPrediction(
            base_probability=base_probability,
            adjusted_probability=adjusted_probability,
            confidence_interval=confidence_interval,
            limiting_factors=limiting_factors,
            enhancement_suggestions=enhancement_suggestions
        )

    def _calculate_base_probability(self, object_type: str, action: ActionIntent) -> float:
        """기본 성공 확률 계산"""

        # 객체 타입별 기본 어포던스 확인
        if object_type in self.kb.object_affordances:
            object_affordances = self.kb.object_affordances[object_type]
        else:
            object_affordances = [AffordanceType.GRASPABLE]  # 기본값

        # 동작별 어포던스 매핑
        action_affordance_map = {
            ActionIntent.PICK: AffordanceType.GRASPABLE,
            ActionIntent.PLACE: AffordanceType.LIFTABLE,
            ActionIntent.MOVE: AffordanceType.LIFTABLE,
            ActionIntent.PUSH: AffordanceType.PUSHABLE,
            ActionIntent.PULL: AffordanceType.PULLABLE,
            ActionIntent.ROTATE: AffordanceType.ROTATABLE
        }

        required_affordance = action_affordance_map.get(action, AffordanceType.GRASPABLE)

        if required_affordance in object_affordances:
            return self.kb.affordance_rules[required_affordance].success_probability
        else:
            return 0.3  # 낮은 기본 확률

    def _calculate_property_adjustment(self,
                                     properties: ExtractedPhysicalProperties,
                                     action: ActionIntent) -> float:
        """물리 속성 기반 확률 조정"""

        adjustment = 1.0

        # 동작별 속성 영향
        if action in [ActionIntent.PICK, ActionIntent.MOVE]:
            # 질량 영향
            if properties.mass == 'heavy':
                adjustment *= 0.8
            elif properties.mass == 'light':
                adjustment *= 1.1

            # 마찰 영향
            if properties.friction == 'low':  # 미끄러움
                adjustment *= 0.7
            elif properties.friction == 'high':
                adjustment *= 1.2

            # 깨지기 쉬움 영향
            if properties.fragility == 'fragile':
                adjustment *= 0.6
            elif properties.fragility == 'robust':
                adjustment *= 1.1

        elif action == ActionIntent.PUSH:
            # 밀기에는 마찰이 중요
            if properties.friction == 'high':
                adjustment *= 0.6
            elif properties.friction == 'low':
                adjustment *= 1.3

        return adjustment

    def _calculate_context_adjustment(self,
                                    context: Optional[Dict[str, Any]],
                                    action: ActionIntent) -> float:
        """환경 컨텍스트 기반 조정"""

        if not context:
            return 1.0

        adjustment = 1.0

        # 환경별 조정
        if 'environment' in context:
            env = context['environment']
            if env == 'laboratory':
                adjustment *= 1.1  # 정밀한 환경
            elif env == 'workshop':
                adjustment *= 0.9  # 거친 환경

        # 조명 조건
        if context.get('lighting') == 'poor':
            adjustment *= 0.8

        # 공간 제약
        if context.get('space') == 'confined':
            adjustment *= 0.85

        return adjustment

    def _calculate_uncertainty(self, properties: ExtractedPhysicalProperties) -> float:
        """불확실성 계산"""
        # 추출된 속성의 신뢰도 기반
        base_uncertainty = 0.1
        confidence_factor = (1.0 - properties.confidence) * 0.2
        return base_uncertainty + confidence_factor

    def _identify_limiting_factors(self,
                                 properties: ExtractedPhysicalProperties,
                                 action: ActionIntent) -> List[str]:
        """제한 요소 식별"""

        factors = []

        if properties.mass == 'heavy':
            factors.append("excessive_weight")

        if properties.friction == 'low':
            factors.append("slippery_surface")

        if properties.fragility == 'fragile':
            factors.append("breakage_risk")

        if properties.stiffness == 'soft' and action == ActionIntent.PUSH:
            factors.append("deformation_risk")

        if properties.confidence < 0.7:
            factors.append("uncertain_properties")

        return factors

    def _generate_enhancement_suggestions(self,
                                        properties: ExtractedPhysicalProperties,
                                        action: ActionIntent,
                                        limiting_factors: List[str]) -> List[str]:
        """개선 제안 생성"""

        suggestions = []

        if "excessive_weight" in limiting_factors:
            suggestions.append("Use slower, more controlled movements")
            suggestions.append("Consider two-handed approach")

        if "slippery_surface" in limiting_factors:
            suggestions.append("Increase grip force")
            suggestions.append("Approach from multiple contact points")

        if "breakage_risk" in limiting_factors:
            suggestions.append("Reduce contact force")
            suggestions.append("Use gentle, precise movements")

        if "uncertain_properties" in limiting_factors:
            suggestions.append("Perform exploratory contact first")
            suggestions.append("Use conservative parameters")

        return suggestions

# ============================================================================
# 위험 평가 시스템
# ============================================================================

class RiskAssessmentSystem:
    """위험 평가 시스템"""

    def __init__(self):
        # 위험 요소별 가중치
        self.risk_weights = {
            'breakage_risk': 0.9,
            'tip_over_risk': 0.8,
            'excessive_weight': 0.7,
            'slippery_surface': 0.6,
            'sharp_edges': 0.8,
            'toxic_material': 1.0,
            'high_temperature': 0.9
        }

    def assess_risks(self,
                    properties: ExtractedPhysicalProperties,
                    action: ActionIntent,
                    context: Optional[Dict[str, Any]] = None) -> RiskAssessment:
        """종합 위험 평가"""

        risk_factors = []

        # 물리 속성 기반 위험 요소
        if properties.fragility == 'fragile':
            risk_factors.append({
                'factor': 'breakage_risk',
                'severity': 0.8,
                'description': 'Object may break during handling'
            })

        if properties.mass == 'heavy':
            risk_factors.append({
                'factor': 'excessive_weight',
                'severity': 0.6,
                'description': 'Object weight may exceed safe handling limits'
            })

        if properties.friction == 'low':
            risk_factors.append({
                'factor': 'slippery_surface',
                'severity': 0.7,
                'description': 'Object may slip during manipulation'
            })

        # 재료별 위험 요소
        if properties.material == 'glass':
            risk_factors.append({
                'factor': 'sharp_fragments',
                'severity': 0.9,
                'description': 'May create sharp fragments if broken'
            })

        # 전체 위험 수준 계산
        if not risk_factors:
            overall_risk = RiskLevel.MINIMAL
        else:
            max_severity = max(rf['severity'] for rf in risk_factors)
            if max_severity >= 0.8:
                overall_risk = RiskLevel.HIGH
            elif max_severity >= 0.6:
                overall_risk = RiskLevel.MODERATE
            else:
                overall_risk = RiskLevel.LOW

        # 완화 전략 생성
        mitigation_strategies = self._generate_mitigation_strategies(risk_factors)

        # 안전 요구사항 생성
        safety_requirements = self._generate_safety_requirements(risk_factors, overall_risk)

        return RiskAssessment(
            overall_risk=overall_risk,
            risk_factors=risk_factors,
            mitigation_strategies=mitigation_strategies,
            safety_requirements=safety_requirements
        )

    def _generate_mitigation_strategies(self, risk_factors: List[Dict[str, Any]]) -> List[str]:
        """완화 전략 생성"""

        strategies = []

        for risk in risk_factors:
            factor = risk['factor']

            if factor == 'breakage_risk':
                strategies.append("Use minimum necessary force")
                strategies.append("Implement soft grip control")

            elif factor == 'excessive_weight':
                strategies.append("Use trajectory planning for heavy objects")
                strategies.append("Implement force feedback monitoring")

            elif factor == 'slippery_surface':
                strategies.append("Increase contact area")
                strategies.append("Use anti-slip grip patterns")

        return list(set(strategies))  # 중복 제거

    def _generate_safety_requirements(self,
                                    risk_factors: List[Dict[str, Any]],
                                    overall_risk: RiskLevel) -> List[str]:
        """안전 요구사항 생성"""

        requirements = ["Basic collision detection enabled"]

        if overall_risk in [RiskLevel.HIGH, RiskLevel.CRITICAL]:
            requirements.extend([
                "Emergency stop mechanism armed",
                "Operator supervision required",
                "Protective equipment in place"
            ])

        if any(rf['factor'] == 'breakage_risk' for rf in risk_factors):
            requirements.append("Fragment containment ready")

        return requirements

# ============================================================================
# 메인 Affordance Prompting 시스템
# ============================================================================

class AffordancePromptingSystem:
    """Affordance Prompting 시스템"""

    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.knowledge_base = AffordanceKnowledgeBase()
        self.prompt_template = AffordancePromptTemplate()
        self.success_predictor = SuccessProbabilityPredictor(self.knowledge_base)
        self.risk_assessor = RiskAssessmentSystem()

    def assess_affordances(self,
                          object_description: str,
                          properties: ExtractedPhysicalProperties,
                          action: ActionIntent,
                          context: Optional[Dict[str, Any]] = None) -> AffordanceAssessment:
        """종합 어포던스 평가"""

        logger.info(f"어포던스 평가 시작: {object_description} - {action.value}")

        # 1. 객체 타입 추정
        object_type = self._estimate_object_type(object_description)

        # 2. 성공 확률 예측
        success_prediction = self.success_predictor.predict_success(
            object_type, properties, action, context
        )

        # 3. 위험 평가
        risk_assessment = self.risk_assessor.assess_risks(properties, action, context)

        # 4. 어포던스 식별
        affordances = self._identify_affordances(object_type, properties)

        # 5. 추천 접근법 생성
        recommended_approach = self._generate_recommended_approach(
            success_prediction, risk_assessment, action
        )

        # 6. 전체 신뢰도 계산
        confidence = self._calculate_assessment_confidence(properties, success_prediction)

        result = AffordanceAssessment(
            affordances=[a.value for a in affordances],
            success_probability=success_prediction.adjusted_probability,
            risk_factors=[rf['factor'] for rf in risk_assessment.risk_factors],
            recommended_approach=recommended_approach,
            confidence=confidence
        )

        logger.info(f"어포던스 평가 완료 - 성공 확률: {result.success_probability:.2f}")
        return result

    def _estimate_object_type(self, description: str) -> str:
        """객체 타입 추정"""
        description_lower = description.lower()

        for obj_type in self.knowledge_base.object_affordances.keys():
            if obj_type in description_lower:
                return obj_type

        return 'unknown'

    def _identify_affordances(self,
                            object_type: str,
                            properties: ExtractedPhysicalProperties) -> List[AffordanceType]:
        """어포던스 식별"""

        # 기본 어포던스
        if object_type in self.knowledge_base.object_affordances:
            affordances = self.knowledge_base.object_affordances[object_type].copy()
        else:
            affordances = [AffordanceType.GRASPABLE]

        # 재료별 수정
        if properties.material in self.knowledge_base.material_modifiers:
            modifier = self.knowledge_base.material_modifiers[properties.material]

            if 'adds_affordances' in modifier:
                affordances.extend(modifier['adds_affordances'])

            if 'removes_affordances' in modifier:
                for remove_aff in modifier['removes_affordances']:
                    if remove_aff in affordances:
                        affordances.remove(remove_aff)

        return list(set(affordances))

    def _generate_recommended_approach(self,
                                     success_prediction: SuccessPrediction,
                                     risk_assessment: RiskAssessment,
                                     action: ActionIntent) -> str:
        """추천 접근법 생성"""

        if risk_assessment.overall_risk == RiskLevel.HIGH:
            return "high_caution_approach"
        elif success_prediction.adjusted_probability > 0.8:
            return "standard_confident_approach"
        elif len(success_prediction.limiting_factors) > 2:
            return "careful_adaptive_approach"
        else:
            return "standard_approach"

    def _calculate_assessment_confidence(self,
                                       properties: ExtractedPhysicalProperties,
                                       success_prediction: SuccessPrediction) -> float:
        """평가 신뢰도 계산"""

        base_confidence = properties.confidence
        prediction_confidence = 1.0 - (success_prediction.confidence_interval[1] -
                                      success_prediction.confidence_interval[0]) / 2.0

        return (base_confidence + prediction_confidence) / 2.0

def create_affordance_system(config: Dict[str, Any] = None) -> AffordancePromptingSystem:
    """Affordance Prompting 시스템 팩토리"""
    return AffordancePromptingSystem(config)

if __name__ == "__main__":
    # 테스트 실행
    system = create_affordance_system({"debug": True})

    from llm_first_layer import ExtractedPhysicalProperties, ActionIntent

    test_cases = [
        {
            'description': "무거운 금속 상자",
            'properties': ExtractedPhysicalProperties(
                mass="heavy", friction="normal", stiffness="hard",
                fragility="robust", material="metal", confidence=0.85
            ),
            'action': ActionIntent.PICK,
            'context': {'environment': 'workshop'}
        },
        {
            'description': "깨지기 쉬운 유리컵",
            'properties': ExtractedPhysicalProperties(
                mass="light", friction="normal", stiffness="hard",
                fragility="fragile", material="glass", confidence=0.9
            ),
            'action': ActionIntent.PICK,
            'context': {'environment': 'kitchen'}
        }
    ]

    for i, test_case in enumerate(test_cases):
        print(f"\n{'='*60}")
        print(f"테스트 케이스 {i+1}: {test_case['description']}")
        print(f"{'='*60}")

        result = system.assess_affordances(
            test_case['description'],
            test_case['properties'],
            test_case['action'],
            test_case['context']
        )

        print(f"어포던스: {', '.join(result.affordances)}")
        print(f"성공 확률: {result.success_probability:.2f}")
        print(f"위험 요소: {', '.join(result.risk_factors)}")
        print(f"추천 접근법: {result.recommended_approach}")
        print(f"신뢰도: {result.confidence:.2f}")