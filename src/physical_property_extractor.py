#!/usr/bin/env python3
"""
물리 속성 추론 엔진 (Physical Property Inference Engine)

졸업논문: "LLM-First 기반 물리 속성 추출 로봇 제어"
고급 물리 속성 추론 및 불확실성 정량화

질량(Mass), 마찰(Friction), 강성(Stiffness) 등의 속성을
자연어에서 정확하게 추론하고 신뢰도를 계산
"""

import re
import json
import numpy as np
import logging
from typing import Dict, List, Optional, Tuple, Any, Set
from dataclasses import dataclass, asdict
from enum import Enum
from collections import defaultdict
import math

from llm_first_layer import ExtractedPhysicalProperties

logger = logging.getLogger(__name__)

# ============================================================================
# 물리 속성 지식 베이스
# ============================================================================

@dataclass
class MaterialProperties:
    """재료별 물리 속성 데이터베이스"""
    density: float          # 밀도 (kg/m³)
    friction_coeff: float   # 마찰계수
    youngs_modulus: float   # 영률 (Pa)
    hardness: float         # 경도 (0-1)
    fragility: float        # 깨지기 쉬움 (0-1)
    thermal_conductivity: float  # 열전도율

class PhysicalPropertyDatabase:
    """물리 속성 데이터베이스"""

    def __init__(self):
        self.materials = {
            # 금속류
            'steel': MaterialProperties(7850, 0.74, 200e9, 0.9, 0.1, 50),
            'aluminum': MaterialProperties(2700, 0.61, 70e9, 0.7, 0.2, 237),
            'iron': MaterialProperties(7870, 0.74, 211e9, 0.85, 0.15, 80),
            'copper': MaterialProperties(8960, 0.53, 110e9, 0.6, 0.3, 401),

            # 플라스틱류
            'plastic': MaterialProperties(950, 0.35, 2e9, 0.4, 0.6, 0.2),
            'polymer': MaterialProperties(1200, 0.42, 3e9, 0.5, 0.5, 0.15),
            'pvc': MaterialProperties(1380, 0.45, 3.5e9, 0.6, 0.4, 0.16),

            # 목재류
            'wood': MaterialProperties(600, 0.65, 12e9, 0.5, 0.7, 0.12),
            'bamboo': MaterialProperties(350, 0.70, 20e9, 0.6, 0.8, 0.15),

            # 유리/세라믹
            'glass': MaterialProperties(2500, 0.94, 70e9, 0.9, 0.9, 1.05),
            'ceramic': MaterialProperties(2300, 0.85, 300e9, 0.95, 0.85, 2.0),

            # 직물/고무
            'fabric': MaterialProperties(200, 0.40, 0.01e9, 0.1, 0.3, 0.04),
            'rubber': MaterialProperties(920, 0.90, 0.002e9, 0.2, 0.1, 0.16),
            'foam': MaterialProperties(30, 0.30, 0.0001e9, 0.05, 0.2, 0.03),

            # 식품류
            'food_solid': MaterialProperties(800, 0.50, 0.1e9, 0.3, 0.8, 0.5),
            'food_liquid': MaterialProperties(1000, 0.20, 0.000001e9, 0.1, 0.9, 0.6),
        }

        # 크기별 질량 추정 (상대적)
        self.size_mass_mapping = {
            'tiny': 0.1,
            'small': 0.3,
            'medium': 1.0,
            'large': 3.0,
            'huge': 10.0
        }

        # 형용사별 속성 매핑
        self.adjective_mappings = {
            # 질량 관련
            'heavy': {'mass_multiplier': 2.5, 'confidence': 0.9},
            'light': {'mass_multiplier': 0.3, 'confidence': 0.9},
            'weighty': {'mass_multiplier': 2.0, 'confidence': 0.8},
            'massive': {'mass_multiplier': 5.0, 'confidence': 0.95},
            'feather': {'mass_multiplier': 0.1, 'confidence': 0.9},

            # 표면 특성
            'smooth': {'friction_coeff': 0.2, 'confidence': 0.8},
            'rough': {'friction_coeff': 0.8, 'confidence': 0.8},
            'slippery': {'friction_coeff': 0.1, 'confidence': 0.9},
            'grippy': {'friction_coeff': 0.9, 'confidence': 0.8},
            'sticky': {'friction_coeff': 1.2, 'confidence': 0.7},

            # 경도
            'soft': {'hardness': 0.2, 'youngs_modulus_factor': 0.1, 'confidence': 0.8},
            'hard': {'hardness': 0.9, 'youngs_modulus_factor': 10.0, 'confidence': 0.8},
            'rigid': {'hardness': 0.95, 'youngs_modulus_factor': 20.0, 'confidence': 0.9},
            'flexible': {'hardness': 0.3, 'youngs_modulus_factor': 0.2, 'confidence': 0.8},

            # 깨지기 쉬움
            'fragile': {'fragility': 0.9, 'confidence': 0.9},
            'delicate': {'fragility': 0.85, 'confidence': 0.8},
            'breakable': {'fragility': 0.8, 'confidence': 0.7},
            'sturdy': {'fragility': 0.2, 'confidence': 0.8},
            'robust': {'fragility': 0.1, 'confidence': 0.85},
        }

class ContextualAnalyzer:
    """문맥 분석기"""

    def __init__(self):
        # 문맥 단서 패턴
        self.context_patterns = {
            'kitchen': {
                'objects': ['cup', 'plate', 'bowl', 'knife', 'pan'],
                'material_bias': {'ceramic': 1.5, 'glass': 1.3, 'metal': 1.2}
            },
            'workshop': {
                'objects': ['tool', 'screw', 'bolt', 'wrench', 'hammer'],
                'material_bias': {'metal': 2.0, 'steel': 1.8}
            },
            'office': {
                'objects': ['paper', 'pen', 'book', 'folder', 'stapler'],
                'material_bias': {'plastic': 1.5, 'paper': 2.0}
            },
            'laboratory': {
                'objects': ['beaker', 'flask', 'tube', 'sample'],
                'material_bias': {'glass': 2.0, 'plastic': 1.5}
            }
        }

        # 크기 추정 단서
        self.size_indicators = {
            'tiny': ['tiny', 'small', 'mini', '작은', '소형'],
            'small': ['small', 'little', '작은', '소형'],
            'medium': ['normal', 'regular', 'standard', '보통', '일반'],
            'large': ['large', 'big', 'huge', '큰', '대형'],
            'huge': ['massive', 'enormous', 'giant', '거대한', '초대형']
        }

class UncertaintyQuantifier:
    """불확실성 정량화"""

    def __init__(self):
        self.base_confidence = 0.7

    def calculate_confidence(self,
                           evidence_count: int,
                           context_consistency: float,
                           knowledge_coverage: float) -> float:
        """종합 신뢰도 계산"""

        # 증거 개수에 따른 가중치
        evidence_weight = min(1.0, evidence_count / 3.0)

        # 문맥 일관성 가중치
        context_weight = context_consistency

        # 지식 베이스 커버리지 가중치
        knowledge_weight = knowledge_coverage

        # 가중 평균
        confidence = (
            0.4 * evidence_weight +
            0.3 * context_weight +
            0.3 * knowledge_weight
        )

        return max(0.3, min(0.95, confidence))

    def estimate_uncertainty_bounds(self,
                                  value: float,
                                  confidence: float) -> Tuple[float, float]:
        """불확실성 구간 추정"""
        uncertainty = (1.0 - confidence) * 0.5
        lower_bound = value * (1.0 - uncertainty)
        upper_bound = value * (1.0 + uncertainty)
        return lower_bound, upper_bound

# ============================================================================
# 고급 물리 속성 추론 엔진
# ============================================================================

class AdvancedPhysicalPropertyExtractor:
    """고급 물리 속성 추론 엔진"""

    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.database = PhysicalPropertyDatabase()
        self.context_analyzer = ContextualAnalyzer()
        self.uncertainty_quantifier = UncertaintyQuantifier()

        # 추론 결과 캐시
        self.inference_cache = {}

    def extract_properties(self,
                         text: str,
                         context: Optional[Dict[str, Any]] = None) -> ExtractedPhysicalProperties:
        """고급 물리 속성 추론"""

        # 캐시 확인
        cache_key = f"{text}_{hash(str(context))}"
        if cache_key in self.inference_cache:
            logger.info("캐시된 추론 결과 사용")
            return self.inference_cache[cache_key]

        logger.info(f"물리 속성 추론 시작: {text}")

        # 1. 텍스트 전처리 및 토큰화
        tokens = self._preprocess_text(text)

        # 2. 재료 인식
        material_info = self._identify_material(tokens, context)

        # 3. 형용사 기반 속성 추론
        adjective_properties = self._extract_adjective_properties(tokens)

        # 4. 문맥 기반 보정
        context_adjusted = self._apply_contextual_adjustment(
            adjective_properties, material_info, context
        )

        # 5. 크기 기반 질량 추정
        size_mass = self._estimate_size_based_mass(tokens, material_info)

        # 6. 통합 추론
        final_properties = self._integrate_inferences(
            material_info, context_adjusted, size_mass
        )

        # 7. 불확실성 정량화
        confidence = self._calculate_overall_confidence(tokens, material_info, final_properties)

        result = ExtractedPhysicalProperties(
            mass=final_properties['mass'],
            friction=final_properties['friction'],
            stiffness=final_properties['stiffness'],
            fragility=final_properties['fragility'],
            material=material_info['material'],
            confidence=confidence
        )

        # 캐시 저장
        self.inference_cache[cache_key] = result

        logger.info(f"추론 완료 - 신뢰도: {confidence:.3f}")
        return result

    def _preprocess_text(self, text: str) -> List[str]:
        """텍스트 전처리"""
        # 소문자 변환 및 구두점 제거
        text = re.sub(r'[^\w\s]', ' ', text.lower())
        tokens = text.split()
        return [token for token in tokens if len(token) > 1]

    def _identify_material(self,
                          tokens: List[str],
                          context: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """재료 식별"""

        # 직접적인 재료 언급 찾기
        identified_material = 'unknown'
        material_confidence = 0.0

        for token in tokens:
            if token in self.database.materials:
                identified_material = token
                material_confidence = 0.9
                break

        # 간접적 재료 추론
        if identified_material == 'unknown':
            # 재료 키워드 매칭
            material_keywords = {
                'metal': ['metal', 'steel', 'iron', 'aluminum', '금속', '철', '강철'],
                'plastic': ['plastic', 'polymer', '플라스틱'],
                'wood': ['wood', 'wooden', '나무', '목재'],
                'glass': ['glass', 'crystal', '유리', '크리스탈'],
                'fabric': ['fabric', 'cloth', 'textile', '천', '직물']
            }

            for material, keywords in material_keywords.items():
                if any(keyword in ' '.join(tokens) for keyword in keywords):
                    identified_material = material
                    material_confidence = 0.7
                    break

        # 문맥 기반 재료 추론
        if context and 'environment' in context:
            env = context['environment']
            if env in self.context_analyzer.context_patterns:
                pattern = self.context_analyzer.context_patterns[env]
                for material, bias in pattern.get('material_bias', {}).items():
                    if any(obj in tokens for obj in pattern['objects']):
                        if identified_material == 'unknown':
                            identified_material = material
                            material_confidence = 0.6 * bias

        return {
            'material': identified_material,
            'confidence': material_confidence,
            'properties': self.database.materials.get(identified_material,
                                                    self.database.materials['plastic'])
        }

    def _extract_adjective_properties(self, tokens: List[str]) -> Dict[str, Any]:
        """형용사 기반 속성 추론"""

        extracted = {
            'mass_factors': [],
            'friction_factors': [],
            'hardness_factors': [],
            'fragility_factors': []
        }

        for token in tokens:
            if token in self.database.adjective_mappings:
                adj_data = self.database.adjective_mappings[token]

                if 'mass_multiplier' in adj_data:
                    extracted['mass_factors'].append(adj_data)
                if 'friction_coeff' in adj_data:
                    extracted['friction_factors'].append(adj_data)
                if 'hardness' in adj_data:
                    extracted['hardness_factors'].append(adj_data)
                if 'fragility' in adj_data:
                    extracted['fragility_factors'].append(adj_data)

        return extracted

    def _apply_contextual_adjustment(self,
                                   adjective_props: Dict[str, Any],
                                   material_info: Dict[str, Any],
                                   context: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """문맥 기반 속성 조정"""

        base_props = material_info['properties']
        adjusted = {
            'mass_multiplier': 1.0,
            'friction_coeff': base_props.friction_coeff,
            'hardness': base_props.hardness,
            'fragility': base_props.fragility
        }

        # 형용사 요소들의 가중 평균
        for prop_type, factors in adjective_props.items():
            if not factors:
                continue

            if prop_type == 'mass_factors':
                weights = [f['confidence'] for f in factors]
                values = [f['mass_multiplier'] for f in factors]
                adjusted['mass_multiplier'] = np.average(values, weights=weights)

            elif prop_type == 'friction_factors':
                weights = [f['confidence'] for f in factors]
                values = [f['friction_coeff'] for f in factors]
                adjusted['friction_coeff'] = np.average(values, weights=weights)

            elif prop_type == 'hardness_factors':
                weights = [f['confidence'] for f in factors]
                values = [f['hardness'] for f in factors]
                adjusted['hardness'] = np.average(values, weights=weights)

            elif prop_type == 'fragility_factors':
                weights = [f['confidence'] for f in factors]
                values = [f['fragility'] for f in factors]
                adjusted['fragility'] = np.average(values, weights=weights)

        return adjusted

    def _estimate_size_based_mass(self,
                                tokens: List[str],
                                material_info: Dict[str, Any]) -> float:
        """크기 기반 질량 추정"""

        # 크기 지시어 찾기
        estimated_size = 'medium'  # 기본값

        for size, indicators in self.context_analyzer.size_indicators.items():
            if any(indicator in tokens for indicator in indicators):
                estimated_size = size
                break

        # 기본 밀도에서 크기 기반 질량 계산
        base_density = material_info['properties'].density
        size_factor = self.database.size_mass_mapping[estimated_size]

        # 상대적 질량 (실제 부피 정보 없이 추정)
        estimated_mass = (base_density / 1000.0) * size_factor

        return estimated_mass

    def _integrate_inferences(self,
                            material_info: Dict[str, Any],
                            adjusted_props: Dict[str, Any],
                            estimated_mass: float) -> Dict[str, Any]:
        """추론 결과 통합"""

        # 질량 분류
        mass_value = estimated_mass * adjusted_props['mass_multiplier']

        if mass_value < 0.5:
            mass_category = 'light'
        elif mass_value > 2.0:
            mass_category = 'heavy'
        else:
            mass_category = 'medium'

        # 마찰 분류
        friction_value = adjusted_props['friction_coeff']
        if friction_value < 0.3:
            friction_category = 'low'
        elif friction_value > 0.7:
            friction_category = 'high'
        else:
            friction_category = 'normal'

        # 강성 분류
        hardness_value = adjusted_props['hardness']
        if hardness_value < 0.4:
            stiffness_category = 'soft'
        elif hardness_value > 0.7:
            stiffness_category = 'hard'
        else:
            stiffness_category = 'medium'

        # 깨지기 쉬움 분류
        fragility_value = adjusted_props['fragility']
        if fragility_value > 0.7:
            fragility_category = 'fragile'
        elif fragility_value < 0.3:
            fragility_category = 'robust'
        else:
            fragility_category = 'normal'

        return {
            'mass': mass_category,
            'friction': friction_category,
            'stiffness': stiffness_category,
            'fragility': fragility_category,
            'mass_value': mass_value,
            'friction_value': friction_value,
            'hardness_value': hardness_value,
            'fragility_value': fragility_value
        }

    def _calculate_overall_confidence(self,
                                    tokens: List[str],
                                    material_info: Dict[str, Any],
                                    final_props: Dict[str, Any]) -> float:
        """전체 신뢰도 계산"""

        # 증거 개수 계산
        evidence_count = 0
        evidence_count += len([t for t in tokens if t in self.database.adjective_mappings])
        evidence_count += 1 if material_info['material'] != 'unknown' else 0

        # 문맥 일관성 (간단한 휴리스틱)
        context_consistency = material_info['confidence']

        # 지식 베이스 커버리지
        knowledge_coverage = 0.8 if material_info['material'] in self.database.materials else 0.4

        confidence = self.uncertainty_quantifier.calculate_confidence(
            evidence_count, context_consistency, knowledge_coverage
        )

        return confidence

    def get_property_uncertainty(self,
                               properties: ExtractedPhysicalProperties) -> Dict[str, Tuple[float, float]]:
        """속성별 불확실성 구간 반환"""

        uncertainty_bounds = {}

        # 수치형 속성들에 대한 불확실성 구간
        if hasattr(properties, 'mass_value'):
            bounds = self.uncertainty_quantifier.estimate_uncertainty_bounds(
                properties.mass_value, properties.confidence
            )
            uncertainty_bounds['mass'] = bounds

        return uncertainty_bounds

# ============================================================================
# 통합 인터페이스
# ============================================================================

def create_advanced_extractor(config: Dict[str, Any] = None) -> AdvancedPhysicalPropertyExtractor:
    """고급 물리 속성 추론기 팩토리"""
    return AdvancedPhysicalPropertyExtractor(config)

def analyze_extraction_quality(extraction_result: ExtractedPhysicalProperties) -> Dict[str, Any]:
    """추론 품질 분석"""

    quality_metrics = {
        'overall_confidence': extraction_result.confidence,
        'completeness': 0.0,
        'consistency': 0.0,
        'reliability': 'high' if extraction_result.confidence > 0.8 else 'medium' if extraction_result.confidence > 0.6 else 'low'
    }

    # 완전성 계산 (속성이 얼마나 구체적으로 추출되었는가)
    completeness = 0
    if extraction_result.mass != 'medium':
        completeness += 0.25
    if extraction_result.friction != 'normal':
        completeness += 0.25
    if extraction_result.stiffness != 'medium':
        completeness += 0.25
    if extraction_result.material != 'unknown':
        completeness += 0.25

    quality_metrics['completeness'] = completeness

    # 일관성 (속성들 간의 논리적 일관성)
    consistency_score = 1.0

    # 예: 금속인데 soft라면 일관성 낮음
    if extraction_result.material == 'metal' and extraction_result.stiffness == 'soft':
        consistency_score -= 0.3

    # 예: 유리인데 robust라면 일관성 낮음
    if extraction_result.material == 'glass' and extraction_result.fragility == 'robust':
        consistency_score -= 0.3

    quality_metrics['consistency'] = max(0.0, consistency_score)

    return quality_metrics

if __name__ == "__main__":
    # 테스트 실행
    extractor = create_advanced_extractor({"debug": True})

    test_cases = [
        {
            'text': "무거운 철제 상자를 조심스럽게 들어올려",
            'context': {'environment': 'workshop'}
        },
        {
            'text': "부드럽고 말랑한 스펀지를 부드럽게 눌러",
            'context': {'environment': 'kitchen'}
        },
        {
            'text': "깨지기 쉬운 유리컵을 안전하게 잡아",
            'context': {'environment': 'laboratory'}
        },
        {
            'text': "거친 표면의 나무 블록을 옮겨",
            'context': {'environment': 'workshop'}
        }
    ]

    for i, test_case in enumerate(test_cases):
        print(f"\n{'='*60}")
        print(f"테스트 케이스 {i+1}: {test_case['text']}")
        print(f"문맥: {test_case['context']}")
        print(f"{'='*60}")

        result = extractor.extract_properties(test_case['text'], test_case['context'])
        quality = analyze_extraction_quality(result)

        print(f"추출된 속성:")
        print(f"  질량: {result.mass}")
        print(f"  마찰: {result.friction}")
        print(f"  강성: {result.stiffness}")
        print(f"  깨지기 쉬움: {result.fragility}")
        print(f"  재질: {result.material}")
        print(f"  신뢰도: {result.confidence:.3f}")

        print(f"\n품질 분석:")
        print(f"  완전성: {quality['completeness']:.2f}")
        print(f"  일관성: {quality['consistency']:.2f}")
        print(f"  신뢰성: {quality['reliability']}")