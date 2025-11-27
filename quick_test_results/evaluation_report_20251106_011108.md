# 비교 실험 평가 보고서

## 전체 통계
- 총 실험 수: 27
- 전체 성공률: 44.4%
- 평균 추론 시간: 9965.6ms

## 시나리오별 분석
### physical_property_reasoning
- 실행 수: 9
- 성공률: 44.4%
- 컨트롤러별 성능:
  - LLM-First: 66.7% 성공률, 29433.5ms 추론시간
  - RL-Based: 33.3% 성공률, 3.4ms 추론시간
  - Rule-Based: 33.3% 성공률, 0.0ms 추론시간

### multi_step_rearrangement
- 실행 수: 9
- 성공률: 0.0%
- 컨트롤러별 성능:
  - LLM-First: 0.0% 성공률, 28910.0ms 추론시간
  - RL-Based: 0.0% 성공률, 3.0ms 추론시간
  - Rule-Based: 0.0% 성공률, 0.0ms 추론시간

### object_sorting
- 실행 수: 9
- 성공률: 88.9%
- 컨트롤러별 성능:
  - LLM-First: 100.0% 성공률, 31333.8ms 추론시간
  - RL-Based: 100.0% 성공률, 7.0ms 추론시간
  - Rule-Based: 66.7% 성공률, 0.0ms 추론시간

## 컨트롤러별 분석
### LLM-First
- 실행 수: 9
- 성공률: 55.6%
- 실패 분석:
  - material_detection: 4건

### RL-Based
- 실행 수: 9
- 성공률: 44.4%
- 실패 분석:
  - material_detection: 5건

### Rule-Based
- 실행 수: 9
- 성공률: 33.3%
- 실패 분석:
  - material_detection: 6건
