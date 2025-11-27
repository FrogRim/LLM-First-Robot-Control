# 비교 실험 시각화 보고서

## 1. 성공률 비교
![성공률 비교](success_rate_comparison.png)

## 2. 실행 효율성 비교
![효율성 비교](efficiency_comparison.png)

## 3. 실패 유형 분석
![실패 분석](failure_analysis.png)

## 4. 시나리오별 성능 히트맵
![성능 히트맵](scenario_performance_heatmap.png)

## 5. LLM 출력 예시

### 시나리오: Object Sorting
**명령어:** Sort the plastic bottle to the left side

**물리 분석 결과:**
```json
{
  "material_inference": "plastic",
  "mass_category": "light",
  "friction_coefficient": "high",
  "fragility": "low",
  "stiffness": "soft",
  "confidence": 0.85
}
```

**제어 파라미터:**
```json
{
  "grip_force": 0.2,
  "lift_speed": 0.71,
  "approach_angle": 0.0,
  "contact_force": 0.1,
  "safety_margin": 0.8
}
```

**추론 근거:** The plastic material is light with excellent surface grip. Lower grip force (0.2N) is sufficient. Faster lift speed (0.7142857142857143 m/s) is safe given material resilience.

**평가 결과:** ✅ 성공
---


### 시나리오: Object Sorting
**명령어:** Place the metal can on the right table

**물리 분석 결과:**
```json
{
  "material_inference": "metal",
  "mass_category": "heavy",
  "friction_coefficient": "high",
  "fragility": "low",
  "stiffness": "hard",
  "confidence": 0.85
}
```

**제어 파라미터:**
```json
{
  "grip_force": 0.7,
  "lift_speed": 0.4,
  "approach_angle": 0.0,
  "contact_force": 0.5,
  "safety_margin": 0.8
}
```

**추론 근거:** The metal nature indicates heavy mass. Standard grip force (0.7N) ensures secure grasp despite weight. Slower lift speed (0.4 m/s) maintains stability.

**평가 결과:** ✅ 성공
---


### 시나리오: Object Sorting
**명령어:** Move the glass cup to the right side carefully

**물리 분석 결과:**
```json
{
  "material_inference": "glass",
  "mass_category": "light",
  "friction_coefficient": "high",
  "fragility": "high",
  "stiffness": "hard",
  "confidence": 0.85
}
```

**제어 파라미터:**
```json
{
  "grip_force": 0.2,
  "lift_speed": 0.3,
  "approach_angle": 0.0,
  "contact_force": 0.1,
  "safety_margin": 1.6
}
```

**추론 근거:** The glass material is fragile. I reduce grip force to 0.2N and lift speed to 0.3 m/s to minimize breakage risk.

**평가 결과:** ✅ 성공
---


### 시나리오: Physical Property Reasoning
**명령어:** Pick up the glass cup very carefully without breaking it

**물리 분석 결과:**
```json
{
  "material_inference": "glass",
  "mass_category": "light",
  "friction_coefficient": "high",
  "fragility": "very_high",
  "stiffness": "hard",
  "confidence": 0.87
}
```

**제어 파라미터:**
```json
{
  "grip_force": 0.2,
  "lift_speed": 0.2,
  "approach_angle": 0.0,
  "contact_force": 0.1,
  "safety_margin": 1.0
}
```

**추론 근거:** The glass material is fragile. Minimal grip force (0.2N) prevents breakage and slow lift speed (0.2 m/s) maintains safety.

**평가 결과:** ✅ 성공
---


### 시나리오: Physical Property Reasoning
**명령어:** Handle the heavy metal object with strong grip

**물리 분석 결과:**
```json
{
  "material_inference": "metal",
  "mass_category": "heavy",
  "friction_coefficient": "high",
  "fragility": "low",
  "stiffness": "hard",
  "confidence": 0.85
}
```

**제어 파라미터:**
```json
{
  "grip_force": 1.0,
  "lift_speed": 0.3,
  "approach_angle": 0.0,
  "contact_force": 0.6,
  "safety_margin": 0.8
}
```

**추론 근거:** The metal nature indicates heavy mass. I apply maximum grip force to ensure secure grasp despite weight.

**평가 결과:** ❌ 실패
---


### 시나리오: Physical Property Reasoning
**명령어:** Move the light plastic item gently

**물리 분석 결과:**
```json
{
  "material_inference": "plastic",
  "mass_category": "light",
  "friction_coefficient": "high",
  "fragility": "low",
  "stiffness": "soft",
  "confidence": 0.85
}
```

**제어 파라미터:**
```json
{
  "grip_force": 0.2,
  "lift_speed": 0.8,
  "approach_angle": 0.0,
  "contact_force": 0.1,
  "safety_margin": 0.8
}
```

**추론 근거:** The light_plastic material is light with soft characteristics. Minimal grip force (0.2N) prevents damage while 0.8 m/s lift speed is safe due to robustness.

**평가 결과:** ✅ 성공
---

