# LLM-First Robot Control

자연어 명령에서 물리 속성을 추론하고, 로봇 제어 파라미터로 변환하는 LLM-first 로봇 제어 연구 프로젝트입니다.

> Portfolio position: robotics + LLM reasoning, structured control output, simulation-based validation.

## Problem

로봇 제어는 보통 명시적인 파라미터와 규칙을 요구합니다. 하지만 실제 사용자는 “조심히 들어 올려”, “깨지기 쉬운 물체를 옮겨”처럼 자연어로 의도를 전달합니다. 이 프로젝트는 자연어 명령에서 물체의 물리 속성과 affordance를 추론하고, `grip_force`, `lift_speed`, `safety_margin` 같은 제어 파라미터로 구조화하는 방식을 실험했습니다.

## What I Built

- DROID 공개 데이터 기반 instruction/control dataset 구성
- Qwen2.5-14B QLoRA fine-tuning 실험
- JSON schema 기반 제어 출력 구조화
- 물리 속성 추론 및 affordance 평가 모듈
- Genesis 기반 비교 실험과 Isaac Sim demo path
- 로봇 제어 파라미터를 `grip_force`, `lift_speed`, `density`, `friction` 등으로 매핑

## My Role

졸업논문 프로젝트로 데이터 구성, 모델 학습, 출력 schema 설계, 비교 실험, 시뮬레이션 검증을 주도했습니다. 핵심 목표는 “LLM 응답을 설명 텍스트가 아니라 검증 가능한 제어 파라미터”로 만드는 것이었습니다.

## Stack

| Area | Stack |
| --- | --- |
| Language model | Qwen2.5-14B, QLoRA |
| Robotics data | DROID dataset, instruction dataset |
| Simulation | Genesis, Isaac Sim demo path |
| Implementation | Python, JSON schema, evaluation scripts |
| Output contract | Structured physical/control parameters |

## Run

```bash
# core module smoke/integration checks
python test_phase2_integration.py

# DROID analysis and conversion path
python droid_dataset_analyzer.py
python droid_to_genesis_pipeline.py
python integrated_droid_llm_pipeline.py
```

Isaac Sim demo path:

```bash
# Isaac Sim 5.1.0 설치 후
cd ~/IsaacSim/_build/linux-x86_64/release/
./python.sh /path/to/isaacSim/demo_portfolio.py
```

## Validation Evidence

| Metric | Result |
| --- | --- |
| Proposed-method task success | 55.6% |
| Physical inference accuracy | 66.7% |
| Safety margin | 1.6 |
| JSON parsing compliance | 100% |
| Required-field compliance | 100% |
| Numeric-range compliance | 100% |

## Demo / Screenshots

![Isaac Sim Demo](isaacSim/assets/demo_preview.gif)

## Notes

이 README는 포트폴리오 판단에 필요한 핵심 구조와 검증 지표만 압축했습니다. 세부 실험/확장 아이디어는 저장소의 개별 scripts와 reports를 기준으로 확인합니다.
