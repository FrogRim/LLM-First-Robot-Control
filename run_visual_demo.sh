#!/bin/bash
# Genesis AI 시각적 데모 빠른 실행 스크립트

set -e

echo "=================================="
echo "Genesis AI 시각적 데모"
echo "=================================="

# 색상 정의
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# 환경 확인
echo -e "\n${YELLOW}[1/4] 환경 확인 중...${NC}"

# Python 확인
if ! command -v python &> /dev/null; then
    echo -e "${RED}✗ Python을 찾을 수 없습니다${NC}"
    exit 1
fi
echo -e "${GREEN}✓ Python 확인됨${NC}"

# 모델 디렉토리 확인
MODEL_DIR="./droid-physics-qwen14b-qlora"
if [ ! -d "$MODEL_DIR" ]; then
    echo -e "${RED}✗ 모델 디렉토리를 찾을 수 없습니다: $MODEL_DIR${NC}"
    echo "   학습을 먼저 완료해주세요: ./run_training.sh"
    exit 1
fi
echo -e "${GREEN}✓ 모델 디렉토리 확인됨: $MODEL_DIR${NC}"

# adapter_model.safetensors 확인
if [ ! -f "$MODEL_DIR/adapter_model.safetensors" ]; then
    echo -e "${RED}✗ 모델 파일을 찾을 수 없습니다: $MODEL_DIR/adapter_model.safetensors${NC}"
    exit 1
fi
echo -e "${GREEN}✓ 모델 파일 확인됨${NC}"

# Genesis AI 확인
echo -e "\n${YELLOW}[2/4] Genesis AI 설치 확인 중...${NC}"
GENESIS_CHECK=$(python -c "import genesis as gs; print('OK')" 2>/dev/null || echo "NOT_INSTALLED")

if [ "$GENESIS_CHECK" = "OK" ]; then
    echo -e "${GREEN}✓ Genesis AI 설치 확인됨 (시각화 활성화)${NC}"
    USE_GENESIS="yes"
else
    echo -e "${YELLOW}⚠ Genesis AI가 설치되지 않았습니다 (LLM 추론만 실행)${NC}"
    USE_GENESIS="no"
fi

# 모드 선택
echo -e "\n${YELLOW}[3/4] 실행 모드 선택${NC}"
echo "1) basic    - 기본 4개 시나리오 (권장)"
echo "2) advanced - 고급 4개 시나리오 (한국어 포함)"
echo "3) stress   - 스트레스 테스트 3개"
echo "4) all      - 전체 11개 시나리오"
echo "5) llm-only - LLM 추론만 (Genesis 없이)"

read -p "모드를 선택하세요 [1-5, 기본값=1]: " MODE_CHOICE

case $MODE_CHOICE in
    1|"")
        MODE="basic"
        ;;
    2)
        MODE="advanced"
        ;;
    3)
        MODE="stress"
        ;;
    4)
        MODE="all"
        ;;
    5)
        MODE="basic"
        USE_GENESIS="no"
        ;;
    *)
        echo -e "${RED}✗ 잘못된 선택입니다${NC}"
        exit 1
        ;;
esac

# 실행 명령 구성
CMD="python visual_demo.py --mode $MODE"

if [ "$USE_GENESIS" = "no" ]; then
    CMD="$CMD --no-genesis"
fi

# 실행
echo -e "\n${YELLOW}[4/4] 데모 실행 중...${NC}"
echo -e "${GREEN}명령: $CMD${NC}\n"

# 실행
$CMD

# 결과
echo -e "\n${GREEN}=================================="
echo "✅ 데모 완료!"
echo "==================================${NC}"
echo ""
echo "결과 파일 위치: ./demo_results/"
echo ""
echo "다음 단계:"
echo "  - 결과 파일 확인: ls -lh demo_results/"
echo "  - 추론 벤치마크: python scripts/benchmark_inference.py"
echo "  - JSON 평가: python scripts/eval_physics_json.py"
echo ""
