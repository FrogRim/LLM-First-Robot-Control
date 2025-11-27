"""
JSON Sanitizer - 다양한 응답 포맷에서 JSON 추출 및 정리
"""
import re
import json
from typing import Optional


def sanitize_and_extract_json(text: str, debug: bool = False) -> Optional[str]:
    """
    응답 텍스트에서 JSON을 추출하고 정리
    
    처리 순서:
    1. 코드펜스 제거 (```json, ```)
    2. 역할 토큰 제거 (assistant:, <|im_start|>, etc.)
    3. 첫 '{'부터 마지막 '}'까지 추출
    4. 트레일링 콤마 정리
    5. json.loads 검증
    """
    if debug:
        print(f"\n[DEBUG] Original text length: {len(text)}")
        print(f"[DEBUG] First 200 chars: {text[:200]}")
    
    # Step 1: 코드펜스 제거
    text = re.sub(r"```json\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"```\s*", "", text)
    
    # Step 2: 역할 토큰 제거
    text = re.sub(r"<\|im_start\|>assistant\s*", "", text)
    text = re.sub(r"<\|im_end\|>", "", text)
    text = re.sub(r"^assistant:\s*", "", text, flags=re.MULTILINE)
    text = re.sub(r"^Assistant:\s*", "", text, flags=re.MULTILINE)
    
    # Step 3: 첫 '{'부터 마지막 '}'까지 추출
    match = re.search(r"\{[\s\S]*\}", text)
    if not match:
        if debug:
            print("[DEBUG] No JSON block found")
        return None
    
    json_str = match.group(0)
    
    if debug:
        print(f"[DEBUG] Extracted JSON length: {len(json_str)}")
        print(f"[DEBUG] First 200 chars: {json_str[:200]}")
    
    # Step 4: 트레일링 콤마 정리 (간단한 패턴만)
    json_str = re.sub(r",(\s*[}\]])", r"\1", json_str)
    
    # Step 5: 검증 (불완전한 JSON 처리 시도)
    try:
        parsed = json.loads(json_str)
        if debug:
            print(f"[DEBUG] JSON parsed successfully, keys: {list(parsed.keys()) if isinstance(parsed, dict) else 'not a dict'}")
        return json_str
    except json.JSONDecodeError as e:
        # 불완전한 JSON 시도: 마지막 불완전 필드 제거 후 재시도
        if debug:
            print(f"[DEBUG] JSON parse failed: {e}")
            print(f"[DEBUG] Attempting to fix incomplete JSON...")
        
        # 마지막 불완전한 줄/필드 제거
        lines = json_str.rsplit('\n', 1)[0] if '\n' in json_str else json_str
        # 마지막 콤마 제거 및 닫기 괄호 추가
        lines = lines.rstrip().rstrip(',')
        
        # 중첩 레벨 계산하여 필요한 닫기 괄호 추가
        open_braces = lines.count('{') - lines.count('}')
        open_brackets = lines.count('[') - lines.count(']')
        
        fixed_json = lines
        for _ in range(open_brackets):
            fixed_json += ']'
        for _ in range(open_braces):
            fixed_json += '}'
        
        try:
            parsed = json.loads(fixed_json)
            if debug:
                print(f"[DEBUG] Fixed JSON parsed successfully!")
            return fixed_json
        except:
            if debug:
                print(f"[DEBUG] Could not fix JSON")
                print(f"[DEBUG] Problematic JSON: {json_str[:500]}")
            return None


def extract_json_robust(text: str, debug: bool = False) -> Optional[dict]:
    """
    견고한 JSON 추출 및 파싱
    
    Returns:
        파싱된 dict 또는 None
    """
    json_str = sanitize_and_extract_json(text, debug=debug)
    if not json_str:
        return None
    
    try:
        return json.loads(json_str)
    except:
        return None

