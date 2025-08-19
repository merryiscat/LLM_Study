# save as json_to_excel.py
import json, sys, pandas as pd
from pandas import json_normalize
from pathlib import Path

def to_sheets(obj):
    """
    JSON 형태에 따라 DataFrame(s)로 변환:
    - [ {...}, {...} ]  : 단일 시트
    - { "name":[...], "age":[...] } : 단일 시트(열로 맞춤)
    - { "users":[...], "logs":[...] } : 키마다 시트
    - 중첩 구조는 열을 펼침(sep='.')
    """
    if isinstance(obj, list):
        return {"data": json_normalize(obj, sep=".")}
    if isinstance(obj, dict):
        # dict의 각 값이 '레코드 리스트'면 시트로, 아니면 한 줄짜리 DF로
        sheets = {}
        complex_key_found = False
        for k, v in obj.items():
            if isinstance(v, list) and all(isinstance(x, dict) for x in v):
                sheets[k] = json_normalize(v, sep=".")
                complex_key_found = True
        if sheets:
            return sheets
        # 그 외 케이스: dict을 한 행짜리로
        return {"data": json_normalize(obj, sep=".")}
    # 그 외는 문자열/숫자 단일 값
    return {"data": pd.DataFrame([{"value": obj}])}

def main(in_path, out_path=None):
    in_path = Path(in_path)
    if out_path is None:
        out_path = in_path.with_suffix(".xlsx")

    with open(in_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    sheets = to_sheets(data)

    with pd.ExcelWriter(out_path, engine="xlsxwriter") as writer:
        for name, df in sheets.items():
            # 너무 긴 시트명 방지
            sheet = (str(name)[:31] or "data")
            # 리스트/스칼라 컬럼도 안전하게 변환
            for col in df.columns:
                df[col] = df[col].apply(lambda x: json.dumps(x, ensure_ascii=False) if isinstance(x, (list, dict)) else x)
            df.to_excel(writer, sheet_name=sheet, index=False)

    print(f"Saved -> {out_path}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python json_to_excel.py input.json [output.xlsx]")
        sys.exit(1)
    main(*sys.argv[1:3])
