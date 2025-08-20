import json, sys, pandas as pd
from pandas import json_normalize
from pathlib import Path

MAX_ROWS = 1_000_000  # 엑셀 시트당 행 제한

def to_sheets(obj):
    """JSON → dict[str, DataFrame] 변환"""
    if isinstance(obj, list):
        return {"data": json_normalize(obj, sep=".")}

    if isinstance(obj, dict):
        sheets = {}
        for k, v in obj.items():
            if isinstance(v, list) and all(isinstance(x, dict) for x in v):
                sheets[k] = json_normalize(v, sep=".")
        if sheets:
            return sheets
        return {"data": json_normalize(obj, sep=".")}

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
            base = (str(name)[:31] or "data")

            # 리스트/스칼라 컬럼 안전 변환
            for col in df.columns:
                df[col] = df[col].apply(
                    lambda x: json.dumps(x, ensure_ascii=False) if isinstance(x, (list, dict)) else x
                )

            # 시트 분할 저장
            part = 1
            for start in range(0, len(df), MAX_ROWS):
                chunk = df.iloc[start:start + MAX_ROWS]
                sheet_name = base if part == 1 else f"{base}_{part}"
                sheet_name = sheet_name[:31]
                chunk.to_excel(writer, sheet_name=sheet_name, index=False)
                part += 1

    print(f"Saved -> {out_path}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python json_to_excel.py input.json [output.xlsx]")
        sys.exit(1)
    main(*sys.argv[1:3])
