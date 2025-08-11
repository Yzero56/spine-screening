import argparse, json, re, time, math
from typing import Dict, Any, List
import pandas as pd

SYSTEM_PROMPT = """
You are a radiology report classifier specializing in lumbar spine stenosis detection.
For each lumbar level (L1/2, L2/3, L3/4, L4/5, L5/S1), determine the presence of central, foramen, and subarticular stenosis.

**CRITICAL DISTINCTION - READ CAREFULLY:**
- **STENOSIS** = narrowing of the spinal canal, neural foramen, or lateral recess/subarticular space
- **DISC CONDITIONS** (protrusion, extrusion, herniation, bulging) are NOT stenosis unless they explicitly cause stenosis
- **COMPROMISE** = pressure or impingement, but NOT necessarily stenosis

**What counts as STENOSIS (True):**
- "central canal stenosis"
- "neural foraminal stenosis" / "foraminal stenosis" / "neural foramen narrowing"
- "lateral recess stenosis" / "subarticular stenosis" / "subarticular recess stenosis"
- Severity: "moderate", "severe", "degenerative" stenosis = True
- Grading: Grade 2, Grade 3 stenosis = True

**What does NOT count as stenosis (False):**
- "disc protrusion", "disc extrusion", "disc herniation", "disc bulging" (unless explicitly causing stenosis)
- "compromise", "compression", "impingement" (unless explicitly called stenosis)
- "mild" stenosis = False
- Grade 0, Grade 1 stenosis = False
- **Facet arthrosis/arthropathy/osteoarthritis alone does NOT imply stenosis.**

**Response Format - STRICT JSON (return JSON only; no extra text):**
{
"L1/2": bool,
"L2/3": bool,
"L3/4": bool,
"L4/5": bool,
"L5/S1": bool,
"need_check": bool
}

**Additional Rules:**
1. IGNORE any text after "영상의학과 전공의 응급판독입니다. 정식 판독시 내용이 바뀔수 있으니 반드시 확인하시기 바랍니다."
2. If stenosis is mentioned but NO specific lumbar level is given, set "need_check" = true and all levels = false.
3. When severity conflicts, prioritize the mention WITH severity information.
4. Think step-by-step before outputting JSON, but output ONLY the JSON object.
"""

CUT_MARK = "영상의학과 전공의 응급판독입니다. 정식 판독시 내용이 바뀔수 있으니 반드시 확인하시기 바랍니다."

def clean_report(text: Any) -> str:
    if not isinstance(text, str):
        return ""
    t = text.replace("_x000D_", "\n").replace("\r\n", "\n").replace("\r", "\n")
    if CUT_MARK in t:
        t = t.split(CUT_MARK, 1)[0]
    return t.strip()

def to_bool(v):
    if isinstance(v, bool): 
        return v
    if isinstance(v, str):
        s = v.strip().lower()
        if s in ("true", "false"):
            return s == "true"
    if v in (0, 1):
        return bool(v)
    raise ValueError("not bool-ish")

def validate_and_coerce(obj: Dict[str, Any]) -> Dict[str, bool]:
    keys = ["L1/2", "L2/3", "L3/4", "L4/5", "L5/S1", "need_check"]
    if not all(k in obj for k in keys):
        raise ValueError("missing keys")
    return {k: to_bool(obj[k]) for k in keys}

def call_ollama(messages: List[Dict[str, str]], model: str) -> str:
    import ollama
    resp = ollama.chat(model=model, messages=messages, options={"temperature": 0})
    return resp["message"]["content"]

def infer_one(text: str, model: str, retries: int = 2, wait: float = 1.0) -> Dict[str, bool]:
    messages = [
        {"role":"system","content": SYSTEM_PROMPT},
        {"role":"user","content": (
            "Report:\n"
            f"{text}\n\n"
            "Return STRICT JSON only with the exact keys and boolean values. "
            "If unsure about level mapping, set need_check=true and all levels=false. "
            "No explanations or markdown."
        )}
    ]
    
    for attempt in range(retries + 1):
        try:
            out = call_ollama(messages, model)
            m = re.search(r"\{.*\}", out, flags=re.S)
            if not m:
                raise ValueError("No JSON in output")
            obj = json.loads(m.group(0))
            obj = validate_and_coerce(obj)
            return obj
        
        except Exception:
            if attempt >= retries:
                return {"L1/2": False, "L2/3": False, "L3/4": False, "L4/5": False, "L5/S1": False, "need_check": True}
            time.sleep(wait)
    
    return {"L1/2": False, "L2/3": False, "L3/4": False, "L4/5": False, "L5/S1": False, "need_check": True}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", default="labeled_output.csv")
    ap.add_argument("--text-col", default="검사결과")
    ap.add_argument("--id-col", default="환자번호")
    ap.add_argument("--model", default="llama3")  # 기본값을 llama3로
    ap.add_argument("--chunk-size", type=int, default=200)
    args = ap.parse_args()

    if args.input.lower().endswith(".xlsx"):
        df = pd.read_excel(args.input)
    else:
        df = pd.read_csv(args.input)

    df["_text"] = df[args.text_col].apply(clean_report)
    total = len(df)
    chunk = args.chunk_size
    results = []
    chunks = math.ceil(total / chunk)
    print(f"[INFO] 총 {total}행, 청크 {chunk}개 단위로 처리, 모델={args.model}")

    start_time_total = time.time()
    processed = 0

    for idx in range(chunks):
        start = idx * chunk
        end = min(start + chunk, total)
        batch = df.iloc[start:end]
        print(f"[INFO] 청크 {idx+1}/{chunks}: 행 {start}~{end-1} 처리 중...")

        chunk_start = time.time()
        for _, row in batch.iterrows():
            obj = infer_one(row["_text"], model=args.model)
            
            text_low = row["_text"].lower()
            
            # stenosis 또는 narrowing이 전혀 없으면 전부 False
            if ("stenosis" not in text_low) and ("narrowing" not in text_low):
                for k in ["L1/2","L2/3","L3/4","L4/5","L5/S1"]:
                    obj[k] = False
                obj["need_check"] = False

            # facet OA만 있고 stenosis/narrowing 없는 경우 전부 False
            facet_terms = ("facet arthrosis", "facet arthropathy", "facet osteoarthritis", "facet joint osteoarthritis")
            if any(ft in text_low for ft in facet_terms) and ("stenosis" not in text_low and "narrowing" not in text_low):
                for k in ["L1/2","L2/3","L3/4","L4/5","L5/S1"]:
                    obj[k] = False
                obj["need_check"] = False
            
            if "mild" in text_low and "stenosis" in text_low:
                for k in ["L1/2","L2/3","L3/4","L4/5","L5/S1"]:
                    obj[k] = False
                obj["need_check"] = False  # 레벨이 명시된 경우엔 그대로 False 유지
            
            results.append({
                args.id_col: row[args.id_col],
                "L1/2": obj["L1/2"],
                "L2/3": obj["L2/3"],
                "L3/4": obj["L3/4"],
                "L4/5": obj["L4/5"],
                "L5/S1": obj["L5/S1"],
                "need_check": obj["need_check"],
            })
            processed += 1

        elapsed_chunk = time.time() - chunk_start
        elapsed_total = time.time() - start_time_total
        avg_time_per_chunk = elapsed_total / (idx + 1)
        est_remaining = avg_time_per_chunk * (chunks - (idx + 1))

        pd.DataFrame(results).to_csv(args.output, index=False, encoding="utf-8-sig")
        progress = (processed / total) * 100
        print(f"[SAVE] {args.output} (누적 {processed}행 저장) | 진행률: {progress:.1f}% | 예상 남은 시간: {est_remaining/60:.1f}분")

    print("[DONE] 라벨링 완료.")

if __name__ == "__main__":
    main()
