#!/usr/bin/env python3
"""Offline structural check; no model, predictions, or third-party imports."""
import argparse
import ast
import json
from pathlib import Path

def readl(p):
    with p.open(encoding="utf-8") as f:
        return [json.loads(x) for x in f if x.strip()]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", type=Path, default=Path(__file__).resolve().parent.parent / "data")
    args = ap.parse_args()
    d = args.data_dir
    tests = [r for p in sorted((d/"ti_mmlu/test").glob("*.jsonl")) for r in readl(p)]
    assert len(tests) == 669
    assert len({r["loc"] for r in tests}) == 669
    assert all(r["polished_ti_content"].strip() and str(r["answer"]).upper() in "ABCD" for r in tests)
    demo_files = sorted((d/"ti_mmlu/demos").glob("*.jsonl"))
    assert len(demo_files) == 67
    assert all(len(readl(p)) == 2 for p in demo_files)
    g = json.loads((d/"gsm8k_bo.json").read_text(encoding="utf-8"))
    assert len(g["test"]) == 500 and len(g["demos"]) == 2
    assert len({r["item_id"] for r in g["test"]}) == 500
    assert not {r["item_id"] for r in g["test"]} & {r["item_id"] for r in g["demos"]}
    assert all({"question_bo", "answer_bo", "answer_only", "item_id"} <= r.keys() for r in g["test"]+g["demos"])
    b = readl(d/"belebele_bod.jsonl")
    assert len(b) == 900
    assert all(int(r["correct_answer_num"]) in range(1,5) for r in b)
    m = json.loads((d/"zh_bo_pairs.json").read_text(encoding="utf-8"))
    assert len(m) == 3000 and all(r["zh"].strip() and r["bo"].strip() for r in m)
    for filename, count in (("mmlu_test.jsonl",14042),("ceval_val.jsonl",1346)):
        rows = readl(d/"retention"/filename)
        assert len(rows) == count and len({r["item_id"] for r in rows}) == count
        assert all(len(r["choices"]) == 4 and r["answer"] in "ABCD" for r in rows)
    files = list(Path(__file__).resolve().parent.rglob("*.py"))
    for p in files:
        ast.parse(p.read_text(encoding="utf-8"), filename=p.name)
    print(json.dumps({"status":"passed", "ti_mmlu":669, "gsm8k_bo":500,
                      "belebele":900,"translation_pairs":3000,"mmlu":14042,
                      "ceval":1346,"python_files":len(files)}))

if __name__ == "__main__":
    main()
