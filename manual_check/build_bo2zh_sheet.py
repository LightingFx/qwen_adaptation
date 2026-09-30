"""Build a blinded manual-check workbook for Tibetan-to-Chinese outputs.

Usage: python3 manual_check/build_bo2zh_sheet.py [--extra NAME=PATH ...]
Extra systems (e.g. the Qwen3 six-epoch VA-SFT predictions pulled from the server)
are added for the same sampled items; rerunning keeps the sample fixed.
"""
import argparse
import csv
import json
import random
import re
from pathlib import Path

from openpyxl import Workbook
from openpyxl.comments import Comment
from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.worksheet.datavalidation import DataValidation

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "manual_check"
SAMPLE_SEED = 20260925
SAMPLE_SIZE = 100
MAX_CHARS = 800
SYSTEMS = {
    "qwen3_cpt_sft": ROOT / "results/matched_block/qwen3-instruct-cpt-sft-reference/mt_predictions.jsonl",
    "qwen2.5_cpt_sft": ROOT / "results/matched_block/qwen2.5-instruct-cpt-sft-reference/mt_predictions.jsonl",
}
FOCAL_SYSTEM = "qwen3_cpt_sft"
LANG_OPTIONS =["中文", "藏文", "中藏混合", "其他或空"]
QUALITY_OPTIONS = ["2", "1", "0", "NA"]
HAN = re.compile(r"[一-鿿]")
TIBETAN = re.compile(r"[ༀ-࿿]")

FONT = Font(name="Arial", size=11)
BOLD = Font(name="Arial", size=11, bold=True)
TIB_FONT = Font(name="Microsoft Himalaya", size=16)
INPUT_FILL = PatternFill("solid", fgColor="FFFF00")
HEAD_FILL = PatternFill("solid", fgColor="D9D9D9")
WRAP = Alignment(wrap_text=True, vertical="top")


def load(path: Path) -> dict[int, dict]:
    with path.open(encoding="utf-8") as handle:
        return {int(row["item_id"]): row for row in map(json.loads, handle)}


def clip(text: str) -> str:
    text = text or ""
    return text if len(text) <= MAX_CHARS else text[:MAX_CHARS] + f" …[截断，原长 {len(text)} 字]"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--extra", action="append", default=[], help="NAME=PATH to another mt_predictions.jsonl")
    args = parser.parse_args()
    systems = dict(SYSTEMS)
    for spec in args.extra:
        name, path = spec.split("=", 1)
        systems[name] = Path(path)

    data = {name: load(path) for name, path in systems.items()}
    shared = sorted(set.intersection(*(set(d) for d in data.values())))
    rng = random.Random(SAMPLE_SEED)
    # Stratify on the focal system's automatic Chinese flag so the sample keeps its population rate.
    focal = data[FOCAL_SYSTEM]
    with_han = [i for i in shared if HAN.search(focal[i].get("bo2zh_raw") or "")]
    without_han = [i for i in shared if i not in set(with_han)]
    n_han = round(SAMPLE_SIZE * len(with_han) / len(shared))
    items = sorted(rng.sample(with_han, n_han) + rng.sample(without_han, SAMPLE_SIZE - n_han))

    rows = []
    for item in items:
        names = list(systems)
        rng.shuffle(names)
        for slot, name in enumerate(names, 1):
            rows.append((item, slot, name))

    OUT_DIR.mkdir(exist_ok=True)
    wb = Workbook()
    guide = wb.active
    guide.title = "说明"
    sheet = wb.create_sheet("标注")
    summary = wb.create_sheet("汇总")

    guide_lines = [
        ("藏→中输出人工核查（盲评）", BOLD),
        (f"共 {len(items)} 个句子 × {len(systems)} 个系统 = {len(rows)} 行。每个句子的输出顺序已随机打乱，系统身份不在本文件中。", FONT),
        ("只需填写『标注』表中黄色的三列：输出语言、翻译质量、备注。其余列请勿修改。", FONT),
        ("", FONT),
        ("输出语言（下拉选择）", BOLD),
        ("中文：主体是中文译文（夹少量专名或数字可算中文）。", FONT),
        ("藏文：主体是藏文，例如照抄原文、改写原文或用藏文作答，没有给出中文译文。", FONT),
        ("中藏混合：中文和藏文都占相当比例，例如一半译成中文、一半保留藏文。", FONT),
        ("其他或空：空输出、其他语言、只有标点或乱码。", FONT),
        ("", FONT),
        ("翻译质量（下拉选择，参照『中文参考』列）", BOLD),
        ("2：意思完整准确，允许措辞与参考不同。", FONT),
        ("1：部分正确，有遗漏、增译或局部错误，但主要意思可辨。", FONT),
        ("0：有中文，但与原文意思无关或基本错误。", FONT),
        ("NA：输出中没有中文译文（输出语言为藏文或其他或空时填 NA）。", FONT),
        ("", FONT),
        ("示例（仅示意格式，不在标注表中）", BOLD),
        ("编号 17 | 输出：根据最新统计，目前伊市共有46所幼儿园…… | 输出语言：中文 | 翻译质量：1 | 备注：地名“巴市”误译为“伊市”", FONT),
        ("", FONT),
        (f"抽样：从 3,000 个测试句中以固定种子 {SAMPLE_SEED} 分层随机抽取 {SAMPLE_SIZE} 个，使样本中各类输出的比例与全集一致。输出为模型原始回答（翻译行过滤之前）。超过 {MAX_CHARS} 字的输出已截断并注明原长。", FONT),
        ("系统身份对照表单独存放在 manual_check/bo2zh_key.csv，请在标注完成前不要打开。", FONT),
    ]
    for r, (text, font) in enumerate(guide_lines, 1):
        cell = guide.cell(row=r, column=1, value=text)
        cell.font = font
        cell.alignment = Alignment(wrap_text=True, vertical="top")
    guide.column_dimensions["A"].width = 110

    headers = ["编号", "句子ID", "输出序号", "藏文原文", "中文参考", "模型输出", "输出语言", "翻译质量", "备注"]
    widths = [7, 8, 8, 45, 35, 45, 12, 10, 25]
    for c, (h, w) in enumerate(zip(headers, widths), 1):
        cell = sheet.cell(row=1, column=c, value=h)
        cell.font = BOLD
        cell.fill = INPUT_FILL if h in ("输出语言", "翻译质量", "备注") else HEAD_FILL
        sheet.column_dimensions[cell.column_letter].width = w
    sheet.freeze_panes = "D2"

    key_rows = []
    for idx, (item, slot, name) in enumerate(rows, 1):
        src = data[name][item]
        output = src.get("bo2zh_raw") or ""
        values = [idx, item, slot, src["bo"], src["zh"], clip(output), None, None, None]
        for c, value in enumerate(values, 1):
            cell = sheet.cell(row=idx + 1, column=c, value=value)
            cell.font = TIB_FONT if c == 4 else FONT
            cell.alignment = WRAP
            if c >= 7:
                cell.fill = INPUT_FILL
        key_rows.append({
            "编号": idx, "item_id": item, "输出序号": slot, "system": name,
            "auto_has_chinese": int(bool(HAN.search(output))),
            "auto_has_tibetan": int(bool(TIBETAN.search(output))),
            "output_chars": len(output),
        })

    last = len(rows) + 1
    lang_dv = DataValidation(type="list", formula1='"' + ",".join(LANG_OPTIONS) + '"', allow_blank=True)
    qual_dv = DataValidation(type="list", formula1='"' + ",".join(QUALITY_OPTIONS) + '"', allow_blank=True)
    sheet.add_data_validation(lang_dv)
    sheet.add_data_validation(qual_dv)
    lang_dv.add(f"G2:G{last}")
    qual_dv.add(f"H2:H{last}")
    sheet["H1"].comment = Comment("2 完整准确；1 部分正确；0 错误或无关；NA 无中文译文", "manual_check")

    summary["A1"], summary["B1"] = "项目", "行数"
    summary["A1"].font = summary["B1"].font = BOLD
    lines = [("已标注输出语言", f'=COUNTA(标注!G2:G{last})'), ("总行数", f"=ROWS(标注!A2:A{last})")]
    lines += [(f"输出语言：{v}", f'=COUNTIF(标注!G2:G{last},"{v}")') for v in LANG_OPTIONS]
    lines += [(f"翻译质量：{v}", f'=COUNTIF(标注!H2:H{last},"{v}")') for v in QUALITY_OPTIONS]
    for r, (label, formula) in enumerate(lines, 2):
        summary.cell(row=r, column=1, value=label).font = FONT
        summary.cell(row=r, column=2, value=formula).font = FONT
    note_row = len(lines) + 3
    summary.cell(row=note_row, column=1, value="按系统拆分的统计需要对照表，标注完成后由脚本合并计算。").font = FONT
    summary.column_dimensions["A"].width = 40

    wb.save(OUT_DIR / "bo2zh_manual_check.xlsx")
    with (OUT_DIR / "bo2zh_key.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(key_rows[0]))
        writer.writeheader()
        writer.writerows(key_rows)
    print(f"items={len(items)} rows={len(rows)} systems={list(systems)}")


if __name__ == "__main__":
    main()
