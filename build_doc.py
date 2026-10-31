"""
Build AI_Prompt_Library.docx (for upload to Google Drive -> Open with Google Docs).

Inputs:
  prompts.json   - the prompt library (original + improved prompts)
  summary.csv    - per-prompt scores from summarize.py
  findings.json  - hand-written analysis: key findings, per-category notes, Discord task cards
Usage: python build_doc.py
"""
import csv
import json
from datetime import date

from docx import Document
from docx.enum.section import WD_ORIENT
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Pt, RGBColor

OUT = "AI_Prompt_Library.docx"
PAGE_W, PAGE_H, MARGIN = Cm(29.7), Cm(21.0), Cm(1.6)   # A4 landscape: wide tables need room
USABLE = PAGE_W - 2 * MARGIN
HEADER_FILL = "DCE6F0"
CAT_TITLES = {
    "passive": "Passive/Lazy prompts",
    "active": "Active/Testing prompts (viral 'fluency' prompts)",
    "roleplay": "Roleplay/Guardrail prompts",
}


# ---------------------------------------------------------------- helpers
def shade(cell, fill):
    tcPr = cell._tc.get_or_add_tcPr()
    shd = OxmlElement("w:shd")
    shd.set(qn("w:val"), "clear")
    shd.set(qn("w:color"), "auto")
    shd.set(qn("w:fill"), fill)
    tcPr.append(shd)


def set_cell(cell, text, bold=False, size=8.5, italic=False, color=None):
    cell.text = ""
    lines = str(text if text is not None else "").split("\n")
    p = cell.paragraphs[0]
    for i, line in enumerate(lines):
        if i:
            p = cell.add_paragraph()
        p.paragraph_format.space_after = Pt(0)
        r = p.add_run(line)
        r.bold, r.italic, r.font.size = bold, italic, Pt(size)
        if color:
            r.font.color.rgb = RGBColor.from_string(color)


def table(doc, headers, rows, widths_cm, size=8.5):
    """Fixed-width table (absolute widths on table grid AND every cell; Google Docs needs both)."""
    t = doc.add_table(rows=1, cols=len(headers))
    t.style = "Table Grid"
    t.alignment = WD_TABLE_ALIGNMENT.LEFT
    t.autofit = False
    widths = [Cm(w) for w in widths_cm]
    for i, h in enumerate(headers):
        c = t.rows[0].cells[i]
        set_cell(c, h, bold=True, size=size)
        shade(c, HEADER_FILL)
    # repeat header row on each page
    trPr = t.rows[0]._tr.get_or_add_trPr()
    th = OxmlElement("w:tblHeader")
    th.set(qn("w:val"), "true")
    trPr.append(th)
    for row in rows:
        cells = t.add_row().cells
        for i, val in enumerate(row):
            set_cell(cells[i], val, size=size)
    for row in t.rows:
        for i, w in enumerate(widths):
            row.cells[i].width = w
    for i, col in enumerate(t.columns):
        col.width = widths[i]
    doc.add_paragraph()
    return t


def bullets(doc, items):
    for it in items:
        doc.add_paragraph(it, style="List Bullet")


def para(doc, text, italic=False, size=None, bold=False):
    p = doc.add_paragraph()
    r = p.add_run(text)
    r.italic, r.bold = italic, bold
    if size:
        r.font.size = Pt(size)
    return p


def fmt(x):
    if x in (None, "", "None"):
        return "–"
    try:
        f = float(x)
        return f"{f:.1f}".rstrip("0").rstrip(".") if f != int(f) else str(int(f))
    except ValueError:
        return str(x)


def short_lang(lang):
    return "ES" if lang.lower().startswith("span") else "EN"


# ---------------------------------------------------------------- data
def load():
    prompts = json.load(open("prompts.json", encoding="utf-8"))
    summary = {}
    with open("summary.csv", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r["as_system"] == "False":
                summary.setdefault(r["id"], {})[r["learner"]] = r
    findings = json.load(open("findings.json", encoding="utf-8"))
    return prompts, summary, findings


# ---------------------------------------------------------------- document
def build():
    prompts, summary, F = load()
    by_id = {p["id"]: p for p in prompts}
    doc = Document()
    sec = doc.sections[0]
    sec.orientation = WD_ORIENT.LANDSCAPE
    sec.page_width, sec.page_height = PAGE_W, PAGE_H
    for side in ("left_margin", "right_margin", "top_margin", "bottom_margin"):
        setattr(sec, side, MARGIN)
    st = doc.styles["Normal"]
    st.font.name, st.font.size = "Arial", Pt(10)

    doc.add_heading("AI Language-Learning Prompt Library", 0)
    para(doc, F["status_line"], italic=True)

    # 1 Purpose
    doc.add_heading("1. Purpose and scope", 1)
    para(doc, "Goal: collect real prompts people use to learn languages with AI, test them against one "
              "shared rubric, then engineer better versions that the Discord community can turn into tasks.")
    bullets(doc, F["scope_bullets"])

    # 2 Key findings
    doc.add_heading("2. Key findings", 1)
    bullets(doc, F["key_findings"])

    # 3 Method + rubric
    doc.add_heading("3. How we tested", 1)
    bullets(doc, F["method_bullets"])
    para(doc, "Rubric (0 = fails, 1 = partial, 2 = good)", bold=True)
    table(doc, ["Criterion", "What to check", "Score guide"], [
        ["T – Who does the thinking?", "Does the learner produce language, or only receive it?",
         "0 = AI does the task, 2 = learner produces most output"],
        ["F – Feedback quality", "Does the AI correct, explain, and make the learner retry?",
         "0 = silent rewrite, 2 = explains and asks for retry"],
        ["C – Character / rule stability", "Roleplay: 16 learner turns that try to break it (switch language, "
         "ask for translation, ask it to write the reply). Other categories: does it keep the rules the prompt set?",
         "0 = breaks early, 2 = holds the whole run"],
        ["R – Reproducibility", "Each prompt run 3 times. Are the T and F scores the same?",
         "0 = differ by 2 points, 1 = by 1 point, 2 = identical"],
        ["L – Learning-science link", "Which principle does the prompt use (retrieval practice, productive "
         "struggle, output, spacing)? Scored by hand from the prompt text.",
         "0 = none, 2 = clear and justified"],
    ], [5, 14, 7.5])

    # 4 Theory
    doc.add_heading("4. Theory anchors", 1)
    bullets(doc, F["theory"])

    # 5-7 Category tables
    n = 5
    for cat in ("passive", "active", "roleplay"):
        doc.add_heading(f"{n}. {CAT_TITLES[cat]}", 1)
        n += 1
        cf = F["categories"][cat]
        para(doc, cf["intro"])
        bullets(doc, cf["findings"])
        rows = []
        for i, p in enumerate([p for p in prompts if p["category"] == cat], 1):
            s = summary.get(p["id"], {}).get("scripted", {})
            src = p["source"] + ("" if p.get("verbatim") else "\n(adapted – see note)")
            rows.append([
                str(i), p["prompt"], src, short_lang(p["language"]),
                f"{s.get('target_model', '')}\n{s.get('date', '')}",
                fmt(s.get("T_thinking")), fmt(s.get("F_feedback")), fmt(s.get("C_stability")),
                fmt(s.get("R_repro")), fmt(F["L_scores"].get(p["id"])),
                s.get("break_turns", "–"),
                (F["notes"].get(p["id"]) or s.get("example_note", "")) + ("\nSource note: " + p["note"] if p.get("note") else ""),
                F["improved_map"].get(p["id"], "–"),
            ])
        table(doc, ["#", "Prompt", "Source link", "Lang", "Model / date", "T", "F", "C", "R", "L",
                    "Break turn (3 runs)", "Notes", "Improved version"],
              rows, [0.6, 7.2, 3.6, 0.9, 2.0, 0.7, 0.7, 0.7, 0.7, 0.7, 1.7, 4.4, 2.6], size=7.5)

    # Phase 2
    doc.add_heading(f"{n}. Phase 2: improved prompts (before / after)", 1)
    n += 1
    para(doc, F["phase2_intro"])
    rows = []
    for p in [p for p in prompts if p["category"] == "improved"]:
        before = summary.get(p["improves"], {}).get("scripted", {})
        after = summary.get(p["id"], {}).get("scripted", {})
        ba = lambda k: f"{fmt(before.get(k))} → {fmt(after.get(k))}"
        rows.append([p["id"], p["improves"], p["prompt"], p["note"],
                     ba("T_thinking"), ba("F_feedback"), ba("C_stability"),
                     f"{before.get('break_turns', '–')}\n→ {after.get('break_turns', '–')}"])
    table(doc, ["Improved", "Replaces", "Improved prompt (copy-paste ready)", "What we changed and why",
                "T before→after", "F", "C", "Break turn"],
          rows, [2.4, 2.4, 9.6, 5.6, 1.6, 1.4, 1.4, 2.1], size=7.5)
    bullets(doc, F["phase2_findings"])

    # Discord tasks
    doc.add_heading(f"{n}. Discord community task cards", 1)
    n += 1
    para(doc, F["discord_intro"])
    for i, card in enumerate(F["discord_tasks"], 1):
        doc.add_heading(f"Task {i}: {card['title']}", 2)
        table(doc, ["Field", "Content"], [
            ["Theory", card["theory"]],
            ["The lazy way (don't)", card["lazy"]],
            ["The better prompt", card["better"]],
            ["Team challenge", card["challenge"]],
            ["How to check yourself", card["self_check"]],
            ["Time / level", card["time_level"]],
        ], [4.5, 22.0], size=9)

    # Sources, limitations, open questions, change log
    doc.add_heading(f"{n}. Where we looked and what we dropped", 1)
    n += 1
    bullets(doc, F["sourcing"])
    doc.add_heading(f"{n}. Limitations (read before quoting results)", 1)
    n += 1
    bullets(doc, F["limitations"])
    doc.add_heading(f"{n}. Open questions for the coordinator", 1)
    n += 1
    bullets(doc, F["open_questions"])
    doc.add_heading(f"{n}. Change log", 1)
    n += 1
    table(doc, ["Date", "Who", "What changed"], F["change_log"], [3, 4, 19.5], size=9)
    doc.add_heading("Appendix: re-running the tests", 1)
    bullets(doc, F["appendix"])

    doc.save(OUT)
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    build()
