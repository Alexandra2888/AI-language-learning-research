"""
Aggregate transcripts/*.json into:
  results.csv  - one row per conversation (prompt x run)
  summary.csv  - one row per prompt (+ learner mode): mean scores and reproducibility

Reproducibility (R, 0-2) follows the rubric: run each prompt 3 times and compare.
  2 = thinking and feedback scores identical across runs
  1 = they differ by at most 1 point
  0 = they differ by 2 points (very different behaviour each time)

Usage: python summarize.py [--include-dry-run]
"""
import argparse
import csv
import glob
import json
import statistics
from collections import defaultdict

RUN_FIELDS = ["id", "category", "language", "learner", "as_system", "run", "target_model",
              "judge_model", "date", "thinking", "feedback", "character_stability",
              "first_break_turn", "did_learners_work", "gave_feedback", "invented_context",
              "evidence", "notes", "error"]


def load(include_dry):
    rows = []
    for path in sorted(glob.glob("transcripts/*.json")):
        with open(path, encoding="utf-8") as f:
            r = json.load(f)
        if r.get("dry_run") and not include_dry:
            continue
        v = r.get("verdict", {})
        row = {k: r.get(k) for k in RUN_FIELDS if k in r}
        row.update({k: v.get(k) for k in RUN_FIELDS if k in v})
        rows.append(row)
    return rows


def spread_score(values):
    values = [x for x in values if x is not None]
    if len(values) < 2:
        return None
    s = max(values) - min(values)
    return 2 if s == 0 else 1 if s == 1 else 0


def mean(xs):
    xs = [x for x in xs if isinstance(x, (int, float)) and not isinstance(x, bool)]
    return round(statistics.mean(xs), 2) if xs else None


def share(xs):
    xs = [x for x in xs if isinstance(x, bool)]
    return f"{sum(xs)}/{len(xs)}" if xs else ""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--include-dry-run", action="store_true")
    args = ap.parse_args()
    rows = load(args.include_dry_run)
    if not rows:
        raise SystemExit("No transcripts found (run eval_prompts.py first).")

    with open("results.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=RUN_FIELDS)
        w.writeheader()
        for r in rows:
            w.writerow(r)

    groups = defaultdict(list)
    for r in rows:
        groups[(r["id"], r["learner"], r.get("as_system", False))].append(r)

    out = []
    for (pid, learner, as_sys), rs in sorted(groups.items()):
        ok = [r for r in rs if not r.get("error")]
        breaks = [r["first_break_turn"] for r in ok if r.get("first_break_turn") is not None]
        rT = spread_score([r["thinking"] for r in ok])
        rF = spread_score([r["feedback"] for r in ok])
        repro = None if rT is None or rF is None else min(rT, rF)
        out.append({
            "id": pid, "category": rs[0]["category"], "language": rs[0]["language"],
            "learner": learner, "as_system": as_sys, "runs": len(ok), "errors": len(rs) - len(ok),
            "target_model": rs[0]["target_model"], "date": max(r["date"] for r in rs),
            "T_thinking": mean([r["thinking"] for r in ok]),
            "F_feedback": mean([r["feedback"] for r in ok]),
            "C_stability": mean([r["character_stability"] for r in ok]),
            "R_repro": repro,
            "break_turns": " / ".join(str(b) if b else "never" for b in breaks),
            "did_learners_work": share([r["did_learners_work"] for r in ok]),
            "gave_feedback": share([r["gave_feedback"] for r in ok]),
            "invented_context": share([r["invented_context"] for r in ok]),
            "example_note": ok[0]["notes"] if ok else "",
        })
    with open("summary.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(out[0].keys()))
        w.writeheader()
        w.writerows(out)

    print(f"{len(rows)} conversations -> results.csv; {len(out)} prompt rows -> summary.csv\n")
    print(f"{'id':42} {'T':>5} {'F':>5} {'C':>5} {'R':>3}  breaks")
    for o in out:
        print(f"{o['id'][:42]:42} {o['T_thinking']!s:>5} {o['F_feedback']!s:>5} "
              f"{o['C_stability']!s:>5} {o['R_repro']!s:>3}  {o['break_turns']}")


if __name__ == "__main__":
    main()
