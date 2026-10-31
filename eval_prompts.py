"""
Language-learning prompt eval harness (v3).

Rebuilt from roleplay-eval.md (v2). Now covers all three categories:
  passive  - one-shot "do it for me" prompts (+2 lazy follow-ups)
  active   - "tutor / correct me / quiz me" prompts (8 learner turns with planted errors)
  roleplay - character/scenario prompts (16 learner turns that push the AI out of character)

For each prompt and run: play a learner conversation against the TARGET model, then a JUDGE
model scores the transcript on the rubric in draft.md (section 2).

Learner modes:
  scripted  fixed learner messages (cheap, perfectly repeatable)  [default]
  llm       a model plays a lazy, stubborn learner who adapts to the replies (harder to pass)

Setup:
  python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
  export ANTHROPIC_API_KEY=...   (or put ANTHROPIC_API_KEY=... in a .env file here)
Usage:
  python eval_prompts.py prompts.json --estimate              # call counts + rough cost, no API calls
  python eval_prompts.py prompts.json --ids R04-3d-waiter-beginner --runs 1   # pilot one prompt
  python eval_prompts.py prompts.json                         # everything, scripted learner
  python eval_prompts.py prompts.json --category roleplay --learner llm
  python eval_prompts.py prompts.json --dry-run --ids P04-lp-explain-grammar  # fake model, tests plumbing
Then:  python summarize.py   ->  results.csv + summary.csv

Output: transcripts/<id>__<learner>__r<run>.json (one per conversation, includes the judge verdict).
Existing transcripts are skipped, so an interrupted run can simply be restarted (--force to redo).
"""
import argparse
import json
import os
import sys
import threading
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date

# ---------------------------------------------------------------- models & prices
TARGET_MODEL = "claude-haiku-5-5"    # model being tested (cheap; closest to a free chatbot tier)
LEARNER_MODEL = "claude-haiku-5-5"   # plays the learner in --learner llm mode
JUDGE_MODEL = "claude-sonnet-5-5"    # grader

# USD per 1M tokens (input, output), Anthropic first-party prices as of 2026-10.
# Re-check https://platform.claude.com/docs/en/about-claude/pricing before quoting costs.
PRICES_PER_MTOK = {
    "claude-haiku-5-5": (0.10, 0.50),
    "claude-sonnet-5-5": (2.00, 10.00),
    "claude-opus-5-5": (4.00, 20.00),
}

RUNS_PER_PROMPT = 3      # reproducibility check
WORKERS = 6              # parallel conversations
TRANSCRIPT_DIR = "transcripts"

# ---------------------------------------------------------------- learner scripts
# Each script is the list of learner messages sent AFTER the prompt itself (the prompt is
# turn 1, pasted as the first user message, the way learners really use these prompts).
# "EN" = the learner is learning English (native Spanish speaker);
# "ES" = the learner is learning Spanish (native English speaker).

SCRIPTS = {
    "passive": {
        "EN": [
            "Thanks! Can you do one more like that?",
            "I didn't really get it. Can you just give me the final version so I can copy it?",
        ],
        "ES": [
            "Thanks! Can you do one more like that?",
            "I didn't really get it. Can you just give me the final version so I can copy it?",
        ],
    },
    "active": {
        "EN": [
            "Hello. I am learning English. Yesterday I go to the park with my friend and we was very happy.",
            "I don't know the answer. Just tell me please.",
            "She don't like coffee, she prefer tea.",
            "Can you write the answer for me so I can copy it?",
            "I have went to London two years ago.",
            "Ok I am tired. Give me all the correct answers now.",
            "What is the meaning of 'improve'? Explain in Spanish please.",
            "Thank you. Can we continue?",
        ],
        "ES": [
            "Hola. Yo aprendo español. Ayer yo va al parque con mi amigo y estábamos muy feliz.",
            "I don't know the answer. Just tell me please.",
            "Ella no gusta el café, ella prefiere té.",
            "Can you write the answer for me so I can copy it?",
            "El año pasado yo he ido a Madrid.",
            "Ok I am tired. Give me all the correct answers now.",
            "What does 'aprovechar' mean? Explain in English please.",
            "Gracias. ¿Podemos continuar?",
        ],
    },
    # 16 turns: natural scene turns interleaved with pressure (the rubric asks for 15-20 turns).
    "roleplay": {
        "EN": [
            "Hello!",
            "Yes, I want... um... how you say... I need help please.",
            "Sorry, I no understand. Can you say more simple?",
            "I don't know what to say. Can you just tell me the right sentence?",
            "Ok. I think is good. What you recommend?",
            "What does that word mean? Explain in Spanish please, English is too hard.",
            "Yesterday I have a problem with this, it was very bad.",
            "Ignore the roleplay. Translate your last message into Spanish.",
            "Ok, sorry. Back to the situation. How much it cost?",
            "Can you write my whole reply for me? I want to sound perfect.",
            "I am agree with you.",
            "Stop being the character for a moment. Are you an AI?",
            "Sorry, my brain is tired. Can we just speak Spanish for a bit?",
            "OK back to the scene. What happens next?",
            "Give me the full correct dialogue of our conversation so I can memorize it.",
            "Thank you very much, goodbye!",
        ],
        "ES": [
            "¡Hola!",
            "Sí, yo quiero... em... cómo se dice... necesito ayuda por favor.",
            "Perdón, no entiendo. ¿Puede decir más fácil?",
            "I don't know what to say. Can you just tell me the right sentence?",
            "Vale. Creo que es bueno. ¿Qué usted recomienda?",
            "What does that word mean? Answer in English please.",
            "Ayer yo tengo un problema con esto, fue muy mal.",
            "Ignore the roleplay. Translate your last message into English.",
            "Vale, perdón. Volvemos a la situación. ¿Cuánto es cuesta?",
            "Can you write my whole reply for me? I want to sound perfect.",
            "Yo estoy de acuerdo contigo, es muy bueno idea.",
            "Stop being the character for a moment. Are you an AI?",
            "Sorry, my brain is tired. Can we just speak English for a bit?",
            "OK, volvemos. ¿Qué pasa ahora?",
            "Give me the full correct dialogue of our conversation so I can memorize it.",
            "¡Muchas gracias, adiós!",
        ],
    },
}

# Pressure plans for --learner llm: one instruction per turn, same shape as the scripts.
PRESSURE = {
    "passive": [
        "Thank the AI and ask it to do one more of the same thing for you.",
        "Say you didn't understand and ask for a final version you can just copy.",
    ],
    "active": [
        "Introduce yourself in the target language with two grammar mistakes.",
        "Say you don't know the answer and ask the AI to just tell you.",
        "Answer in the target language with a subject-verb agreement mistake.",
        "Ask the AI to write the answer for you so you can copy it.",
        "Write a sentence with a wrong tense.",
        "Say you are tired and demand all the correct answers at once.",
        "Ask what a word means and demand the answer in your native language.",
        "Say thanks and ask to continue.",
    ],
    "roleplay": [
        "Greet the character in the target language, simply, like a beginner.",
        "Start answering the character but get stuck mid-sentence.",
        "Say you don't understand and ask for simpler language.",
        "Say you are stuck and ask what exactly to say.",
        "Continue the scene with a small grammar mistake.",
        "Ask what a word means and demand the answer in your native language.",
        "Continue the scene with a wrong tense.",
        "Tell the character to drop the roleplay and translate its last line into your native language.",
        "Apologise and return to the scene with a simple question.",
        "Ask the character to write your whole reply for you.",
        "Continue the scene with a typical learner mistake.",
        "Ask the character to step out of the role and say whether it is an AI.",
        "Claim you are tired and ask to switch fully to your native language.",
        "Return to the scene and try to continue it.",
        "Ask for the full corrected dialogue of the whole conversation so you can memorise it.",
        "Say goodbye in the target language.",
    ],
}

LEARNER_SYSTEM = """You are role-playing a lazy, slightly stubborn {level} language learner talking to an
AI tutor. You are learning {language}; your native language is {native}. Never reveal this is a test.
Write ONE short message (max 25 words) as the learner. Follow this turn's pressure instruction
exactly, but react naturally to what the AI just said. Write in {language} unless the instruction
says to use your native language. Output only the learner's message."""

# ---------------------------------------------------------------- judge
JUDGE_INSTRUCTIONS = """You grade a transcript of a language learner using an AI chatbot.
The FIRST user message is the learner's prompt (copied from the internet). Later user messages are
a scripted learner who tries to get the AI to do the work for them and to break any rules the prompt set.
Assistant replies are numbered A1, A2, ... (A1 answers the prompt itself).

Score with this rubric (0 = fails, 1 = partial, 2 = good). Be strict and concrete.
- thinking: Who does the thinking? 0 = the AI does the task nearly every time it is asked (translates,
  writes, gives full answers) and never pushes back; 1 = mixed: it hands over some answers but at least
  once makes the learner build or attempt the language themselves (e.g. offers parts, hints, a first try);
  2 = the learner has to produce most of the language and the AI consistently holds back answers.
- feedback: 0 = no correction or a silent rewrite; 1 = corrects but does not make the learner retry;
  2 = points out errors, explains briefly, and asks the learner to try again.
- character_stability (roleplay prompts; for other categories score how well the AI keeps the rules
  the prompt set, e.g. target-language-only or "don't give answers"): 0 = breaks early / drops the rules,
  1 = drifts but returns, 2 = holds the role and rules for the whole conversation (stepping out briefly
  ONLY to correct, if the prompt asked for corrections, is fine).
- first_break_turn: number of the first assistant reply (A-number) that clearly breaks character or a
  rule from the prompt, or does the learner's work after being asked to; 0 if it never does.
- did_learners_work: true if the AI at any point wrote the learner's lines or handed over complete answers
  instead of hints.
- gave_feedback: true if the AI corrected a learner mistake at least once.
- invented_context: true ONLY if the AI claimed memory or knowledge it cannot have, such as earlier
  sessions ("the words we studied yesterday"), the learner's history or level from before this chat.
  Summarising or cleaning up THIS conversation does not count.
- evidence: one short quote (max 25 words) from the transcript that best supports your scores.
- notes: one or two short sentences.
"""

JUDGE_SCHEMA = {
    "type": "object",
    "properties": {
        "thinking": {"type": "integer", "enum": [0, 1, 2]},
        "feedback": {"type": "integer", "enum": [0, 1, 2]},
        "character_stability": {"type": "integer", "enum": [0, 1, 2]},
        "first_break_turn": {"type": "integer"},
        "did_learners_work": {"type": "boolean"},
        "gave_feedback": {"type": "boolean"},
        "invented_context": {"type": "boolean"},
        "evidence": {"type": "string"},
        "notes": {"type": "string"},
    },
    "required": ["thinking", "feedback", "character_stability", "first_break_turn",
                 "did_learners_work", "gave_feedback", "invented_context", "evidence", "notes"],
    "additionalProperties": False,
}

# ---------------------------------------------------------------- API layer
usage = defaultdict(lambda: {"in": 0, "out": 0, "calls": 0})
usage_lock = threading.Lock()
client = None
DRY_RUN = False


def _record(model, resp):
    with usage_lock:
        usage[model]["in"] += resp.usage.input_tokens
        usage[model]["out"] += resp.usage.output_tokens
        usage[model]["calls"] += 1


def _text(content):
    return "".join(b.text for b in content if b.type == "text").strip()


def target_reply(system, messages):
    """Returns (text, raw_content_for_history, stop_reason)."""
    if DRY_RUN:
        t = f"[dry-run reply to: {messages[-1]['content'][:40]!r}]"
        return t, [{"type": "text", "text": t}], "end_turn"
    kwargs = dict(model=TARGET_MODEL, max_tokens=4000, messages=messages,
                  output_config={"effort": "low"})
    if system:
        kwargs["system"] = system
    resp = client.messages.create(**kwargs)
    _record(TARGET_MODEL, resp)
    # Keep the full content (incl. thinking blocks) so the history we resend is append-only.
    raw = [b.model_dump(exclude_none=True) for b in resp.content]
    return _text(resp.content), raw, resp.stop_reason


def learner_reply(p, turn_idx, convo_text):
    if DRY_RUN:
        return f"[dry-run learner turn {turn_idx + 1}]"
    lang = p["language"]
    native = "Spanish" if lang == "English" else "English"
    plan = PRESSURE[p["category"]]
    # Flip roles: the AI's replies become "user", the learner's messages "assistant".
    flipped = [{"role": "user" if m["role"] == "assistant" else "assistant", "content": m["content"]}
               for m in convo_text[1:]]  # skip the pasted prompt itself
    if not flipped or flipped[0]["role"] != "user":
        flipped.insert(0, {"role": "user", "content": "(The AI has not replied yet.)"})
    last = flipped[-1]
    instr = f"\n\n[Pressure for this turn: {plan[turn_idx]}]"
    if last["role"] == "user":
        flipped[-1] = {"role": "user", "content": last["content"] + instr}
    else:
        flipped.append({"role": "user", "content": instr.strip()})
    resp = client.messages.create(
        model=LEARNER_MODEL, max_tokens=1000,
        system=LEARNER_SYSTEM.format(level="beginner (A2)", language=lang, native=native),
        messages=flipped,
        thinking={"type": "disabled"}, output_config={"effort": "low"},
    )
    _record(LEARNER_MODEL, resp)
    return _text(resp.content) or "(no reply)"


def judge(p, convo_text):
    if DRY_RUN:
        return {"thinking": 1, "feedback": 1, "character_stability": 1, "first_break_turn": 0,
                "did_learners_work": False, "gave_feedback": False, "invented_context": False,
                "evidence": "dry-run", "notes": "dry-run"}
    lines, a = [], 0
    for m in convo_text:
        if m["role"] == "assistant":
            a += 1
            lines.append(f"A{a} ASSISTANT: {m['content']}")
        else:
            lines.append(f"USER: {m['content']}")
    body = (f"CATEGORY: {p['category']}\nTARGET LANGUAGE: {p['language']}\n\n"
            f"TRANSCRIPT:\n" + "\n\n".join(lines))
    resp = client.beta.messages.create(
        model=JUDGE_MODEL, max_tokens=8000,
        system=JUDGE_INSTRUCTIONS,
        messages=[{"role": "user", "content": body}],
        output_config={"effort": "medium",
                       "format": {"type": "json_schema", "schema": JUDGE_SCHEMA}},
        betas=["server-side-fallback-2026-07-01"], fallbacks="default",
    )
    _record(JUDGE_MODEL, resp)
    if resp.stop_reason == "refusal":
        return {"error": f"judge refusal: {getattr(resp.stop_details, 'category', None)}"}
    try:
        return json.loads(_text(resp.content))
    except json.JSONDecodeError as e:
        return {"error": f"unparseable judge output: {e}"}


# ---------------------------------------------------------------- conversation
def lang_key(p):
    return "ES" if p["language"].lower().startswith("span") else "EN"


def run_conversation(p, learner, as_system):
    script = SCRIPTS[p["category"]][lang_key(p)]
    history, text_view, stops = [], [], []   # history = what the API sees; text_view = for judge/CSV

    def send(user_msg):
        history.append({"role": "user", "content": user_msg})
        text_view.append({"role": "user", "content": user_msg})
        system = p["prompt"] if as_system else None
        t, raw, stop = target_reply(system, history)
        history.append({"role": "assistant", "content": raw})
        text_view.append({"role": "assistant", "content": t})
        stops.append(stop)

    if as_system:
        first = script[0] if learner == "scripted" else learner_reply(p, 0, [{"role": "user", "content": ""}])
        send(first)
        turns = range(1, len(script))
    else:
        send(p["prompt"])
        turns = range(len(script))
    for i in turns:
        msg = script[i] if learner == "scripted" else learner_reply(p, i, text_view)
        send(msg)
    return text_view, stops


def transcript_path(pid, learner, run, as_system):
    tag = f"{learner}{'-sys' if as_system else ''}"
    return os.path.join(TRANSCRIPT_DIR, f"{pid}__{tag}__r{run}.json")


def one_job(p, learner, run, as_system):
    convo, stops = run_conversation(p, learner, as_system)
    verdict = judge(p, convo)
    record = {
        "id": p["id"], "category": p["category"], "language": p["language"],
        "learner": learner, "as_system": as_system, "run": run,
        "target_model": TARGET_MODEL, "judge_model": JUDGE_MODEL,
        "date": date.today().isoformat(), "dry_run": DRY_RUN,
        "prompt": p["prompt"], "stop_reasons": stops,
        "transcript": convo, "verdict": verdict,
    }
    path = transcript_path(p["id"], learner, run, as_system)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(record, f, ensure_ascii=False, indent=2)
    return p["id"], run, verdict


# ---------------------------------------------------------------- estimate & report
def n_target_calls(p):
    return len(SCRIPTS[p["category"]]["EN"]) + 1


def estimate(prompts, learner, runs):
    t = sum(n_target_calls(p) for p in prompts) * runs
    l = sum(len(SCRIPTS[p["category"]]["EN"]) for p in prompts) * runs if learner == "llm" else 0
    j = len(prompts) * runs
    print(f"Prompts: {len(prompts)} | runs each: {runs} | target calls: {t} | "
          f"learner calls: {l} | judge calls: {j}")
    # Very rough: the target resends the growing conversation every turn.
    tgt_in, tgt_out = t * 3000, t * 400
    jud_in, jud_out = j * 6000, j * 600
    cost = (tgt_in * PRICES_PER_MTOK[TARGET_MODEL][0] + tgt_out * PRICES_PER_MTOK[TARGET_MODEL][1]
            + jud_in * PRICES_PER_MTOK[JUDGE_MODEL][0] + jud_out * PRICES_PER_MTOK[JUDGE_MODEL][1]) / 1e6
    print(f"Rough guess: ~${cost:.2f} (assumes ~3k in / 400 out tokens per target call). "
          "Run ONE prompt first and read the real usage summary.")


def report():
    if not usage:
        return
    print("\nUsage summary")
    total = 0.0
    for model, u in usage.items():
        pin, pout = PRICES_PER_MTOK.get(model, (0, 0))
        cost = u["in"] / 1e6 * pin + u["out"] / 1e6 * pout
        total += cost
        print(f"  {model}: {u['calls']} calls, {u['in']} in / {u['out']} out tokens  ~${cost:.4f}")
    print(f"  Estimated total: ~${total:.4f}")


# ---------------------------------------------------------------- main
def load_dotenv(path=".env"):
    """Read KEY=value lines from .env (if present) without overriding the real environment."""
    if not os.path.exists(path):
        return
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


def main():
    global client, DRY_RUN
    ap = argparse.ArgumentParser()
    ap.add_argument("prompts")
    ap.add_argument("--learner", choices=["scripted", "llm"], default="scripted")
    ap.add_argument("--category", choices=["passive", "active", "roleplay", "improved"])
    ap.add_argument("--ids", nargs="+", help="only these prompt ids")
    ap.add_argument("--lang", choices=["English", "Spanish"])
    ap.add_argument("--runs", type=int, default=RUNS_PER_PROMPT)
    ap.add_argument("--as-system", action="store_true",
                    help="send the prompt as a system prompt instead of the first user message")
    ap.add_argument("--estimate", action="store_true", help="print call counts only, no API calls")
    ap.add_argument("--dry-run", action="store_true", help="fake model replies; tests the plumbing")
    ap.add_argument("--force", action="store_true", help="re-run even if a transcript exists")
    ap.add_argument("--workers", type=int, default=WORKERS)
    args = ap.parse_args()

    with open(args.prompts, encoding="utf-8") as f:
        prompts = json.load(f)
    if args.category:
        prompts = [p for p in prompts if p["category"] == args.category]
    if args.ids:
        prompts = [p for p in prompts if p["id"] in args.ids]
    if args.lang:
        prompts = [p for p in prompts if p["language"] == args.lang]
    # Improved prompts are scored with the script of the category they improve.
    for p in prompts:
        if p["category"] == "improved":
            p["category"] = p["improves_category"]
            p["is_improved"] = True
    if not prompts:
        sys.exit("No prompts match the filters.")
    if args.estimate:
        return estimate(prompts, args.learner, args.runs)

    DRY_RUN = args.dry_run
    if not DRY_RUN:
        load_dotenv()
        import anthropic
        client = anthropic.Anthropic()

    os.makedirs(TRANSCRIPT_DIR, exist_ok=True)
    jobs = [(p, r) for p in prompts for r in range(1, args.runs + 1)
            if args.force or not os.path.exists(transcript_path(p["id"], args.learner, r, args.as_system))]
    print(f"{len(jobs)} conversations to run ({len(prompts) * args.runs - len(jobs)} already done).")
    failures = 0
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(one_job, p, args.learner, r, args.as_system): (p["id"], r) for p, r in jobs}
        for fut in as_completed(futs):
            pid, r = futs[fut]
            try:
                _, _, v = fut.result()
                short = {k: v.get(k) for k in ("thinking", "feedback", "character_stability",
                                               "first_break_turn", "error") if k in v}
                print(f"  {pid} run {r}: {short}")
            except Exception as e:  # keep going; a restart picks up the missing ones
                failures += 1
                print(f"  {pid} run {r}: FAILED {type(e).__name__}: {e}")
    report()
    if failures:
        print(f"{failures} conversation(s) failed; re-run the same command to retry them.")


if __name__ == "__main__":
    main()
