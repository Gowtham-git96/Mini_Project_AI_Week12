# MP3 · FAQ

Answers to the questions that come up most. If you're stuck for more than ~30
minutes, skim here, then ask in the cohort chat.

---

## Setup

**Do I need to have finished MP2 first?**
No. The MP3 starter ships the whole MP2 pipeline already working — you only fill
in the four new functions. If you *did* do MP2 and want to reuse your own
pipeline, you're welcome to drop it in; just keep the four MP3 functions and the
CLI.

**`selftest` needs no API key — really?**
Really. `selftest` only exercises the four functions you write, using synthetic
data. It's your fast feedback loop. `ingest`/`ask`/`eval` need real OpenAI +
Qdrant credentials; `selftest` needs neither.

**The `export` keyword in `.env` again?**
Yes — `export OPENAI_API_KEY=...`, not `OPENAI_API_KEY=...`, then `source .env`.
Without `export`, your Python process won't see the variables. (Same footgun as
MP2/W7.)

---

## Citations

**What exactly should `format_context` return?**
A tuple: `(context_string, sources)`. The context string numbers each chunk so
the model can refer to it — e.g. `[1] (from "The Speckled Band" — ...)\n<text>`.
`sources` is a parallel list of dicts (`n, source, title, section, score`) so you
can later check the answer's `[n]` markers against real chunks.

**Why validate citations at all — won't the model just cite correctly?**
Usually, but not always. Models sometimes emit a `[4]` when only 3 chunks were
provided, or cite a chunk that doesn't actually support the claim. Catching the
out-of-range case (`validate_citations`) is a cheap, real guard — and finding one
in the wild is great reflection material.

**Should every sentence have a citation?**
Aim for every *claim*. The provided system prompt asks the model to cite after
each claim. You're grading whether citations are present and valid, not policing
exact placement.

---

## Abstention

**What threshold should I use?**
Start at the provided `RETRIEVAL_THRESHOLD = 0.30` (cosine). Then tune: ask the
unanswerable questions and print the top retrieved score. If a "Moriarty"
question retrieves a Roylott chunk at 0.28 and gets answered, raise the threshold.
If a real question abstains because its best chunk is 0.31, lower it. The Stretch
step is exactly this sweep.

**My assistant abstains on everything / nothing.**
- Everything → threshold too high, or `should_abstain` is comparing the wrong
  thing. Print `[c["score"] for c in chunks]`.
- Nothing → threshold too low (0.0?), or `should_abstain` always returns `False`.

**The model wrote a paragraph instead of the exact abstain phrase.**
The provided `answer()` also treats an answer *containing* the abstain phrase as
an abstention, so minor wording is fine. If the model answers from outside
knowledge instead of abstaining, that's a real hallucination — the score gate is
your primary defence, the prompt is the backup.

---

## Evaluation

**What are the four numbers?**
- **Retrieval hit rate** — of the answerable questions, how many had the expected
  source in the retrieved top-K. (Isolates retrieval quality.)
- **Grounded response rate** — of the answerable questions, how many were answered
  with a valid citation to the right source. (KPI.)
- **Abstention accuracy** — of the should-abstain questions, how many correctly
  refused.
- **Hallucination rate** — of the should-abstain questions, how many got answered
  anyway. (KPI. Lower is better.)

**How does `evaluate` know which questions should abstain?**
Each question in the JSONL is either answerable (`expected_source` set) or
should-abstain (`should_abstain: true` or `expected_source: null`). Your
`evaluate` branches on that.

**`format_eval_table` is provided — do I have to use it?**
It's provided so you don't spend time on ASCII formatting. It expects `evaluate`
to return `{"rows": [...], "summary": {...}}` with specific keys (documented in
the TODO). Match those keys and the table just works.

**Can I add more questions?**
Yes — that's Step 5. Two answerable + one abstain, in `learner_questions.jsonl`.
More is fine. Spread them across stories so retrieval has to discriminate.

---

## Submission

**What goes in the demo?**
2–3 minutes: run `ask`, show one answerable question producing a cited answer,
and one unanswerable question producing the "I don't know". That's the whole
point of the project in 60 seconds.

**Do I submit `mp3_eval.txt`?**
Yes — pipe the eval output to it: `python mp3_grounded_qa.py eval > mp3_eval.txt`.
It's the main graded artifact.

---

## Beyond MP3

**How does this map to the capstone?**
Directly. Grounding + abstention is exactly what the capstone's Enterprise
Knowledge Assistant does (don't invent HR policy). The four rates here are small
versions of the capstone's KPI dashboard — grounded response rate and
hallucination rate are two of the eight KPIs you report at every milestone.

**What would make this production-grade?**
A bigger, versioned question set; a proper eval framework (RAGAS, which the
capstone uses in W11); measuring citation *correctness* (does the cited chunk
actually support the claim?), not just presence; and tuning the threshold against
a measured false-abstention rate. All of that is capstone territory — MP3 is the
portfolio-sized taste.
