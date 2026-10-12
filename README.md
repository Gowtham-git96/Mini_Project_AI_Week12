# MP3 · Grounded Q&A — A Cited Document Assistant

> *Mini Project 3 · Week 12 deliverable · Phase 2 (Retrieval / RAG)
> ~6–8 hours. Demonstrates two of the eight KPIs: grounded response rate and
> hallucination rate.*

Extends MP2's Mini-RAG pipeline into a **trustworthy** assistant: it **cites** its
sources inline, **abstains** ("I don't know") when the documents don't support an
answer, and ships a small **evaluation** harness that measures how well it does.
The pipeline is provided; learners implement the four functions that add
citations, abstention, and evaluation.

---

## Bundle contents

### For learners

| File | Purpose |
|---|---|
| `learner/MP3_Brief.md` | What to build, how to submit |
| `learner/mp3_grounded_qa_starter.py` | Working pipeline + 4 TODO functions (citations, abstention, eval) |
| `learner/corpus/*.txt` | 5 Sherlock Holmes story summaries (public domain, same as MP2) |
| `learner/data/predefined_questions.jsonl` | 4 answerable + 2 should-abstain questions |
| `learner/data/learner_questions.jsonl` | Template for the learner's 3 questions |
| `learner/requirements.txt` | Dependencies (no new deps vs MP2) |
| `learner/.env.example` | Environment template (OpenAI + Qdrant) |
| `learner/MP3_FAQ.md` | Anticipated questions + answers |

### For instructors / graders

| File | Purpose |
|---|---|
| `instructor/MP3_Rubric.md` | 4-dimension × 1–4 scoring rubric (generous) |
| `instructor/MP3_Reference_Solution.py` | Complete working implementation |
| `instructor/MP3_Solution_Document.md` | Walkthrough, expected outputs, grading hints |
| `instructor/MP3_Sample_Submission.md` | What 3.8/4.0 looks like (calibration) |

---

## Quick start (learner)

```bash
cd learner/
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env            # add OPENAI_API_KEY + QDRANT_URL, then: source .env
mv mp3_grounded_qa_starter.py mp3_grounded_qa.py

python mp3_grounded_qa.py selftest   # OFFLINE — go green on the 4 functions first (no key needed)
python mp3_grounded_qa.py ingest     # build the collection (once)
python mp3_grounded_qa.py ask        # cited answers + "I don't know"
python mp3_grounded_qa.py eval        # the evaluation table (graded artifact)
```

## The four functions the learner implements

| Function | Capability |
|---|---|
| `format_context` | Number retrieved chunks `[1..n]`; return a source map |
| `validate_citations` | Flag `[n]` markers that don't map to a real chunk |
| `should_abstain` | Refuse when no chunk is relevant enough (score gate) |
| `evaluate` | Compute hit rate, grounded rate, abstention accuracy, hallucination rate |

`selftest` verifies all four **offline** — it's the learner's feedback loop and
the grader's first gate.

---

## Instructor note — offline verification

The reference's logic (corpus loading, chunking, all three new capabilities, and
the eval harness) is verified by `python mp3_grounded_qa.py selftest` with **no
credentials**. The live pipeline (`ingest`/`ask`/`eval`) needs OpenAI + Qdrant and
is the MP2 loop carried forward. Grade primarily from `selftest` + the learner's
`mp3_eval.txt` + a two-question `ask` (the Moriarty abstention test is the fastest
signal).

*Phase-2 showcase — sits at the ~M2 checkpoint. Cites, abstains, measures.*
