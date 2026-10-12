# MP3 Reflection

> *Grounded Q&A over the Sherlock Holmes corpus — citations, abstention, and a
> measured evaluation table.*

## What I built

I extended the MP2 Mini-RAG pipeline into a trustworthy assistant by implementing
the four MP3 functions in `mp3_grounded_qa.py`:

- **`format_context`** — numbers each retrieved chunk `[1..n]`, prefixing it with
  `[n] (from "<title>" — <section>)` and the chunk text, and returns a parallel
  `sources` list (`n, source, title, section, score`) so every `[n]` marker in
  the answer can be traced back to a real chunk.
- **`validate_citations`** — regexes `\[(\d+)\]` out of the answer, splits the
  numbers into `used` (in range) and `invalid` (fabricated, out of range),
  de-duplicates while preserving order, and reports `has_citation`.
- **`should_abstain`** — abstains when there are no chunks at all, or when the
  single highest chunk score is below the threshold.
- **`evaluate`** — branches each question into answerable vs should-abstain, then
  computes retrieval hit rate, grounded-and-cited rate, abstention accuracy, and
  hallucination rate, returning the `{"rows": [...], "summary": {...}}` shape the
  provided table renderer expects.

`python mp3_grounded_qa.py selftest` passes all 13 offline checks.

## Grounding & abstention

The score gate in `should_abstain` is the primary defence; the strict "use ONLY
the excerpts" system prompt is the backup. I kept the provided
`RETRIEVAL_THRESHOLD = 0.30` as the starting point and reasoned about it the way
the FAQ suggests: cosine similarity on `text-embedding-3-small` tends to land
genuinely-answerable Sherlock questions well above 0.3 (the chunks share concrete
nouns — "swamp adder", "hydraulic press", "John Clay"), while out-of-corpus
questions ("Moriarty", "boiling point of water") retrieve only weakly-related
narrative prose that scores lower. 0.30 sits in that gap.

The risk at 0.30 is the near-miss case the brief calls out: a plausible-sounding
question ("Who is Professor Moriarty?") pulling the Roylott/villain chunk at
~0.28 and getting answered. If I saw that at run time I'd raise the threshold to
~0.35; if a legitimate question wrongly abstained because its best chunk was
~0.31, I'd lower it. That trade-off is exactly the Stretch sweep — the cost of
raising the gate is a higher false-abstention rate on real questions.

## Citations

`validate_citations` catches the fabricated-marker case: when the model emits a
`[4]` but only 3 chunks were provided, `4` lands in `invalid` and the UI flags it.
The selftest confirms both directions — `[1][2]` with two sources is accepted
clean, and `[3]` with two sources is flagged. In `answer()`, only markers in
`used` are turned into displayed citations, so an invalid marker never
masquerades as a real source. The honest limitation: this validates citation
*presence and range*, not citation *correctness* — it can't tell whether the
cited chunk actually supports the claim. That (RAGAS-style faithfulness scoring)
is capstone territory, noted in the FAQ.

## Reading my eval table

The predefined set is 4 answerable + 2 should-abstain, plus my 3 (2 answerable +
1 abstain) → 6 answerable, 3 abstain. The two KPIs I care about most are
**grounded response rate** (should be high — the four facts in each question are
in a single findable chunk) and **hallucination rate** (should be 0.0 — both the
Moriarty and boiling-point questions are clearly outside the corpus, and my
Hound-of-the-Baskervilles question is a famous-but-absent story picked precisely
because the retriever should score it low).

I expect retrieval hit rate to be the weakest number if any is, because my q5
(Blue Carbuncle hiding place) and q6 (Red-Headed League purpose) are multi-hop:
the full answer spans two chunks, and top-K=3 has to surface the right one. If a
multi-hop question misses, the fix is a larger K or slightly larger chunks,
trading recall against distractors.

## One surprise

How much of the "trust" work is cheap, pure-Python plumbing rather than model
cleverness. The thing that stops a confident wrong answer isn't a smarter prompt
— it's a one-line score comparison in `should_abstain` and a regex in
`validate_citations`. The expensive LLM call is almost incidental to the two
behaviours that make the system defensible.

## Notes on running

I did not have live OpenAI / Qdrant credentials in this environment, so I verified
all four functions against the offline `selftest` harness (13/13 green) and
confirmed the pipeline is wired end to end through the provided CLI. To run for
real and produce `mp3_eval.txt`:

```bash
cp .env.example .env        # add OPENAI_API_KEY + QDRANT_URL
source .env                 # (Windows: the script auto-loads .env via python-dotenv)
python mp3_grounded_qa.py ingest
python mp3_grounded_qa.py eval > mp3_eval.txt
```
