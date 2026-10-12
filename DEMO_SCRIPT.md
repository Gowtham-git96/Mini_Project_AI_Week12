# MP3 Demo Script (2–3 minutes)

> A runbook to record the required demo: a couple of **cited answers** and one
> correct **"I don't know"** abstention. Follow the beats; the whole thing fits
> in under 3 minutes.

## Before you hit record (one-time setup)

```bash
cd "Week 12_Graded Mini Project_Gowtham P"
python -m venv .venv
.venv\Scripts\activate            # Windows
# source .venv/bin/activate       # macOS / Linux
pip install -r requirements.txt

copy .env.example .env            # Windows  (cp on mac/linux)
# edit .env: OPENAI_API_KEY + OPENAI_BASE_URL (Vocareum), QDRANT_URL, QDRANT_API_KEY
python mp3_grounded_qa.py ingest  # build the collection once (not shown in the recording)
```

Confirm the collection is built before recording so you don't wait on ingest.

---

## The recording (target ~2.5 min)

### Beat 1 — Frame it (0:00–0:20)

> "This is MP3, a grounded Q&A assistant over five Sherlock Holmes stories. It
> cites every claim with a numbered marker, and when the documents don't support
> an answer it abstains instead of guessing. I'll show both, then the eval table."

Optionally show the offline logic is sound (no keys needed):

```bash
python mp3_grounded_qa.py selftest
```

Point at **"✓ selftest passed"** — all four functions verified offline.

### Beat 2 — A cited answer (0:20–1:00)

```bash
python mp3_grounded_qa.py ask
```

Type an **answerable** question:

```
What killed Julia Stoner, and how was the method delivered?
```

Point at the inline `[1]`/`[2]` markers and the **Sources** block mapping each
marker to a real chunk.

### Beat 3 — A second cited answer, multi-hop (1:00–1:30)

```
In The Blue Carbuncle, where did James Ryder hide the stolen gem, and how did it end up with Henry Baker?
```

Point out the answer stitches facts across the Blue Carbuncle chunk(s) and still
cites its source.

### Beat 4 — A correct abstention (1:30–2:00)

```
Who is Professor Moriarty, and how does he meet his end?
```

Point at the exact line:

> **I don't know based on the provided documents.**
> (abstained — no supporting excerpt)

Press Enter on an empty line to exit `ask`.

### Beat 5 — The evaluation table (2:00–2:30)

```bash
python mp3_grounded_qa.py eval
```

Walk the four numbers: retrieval hit rate, grounded & cited rate (KPI),
abstention accuracy, hallucination rate (KPI, target 0.0).

---

## Questions cheat-sheet (copy/paste during the demo)

**Answerable (expect citations):**
- `What killed Julia Stoner, and how was the method delivered?`
- `In The Blue Carbuncle, where did James Ryder hide the stolen gem, and how did it end up with Henry Baker?`
- `What was the real purpose of the Red-Headed League, and how did Holmes identify the pawnbroker's assistant?`

**Must abstain (expect "I don't know"):**
- `Who is Professor Moriarty, and how does he meet his end?`
- `What is the boiling point of water in degrees Celsius?`
- `What breed was the spectral hound in The Hound of the Baskervilles, and who was behind the haunting?`

---

## If something goes sideways on camera

- **Qdrant slow / timeout** — free-tier clusters sleep; wait ~60s and retry. Pre-warm it before recording.
- **TLS reset on port 6333** — your network may block TLS on that port; switch networks or run Qdrant locally via Docker (`QDRANT_URL="http://localhost:6333"`).
- **It answers when it should abstain** — raise `RETRIEVAL_THRESHOLD` toward 0.35 in `.env`.
- **A valid question abstains** — lower `RETRIEVAL_THRESHOLD` toward 0.25.
