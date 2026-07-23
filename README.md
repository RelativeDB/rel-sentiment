# rel-sentiment

Does RelativeDB's pretrained **RT-J** relational transformer have an innate
sense of **sentiment** — with no training and, ideally, no examples at all?

Short answer: **yes.** Given a one-table "database" of review sentences and
nothing but a column literally named `sentiment`, the frozen model separates
positive from negative sentences perfectly. It never sees a label, a head is
never fitted, nothing is fine-tuned.

## What it does

We build a `reviews` table with a `text` column and a masked target, then ask
the two **head-free** checkpoints that ship with the model to fill the target
in for 12 held-out sentences:

- **Binary classification** — target is a `BOOLEAN` cell (`sentiment`), scored
  straight through the classification checkpoint's sigmoid.
- **Regression** — target is a `NUMBER` cell (`sentiment_rating`, 0=neg .. 1=pos),
  scored through the regression checkpoint.

```sql
PREDICT reviews.sentiment        WHERE reviews.sentiment        IS NULL   -- classification
PREDICT reviews.sentiment_rating WHERE reviews.sentiment_rating IS NULL   -- regression
```

The multiclass path (masking a `TEXT` cell) is deliberately avoided because it
needs a fitted head. Both tasks here are pure inference.

Each held-out sentence has its target cell masked; the model reads the sentence
(embedded by the bundled MiniLM encoder) plus the semantics of the column name,
and predicts. We report results with **zero examples** and with a single
labeled pair (**2-shot**) for calibration.

## Results (12 held-out sentences)

| Regime                | Task           | Accuracy @0.5 | pos vs neg mean | separation |
| --------------------- | -------------- | ------------- | --------------- | ---------- |
| **0-shot** (name only) | classification | **12/12 (100%)** | 0.630 vs 0.275 | +0.354 |
| **0-shot** (name only) | regression     | 8/12 (67%)   | +0.439 vs −0.509 | +0.948 |
| **2-shot** (1 pos + 1 neg) | classification | **12/12 (100%)** | 0.661 vs 0.190 | +0.471 |
| **2-shot** (1 pos + 1 neg) | regression | **12/12 (100%)** | +0.816 vs +0.020 | +0.796 |

**Binary classification is already perfect with zero examples** — the column
name `sentiment` alone is enough to separate positive (p ≈ 0.63) from negative
(p ≈ 0.28). **Regression** ranks polarity correctly zero-shot too (a +0.95
mean gap between positives and negatives), but its numeric scale is
uncalibrated — several positives land just under 0.5 — so a fixed 0.5 threshold
only catches 8/12. **One labeled pair fixes that**, pulling regression to 100%
with positives near +0.8 and negatives near 0.

### Zero-shot scores (column name only, no examples)

```
classification (prob positive)          regression (sentiment_rating)
 p+     true  sentence                    r       true  sentence
 0.616  pos   I really enjoyed this...    +0.334  pos   I really enjoyed this...
 0.638  pos   An outstanding product...   +0.389  pos   An outstanding product...
 0.615  pos   Lovely atmosphere...        +0.513  pos   Lovely atmosphere...
 0.662  pos   So glad I tried it...       +0.553  pos   So glad I tried it...
 0.614  pos   A wonderful surprise...     +0.485  pos   A wonderful surprise...
 0.632  pos   Hands down the best...      +0.358  pos   Hands down the best...
 0.202  neg   Horrible from start...      -0.247  neg   Horrible from start...
 0.114  neg   The quality is garbage...   -0.403  neg   The quality is garbage...
 0.358  neg   What a disappointing...     +0.067  neg   What a disappointing...
 0.130  neg   Terrible support...         -2.094  neg   Terrible support...
 0.481  neg   I would never recommend...  +0.236  neg   I would never recommend...
 0.367  neg   Painfully dull...           -0.615  neg   Painfully dull...
```

### 2-shot scores (one positive + one negative example)

```
classification (prob positive)          regression (sentiment_rating)
 p+     true  sentence                    r       true  sentence
 0.652  pos   I really enjoyed this...    +0.743  pos   I really enjoyed this...
 0.678  pos   An outstanding product...   +0.753  pos   An outstanding product...
 0.656  pos   Lovely atmosphere...        +0.868  pos   Lovely atmosphere...
 0.693  pos   So glad I tried it...       +0.897  pos   So glad I tried it...
 0.656  pos   A wonderful surprise...     +0.837  pos   A wonderful surprise...
 0.631  pos   Hands down the best...      +0.797  pos   Hands down the best...
 0.141  neg   Horrible from start...      -0.227  neg   Horrible from start...
 0.123  neg   The quality is garbage...   -0.094  neg   The quality is garbage...
 0.170  neg   What a disappointing...     +0.205  neg   What a disappointing...
 0.131  neg   Terrible support...         -0.150  neg   Terrible support...
 0.300  neg   I would never recommend...  +0.459  neg   I would never recommend...
 0.275  neg   Painfully dull...           -0.074  neg   Painfully dull...
```

These are clear-cut sentences by design — the point is to locate the signal,
not to benchmark hard cases.

## Run

**One command** (creates a venv, installs deps, runs everything):

```bash
./run.sh
```

**Or manually:**

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python sentiment.py
```

Requirements:

- **Python ≥ 3.10** on **macOS arm64 (Apple Silicon)** — the `relativedb` wheel
  bundles the native inference engine there. On other platforms, build
  `librt_c` from the [RelQL repo](https://github.com/RelativeDB/RelQL) and set
  `RELATIVEDB_RT_LIB` to point at it.
- First run downloads the RT-J checkpoints (~350MB) and the MiniLM encoder into
  `~/.cache/huggingface`; later runs are offline.

Everything else (numpy, sentence-transformers, huggingface_hub) is pulled in by
`relativedb` — see `requirements.txt`.

## Files

- `data.py` — the labeled sentences and the 12 held-out test sentences.
- `sentiment.py` — builds the schema/wiring, runs both tasks at 0- and 2-shot,
  prints the report.
- `requirements.txt` — Python dependencies.
- `run.sh` — one-shot: create venv, install deps, run.
