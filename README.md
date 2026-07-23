# rel-sentiment

A quick probe: does RelativeDB's pretrained **RT-J** relational transformer
have any innate sense of **sentiment**, with zero training?

We never fit a head or fine-tune. We hand it a one-table "database" of
review sentences, mark some as demonstrations, mask the sentiment of a few
held-out sentences, and let the frozen checkpoints fill them in — using only
the two **head-free** task types:

- **Binary classification** — target is a `BOOLEAN` cell (`sentiment`), scored
  directly through the classification checkpoint's sigmoid.
- **Regression** — target is a `NUMBER` cell (`rating`, 0=neg .. 1=pos), scored
  through the regression checkpoint.

The multiclass path (mask a `TEXT` cell) is deliberately avoided because it
needs a fitted head; both tasks here are pure zero-shot inference.

## How it works

The model reads a **shared context**: the 24 labeled seed sentences become
in-context demonstrations (their sentiment cell stays visible), while each of
the 12 held-out sentences has its sentiment cell masked and predicted. Text is
embedded by the bundled MiniLM encoder, so polarity is already partly present
in the embedding space — the RT model just has to separate it.

```sql
PREDICT reviews.sentiment WHERE reviews.sentiment IS NULL   -- classification
PREDICT reviews.rating    WHERE reviews.rating    IS NULL   -- regression
```

## Run

**One command** (creates a venv, installs deps, runs the probe):

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

## Results (12 held-out sentences)

Both tasks classify **all 12** held-out sentences correctly, with confident,
well-separated scores.

| Task                  | Metric                | Score |
| --------------------- | --------------------- | ----- |
| Binary classification | accuracy @0.5         | **12/12 (100%)** |
| Regression            | sign accuracy @0.5    | **12/12 (100%)** |
| Regression            | mean rating, positive | +0.92 |
| Regression            | mean rating, negative | −0.06 |
| Regression            | separation (pos−neg)  | +0.98 |

### Binary classification — probability positive (threshold 0.5)

```
 p+     pred  true  sentence
 0.689  pos   pos   I really enjoyed this, it made my whole day better.
 0.704  pos   pos   An outstanding product that exceeded my expectations.
 0.709  pos   pos   Lovely atmosphere and the meal was superb.
 0.722  pos   pos   So glad I tried it, pure delight from start to finish.
 0.700  pos   pos   A wonderful surprise, everything went perfectly.
 0.687  pos   pos   Hands down the best I have ever seen.
 0.181  neg   neg   Horrible from start to finish, I want a refund.
 0.159  neg   neg   The quality is garbage and it stopped working immediately.
 0.179  neg   neg   What a disappointing mess, I hated every second.
 0.152  neg   neg   Terrible support and a product that simply does not work.
 0.233  neg   neg   I would never recommend this to anyone, avoid it.
 0.254  neg   neg   Painfully dull and not worth a single penny.
```
Positives cluster at **p ≈ 0.69–0.72**, negatives at **p ≈ 0.15–0.25** — a
clean gap with nothing near the 0.5 boundary.

### Regression — predicted rating (0 = neg .. 1 = pos)

```
 rating  pred  true  sentence
 +0.861  pos   pos   I really enjoyed this, it made my whole day better.
 +0.896  pos   pos   An outstanding product that exceeded my expectations.
 +0.965  pos   pos   Lovely atmosphere and the meal was superb.
 +0.970  pos   pos   So glad I tried it, pure delight from start to finish.
 +0.930  pos   pos   A wonderful surprise, everything went perfectly.
 +0.920  pos   pos   Hands down the best I have ever seen.
 -0.099  neg   neg   Horrible from start to finish, I want a refund.
 -0.111  neg   neg   The quality is garbage and it stopped working immediately.
 -0.029  neg   neg   What a disappointing mess, I hated every second.
 -0.137  neg   neg   Terrible support and a product that simply does not work.
 +0.079  neg   neg   I would never recommend this to anyone, avoid it.
 -0.059  neg   neg   Painfully dull and not worth a single penny.
```
Positives land near **+0.9**, negatives near **0.0** — a **+0.98** mean
separation.

### Takeaway

On clear-cut sentences the pretrained model separates positive from negative
cleanly, out of the box — so yes, there is a real innate sentiment signal, with
**zero training and no head fitting**. These are easy cases by design; the point
is to show the signal exists, not to benchmark hard ones.

## Files

- `data.py` — the labeled seed sentences and held-out test sentences.
- `sentiment.py` — builds the schema/wiring, runs both tasks, prints a report.
- `requirements.txt` — Python dependencies.
- `run.sh` — one-shot: create venv, install deps, run the probe.
