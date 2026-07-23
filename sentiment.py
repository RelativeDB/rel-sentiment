"""Zero-shot sentiment probing of the RT-J relational transformer.

We build a one-table "database" of movie/product-review sentences and ask the
*pretrained* model to fill in a masked sentiment cell for held-out sentences.
No training, no head fitting — just the two frozen checkpoints that ship with
the model:

  binary classification  -> hf://stanford-star/rt-j/classification   (sigmoid prob)
  regression             -> hf://stanford-star/rt-j/regression       (continuous)

Both are "head-free": the target cell is a BOOLEAN (classification) or NUMBER
(regression) column, so the model scores it directly. (The multiclass path,
which masks a TEXT cell and needs a fitted head, is deliberately avoided.)

How little context can it do this with?

  shots=0  FULLY ZERO-SHOT. No labeled examples at all. The ONLY hint about
           what to predict is the column *name* — `is_positive` / `positivity` —
           which the RT model embeds as text. Every row's target is masked;
           the model infers polarity from the column name + the sentence.
  shots=2  ONE positive + ONE negative demonstration in context.

Run:  python sentiment.py
"""
from __future__ import annotations

from datetime import datetime, timezone

from relativedb import (Engine, ExecutionInput, RetrieverWiring, Row,
                        RtNativeBackend, Schema, TableDef, TemporalBound,
                        ValueType)

from data import SEED, TEST

ANCHOR = datetime(2026, 7, 23, tzinfo=timezone.utc)   # rows are static; any anchor works

# Target-column names. In the zero-shot regime this name is the only signal
# the model has for "what am I predicting", so it carries real weight.
CLF_COL = "sentiment"          # BOOLEAN: True = positive sentiment
REG_COL = "sentiment_rating"   # NUMBER : 0 = negative .. 1 = positive


def _balanced_seeds(n: int):
    """Pick n demonstrations, alternating positive/negative for balance."""
    pos = [e for e in SEED if e.positive]
    neg = [e for e in SEED if not e.positive]
    out = []
    for i in range(n):
        out.append(pos[i // 2] if i % 2 == 0 else neg[i // 2])
    return out


# --------------------------------------------------------------------------
# Build the one-table "reviews" database for a given task and shot count.
# Seed rows carry the real label; test rows carry None (the masked target).
# shots=0 -> no seed rows: prediction rests entirely on the column name.
# --------------------------------------------------------------------------
def build(task: str, shots: int):
    is_clf = task == "classification"
    target = CLF_COL if is_clf else REG_COL
    ttype = ValueType.BOOLEAN if is_clf else ValueType.NUMBER

    schema = (Schema.new_schema()
              .table(TableDef.new_table("reviews")
                     .column("text", ValueType.TEXT)
                     .column(target, ttype)
                     .primary_key("review_id").build())
              .build())

    def label(ex):
        if is_clf:
            return ex.positive
        return 1.0 if ex.positive else 0.0

    rows = []
    for i, ex in enumerate(_balanced_seeds(shots)):
        rows.append(Row("reviews", f"seed{i}",
                        {"text": ex.text, target: label(ex)}))
    for i, ex in enumerate(TEST):
        rows.append(Row("reviews", f"test{i}",
                        {"text": ex.text, target: None}))   # masked

    by_id = {r.id: r for r in rows}

    def entities(table, ids, bound: TemporalBound):
        return [by_id[i] for i in ids if i in by_id]

    def scan(table, bound: TemporalBound):
        yield from rows

    wiring = (RetrieverWiring.new_wiring()
              .entities("reviews", entities)
              .scanner("reviews", scan)
              .build())
    return schema, wiring, target


def run(task: str, shots: int):
    schema, wiring, target = build(task, shots)
    engine = Engine(schema, wiring,
                    model_backend=RtNativeBackend(schema=schema))
    result = engine.execute(ExecutionInput(
        query=f"PREDICT reviews.{target} WHERE reviews.{target} IS NULL",
        anchor_time=ANCHOR))
    by_id = {p.id: p for p in result.predictions}
    out = []
    for i, ex in enumerate(TEST):
        p = by_id[f"test{i}"]
        score = p.probability if task == "classification" else p.value
        out.append((ex, float(score)))
    return out


def report_classification(rows):
    print("  -- binary classification (prob positive, threshold 0.5) --")
    correct = 0
    for ex, prob in rows:
        pred = prob >= 0.5
        ok = pred == ex.positive
        correct += ok
        mark = "OK " if ok else "XX "
        print(f"    {mark} p+={prob:.3f}  pred={'pos' if pred else 'neg'}  "
              f"true={'pos' if ex.positive else 'neg'}  | {ex.text}")
    print(f"    accuracy: {correct}/{len(rows)} = {correct/len(rows):.0%}")


def report_regression(rows):
    print("  -- regression (predicted positivity, 0=neg .. 1=pos) --")
    pos = [s for ex, s in rows if ex.positive]
    neg = [s for ex, s in rows if not ex.positive]
    correct = 0
    for ex, score in rows:
        pred = score >= 0.5
        ok = pred == ex.positive
        correct += ok
        mark = "OK " if ok else "XX "
        print(f"    {mark} r={score:+.3f}  pred={'pos' if pred else 'neg'}  "
              f"true={'pos' if ex.positive else 'neg'}  | {ex.text}")
    print(f"    sign accuracy @0.5: {correct}/{len(rows)} = {correct/len(rows):.0%}")
    if pos and neg:
        print(f"    mean positive: {sum(pos)/len(pos):+.3f}   "
              f"mean negative: {sum(neg)/len(neg):+.3f}   "
              f"separation: {sum(pos)/len(pos)-sum(neg)/len(neg):+.3f}")


if __name__ == "__main__":
    for shots in (0, 2):
        tag = ("FULLY ZERO-SHOT (column name only, no demonstrations)"
               if shots == 0 else f"{shots}-SHOT (1 pos + 1 neg demonstration)")
        print(f"\n================ {tag} ================")
        report_classification(run("classification", shots))
        report_regression(run("regression", shots))
