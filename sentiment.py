"""Zero-shot sentiment probing of the RT-J relational transformer.

We build a one-table "database" of movie/product-review sentences and ask the
*pretrained* model to fill in a masked sentiment cell for held-out sentences,
using the labeled sentences as in-context demonstrations. No training, no head
fitting — just the two frozen checkpoints that ship with the model:

  binary classification  -> hf://stanford-star/rt-j/classification   (sigmoid prob)
  regression             -> hf://stanford-star/rt-j/regression       (continuous)

Both are "head-free": the target cell is a BOOLEAN (classification) or NUMBER
(regression) column, so the model scores it directly. (The multiclass path,
which masks a TEXT cell and needs a fitted head, is deliberately avoided.)

Run:  python sentiment.py
"""
from __future__ import annotations

from datetime import datetime, timezone

from relativedb import (Engine, ExecutionInput, LinkDef, RetrieverWiring, Row,
                        RtNativeBackend, Schema, TableDef, TemporalBound,
                        ValueType)

from data import SEED, TEST

ANCHOR = datetime(2026, 7, 23, tzinfo=timezone.utc)   # rows are static; any anchor works


# --------------------------------------------------------------------------
# Build the one-table "reviews" database for a given task.
#   task="classification" -> target column `sentiment` : BOOLEAN
#   task="regression"     -> target column `rating`    : NUMBER (0=neg, 1=pos)
# Seed rows carry the real label; test rows carry None (the masked target).
# --------------------------------------------------------------------------
def build(task: str):
    is_clf = task == "classification"
    target = "sentiment" if is_clf else "rating"
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
    for i, ex in enumerate(SEED):
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


def run(task: str):
    schema, wiring, target = build(task)
    engine = Engine(schema, wiring,
                    model_backend=RtNativeBackend(schema=schema))
    result = engine.execute(ExecutionInput(
        query=f"PREDICT reviews.{target} WHERE reviews.{target} IS NULL",
        anchor_time=ANCHOR))
    # Map predictions (id -> score) back onto the TEST rows in order.
    by_id = {p.id: p for p in result.predictions}
    out = []
    for i, ex in enumerate(TEST):
        p = by_id[f"test{i}"]
        score = p.probability if task == "classification" else p.value
        out.append((ex, float(score)))
    return out


def report_classification(rows):
    print("\n=== BINARY CLASSIFICATION  (prob positive, threshold 0.5) ===")
    correct = 0
    for ex, prob in rows:
        pred = prob >= 0.5
        ok = pred == ex.positive
        correct += ok
        mark = "OK " if ok else "XX "
        print(f"  {mark} p+={prob:.3f}  pred={'pos' if pred else 'neg'}  "
              f"true={'pos' if ex.positive else 'neg'}  | {ex.text}")
    print(f"  accuracy: {correct}/{len(rows)} = {correct/len(rows):.0%}")


def report_regression(rows):
    print("\n=== REGRESSION  (predicted rating, 0=neg .. 1=pos) ===")
    pos = [s for ex, s in rows if ex.positive]
    neg = [s for ex, s in rows if not ex.positive]
    correct = 0
    for ex, score in rows:
        pred = score >= 0.5
        ok = pred == ex.positive
        correct += ok
        mark = "OK " if ok else "XX "
        print(f"  {mark} r={score:+.3f}  pred={'pos' if pred else 'neg'}  "
              f"true={'pos' if ex.positive else 'neg'}  | {ex.text}")
    print(f"  sign accuracy @0.5: {correct}/{len(rows)} = {correct/len(rows):.0%}")
    if pos and neg:
        print(f"  mean positive rating: {sum(pos)/len(pos):+.3f}")
        print(f"  mean negative rating: {sum(neg)/len(neg):+.3f}")
        print(f"  separation (pos-neg): {sum(pos)/len(pos)-sum(neg)/len(neg):+.3f}")


if __name__ == "__main__":
    print(f"seed (labeled demos): {len(SEED)}   test (held out): {len(TEST)}")
    report_classification(run("classification"))
    report_regression(run("regression"))
