"""Tiny hand-labeled sentiment dataset.

`SEED` sentences carry a known label and act as in-context demonstrations
(they become the "cohort" the RT model reads while its target cell is left
visible).  `TEST` sentences are held out: the model never sees their label —
we mask it (set to None), let the model predict, then score against `truth`.

Labels:
  polarity  True  = positive, False = negative     (binary classification)
  rating    1.0   = positive, 0.0  = negative       (regression target)

Kept deliberately short and unambiguous: this probes whether the pretrained
model has *any* innate notion of sentiment, not whether it can crack subtle
cases.
"""
from dataclasses import dataclass


@dataclass(frozen=True)
class Example:
    text: str
    positive: bool          # ground-truth polarity


SEED = [
    # --- positive ---
    Example("This movie was absolutely wonderful and I loved every minute.", True),
    Example("Best purchase I have made all year, highly recommend it.", True),
    Example("The food was delicious and the service was fantastic.", True),
    Example("An amazing experience, I felt happy the whole time.", True),
    Example("Beautiful design and it works perfectly.", True),
    Example("I am thrilled with how well this turned out.", True),
    Example("A brilliant, heartwarming story that I will never forget.", True),
    Example("Great value, excellent quality, and it arrived early.", True),
    Example("The staff were friendly and everything was clean and comfortable.", True),
    Example("I couldn't be more satisfied with this decision.", True),
    Example("Fantastic performance, the crowd was cheering with joy.", True),
    Example("This is exactly what I hoped for and more.", True),
    # --- negative ---
    Example("This movie was terrible and a complete waste of my time.", False),
    Example("Worst purchase I have ever made, do not buy it.", False),
    Example("The food was cold and the service was awful.", False),
    Example("A miserable experience, I was frustrated the entire time.", False),
    Example("Cheap design and it broke on the first day.", False),
    Example("I am deeply disappointed with how this turned out.", False),
    Example("A boring, forgettable story that dragged on forever.", False),
    Example("Overpriced, poor quality, and it showed up late.", False),
    Example("The staff were rude and the room was filthy.", False),
    Example("I regret this decision completely.", False),
    Example("Dreadful performance, the crowd booed and left early.", False),
    Example("This is nothing like what was promised, total letdown.", False),
]

TEST = [
    Example("I really enjoyed this, it made my whole day better.", True),
    Example("An outstanding product that exceeded my expectations.", True),
    Example("Lovely atmosphere and the meal was superb.", True),
    Example("So glad I tried it, pure delight from start to finish.", True),
    Example("A wonderful surprise, everything went perfectly.", True),
    Example("Hands down the best I have ever seen.", True),
    Example("Horrible from start to finish, I want a refund.", False),
    Example("The quality is garbage and it stopped working immediately.", False),
    Example("What a disappointing mess, I hated every second.", False),
    Example("Terrible support and a product that simply does not work.", False),
    Example("I would never recommend this to anyone, avoid it.", False),
    Example("Painfully dull and not worth a single penny.", False),
]
