from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TextSplit:
    train_text: str
    val_text: str


def split_text(text: str, train_fraction: float = 0.9) -> TextSplit:
    if not 0.0 < train_fraction < 1.0:
        raise ValueError("train_fraction must be between 0 and 1")
    split_at = int(len(text) * train_fraction)
    return TextSplit(train_text=text[:split_at], val_text=text[split_at:])
