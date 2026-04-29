from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


@dataclass(frozen=True)
class SplitSpec:
    name: str
    data_subtype: str
    image_dir: str
    question_file: str
    annotation_file: str


SPLIT_SPECS = {
    "train": SplitSpec(
        name="train",
        data_subtype="train2014",
        image_dir="train2014",
        question_file="v2_OpenEnded_mscoco_train2014_questions.json",
        annotation_file="v2_mscoco_train2014_annotations.json",
    ),
    "val": SplitSpec(
        name="val",
        data_subtype="val2014",
        image_dir="val2014",
        question_file="v2_OpenEnded_mscoco_val2014_questions.json",
        annotation_file="v2_mscoco_val2014_annotations.json",
    ),
}


def default_mscoco_root() -> Path:
    return Path(__file__).resolve().parent.parent


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as fp:
        return json.load(fp)


def write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def normalize_splits(raw_splits: str) -> list[str]:
    result: list[str] = []
    for raw_split in raw_splits.split(","):
        split = raw_split.strip().lower()
        if not split:
            continue
        if split == "validation":
            split = "val"
        if split not in SPLIT_SPECS:
            allowed = ", ".join(sorted(SPLIT_SPECS))
            raise ValueError(f"Unsupported split: {raw_split}. Allowed: {allowed}")
        if split not in result:
            result.append(split)
    if not result:
        raise ValueError("At least one split is required.")
    return result


def split_tag(splits: Iterable[str]) -> str:
    return "_".join(normalize_splits(",".join(splits)))


def image_relative_path(spec: SplitSpec, image_id: int) -> str:
    return f"{spec.image_dir}/COCO_{spec.data_subtype}_{int(image_id):012d}.jpg"


def clean_answer(value) -> str:
    return str(value).strip().lower()


def answers_from_annotation(annotation: dict) -> list[str]:
    answers = [
        clean_answer(item.get("answer", ""))
        for item in annotation.get("answers", [])
        if clean_answer(item.get("answer", ""))
    ]
    if answers:
        return answers
    fallback = clean_answer(annotation.get("multiple_choice_answer", ""))
    return [fallback] if fallback else []


def canonical_answer(annotation: dict, answers: list[str]) -> str:
    answer = clean_answer(annotation.get("multiple_choice_answer", ""))
    if answer:
        return answer
    return answers[0] if answers else ""
