import argparse
import csv
import json
from pathlib import Path

import pyarrow.parquet as pq


SPLITS = {
    "train": {
        "glob": "train-*.parquet",
        "json_name": "TextVQA_0.5.1_train.json",
        "csv_name": "train.csv",
        "image_dir": "train_images",
        "json_split": "train",
    },
    "validation": {
        "glob": "validation-*.parquet",
        "json_name": "TextVQA_0.5.1_val.json",
        "csv_name": "val.csv",
        "image_dir": "validation_images",
        "json_split": "val",
    },
    "test": {
        "glob": "test-*.parquet",
        "json_name": "TextVQA_0.5.1_test.json",
        "csv_name": "test.csv",
        "image_dir": "test_images",
        "json_split": "test",
    },
}

METADATA_COLUMNS = [
    "image_id",
    "question_id",
    "question",
    "question_tokens",
    "image_width",
    "image_height",
    "flickr_original_url",
    "flickr_300k_url",
    "answers",
    "image_classes",
    "set_name",
    "ocr_tokens",
]

CSV_COLUMNS = [
    "image_id",
    "question_id",
    "question",
    "answers",
    "set_name",
    "image_path",
    "image_width",
    "image_height",
    "question_tokens",
    "ocr_tokens",
    "image_classes",
    "flickr_original_url",
    "flickr_300k_url",
]


def image_extension(image_bytes: bytes, fallback_path: str | None) -> str:
    if image_bytes.startswith(b"\xff\xd8\xff"):
        return ".jpg"
    if image_bytes.startswith(b"\x89PNG\r\n\x1a\n"):
        return ".png"
    if image_bytes.startswith(b"GIF87a") or image_bytes.startswith(b"GIF89a"):
        return ".gif"
    if fallback_path:
        suffix = Path(fallback_path).suffix.lower()
        if suffix:
            return suffix
    return ".bin"


def json_cell(value):
    if isinstance(value, (list, dict)):
        return json.dumps(value, ensure_ascii=False)
    return "" if value is None else value


def export_split(
    split_name: str,
    config: dict,
    data_dir: Path,
    out_dir: Path,
    batch_size: int,
    force: bool,
    written_image_paths: set[str],
):
    parquet_files = sorted(data_dir.glob(config["glob"]))
    if not parquet_files:
        raise FileNotFoundError(f"No parquet files found for split {split_name}: {data_dir / config['glob']}")

    image_dir = out_dir / config["image_dir"]
    csv_dir = out_dir / "csv"
    image_dir.mkdir(parents=True, exist_ok=True)
    csv_dir.mkdir(parents=True, exist_ok=True)

    json_path = out_dir / config["json_name"]
    csv_path = csv_dir / config["csv_name"]

    if not force:
        existing = [path for path in (json_path, csv_path) if path.exists()]
        if existing:
            joined = ", ".join(str(path) for path in existing)
            raise FileExistsError(f"Output already exists. Use --force to overwrite: {joined}")

    rows = []
    image_paths_seen_in_split = set()
    image_write_count = 0
    row_count = 0

    with csv_path.open("w", encoding="utf-8", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=CSV_COLUMNS)
        writer.writeheader()

        for parquet_path in parquet_files:
            parquet_file = pq.ParquetFile(parquet_path)
            columns = METADATA_COLUMNS + ["image"]
            for batch in parquet_file.iter_batches(batch_size=batch_size, columns=columns):
                for item in batch.to_pylist():
                    image = item.pop("image") or {}
                    image_bytes = image.get("bytes")
                    image_path_hint = image.get("path")
                    image_id = item["image_id"]

                    if image_bytes is None:
                        raise ValueError(f"Missing image bytes for image_id={image_id}")

                    extension = image_extension(image_bytes, image_path_hint)
                    relative_image_path = f"{config['image_dir']}/{image_id}{extension}"
                    target_image_path = out_dir / relative_image_path

                    if relative_image_path not in written_image_paths and relative_image_path not in image_paths_seen_in_split:
                        if force or not target_image_path.exists():
                            target_image_path.write_bytes(image_bytes)
                            image_write_count += 1
                        image_paths_seen_in_split.add(relative_image_path)

                    item["image_path"] = relative_image_path
                    rows.append(item)

                    csv_row = {column: json_cell(item.get(column)) for column in CSV_COLUMNS}
                    writer.writerow(csv_row)
                    row_count += 1

    payload = {
        "dataset_type": "TextVQA",
        "dataset_name": "textvqa",
        "version": "0.5.1",
        "split": config["json_split"],
        "data": rows,
    }
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    return {
        "split": split_name,
        "rows": row_count,
        "images_written": image_write_count,
        "unique_images": len({row["image_path"] for row in rows}),
        "json": str(json_path),
        "csv": str(csv_path),
        "image_dir": str(image_dir),
    }


def main():
    parser = argparse.ArgumentParser(description="Export lmms-lab/textvqa parquet files to TextVQA-style files.")
    parser.add_argument("--dataset-dir", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    dataset_dir = args.dataset_dir
    data_dir = dataset_dir / "data"
    out_dir = args.out_dir or dataset_dir / "original_format"
    out_dir.mkdir(parents=True, exist_ok=True)

    results = []
    written_image_paths = set()
    for split_name, config in SPLITS.items():
        print(f"Exporting {split_name}...")
        result = export_split(split_name, config, data_dir, out_dir, args.batch_size, args.force, written_image_paths)
        results.append(result)
        written_image_paths.update(row["image_path"] for row in json.loads(Path(result["json"]).read_text(encoding="utf-8"))["data"])

    manifest_path = out_dir / "manifest.json"
    manifest = {
        "source": "lmms-lab/textvqa parquet export",
        "output_layout": "TextVQA 0.5.1-style annotations plus image folders",
        "splits": results,
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
