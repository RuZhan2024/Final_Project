# Day 2: Build Labels And Fall Spans

## 1. Today's goal

Today we will convert the annotation CSV into two reusable artifacts:

- `labels.json`
- `spans.json`

These files are the first real machine-learning annotations in the project.

## 2. Why this part exists in the full pipeline

The real repository has dataset-specific label builders such as:

- `scripts/make_labels_le2i.py`
- `scripts/make_labels_caucafall.py`
- `scripts/make_labels_muvim.py`

Those scripts all do the same high-level job:

1. decide which sequences are falls
2. decide where the fall spans occur
3. write label artifacts that later stages can consume

Our teaching version does the same thing, but with one simple CSV format.

## 3. What you will finish by the end of today

By the end of today you will have:

- a label builder module
- `data/course_demo/labels.json`
- `data/course_demo/spans.json`

## 4. File tree snapshot for today

```text
course_project/
├── labels/
│   ├── __init__.py
│   └── make_labels.py
└── ...

data/
└── course_demo/
    ├── annotations.csv
    ├── labels.json
    ├── spans.json
    └── pose_npz/
```

## 5. Full code blocks for every file introduced or changed today

### File 1: `course_project/labels/__init__.py`

Why this file exists:

- it makes the `labels` folder importable as a normal package

```python
"""Label builders for the course project."""
```

### File 2: `course_project/labels/make_labels.py`

Why this file exists:

- later stages should not re-parse raw annotations every time
- labels and spans should become stable, inspectable artifacts

```python
"""Build labels.json and spans.json from the demo annotation CSV."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from course_project.common.io import save_json


def load_annotation_rows(path: str | Path) -> list[dict[str, str]]:
    file_path = Path(path)
    with file_path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def build_labels_and_spans(rows: list[dict[str, str]]) -> tuple[dict[str, int], dict[str, list[list[int]]]]:
    labels: dict[str, int] = {}
    spans: dict[str, list[list[int]]] = {}

    for row in rows:
        video_id = str(row["video_id"]).strip()
        label = int(row["label"])
        span_start = int(row["span_start"])
        span_end = int(row["span_end"])

        labels[video_id] = label
        if label == 1 and span_start >= 0 and span_end > span_start:
            spans[video_id] = [[span_start, span_end]]
        else:
            spans[video_id] = []

    return labels, spans


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build labels and spans from a simple CSV.")
    parser.add_argument("--ann_csv", default="data/course_demo/annotations.csv")
    parser.add_argument("--out_labels", default="data/course_demo/labels.json")
    parser.add_argument("--out_spans", default="data/course_demo/spans.json")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = load_annotation_rows(args.ann_csv)
    labels, spans = build_labels_and_spans(rows)
    save_json(labels, args.out_labels)
    save_json(spans, args.out_spans)
    print(f"[ok] wrote labels to {args.out_labels}")
    print(f"[ok] wrote spans to {args.out_spans}")


if __name__ == "__main__":
    main()
```

## 6. Detailed teaching explanation

### Tiny concrete example first

Suppose one row in `annotations.csv` is:

```text
subject01_fall_001,1,20,34
```

That means:

- the sequence is a fall
- the fall begins at frame `20`
- the fall ends at frame `34`

From that one row, we want to produce:

```json
{
  "subject01_fall_001": 1
}
```

in `labels.json`, and:

```json
{
  "subject01_fall_001": [[20, 34]]
}
```

in `spans.json`.

That is exactly what today's code does.

### `load_annotation_rows(...)` line by line

Inputs:

- path to the CSV annotation file

Outputs:

- a list of dictionaries, one dictionary per row

Side effects:

- reads a file from disk

Why it exists:

- we want one clean function responsible for parsing the raw annotation file

Important lines:

`with file_path.open("r", encoding="utf-8", newline="") as f:`

- open the CSV file safely

`return list(csv.DictReader(f))`

- each row becomes a dictionary like `{"video_id": "...", "label": "1", ...}`
- we choose `DictReader` instead of manual string splitting because named columns are easier to inspect and safer to extend

### `build_labels_and_spans(...)` line by line

Inputs:

- the parsed CSV rows

Outputs:

- `labels` dictionary
- `spans` dictionary

Side effects:

- none, this is a pure transformation step

Why it exists:

- it isolates the logic that turns raw annotations into ML-ready artifacts

Important lines:

`video_id = str(row["video_id"]).strip()`

- convert to a clean string and remove accidental whitespace

`label = int(row["label"])`

- CSV data arrives as text, so we convert it to an integer

`if label == 1 and span_start >= 0 and span_end > span_start:`

- only positive examples get a non-empty span
- this guard also prevents bad spans such as `[34, 20]`

`spans[video_id] = [[span_start, span_end]]`

- we store spans as a list of spans, even though the demo uses only one span per video
- this is deliberate
- it makes the teaching format closer to the real repository, where a sequence can contain one or more annotated intervals

### `parse_args()` and `main()`

These functions make the module runnable from the terminal.

That matters because the real repository is orchestration-heavy. Many stages are not notebooks. They are scripts that produce artifacts.

Non-trivial lines in `main()`:

`rows = load_annotation_rows(args.ann_csv)`

- first read the raw annotation source

`labels, spans = build_labels_and_spans(rows)`

- then transform it into stable ML-ready artifacts

`save_json(labels, args.out_labels)`

- persist the cleaned contract instead of making later scripts re-parse the CSV

## 7. Exact run commands

Create the folder:

```bash
mkdir -p course_project/labels
```

Copy today's files into:

- `course_project/labels/__init__.py`
- `course_project/labels/make_labels.py`

Then run:

```bash
PYTHONPATH="$(pwd)" python3 course_project/labels/make_labels.py
```

Inspect the outputs:

```bash
python3 - <<'PY'
import json
print(json.load(open("data/course_demo/labels.json", "r", encoding="utf-8")))
print(json.load(open("data/course_demo/spans.json", "r", encoding="utf-8")))
PY
```

## 8. Expected outputs

Script output:

```text
[ok] wrote labels to data/course_demo/labels.json
[ok] wrote spans to data/course_demo/spans.json
```

Output inspection pattern:

```text
{'subject01_fall_001': 1, 'subject01_walk_001': 0, ...}
{'subject01_fall_001': [[20, 34]], 'subject01_walk_001': [], ...}
```

## 9. Sanity checks

Check that:

1. `labels.json` contains six keys
2. every key in `labels.json` also appears in `spans.json`
3. non-fall videos have empty span lists
4. fall videos have exactly one valid span

## 10. Common bugs and fixes

### Bug: all values in `labels.json` are strings

Fix:

- make sure `label = int(row["label"])` was copied

### Bug: non-fall videos got fake spans

Fix:

- keep the `if label == 1` guard exactly as written

### Bug: `spans.json` is empty

Fix:

- inspect `annotations.csv`
- confirm the fall rows use positive labels and valid span numbers

## 11. Mapping to the original repository

Today's teaching file maps to the real dataset-specific label scripts:

- `scripts/make_labels_le2i.py`
- `scripts/make_labels_urfall.py`
- `scripts/make_labels_caucafall.py`
- `scripts/make_labels_muvim.py`

Teaching simplification:

- one clean CSV schema

Full repository version:

- each dataset uses its own annotation format and custom parsing code

## 12. Tomorrow's preview

Tomorrow we will protect ourselves from one of the biggest mistakes in applied ML:

split leakage.

We will create subject-safe `train`, `val`, and `test` lists so that later results mean something.
