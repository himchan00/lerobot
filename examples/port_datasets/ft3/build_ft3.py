#!/usr/bin/env python
"""FT-3: Square-FT, Wipe-FT and Door-FT merged into one language-conditioned dataset, like a LIBERO suite.

Each source keeps its task string. The task-specific hidden labels (expert.peg_offset, expert.surface,
expert.hinge_side) are dropped so the features match (expert.phase stays), then the sources are merged with the
Array2D-safe aggregate of examples/port_datasets/libero_hf.

    python build_ft3.py --sources <square_ft root> <wipe_ft root> <door_ft root> --out /PublicSSD/himchan/ft3/data
"""

import argparse
import importlib.util
import json
import os
import shutil
from pathlib import Path

import pyarrow.parquet as pq


def strip_labels(src: Path, dst: Path) -> None:
    """Copy of `src` without the expert.* labels other than expert.phase (videos hard-linked)."""
    info = json.loads((src / "meta/info.json").read_text())
    drop = [k for k in info["features"] if k.startswith("expert.") and k != "expert.phase"]
    columns = tuple(drop) + tuple(f"stats/{k}/" for k in drop)  # data columns and per-episode stats
    shutil.rmtree(dst, ignore_errors=True)
    shutil.copytree(src / "videos", dst / "videos", copy_function=os.link)
    shutil.copytree(src / "meta", dst / "meta")
    for f in sorted((src / "data").rglob("*.parquet")) + sorted((src / "meta/episodes").rglob("*.parquet")):
        table = pq.read_table(f)
        table = table.drop_columns([c for c in table.column_names if c.startswith(columns)])
        meta = dict(table.schema.metadata or {})
        if b"huggingface" in meta:
            hf = json.loads(meta[b"huggingface"])
            for k in drop:
                hf["info"]["features"].pop(k, None)
            meta[b"huggingface"] = json.dumps(hf).encode()
        out = dst / f.relative_to(src)
        out.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(table.replace_schema_metadata(meta), out)
    for name in ("info", "stats"):
        d = json.loads((src / f"meta/{name}.json").read_text())
        for k in drop:
            (d["features"] if name == "info" else d).pop(k, None)
        (dst / f"meta/{name}.json").write_text(json.dumps(d, indent=4))
    print(f"{src}: dropped {drop}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--sources", nargs="+", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--repo-id", default="himchan00/ft3")
    a = p.parse_args()
    out = Path(a.out)
    shutil.rmtree(out, ignore_errors=True)
    for i, src in enumerate(a.sources):
        strip_labels(Path(src), out / "shards" / f"{i}_{Path(src).parents[1].name}")

    path = Path(__file__).resolve().parents[1] / "libero_hf" / "regenerate_libero_hf.py"
    spec = importlib.util.spec_from_file_location("regenerate_libero_hf", path)
    regen = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(regen)
    regen.cmd_aggregate(argparse.Namespace(out_dir=out, repo_id=a.repo_id))
    shutil.rmtree(out / "shards")
    print("END")


if __name__ == "__main__":
    main()
