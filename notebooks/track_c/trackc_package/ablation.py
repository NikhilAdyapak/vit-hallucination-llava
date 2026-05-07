#!/usr/bin/env python3
"""
Track C — entropy-stratified VEGAS vs vanilla-LLaVA ablation.

Takes three inputs from Track A + a VEGAS caption CSV:

    entropy_ranks.csv    # 500 rows; per-image ViT rollout entropy + entropy_group
    chair_labels.csv     # 500 rows; vanilla LLaVA caption + chair_i + is_hallucinating
    captions_vegas.csv   # 500 rows; image_id, caption (from Colab notebook)
    instances_val2017.json  # COCO gt (for recomputing CHAIR on VEGAS captions)

Produces:

    trackc_results/merged_baseline.csv          # entropy × vanilla chair
    trackc_results/merged_vegas.csv             # entropy × vegas chair
    trackc_results/ablation_summary.csv         # mean chair_i + halluc rate per stratum per method
    trackc_results/vegas_minus_vanilla_delta.csv  # delta table (main Track C deliverable)
    trackc_results/bootstrap_delta.csv          # 1000× bootstrap CIs per stratum

Run (local):
    python -m trackc.ablation \
        --entropy-csv entropy_ranks.csv \
        --baseline-csv chair_labels.csv \
        --vegas-csv captions_vegas.csv \
        --coco-annotations coco/annotations/instances_val2017.json \
        --out-dir trackc_results
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from trackc.chair_utils import chair_scores_for_captions, load_gt_lookup_from_coco


BINARY_MAP = {
    "low_entropy": "clean",
    "mid_low": "clean",
    "mid_high": "degraded",
    "high_entropy": "degraded",
}

STRATUM_ORDER = [
    "low_entropy",
    "mid_low",
    "mid_high",
    "high_entropy",
    "binary:clean",
    "binary:degraded",
    "all",
]


def _ensure_dirs(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _add_binary_columns(ent: pd.DataFrame) -> pd.DataFrame:
    df = ent.copy()
    df["binary_group"] = df["entropy_group"].map(BINARY_MAP)
    return df


def _group_frames(merged: pd.DataFrame):
    """Yield (stratum_name, subframe) for every stratum we report on."""
    for grp, g in merged.groupby("entropy_group", sort=False):
        yield grp, g
    for grp, g in merged.groupby("binary_group", sort=False):
        yield f"binary:{grp}", g
    yield "all", merged


def summarize(merged: pd.DataFrame, method: str) -> pd.DataFrame:
    rows = []
    for grp, g in _group_frames(merged):
        rows.append(
            {
                "method": method,
                "stratum": grp,
                "n": int(len(g)),
                "mean_chair_i": float(g["chair_i"].mean()),
                "frac_hallucinating": float(g["is_hallucinating"].mean()),
                "sem_chair_i": float(g["chair_i"].std(ddof=1) / np.sqrt(len(g)))
                if len(g) > 1 else 0.0,
            }
        )
    df = pd.DataFrame(rows)
    df["stratum"] = pd.Categorical(df["stratum"], categories=STRATUM_ORDER, ordered=True)
    return df.sort_values(["method", "stratum"]).reset_index(drop=True)


def bootstrap_delta(
    m_base: pd.DataFrame,
    m_veg: pd.DataFrame,
    n_boot: int = 1000,
    seed: int = 0,
) -> pd.DataFrame:
    """Paired bootstrap: for each stratum resample image_ids with
    replacement and recompute Δ mean_chair_i = VEGAS - vanilla.
    """
    rng = np.random.default_rng(seed)
    paired = m_base[["image_id", "entropy_group", "binary_group", "chair_i"]].merge(
        m_veg[["image_id", "chair_i"]],
        on="image_id",
        suffixes=("_vanilla", "_vegas"),
        validate="one_to_one",
    )
    paired["delta"] = paired["chair_i_vegas"] - paired["chair_i_vanilla"]

    def _strata(df: pd.DataFrame):
        for grp, g in df.groupby("entropy_group", sort=False):
            yield grp, g
        for grp, g in df.groupby("binary_group", sort=False):
            yield f"binary:{grp}", g
        yield "all", df

    out = []
    for grp, g in _strata(paired):
        deltas = g["delta"].to_numpy()
        n = len(deltas)
        if n == 0:
            continue
        boot = np.empty(n_boot, dtype=np.float64)
        for b in range(n_boot):
            idx = rng.integers(0, n, size=n)
            boot[b] = deltas[idx].mean()
        out.append({
            "stratum": grp,
            "n": int(n),
            "delta_mean_chair_i": float(deltas.mean()),
            "ci_low_95": float(np.percentile(boot, 2.5)),
            "ci_high_95": float(np.percentile(boot, 97.5)),
            "p_delta_ge_0": float((boot >= 0).mean()),
        })
    df = pd.DataFrame(out)
    df["stratum"] = pd.Categorical(df["stratum"], categories=STRATUM_ORDER, ordered=True)
    return df.sort_values("stratum").reset_index(drop=True)


def run(
    entropy_csv: Path,
    baseline_csv: Path,
    out_dir: Path,
    vegas_csv: Path | None = None,
    coco_ann: Path | None = None,
    bootstrap_iters: int = 1000,
    vanilla_captions_csv: Path | None = None,
) -> dict:
    """Run Track C ablation.

    ``baseline_csv`` is Track A's ``chair_labels.csv`` (already CHAIR-scored).
    If ``vanilla_captions_csv`` is given, we treat that as a re-run of vanilla
    on the same model (e.g. 4-bit on T4) and recompute CHAIR on it against
    COCO — this keeps vanilla-vs-VEGAS at the same quantization level.
    """
    out_dir = Path(out_dir)
    _ensure_dirs(out_dir)

    ent = pd.read_csv(entropy_csv)
    keep = [
        "image_id", "file_name", "entropy", "entropy_quartile",
        "spatial_spread", "entropy_rank", "entropy_percentile", "entropy_group",
    ]
    ent = ent[[c for c in keep if c in ent.columns]]
    ent = _add_binary_columns(ent)

    if vanilla_captions_csv is not None and Path(vanilla_captions_csv).exists():
        if coco_ann is None or not Path(coco_ann).exists():
            raise SystemExit("--coco-annotations is required with --vanilla-captions-csv.")
        from trackc.chair_utils import load_gt_lookup_from_coco as _gt
        gt = _gt(str(coco_ann))
        vc = pd.read_csv(vanilla_captions_csv)
        if not {"image_id", "caption"}.issubset(vc.columns):
            raise SystemExit("--vanilla-captions-csv must have columns image_id, caption")
        v_chair = chair_scores_for_captions(vc, gt)
        base = vc.merge(v_chair, on="image_id", how="left")
    else:
        base = pd.read_csv(baseline_csv)
        if "chair_i" not in base.columns:
            raise SystemExit(
                f"{baseline_csv} has no 'chair_i' column — pass --vanilla-captions-csv "
                f"with --coco-annotations to recompute CHAIR from captions."
            )

    m_base = ent.merge(
        base[["image_id", "caption", "chair_i", "is_hallucinating"]],
        on="image_id",
        how="inner",
        validate="one_to_one",
    )
    if len(m_base) != len(ent):
        raise SystemExit(
            f"Baseline merge mismatch: entropy {len(ent)} rows vs merged {len(m_base)}"
        )
    m_base.to_csv(out_dir / "merged_baseline.csv", index=False)

    s_base = summarize(m_base, "vanilla_llava")
    summary_parts = [s_base]

    if vegas_csv is None or not Path(vegas_csv).exists():
        summary = pd.concat(summary_parts, ignore_index=True)
        summary.to_csv(out_dir / "ablation_summary.csv", index=False)
        print("=== Track C — vanilla LLaVA stratified CHAIR ===")
        print(summary.to_string(index=False))
        print("\n(Supply --vegas-csv + --coco-annotations to compute VEGAS delta.)")
        return {"summary": summary}

    if coco_ann is None or not Path(coco_ann).exists():
        raise SystemExit("--coco-annotations is required when --vegas-csv is set.")

    veg_raw = pd.read_csv(vegas_csv)
    need = {"image_id", "caption"}
    if not need.issubset(veg_raw.columns):
        raise SystemExit(f"--vegas-csv must have columns {need}, got {list(veg_raw.columns)}")

    gt = load_gt_lookup_from_coco(str(coco_ann))
    veg_chair = chair_scores_for_captions(veg_raw, gt)
    veg_full = veg_raw.merge(veg_chair, on="image_id", how="left")

    m_veg = ent.merge(
        veg_full[["image_id", "caption", "chair_i", "is_hallucinating",
                  "n_mentioned", "n_hallucinated"]],
        on="image_id",
        how="inner",
        validate="one_to_one",
    )
    if len(m_veg) != len(ent):
        missing = set(ent["image_id"]) - set(m_veg["image_id"])
        raise SystemExit(
            f"VEGAS caption CSV covers {len(m_veg)}/{len(ent)} images. "
            f"Missing ids (first 10): {sorted(missing)[:10]}"
        )
    m_veg.to_csv(out_dir / "merged_vegas.csv", index=False)

    s_veg = summarize(m_veg, "vegas")
    summary_parts.append(s_veg)
    summary = pd.concat(summary_parts, ignore_index=True)
    summary.to_csv(out_dir / "ablation_summary.csv", index=False)

    # Paired deltas per stratum
    boot = bootstrap_delta(m_base, m_veg, n_boot=bootstrap_iters)
    boot.to_csv(out_dir / "bootstrap_delta.csv", index=False)

    # Legacy delta file (unpaired means, matches prior column names)
    delta_rows = []
    for strat in STRATUM_ORDER:
        b = s_base.loc[s_base["stratum"] == strat, "mean_chair_i"]
        v = s_veg.loc[s_veg["stratum"] == strat, "mean_chair_i"]
        if b.empty or v.empty:
            continue
        hb = s_base.loc[s_base["stratum"] == strat, "frac_hallucinating"].iloc[0]
        hv = s_veg.loc[s_veg["stratum"] == strat, "frac_hallucinating"].iloc[0]
        delta_rows.append({
            "stratum": strat,
            "vanilla_mean_chair_i": float(b.iloc[0]),
            "vegas_mean_chair_i": float(v.iloc[0]),
            "delta_mean_chair_i": float(v.iloc[0] - b.iloc[0]),
            "vanilla_frac_halluc": float(hb),
            "vegas_frac_halluc": float(hv),
            "delta_frac_halluc": float(hv - hb),
        })
    delta = pd.DataFrame(delta_rows)
    delta.to_csv(out_dir / "vegas_minus_vanilla_delta.csv", index=False)

    meta = {
        "entropy_csv": str(entropy_csv),
        "baseline_csv": str(baseline_csv),
        "vegas_csv": str(vegas_csv),
        "coco_ann": str(coco_ann),
        "n_images": int(len(ent)),
        "bootstrap_iters": int(bootstrap_iters),
    }
    (out_dir / "run_meta.json").write_text(json.dumps(meta, indent=2))

    print("\n=== Track C — vanilla vs VEGAS (mean CHAIR_i by stratum) ===")
    print(summary.to_string(index=False))
    print("\n=== Δ = VEGAS − vanilla  (negative => VEGAS helps) ===")
    print(delta.to_string(index=False))
    print("\n=== Paired bootstrap 95% CI on Δ mean CHAIR_i ===")
    print(boot.to_string(index=False))

    return {"summary": summary, "delta": delta, "bootstrap": boot,
            "merged_baseline": m_base, "merged_vegas": m_veg}


def main() -> None:
    ap = argparse.ArgumentParser(description="Track C entropy-stratified CHAIR ablation")
    ap.add_argument("--entropy-csv", type=Path, required=True)
    ap.add_argument("--baseline-csv", type=Path, required=True)
    ap.add_argument("--vegas-csv", type=Path, default=None)
    ap.add_argument("--coco-annotations", type=Path, default=None)
    ap.add_argument("--out-dir", type=Path, default=Path("trackc_results"))
    ap.add_argument("--bootstrap-iters", type=int, default=1000)
    ap.add_argument("--vanilla-captions-csv", type=Path, default=None,
                    help="Alternate vanilla captions (re-scored vs COCO). Use for "
                         "matched-quantization comparisons on T4 4-bit runs.")
    args = ap.parse_args()
    run(
        args.entropy_csv,
        args.baseline_csv,
        args.out_dir,
        args.vegas_csv,
        args.coco_annotations,
        args.bootstrap_iters,
        args.vanilla_captions_csv,
    )


if __name__ == "__main__":
    main()
