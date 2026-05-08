# Locating the Source of Hallucination in LLaVA-1.5

A three-track empirical investigation of the ViT visual encoder.

**CS 639: Introduction to Foundation Models | UW-Madison | Spring 2026**

**Team:** Arihant Jha, Saurabh Rajesh Pandey, Yuvarraj Sriramkumar, Nikhil Adyapak, Asish Das, Siddhartha Pathapati

## Summary

We test whether the CLIP ViT-L/14 visual encoder inside LLaVA-1.5-7B is producing degraded representations on images where the model later hallucinates. Three experiments on 500 COCO 2017 validation images.

- **Track A:** ViT [CLS] attention rollout entropy vs hallucination outcome. Result: null (rho = 0.003, p = 0.94).
- **Track B:** Linear probes on patch embeddings at four ViT layers. Result: probe accuracy is significantly lower on hallucinating images at every layer (delta_acc in [-0.039, -0.027], all Wilcoxon p < 0.005).
- **Track C:** VEGAS attention-based steering and causal patch ablation, in three configurations:
  - Approach 1 (4-bit, entropy-stratified): directional VEGAS backfire on diffuse-attention images, P(delta >= 0) = 0.929.
  - Approach 2 part A (fp16, quality-stratified VEGAS): zero effect, alpha effectively disabled by near-uniform rollout.
  - Approach 2 part B (fp16, causal patch ablation + swap): top-K, bottom-K, and random ablations equivalent at layer 18; random-donor patch swap significantly increases hallucination (p = 0.0028), matched-clean swap does not significantly change it (p = 0.148).

## Repo Structure

```
.
├── notebooks/
│   ├── track_a/                            Three Track A notebooks
│   │   ├── TRACKA_1ST_INFERENCE_CHAIR.ipynb        LLaVA inference + CHAIR scoring
│   │   ├── TrackA_2ND_Attention_Rollout.ipynb      ViT attention rollout extraction
│   │   └── TrackA_3RD_Entropy_Analysis.ipynb       Entropy stats + Track C handoff
│   ├── track_b/
│   │   └── TrackB_Linear_Probing.ipynb     68 probes (4 layers x 17 categories)
│   └── track_c/                            Four Track C variants
│       ├── v0_initial_attempt.ipynb        First attempt, fp16 vs 4-bit comparison issue
│       ├── v1_4bit_entropy_stratified.ipynb        Approach 1
│       ├── v2_fp16_quality_stratified.ipynb        Approach 2 part A
│       ├── v3_causal_ablation_swap.ipynb           Approach 2 part B (NEW)
│       └── trackc_package/                 Python package for Approach 1 pipeline
├── data/
│   ├── track_a_outputs/
│   │   ├── chair_labels.csv                500 LLaVA captions + CHAIR scores
│   │   ├── entropy_ranks.csv               Per-image rollout entropy + quartile
│   │   └── sampled_image_ids.json
│   └── track_c_outputs/
│       ├── captions_vanilla_4bit.csv       4-bit vanilla captions (Approach 1)
│       ├── captions_vegas.csv              4-bit VEGAS captions (Approach 1)
│       └── v1_4bit_results/                Bootstrap CIs, deltas, figures
├── figures/                                Track A figures
└── reports/                                Final report + proposal + AB progress + rubric
```

## How to Reproduce

1. Open the notebooks in Google Colab. Track A and B need A100 (40 GB). Track C Approach 1 runs on T4 (16 GB) with 4-bit quantization. Track C Approach 2 (parts A and B) needs ~50 GB VRAM at fp16 (we used RTX PRO 6000 Blackwell with 102 GB).
2. Run notebooks in order: Track A inference -> Track A rollout -> Track A entropy -> Track B probing -> Track C variants. Each notebook copies its inputs from the prior track's CSVs in `data/`.
3. Track B saves probe outputs to Drive (`probe_results.csv`, `probe_stats.csv`, `patch_embeddings_cache.npz`, four figures). These are not committed because of size; re-run the notebook to regenerate.

## Notes on Track C Versions

- **v0** (`v0_initial_attempt.ipynb`): superseded. Compared fp16 baseline against 4-bit VEGAS, which mixes quantization effect with VEGAS effect.
- **v1** (`v1_4bit_entropy_stratified.ipynb`): Approach 1 in the final report. Re-runs vanilla at the same 4-bit precision so the comparison is fair. Uses raw final-layer [CLS]-to-patch attention as the steering signal at LLM layers 14-15. Entropy-stratified.
- **v2** (`v2_fp16_quality_stratified.ipynb`): Approach 2 part A. Full fp16 model. Uses pre-computed rollout-derived concentration as the steering signal, which produces near-zero alpha for all images and therefore no measurable effect. Track B quality-group stratified.
- **v3** (`v3_causal_ablation_swap.ipynb`): Approach 2 part B. Full fp16. Two experiments: (Exp 1) zero-ablate top-K, bottom-K, or random-K rollout patches at layer 18 across 50 images; (Exp 2) swap top-K patches between matched hallucinating-clean image pairs vs random donors at layer 18.

## Final Report

`reports/CS639_FinalReport.pdf`

The LaTeX source is in `reports/Final_Report.tex`. The earlier draft (`CS639_FinalReport_initial_draft.pdf`) is kept for reference but is superseded.

## License

Coursework. No license claimed.
