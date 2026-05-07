# Locating the Source of Hallucination in LLaVA-1.5

A three-track empirical investigation of the ViT visual encoder.

**CS 639: Introduction to Foundation Models | UW-Madison | Spring 2026**

**Team:** Arihant Jha, Saurabh Rajesh Pandey, Yuvarraj Sriramkumar, Nikhil Adyapak, Asish Das, Siddhartha Pathapati

## Summary

We test whether the CLIP ViT-L/14 visual encoder inside LLaVA-1.5-7B is producing degraded representations on images where the model later hallucinates. Three experiments on 500 COCO 2017 validation images.

- **Track A:** ViT [CLS] attention rollout entropy vs hallucination outcome. Result: null (rho = 0.003, p = 0.94).
- **Track B:** Linear probes on patch embeddings at four ViT layers. Result: probe accuracy is significantly lower on hallucinating images at every layer (delta_acc in [-0.039, -0.027], all Wilcoxon p < 0.005).
- **Track C:** VEGAS attention-based steering, in two configurations. Version 1 (4-bit, entropy-stratified): directional backfire on diffuse-attention images, P(delta >= 0) = 0.929. Version 2 (fp16, quality-stratified): zero effect, alpha effectively disabled by near-uniform rollout.

## Repo Structure

```
.
├── notebooks/
│   ├── track_a/                            Track A inference and rollout
│   │   ├── TRACKA_1ST_INFERENCE_CHAIR.ipynb
│   │   └── TrackA_2ND_Attention_Rollout.ipynb
│   └── track_c/                            Three Track C versions
│       ├── v0_initial_attempt.ipynb        First attempt, fp16 vs 4-bit comparison issue
│       ├── v1_4bit_entropy_stratified.ipynb   Working version, entropy stratification
│       ├── v2_fp16_quality_stratified.ipynb   Full fp16, quality-group stratification
│       └── trackc_package/                 Python package for v1 ablation pipeline
├── data/
│   ├── track_a_outputs/
│   │   ├── chair_labels.csv                500 LLaVA captions + CHAIR scores
│   │   ├── entropy_ranks.csv               Per-image rollout entropy + quartile
│   │   └── sampled_image_ids.json
│   └── track_c_outputs/
│       ├── captions_vanilla_4bit.csv       4-bit vanilla captions
│       ├── captions_vegas.csv              4-bit VEGAS captions
│       └── v1_4bit_results/                Bootstrap CIs, deltas, figures
├── figures/                                Track A figures
└── reports/                                Final report + earlier drafts + proposal
```

## How to Reproduce

1. Open the notebooks in Google Colab. Track A needs A100 (40 GB), Track C v1 runs on T4 (16 GB) with 4-bit quantization, Track C v2 needs ~50 GB VRAM at fp16.
2. Track B was run from a script that is not in this repo. Probe accuracy values are reported in the final report (Table 2). Per-image probe accuracy was not exported.
3. Run notebooks in order: Track A inference → Track A rollout → Track C v1 (or v2). Each notebook copies its inputs from the prior track's CSVs in `data/`.

## Notes on Track C Versions

- **v0** (`v0_initial_attempt.ipynb`): superseded. Compared fp16 baseline against 4-bit VEGAS, which mixes quantization effect with VEGAS effect.
- **v1** (`v1_4bit_entropy_stratified.ipynb`): the version reported as Track C Version 1 in the final report. Re-runs vanilla at the same 4-bit precision so the comparison is fair. Uses raw final-layer [CLS]-to-patch attention as the steering signal.
- **v2** (`v2_fp16_quality_stratified.ipynb`): the version reported as Track C Version 2. Full fp16 model. Uses pre-computed rollout-derived concentration as the steering signal, which produces near-zero alpha for all images and therefore no measurable effect.

## Final Report

`reports/CS639_FinalReport.pdf`

## License

Coursework. No license claimed.
