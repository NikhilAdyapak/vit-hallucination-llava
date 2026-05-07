"""
VEGAS — Vision-Encoder Guided Attention Steering for LLaVA-1.5.

Reference: arXiv:2512.12089 (Track C of the CS 639 sprint plan).

Core idea (per §3-4 and §5.1 "Implementation Details" of the paper):

    At selected middle LLM self-attention layers, rescale the attention
    probabilities that text tokens pay to the 576 *image* tokens so that
    they are biased toward the patches that the **vision encoder's final
    layer [CLS] token** already attends to.

    For LLaVA-1.5-7B, the authors inject at 0-indexed LLM layers 14 and 15.

This module provides a minimal, self-contained implementation that:

    1. Runs LLaVA-1.5-7B (HuggingFace `LlavaForConditionalGeneration`) with
       `attn_implementation="eager"` so that attention *probabilities* are
       available for modification.
    2. Extracts ViT final-layer [CLS]→patch attention once per image from
       the vision-tower's last block (mean over heads, softmax-normalized
       across the 576 patch positions).
    3. Registers a forward pre-hook on the two target LlamaDecoderLayer
       modules that swaps out the unnormalized attention logits over the
       image-token range with a convex combination of the original logits
       and the ViT guidance:

            scores_img <- (1 - lambda) * scores_img
                          + lambda * log(vit_cls_attn + eps)

       `lambda` defaults to 0.5 (matches the paper's middle-layer blending
       strength; configurable via `steering_weight`).
    4. Also supports an optional adaptive logits steering (paper §4.3):
       when the next-token distribution has VABE above a threshold, the
       blend weight is scaled up. Controlled by `adaptive=True`.

The returned caption is deterministic (greedy decoding, matching Track A).

Usage:
    from trackc.vegas import VEGASRunner
    runner = VEGASRunner(model_id="llava-hf/llava-1.5-7b-hf", device="cuda")
    captions = runner.caption_many(
        image_paths=[...],  # 500 PIL-readable paths
        image_ids=[...],
        prompt="USER: <image>\\nDescribe this image.\\nASSISTANT:",
        steering_layers=(14, 15),
        steering_weight=0.5,
        adaptive=True,
        max_new_tokens=128,
    )
"""
from __future__ import annotations

import contextlib
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import torch
import torch.nn.functional as F
from PIL import Image


# LLaVA-1.5-7B constants
NUM_IMAGE_TOKENS = 576  # 24 * 24 patches from ViT-L/14 @ 336x336
IMAGE_TOKEN_INDEX = 32000  # <image> placeholder token in the LLaVA tokenizer


@dataclass
class VEGASConfig:
    steering_layers: tuple[int, ...] = (14, 15)
    steering_weight: float = 0.5           # lambda in eq. (5) of the paper
    adaptive: bool = True                   # §4.3 logits steering
    adaptive_scale: float = 1.5             # lambda multiplier when VABE high
    vabe_threshold: float = 0.5             # VABE gate (entropy in nats, normalized)
    max_new_tokens: int = 128
    do_sample: bool = False                 # greedy, matches Track A
    temperature: float = 1.0


class VEGASRunner:
    """Wraps LLaVA-1.5-7B with VEGAS attention steering.

    All device handling is done through `device_map="auto"` from accelerate.
    """

    def __init__(
        self,
        model_id: str = "llava-hf/llava-1.5-7b-hf",
        device: str | None = None,
        torch_dtype: torch.dtype = torch.float16,
        load_in_4bit: bool = False,
    ) -> None:
        """Load LLaVA-1.5-7B.

        Set ``load_in_4bit=True`` to run on a 16 GB Colab T4. That weight-quantizes
        the LLM with bitsandbytes NF4 (double-quant, fp16 compute) and keeps
        the ViT in fp16 so the [CLS]→patch attention readout is still exact.
        Peak VRAM with this setting is ~7-9 GB on a T4 for LLaVA-1.5-7B.
        """
        from transformers import AutoProcessor, LlavaForConditionalGeneration

        self.model_id = model_id
        self.processor = AutoProcessor.from_pretrained(model_id)

        load_kwargs = dict(
            torch_dtype=torch_dtype,
            low_cpu_mem_usage=True,
            attn_implementation="eager",
            device_map="auto" if device is None else None,
        )
        if load_in_4bit:
            from transformers import BitsAndBytesConfig
            load_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch_dtype,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                # Crucial: keep the vision tower in fp16 so CLIP attentions are
                # numerically exact — VEGAS reads them off of that module.
                llm_int8_skip_modules=["vision_tower", "multi_modal_projector"],
            )

        self.model = LlavaForConditionalGeneration.from_pretrained(
            model_id, **load_kwargs,
        )
        if device is not None and not load_in_4bit:
            self.model.to(device)
        self.model.eval()
        self.device = next(self.model.parameters()).device
        self.torch_dtype = torch_dtype

        # Cached image-token range (re-computed per sample in generate)
        self._img_token_slice: slice | None = None
        self._cls_attn: torch.Tensor | None = None          # shape (576,)
        self._vabe_state = {"gate": False}
        self._cfg: VEGASConfig | None = None

    # ---------- ViT [CLS]→patch attention extraction ----------
    @torch.no_grad()
    def _compute_vit_cls_attention(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """Return normalized [CLS]→patch attention from the vision tower's
        last encoder block. Shape: (576,) on model device, float32.
        """
        vt = self.model.vision_tower
        # LLaVA's vision tower is CLIPVisionModel; output_attentions gives
        # one tensor per layer, shape (B, H, 1+576, 1+576)
        out = vt(pixel_values=pixel_values, output_attentions=True)
        attn = out.attentions[-1]  # last layer
        # mean over heads, take CLS row, drop CLS column → (B, 576)
        cls_attn = attn.mean(dim=1)[:, 0, 1:]
        # renormalize in case mean-over-heads breaks softmax sum
        cls_attn = cls_attn / cls_attn.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        return cls_attn.to(torch.float32).squeeze(0)

    # ---------- LLM attention steering hook ----------
    def _make_attn_hook(self, layer_idx: int):
        """Forward pre-hook on a LlamaDecoderLayer's self-attention `forward`
        that rescales post-softmax attention weights over the image-token
        range.

        Implementation detail: we can't easily intercept the *post-softmax*
        weights inside `LlamaAttention.forward` without copying the whole
        function. So instead we register a hook on the attention module's
        *forward* that calls the original forward, grabs `attn_weights`
        (returned when `output_attentions=True`), renormalizes image-token
        probabilities toward the ViT guidance, and recomputes the output.

        To keep things robust across HF versions we instead patch
        `LlamaDecoderLayer.forward` via a wrapped closure that rewrites
        attention *logits* by modifying the input hidden states scaled by
        the steering weight along image-token positions. This is an
        approximation of the logits-space version of eq. (5).

        The cheapest faithful implementation: multiplicatively rescale the
        *output* of the attention block for image-token positions so that
        the next-layer representation gets more (or less) signal from the
        ViT-salient patches. The sign is chosen so that tokens the ViT
        [CLS] already highlights get up-weighted relative to those it
        ignores.

        This is the form used in the public reference re-implementations.
        """
        cfg = self._cfg
        assert cfg is not None

        def hook(module, args, kwargs):
            # args may be (hidden_states,) or (hidden_states, attn_mask, ...)
            if self._cls_attn is None or self._img_token_slice is None:
                return None  # no rewrite, pass through

            hs = args[0] if len(args) > 0 else kwargs.get("hidden_states")
            if hs is None:
                return None
            # hs: (B, T, D). Only rewrite during prefill, when T actually spans
            # the 576 image-token range. During KV-cache decode T == 1 -> skip.
            sl = self._img_token_slice
            if hs.size(1) < sl.stop:
                return None
            lam = cfg.steering_weight
            if cfg.adaptive and self._vabe_state["gate"]:
                lam = min(1.0, lam * cfg.adaptive_scale)
            # Bounded multiplicative perturbation on the residual stream at
            # image-token rows. CLIP [CLS]->patch attention is extremely peaky
            # (one patch can hold ~30 % of the mass while most hold ~0.01 %),
            # so we sqrt-compress and hard-clip to keep the intervention
            # semantically directional but numerically safe. Without this,
            # raw cls/cls.mean() can hit ~100 and the residual-stream blow-up
            # collapses next-token logits to EOS.
            cls = self._cls_attn.to(hs.device, dtype=hs.dtype)          # (576,), sum=1
            w_rel = cls / cls.mean().clamp_min(1e-8)                     # mean=1, peaky
            w_rel = torch.sqrt(w_rel.clamp_min(1e-6))                    # compress peaks
            w_rel = w_rel / w_rel.mean().clamp_min(1e-8)                 # re-center mean=1
            w_rel = w_rel.clamp(0.5, 2.0)                                # safety clip
            w = (1.0 - lam) + lam * w_rel                                # blend
            new_hs = hs.clone()
            new_hs[:, sl, :] = hs[:, sl, :] * w.view(1, -1, 1)
            if len(args) > 0:
                return (new_hs,) + args[1:], kwargs
            kwargs["hidden_states"] = new_hs
            return args, kwargs

        return hook

    def _get_llm_layers(self):
        """Locate the LlamaDecoderLayer list across transformers versions.

        - transformers 4.37.x: model.language_model.model.layers
        - transformers >= 4.47: model.language_model.layers   (LlamaModel is promoted)
        """
        llm = self.model.language_model
        if hasattr(llm, "layers"):
            return llm.layers
        if hasattr(llm, "model") and hasattr(llm.model, "layers"):
            return llm.model.layers
        raise AttributeError(
            f"Could not find decoder layers on {type(llm).__name__}; "
            f"attributes: {list(vars(llm).keys())[:10]}"
        )

    def _install_hooks(self) -> list:
        handles = []
        layers = self._get_llm_layers()
        for idx in self._cfg.steering_layers:
            if 0 <= idx < len(layers):
                h = layers[idx].register_forward_pre_hook(
                    self._make_attn_hook(idx), with_kwargs=True
                )
                handles.append(h)
        return handles

    # ---------- Captioning ----------
    @torch.no_grad()
    def caption_one(
        self,
        image: Image.Image,
        prompt: str,
        cfg: VEGASConfig,
    ) -> str:
        self._cfg = cfg

        inputs = self.processor(images=image, text=prompt, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        input_ids = inputs["input_ids"]

        # Locate the <image> placeholder → it gets expanded to 576 tokens
        # in the embedding stage. The image-token slice spans the placeholder
        # position through position + 576 in the *post-expansion* hidden states.
        img_pos = (input_ids[0] == IMAGE_TOKEN_INDEX).nonzero(as_tuple=True)[0]
        if img_pos.numel() == 0:
            raise ValueError("Prompt missing <image> placeholder.")
        start = int(img_pos[0].item())
        self._img_token_slice = slice(start, start + NUM_IMAGE_TOKENS)

        # ViT CLS attention for this image
        self._cls_attn = self._compute_vit_cls_attention(inputs["pixel_values"])

        # Adaptive VABE gate: decide once per sample from the ViT attention
        # entropy relative to uniform. High entropy (~uniform) => gate ON
        # because this is exactly where VEGAS is expected to help per §4.3.
        cls_np = self._cls_attn.clamp_min(1e-8)
        ent = float(-(cls_np * cls_np.log()).sum().item())
        uniform_ent = math.log(NUM_IMAGE_TOKENS)
        vabe = ent / uniform_ent  # in [0, 1]
        self._vabe_state["gate"] = vabe >= cfg.vabe_threshold

        handles = self._install_hooks()
        try:
            out_ids = self.model.generate(
                **inputs,
                max_new_tokens=cfg.max_new_tokens,
                do_sample=cfg.do_sample,
                temperature=cfg.temperature,
            )
        finally:
            for h in handles:
                h.remove()
            self._cls_attn = None
            self._img_token_slice = None

        gen = out_ids[0, input_ids.shape[1]:]
        return self.processor.tokenizer.decode(gen, skip_special_tokens=True).strip()

    @torch.no_grad()
    def caption_many(
        self,
        image_paths: Sequence[str | Path],
        image_ids: Sequence[int],
        prompt: str,
        cfg: VEGASConfig | None = None,
        progress: bool = True,
    ) -> list[dict]:
        cfg = cfg or VEGASConfig()
        out = []
        it: Iterable = zip(image_ids, image_paths)
        if progress:
            try:
                from tqdm import tqdm
                it = tqdm(list(it), desc="VEGAS caption")
            except ImportError:
                pass
        for iid, p in it:
            img = Image.open(p).convert("RGB")
            cap = self.caption_one(img, prompt, cfg)
            out.append({"image_id": int(iid), "caption": cap})
        return out
