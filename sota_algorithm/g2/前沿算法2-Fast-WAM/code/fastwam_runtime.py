"""FastWAM inference runtime for G2: image stitching, normalization, action chunks.

Mirrors the official RoboTwin deployment path
(FastWAM/experiments/robotwin/fastwam_policy/deploy_policy.py):
- head camera resized to 256x320 on top, wrists at 128x160 below -> 384x320;
- state goes through FastWAMProcessor transforms + z-score normalization,
  actions are denormalized with the same statistics;
- text uses the DEFAULT_PROMPT template (identical to the T5 cache at training).

Two text-conditioning modes:
- `use_text_encoder=False` (default): load the precomputed T5 cache produced by
  precompute_text.sh and pass `context/context_mask` to `infer_action` — the T5
  encoder (~9.4GB) is never loaded, so the policy fits on a 24GB GPU;
- `use_text_encoder=True`: load T5 and encode prompts on the fly (needs ~22GB+).

Usage (smoke test with random inputs):
    python fastwam_runtime.py \
        --checkpoint FastWAM/runs/<run>/checkpoints/weights/step_007500.pt \
        --dataset-stats FastWAM/runs/<run>/dataset_stats.json --smoke
"""

from __future__ import annotations

import argparse
import hashlib
import inspect
import time
from pathlib import Path

import numpy as np
import torch

from image_layout import concat_robotwin_layout
from settings import (
    ACTION_DIM,
    ACTION_VIDEO_FREQ_RATIO,
    CAMERA_KEYS,
    FASTWAM_ROOT,
    STATE_DIM,
    TEXT_CACHE_ROOT,
)

FASTWAM_CONFIGS = FASTWAM_ROOT / "configs"
G2_TASK = "g2_uncond_3cam384_1e-4"
G2_TEXT_CACHE_DIR = TEXT_CACHE_ROOT / "g2"


class FastWAMPolicy:
    """Load a FastWAM checkpoint and predict G2 action chunks (16D absolute targets)."""

    def __init__(
        self,
        checkpoint: str | Path,
        dataset_stats: str | Path,
        *,
        task: str = G2_TASK,
        device: str = "cuda",
        mixed_precision: str = "bf16",
        action_horizon: int | None = None,
        num_video_frames: int | None = None,
        num_inference_steps: int = 10,
        seed: int | None = None,
        rand_device: str = "cpu",
        action_infer_mode: str | None = None,
        negative_prompt: str = "",
        text_cfg_scale: float = 1.0,
        compile_action_infer: bool = False,
        inference_mode: str = "action",
        measure_model_latency: bool = False,
        use_text_encoder: bool = False,
        text_embed_cache_dir: str | Path | None = None,
    ) -> None:
        if not FASTWAM_CONFIGS.is_dir():
            raise FileNotFoundError(
                f"{FASTWAM_CONFIGS} missing; clone FastWAM first (see setup_env.sh)"
            )
        from hydra import compose, initialize_config_dir
        from hydra.core.global_hydra import GlobalHydra
        from hydra.utils import instantiate
        from omegaconf import OmegaConf

        if GlobalHydra.instance().is_initialized():
            GlobalHydra.instance().clear()
        with initialize_config_dir(version_base="1.3", config_dir=str(FASTWAM_CONFIGS)):
            cfg = compose(config_name="train", overrides=[f"task={task}"])

        model_dtype = {"no": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}[
            mixed_precision
        ]
        model_cfg = OmegaConf.create(OmegaConf.to_container(cfg.model, resolve=True))
        model_cfg.load_text_encoder = bool(use_text_encoder)
        self.model = instantiate(model_cfg, model_dtype=model_dtype, device=device)
        self.model.load_checkpoint(str(checkpoint))
        self.model = self.model.to(device).eval()

        from fastwam.datasets.lerobot.utils.normalizer import load_dataset_stats_from_json

        self.processor = instantiate(cfg.data.train.processor).eval()
        stats = load_dataset_stats_from_json(str(dataset_stats))
        self.processor.set_normalizer_from_stats(stats)

        self.action_horizon = action_horizon or int(cfg.data.train.num_frames) - 1
        self.num_video_frames = num_video_frames or (
            self.action_horizon // ACTION_VIDEO_FREQ_RATIO + 1
        )
        self.num_inference_steps = int(num_inference_steps)
        self.seed = seed
        self.rand_device = rand_device
        self.action_infer_mode = action_infer_mode
        self.negative_prompt = negative_prompt
        self.text_cfg_scale = float(text_cfg_scale)
        self.compile_action_infer = bool(compile_action_infer)
        if inference_mode not in {"action", "joint"}:
            raise ValueError(f"inference_mode must be 'action' or 'joint', got {inference_mode!r}")
        self.inference_mode = inference_mode
        self.measure_model_latency = bool(measure_model_latency)
        self.last_model_latency_s: float | None = None
        self.last_video_frames = None
        self.use_text_encoder = bool(use_text_encoder)
        self.text_embed_cache_dir = Path(
            text_embed_cache_dir if text_embed_cache_dir is not None else G2_TEXT_CACHE_DIR
        )
        if not self.use_text_encoder and not self.text_embed_cache_dir.is_dir():
            raise FileNotFoundError(
                f"{self.text_embed_cache_dir} missing; run precompute_text.sh first, "
                "or pass use_text_encoder=True / text_embed_cache_dir=..."
            )
        self._context_payload_cache: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}

        state_meta = self.processor.shape_meta["state"]
        action_meta = self.processor.shape_meta["action"]
        if len(state_meta) != 1 or len(action_meta) != 1:
            raise ValueError("G2 config must have exactly one merged state/action key")
        self._state_key = state_meta[0]["key"]
        self._action_key = action_meta[0]["key"]

    def _normalize_state(self, state: np.ndarray) -> torch.Tensor:
        state = np.asarray(state, dtype=np.float32)
        if state.shape != (STATE_DIM,):
            raise ValueError(f"state must be ({STATE_DIM},), got {state.shape}")
        batch = {"state": {self._state_key: torch.from_numpy(state).unsqueeze(0)}}
        batch = self.processor.action_state_transform(batch)
        batch = self.processor.normalizer.forward(batch)
        return batch["state"][self._state_key]

    def _denormalize_action(self, action: torch.Tensor) -> np.ndarray:
        if action.ndim == 2:
            action = action.unsqueeze(0)
        normalizer = self.processor.normalizer.normalizers["action"][self._action_key]
        denorm = normalizer.backward(action.to(dtype=torch.float32, device="cpu"))
        return denorm.numpy()[0]

    def _cached_context(self, prompt: str) -> tuple[torch.Tensor, torch.Tensor]:
        """Load the precomputed T5 embedding for a DEFAULT_PROMPT-formatted prompt.

        Follows the training-time reader exactly (robot_video_dataset.py): the
        cache file is sha256(prompt), padded positions of `context` are zeroed
        (Wan2.2 convention), and `context_mask` is all-ones.
        """
        if prompt in self._context_payload_cache:
            return self._context_payload_cache[prompt]
        hashed = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
        matches = sorted(self.text_embed_cache_dir.glob(f"{hashed}.t5_len*.wan22ti2v5b.pt"))
        if not matches:
            raise FileNotFoundError(
                f"no T5 cache for prompt hash {hashed} in {self.text_embed_cache_dir}; "
                f"prompt: {prompt!r}. Run precompute_text.sh first."
            )
        payload = torch.load(matches[0], map_location="cpu")
        context = payload["context"]
        mask = payload["mask"].bool()
        context = context.clone()
        context[~mask] = 0.0
        result = (context, torch.ones_like(mask))
        self._context_payload_cache[prompt] = result
        return result

    def infer(
        self,
        images: dict[str, np.ndarray],
        state: np.ndarray,
        instruction: str,
    ) -> np.ndarray:
        """One three-camera observation + current qpos -> [action_horizon, 16] targets."""
        from fastwam.datasets.lerobot.robot_video_dataset import DEFAULT_PROMPT

        frame = concat_robotwin_layout(images)
        image_tensor = torch.from_numpy(frame).permute(2, 0, 1).unsqueeze(0).to(
            device=self.model.device, dtype=self.model.torch_dtype,
        )
        image_tensor = image_tensor * (2.0 / 255.0) - 1.0
        proprio = self._normalize_state(state)

        prompt = DEFAULT_PROMPT.format(task=instruction)
        kwargs = {
            "input_image": image_tensor,
            "action_horizon": self.action_horizon,
            "proprio": proprio,
            "text_cfg_scale": self.text_cfg_scale,
            "num_inference_steps": self.num_inference_steps,
            "seed": self.seed,
            "rand_device": self.rand_device,
            "compile_action_infer": self.compile_action_infer,
        }
        if self.use_text_encoder:
            kwargs["prompt"] = prompt
            kwargs["negative_prompt"] = self.negative_prompt
        else:
            context, context_mask = self._cached_context(prompt)
            kwargs["prompt"] = None
            kwargs["context"] = context
            kwargs["context_mask"] = context_mask
            kwargs["negative_prompt"] = None
            if self.text_cfg_scale != 1.0:
                raise ValueError(
                    "text_cfg_scale != 1.0 requires the text encoder for the negative "
                    "prompt; pass use_text_encoder=True"
                )
        if self.measure_model_latency:
            torch.cuda.synchronize(self.model.device)
            model_t0 = time.perf_counter()
        with torch.no_grad():
            if self.inference_mode == "joint":
                kwargs["num_video_frames"] = self.num_video_frames
                kwargs["test_action_with_infer_action"] = False
                pred = self.model.infer_joint(**kwargs)
                self.last_video_frames = pred["video"]
            else:
                self.last_video_frames = None
                parameters = inspect.signature(self.model.infer_action).parameters
                if "num_video_frames" in parameters:
                    kwargs["num_video_frames"] = self.num_video_frames
                if "action_infer_mode" in parameters and self.action_infer_mode is not None:
                    kwargs["action_infer_mode"] = self.action_infer_mode
                pred = self.model.infer_action(**kwargs)
        if self.measure_model_latency:
            torch.cuda.synchronize(self.model.device)
            self.last_model_latency_s = time.perf_counter() - model_t0
        action = self._denormalize_action(pred["action"])
        if action.shape != (self.action_horizon, ACTION_DIM):
            raise ValueError(f"unexpected action chunk shape: {action.shape}")
        return action


def main() -> None:
    parser = argparse.ArgumentParser(description="FastWAM G2 inference runtime smoke test")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--dataset-stats", required=True)
    parser.add_argument("--task", default=G2_TASK)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--num-inference-steps", type=int, default=10)
    parser.add_argument("--inference-mode", default="action", choices=["action", "joint"])
    parser.add_argument("--action-infer-mode", default=None, choices=[None, "idm", "first_frame"])
    parser.add_argument("--use-text-encoder", action="store_true",
                        help="load T5 (~9.4GB) and encode prompts online; default uses the precomputed cache")
    parser.add_argument("--text-embed-cache-dir", default=None)
    parser.add_argument("--smoke", action="store_true", help="run one random-input inference")
    args = parser.parse_args()

    policy = FastWAMPolicy(
        args.checkpoint,
        args.dataset_stats,
        task=args.task,
        device=args.device,
        num_inference_steps=args.num_inference_steps,
        inference_mode=args.inference_mode,
        action_infer_mode=args.action_infer_mode,
        seed=42,
        use_text_encoder=args.use_text_encoder,
        text_embed_cache_dir=args.text_embed_cache_dir,
    )
    if not args.smoke:
        print("Model loaded. Pass --smoke to run one random-input inference.")
        return

    rng = np.random.default_rng(0)
    images = {
        key: rng.integers(0, 255, (240, 320, 3), dtype=np.uint8) for key in CAMERA_KEYS
    }
    state = rng.standard_normal(STATE_DIM).astype(np.float32)
    for index in range(3):
        start = time.perf_counter()
        action = policy.infer(images, state, "Pick up the red block and place it into the empty box.")
        elapsed = time.perf_counter() - start
        print(f"iter {index}: action{action.shape} in {elapsed * 1000:.1f}ms "
              f"range [{action.min():.3f}, {action.max():.3f}]")


if __name__ == "__main__":
    main()
