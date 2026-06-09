import os
import shlex
import sys
import warnings
from pathlib import Path

import torch
from PIL import Image

from .llava import LLaVA




def _split_paths(value):
    if value is None:
        return None
    if isinstance(value, (list, tuple)):
        return [str(path) for path in value]

    value = str(value)
    separator = "," if "," in value else os.pathsep
    return [path for path in (part.strip() for part in value.split(separator)) if path]


def _normalise_overrides(overrides):
    if overrides is None:
        return []
    if isinstance(overrides, str):
        return shlex.split(overrides)
    return list(overrides)


def _add_ves_root(ves_root):
    ves_root = ves_root or os.environ.get("VISION_ENCODER_SWAPPING_ROOT")
    if ves_root is None:
        raise ValueError(
            "Pass ves_root=... or set VISION_ENCODER_SWAPPING_ROOT."
        )
    ves_root = Path(ves_root).expanduser()
    if not ves_root.exists():
        raise FileNotFoundError(
            f"VisionEncoderSwapping root does not exist: {ves_root}. "
            "Pass ves_root=... or set VISION_ENCODER_SWAPPING_ROOT."
        )

    ves_root = str(ves_root.resolve())
    if ves_root not in sys.path:
        sys.path.insert(0, ves_root)
    return ves_root


def _resolve_checkpoint_paths(checkpoint_paths=None, checkpoint_path=None):
    checkpoint_paths = (
        checkpoint_paths
        if checkpoint_paths is not None
        else checkpoint_path
        if checkpoint_path is not None
        else os.environ.get("LLAVA_SWAP_CHECKPOINTS")
    )
    checkpoint_paths = _split_paths(checkpoint_paths)
    if not checkpoint_paths:
        raise ValueError(
            "LLaVASwap requires at least one checkpoint. Pass checkpoint_paths=... "
            "or set LLAVA_SWAP_CHECKPOINTS."
        )
    return checkpoint_paths


def _resolve_config_path(config_path=None, config=None):
    if config_path is not None:
        return config_path
    if isinstance(config, (str, os.PathLike)):
        return config
    return os.environ.get("LLAVA_SWAP_CONFIG")


def _load_configs(config=None, config_path=None, checkpoint_paths=None, config_overrides=None):
    from omegaconf import OmegaConf

    from eval_llava_checkpoint import resolve_configs

    overrides = _normalise_overrides(config_overrides)
    if config is not None and not isinstance(config, (str, os.PathLike)):
        cfg = OmegaConf.create(config)
        if overrides:
            cfg = OmegaConf.merge(cfg, OmegaConf.from_cli(overrides))
        return [cfg for _ in checkpoint_paths]

    return resolve_configs(_resolve_config_path(config_path, config), checkpoint_paths, overrides)


class LLaVASwap(LLaVA):
    INSTALL_REQ = True
    INTERLEAVE = True

    def __init__(
        self,
        model_path=None,
        checkpoint_paths=None,
        checkpoint_path=None,
        config_path=None,
        config=None,
        config_overrides=None,
        ves_root=None,
        device=None,
        **kwargs,
    ):
        _add_ves_root(ves_root)

        from eval_llava_checkpoint import build_swapped_llava, resolve_model_path

        checkpoint_paths = _resolve_checkpoint_paths(checkpoint_paths, checkpoint_path)
        configs = _load_configs(
            config=config,
            config_path=config_path,
            checkpoint_paths=checkpoint_paths,
            config_overrides=config_overrides,
        )

        model_path = model_path or os.environ.get("LLAVA_SWAP_MODEL_PATH")
        model_path = resolve_model_path(model_path, configs)
        device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))

        self.backend = build_swapped_llava(configs, model_path, checkpoint_paths, device)
        self.tokenizer = self.backend.tokenizer
        self.model = self.backend.model
        self.image_processor = self.backend.image_processor
        self.context_len = getattr(self.model, "context_len", None)
        self.conv_mode = getattr(self.backend, "conv_mode", "llava_v1")
        self.system_prompt = (
            "A chat between a curious human and an artificial intelligence assistant. "
            "The assistant gives helpful, detailed, and polite answers to the human's questions. "
        )
        self.stop_str = "</s>"

        kwargs_default = dict(
            do_sample=False,
            temperature=0,
            max_new_tokens=2048,
            top_p=None,
            num_beams=1,
            use_cache=True,
        )
        kwargs_default.update(kwargs)
        self.kwargs = kwargs_default
        self.checkpoint_paths = checkpoint_paths
        warnings.warn(f"Following kwargs received: {self.kwargs}, will use as generation config.")

    def generate_inner(self, message, dataset=None):
        content, image_paths = self.concat_tilist(message)
        images = [Image.open(path).convert("RGB") for path in image_paths]
        image = None if not images else images[0] if len(images) == 1 else images

        return self.backend.get_response(content, image=image, **self.kwargs)


LLaVA_Swap = LLaVASwap
