# Copyright 2024 NVIDIA CORPORATION & AFFILIATES
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# SPDX-License-Identifier: Apache-2.0

import argparse
import os
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

os.environ["DISABLE_XFORMERS"] = "1"
os.environ.setdefault("USE_CHUNKWISE_GDN", "1")
os.environ["TOKENIZERS_PARALLELISM"] = "false"
warnings.filterwarnings("ignore")

import imageio
import imageio.v3 as iio
import numpy as np
import pyrallis
import torch
import torch.nn.functional as F

import diffusion.model.nets  # noqa: F401 - register model/attention modules.
from diffusion import FlowEuler
from diffusion.model.builder import build_model, get_tokenizer_and_text_encoder, get_vae, vae_decode, vae_encode
from diffusion.model.utils import get_weight_dtype
from diffusion.scheduler.sana_v2v_streaming_sampler import SANAStreamingSampler
from diffusion.utils.config import AEConfig, ModelConfig, SchedulerConfig, TextEncoderConfig
from sana.tools import resolve_hf_path
from tools.download import find_model

REPO_ROOT = Path(__file__).resolve().parents[2]
RELEASE_REPO_ID = "Yuyang-z/test-streaming"

DEFAULTS = {
    "bidirectional_short": {
        "config": "configs/sana_v2v/sana_v2v_bidirectional_2b_720p.yaml",
        "model_path": f"hf://{RELEASE_REPO_ID}/dit/sana_bidirectional_short.pth",
        "num_frames": 81,
        "step": 50,
        "cfg_scale": 6.0,
    },
    "long_streaming": {
        "config": "configs/sana_v2v/sana_v2v_streaming_2b_720p.yaml",
        "model_path": f"hf://{RELEASE_REPO_ID}/dit/sana_streaming_ar.pth",
        "num_frames": 969,
        "step": 4,
        "cfg_scale": 1.0,
    },
}

DEFAULT_NEGATIVE_PROMPT = (
    "A chaotic sequence with misshapen, deformed limbs in heavy motion blur, sudden disappearance, jump cuts, "
    "jerky movements, rapid shot changes, frames out of sync, inconsistent character shapes, temporal artifacts, "
    "jitter, and ghosting effects, creating a disorienting visual experience."
)


@dataclass
class V2VModelConfig(ModelConfig):
    rope_fhw_dim: Optional[Tuple[int, int, int]] = None
    t_kernel_size: int = 3
    flash_attn_layer_idx: Optional[List[int]] = None
    flash_attn_layer_type: Optional[str] = None
    flash_attn_window_count: Optional[List[int]] = None
    pack_latents: bool = False
    addition_layers_num: int = 0
    cross_attn_image_embeds: bool = False
    chunk_index: Optional[List[int]] = None
    softmax_ratio: Optional[float] = 0.25
    softmax_layer_indices: Optional[List[int]] = None
    softmax_attn_type: str = "V2VGatedSoftmaxAttention"


@dataclass
class InferenceConfig:
    model: V2VModelConfig
    vae: AEConfig
    text_encoder: TextEncoderConfig
    scheduler: SchedulerConfig
    work_dir: str = ""


def model_v2v_init_config(config: InferenceConfig, latent_size: int = 32):
    pred_sigma = getattr(config.scheduler, "pred_sigma", True)
    learn_sigma = getattr(config.scheduler, "learn_sigma", True) and pred_sigma
    return {
        "input_size": latent_size,
        "pe_interpolation": config.model.pe_interpolation,
        "config": config,
        "model_max_length": config.text_encoder.model_max_length,
        "qk_norm": config.model.qk_norm,
        "micro_condition": config.model.micro_condition,
        "caption_channels": config.text_encoder.caption_channels,
        "class_dropout_prob": config.model.class_dropout_prob,
        "y_norm": config.text_encoder.y_norm,
        "attn_type": config.model.attn_type,
        "ffn_type": config.model.ffn_type,
        "mlp_ratio": config.model.mlp_ratio,
        "mlp_acts": list(config.model.mlp_acts),
        "in_channels": config.vae.vae_latent_dim,
        "additional_inchannels": config.vae.vae_latent_dim,
        "use_pe": config.model.use_pe,
        "pos_embed_type": config.model.pos_embed_type,
        "rope_fhw_dim": config.model.rope_fhw_dim,
        "linear_head_dim": config.model.linear_head_dim,
        "pred_sigma": pred_sigma,
        "learn_sigma": learn_sigma,
        "cross_norm": config.model.cross_norm,
        "cross_attn_type": config.model.cross_attn_type,
        "cross_attn_image_embeds": config.model.cross_attn_image_embeds,
        "t_kernel_size": config.model.t_kernel_size,
        "flash_attn_layer_idx": config.model.flash_attn_layer_idx,
        "flash_attn_layer_type": config.model.flash_attn_layer_type,
        "flash_attn_window_count": config.model.flash_attn_window_count,
        "pack_latents": config.model.pack_latents,
        "addition_layers_num": config.model.addition_layers_num,
        "timestep_norm_scale_factor": config.scheduler.timestep_norm_scale_factor,
        "softmax_ratio": config.model.softmax_ratio,
        "softmax_layer_indices": config.model.softmax_layer_indices,
        "softmax_attn_type": config.model.softmax_attn_type,
    }


def str2bool(value):
    if isinstance(value, bool):
        return value
    return str(value).lower() in {"1", "true", "yes", "y"}


def parse_args():
    parser = argparse.ArgumentParser(description="Sana V2V inference.")
    parser.add_argument("--mode", choices=tuple(DEFAULTS), default="long_streaming")
    parser.add_argument("--config", default=None, help="Slim V2V YAML config.")
    parser.add_argument("--model_path", default=None, help="DiT checkpoint, local path or hf:// URI.")
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--video_path", required=True, help="Source video path, local path or hf:// URI.")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--output_name", default="output.mp4")
    parser.add_argument("--num_frames", type=int, default=None)
    parser.add_argument("--height", type=int, default=704)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--fps", type=int, default=16)
    parser.add_argument("--step", type=int, default=None)
    parser.add_argument("--cfg_scale", type=float, default=None)
    parser.add_argument("--flow_shift", type=float, default=None)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--negative_prompt", default=None)
    parser.add_argument("--num_cached_blocks", type=int, default=2)
    parser.add_argument("--sink_token", type=str2bool, default=True)
    args = parser.parse_args()

    mode_defaults = DEFAULTS[args.mode]
    args.config = args.config or mode_defaults["config"]
    args.model_path = args.model_path or mode_defaults["model_path"]
    args.num_frames = args.num_frames or mode_defaults["num_frames"]
    args.step = args.step or mode_defaults["step"]
    args.cfg_scale = args.cfg_scale if args.cfg_scale is not None else mode_defaults["cfg_scale"]
    if args.negative_prompt is None:
        args.negative_prompt = DEFAULT_NEGATIVE_PROMPT if args.mode == "bidirectional_short" else ""
    return args


def resolve_local_path(path, *, for_output=False):
    p = Path(path)
    if p.is_absolute() or str(path).startswith("hf://"):
        return str(p) if not str(path).startswith("hf://") else str(path)

    cwd_path = (Path.cwd() / p).resolve()
    if for_output:
        return str(cwd_path)
    if cwd_path.exists():
        return str(cwd_path)

    repo_path = (REPO_ROOT / p).resolve()
    if repo_path.exists():
        return str(repo_path)
    return str(p)


def resolve_input_video_path(video_path):
    resolved = resolve_local_path(video_path)
    if str(resolved).startswith("hf://"):
        resolved = resolve_hf_path(str(resolved))
    if not Path(resolved).exists():
        raise FileNotFoundError(f"Source video does not exist: {video_path}")
    return str(resolved)


def read_video(video_path, height, width, num_frames):
    frames = []
    local_video_path = resolve_input_video_path(video_path)
    for frame in iio.imiter(local_video_path, plugin="pyav"):
        if frame.shape[-1] > 3:
            frame = frame[..., :3]
        frames.append(frame)
        if len(frames) >= num_frames:
            break
    if len(frames) < num_frames:
        raise RuntimeError(f"{video_path} has {len(frames)} decoded frames, but {num_frames} are required.")

    video = torch.from_numpy(np.stack(frames, axis=0)).permute(0, 3, 1, 2).float() / 255.0
    src_h, src_w = video.shape[-2:]
    scale = max(height / src_h, width / src_w)
    resized_h = int(round(src_h * scale))
    resized_w = int(round(src_w * scale))
    video = F.interpolate(video, size=(resized_h, resized_w), mode="bilinear", align_corners=False)
    top = max((resized_h - height) // 2, 0)
    left = max((resized_w - width) // 2, 0)
    video = video[..., top : top + height, left : left + width]
    return video.mul_(2.0).sub_(1.0)


def save_video(video, output_path, fps):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with imageio.get_writer(output_path, fps=fps, codec="libx264", quality=5) as writer:
        for start in range(0, video.shape[1], 32):
            chunk = video[:, start : start + 32].detach().to("cpu", dtype=torch.float32)
            chunk = torch.clamp(127.5 * chunk + 127.5, 0, 255).to(torch.uint8)
            for frame in chunk.permute(1, 2, 3, 0).contiguous().numpy():
                writer.append_data(frame)


@torch.no_grad()
def encode_prompt(tokenizer, text_encoder, prompt, config, device, *, use_chi_prompt):
    max_length = config.text_encoder.model_max_length
    if use_chi_prompt:
        chi_prompt = "\n".join(config.text_encoder.chi_prompt)
        prompt = chi_prompt + prompt
        max_length = len(tokenizer.encode(chi_prompt)) + max_length - 2

    tokens = tokenizer(
        prompt,
        max_length=max_length,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    ).to(device)
    hidden_states = text_encoder(tokens.input_ids, tokens.attention_mask)[0]
    if use_chi_prompt:
        select_index = [0] + list(range(-config.text_encoder.model_max_length + 1, 0))
        hidden_states = hidden_states[:, select_index]
        attention_mask = tokens.attention_mask[:, select_index]
    else:
        attention_mask = tokens.attention_mask
    return hidden_states[:, None], attention_mask


def normalize_state_dict(checkpoint):
    if "generator" in checkpoint:
        checkpoint = checkpoint["generator"]
    if "state_dict" not in checkpoint:
        checkpoint = {
            "state_dict": {
                key.removeprefix("model.").removeprefix("module."): value for key, value in checkpoint.items()
            }
        }
    return checkpoint["state_dict"]


def load_model(config, latent_size, device, weight_dtype, model_path):
    model_kwargs = model_v2v_init_config(config, latent_size=latent_size)
    model = build_model(
        config.model.model,
        use_fp32_attention=config.model.get("fp32_attention", False),
        **model_kwargs,
    ).to(device)
    state_dict = normalize_state_dict(find_model(model_path))
    if "pos_embed" not in state_dict and "pos_embed" in model.state_dict():
        state_dict["pos_embed"] = model.state_dict()["pos_embed"]
    model.load_state_dict(state_dict, strict=True)
    return model.eval().to(weight_dtype)


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    torch.set_grad_enabled(False)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    config = pyrallis.parse(config_class=InferenceConfig, config_path=resolve_hf_path(args.config), args=[])
    weight_dtype = get_weight_dtype(config.model.mixed_precision)
    vae_dtype = get_weight_dtype(config.vae.weight_dtype)
    vae_stride = config.vae.vae_stride
    latent_t = (args.num_frames - 1) // vae_stride[0] + 1
    latent_h = args.height // vae_stride[1]
    latent_w = args.width // vae_stride[2]
    latent_size = config.model.image_size // config.vae.vae_downsample_rate
    flow_shift = (
        args.flow_shift
        if args.flow_shift is not None
        else config.scheduler.inference_flow_shift
        if config.scheduler.inference_flow_shift is not None
        else config.scheduler.flow_shift
    )

    vae = get_vae(config.vae.vae_type, config.vae.vae_pretrained, device=device, dtype=vae_dtype, config=config.vae)
    tokenizer, text_encoder = get_tokenizer_and_text_encoder(config.text_encoder.text_encoder_name, device=device)
    model = load_model(config, latent_size, device, weight_dtype, args.model_path)

    prompt_embeds, prompt_mask = encode_prompt(
        tokenizer, text_encoder, args.prompt, config, device, use_chi_prompt=True
    )
    negative_embeds, negative_mask = encode_prompt(
        tokenizer, text_encoder, args.negative_prompt, config, device, use_chi_prompt=False
    )

    video = read_video(args.video_path, latent_h * vae_stride[1], latent_w * vae_stride[2], args.num_frames)
    video = video.permute(1, 0, 2, 3).unsqueeze(0).to(device=device, dtype=vae_dtype)
    image_vae_embeds = vae_encode(
        config.vae.vae_type,
        vae,
        video,
        sample_posterior=False,
        device=device,
    ).to(vae_dtype)

    generator = torch.Generator(device=device).manual_seed(args.seed)
    noise = torch.randn(
        1,
        config.vae.vae_latent_dim,
        latent_t,
        latent_h,
        latent_w,
        device=device,
        generator=generator,
    )
    if args.mode == "bidirectional_short":
        hw = torch.tensor([[args.height, args.width]], dtype=torch.float32, device=device)
        model_kwargs = {"data_info": {"img_hw": hw, "image_vae_embeds": image_vae_embeds}, "mask": prompt_mask}
        if args.cfg_scale > 1.0:
            model_kwargs["mask"] = torch.cat([negative_mask, prompt_mask], dim=0)
            model_kwargs["data_info"]["image_vae_embeds"] = torch.cat([image_vae_embeds, image_vae_embeds], dim=0)
        sampler = FlowEuler(
            model,
            condition=prompt_embeds,
            uncondition=negative_embeds,
            cfg_scale=args.cfg_scale,
            flow_shift=flow_shift,
            model_kwargs=model_kwargs,
        )
    else:
        base_chunk_frames = 24 // vae_stride[0]
        sampler = SANAStreamingSampler(
            model,
            condition=prompt_embeds,
            uncondition=negative_embeds,
            cfg_scale=args.cfg_scale,
            flow_shift=flow_shift,
            model_kwargs={"data_info": {"image_vae_embeds": image_vae_embeds}, "mask": prompt_mask},
            base_chunk_frames=base_chunk_frames,
            num_cached_blocks=args.num_cached_blocks,
            cache_strategy="fixed_rope",
            efficient_cache=False,
            sink_token=args.sink_token,
        )

    latents = sampler.sample(noise, steps=args.step).to(vae_dtype)

    samples = vae_decode(config.vae.vae_type, vae, latents)
    output_path = Path(resolve_local_path(args.output_dir, for_output=True)) / args.output_name
    save_video(samples[0], output_path, args.fps)
    print(f"Saved video to {output_path}")


if __name__ == "__main__":
    main()
