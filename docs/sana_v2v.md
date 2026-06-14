# SANA-V2V: Video-to-Video Editing

SANA-V2V is the video-to-video editing variant of SANA-Video. It takes a source
video and an editing prompt, encodes the source video with the LTX-2 VAE, and
generates an edited video with a SANA Video DiT.

This release provides two inference modes:

- `bidirectional_short`: 81-frame short-video editing with a bidirectional GDN DiT.
- `long_streaming`: 969-frame long-video editing with fixed-RoPE state-cached GDN attention.

## Environment Setup

```bash
bash ./environment_setup.sh sana
conda activate sana
```

The V2V inference path requires the fused GDN kernels. The inference script sets
`USE_CHUNKWISE_GDN=1` by default, but it can also be set explicitly:

```bash
export USE_CHUNKWISE_GDN=1
```

## Model Weights

The inference script downloads the default DiT checkpoints from:

```text
hf://Yuyang-z/test-streaming
```

Default checkpoint paths:

| Mode | DiT checkpoint |
|------|----------------|
| `bidirectional_short` | `hf://Yuyang-z/test-streaming/dit/sana_bidirectional_short.pth` |
| `long_streaming` | `hf://Yuyang-z/test-streaming/dit/sana_streaming_ar.pth` |

Both modes use the public LTX-2 VAE from:

```text
Lightricks/LTX-2
```

The model configs are included in the repo:

| Mode | Config |
|------|--------|
| `bidirectional_short` | `configs/sana_v2v/sana_v2v_bidirectional_2b_720p.yaml` |
| `long_streaming` | `configs/sana_v2v/sana_v2v_streaming_2b_720p.yaml` |

## Input Video

`--video_path` should point to a source video with at least the requested number
of frames. The script decodes frames in order, resizes and center-crops them to
the requested resolution, and saves the edited video as MP4.

Default resolution and frame rate:

| Argument | Default |
|----------|---------|
| `--height` | `704` |
| `--width` | `1280` |
| `--fps` | `16` |

## Bidirectional Short-Video Inference

This mode edits 81 frames by default, uses Flow Euler sampling, and uses
classifier-free guidance with a default CFG scale of `6.0`.

```bash
python inference_video_scripts/v2v/inference_sana_v2v.py \
  --mode bidirectional_short \
  --prompt "turn the scene into a cinematic rainy night drive" \
  --video_path /path/to/source.mp4 \
  --output_dir results/sana_v2v_bidirectional \
  --output_name edited.mp4
```

Equivalent explicit command:

```bash
python inference_video_scripts/v2v/inference_sana_v2v.py \
  --mode bidirectional_short \
  --config configs/sana_v2v/sana_v2v_bidirectional_2b_720p.yaml \
  --model_path hf://Yuyang-z/test-streaming/dit/sana_bidirectional_short.pth \
  --prompt "turn the scene into a cinematic rainy night drive" \
  --video_path /path/to/source.mp4 \
  --num_frames 81 \
  --step 50 \
  --cfg_scale 6.0 \
  --output_dir results/sana_v2v_bidirectional \
  --output_name edited.mp4
```

`bidirectional_short` automatically uses the default V2V negative prompt unless
`--negative_prompt` is provided.

## Long Streaming Inference

This mode edits 969 frames by default with the fixed-RoPE streaming sampler.
The distilled streaming checkpoint runs with a default CFG scale of `1.0` and
`4` sampling steps.

```bash
python inference_video_scripts/v2v/inference_sana_v2v.py \
  --mode long_streaming \
  --prompt "turn the road into a futuristic neon city street" \
  --video_path /path/to/source.mp4 \
  --output_dir results/sana_v2v_streaming \
  --output_name edited.mp4
```

Equivalent explicit command:

```bash
python inference_video_scripts/v2v/inference_sana_v2v.py \
  --mode long_streaming \
  --config configs/sana_v2v/sana_v2v_streaming_2b_720p.yaml \
  --model_path hf://Yuyang-z/test-streaming/dit/sana_streaming_ar.pth \
  --prompt "turn the road into a futuristic neon city street" \
  --video_path /path/to/source.mp4 \
  --num_frames 969 \
  --step 4 \
  --cfg_scale 1.0 \
  --num_cached_blocks 2 \
  --sink_token true \
  --output_dir results/sana_v2v_streaming \
  --output_name edited.mp4
```

## Useful Arguments

| Argument | Description |
|----------|-------------|
| `--mode` | `bidirectional_short` or `long_streaming`. Defaults to `long_streaming`. |
| `--prompt` | Editing prompt. |
| `--video_path` | Source video path. |
| `--output_dir` | Directory for generated results. |
| `--output_name` | Output MP4 filename. Defaults to `output.mp4`. |
| `--num_frames` | Number of frames to read and edit. Defaults are mode-specific. |
| `--step` | Sampling steps. Defaults are mode-specific. |
| `--cfg_scale` | CFG scale. Defaults are mode-specific. |
| `--seed` | Random seed. Defaults to `0`. |
| `--model_path` | Override the DiT checkpoint path. Supports local paths and `hf://` URIs. |
| `--config` | Override the V2V YAML config. Supports local paths and `hf://` URIs. |
| `--save_latent` | Save initial noise and generated latents for debugging or comparison. |

## Implementation Notes

The V2V modules are registered through `diffusion.model.nets` and live in:

```text
diffusion/model/nets/sana_multi_scale_video_v2v.py
diffusion/model/nets/sana_v2v_attn_blocks.py
diffusion/scheduler/sana_v2v_streaming_sampler.py
```

The public V2V path keeps only the active attention blocks used by the released
configs: bidirectional GDN, state-cached bidirectional GDN, and their softmax
attention counterparts.
