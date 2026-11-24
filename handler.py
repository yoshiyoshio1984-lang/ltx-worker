"""
RunPod Serverless Worker - 動画生成 (LTX-2.3 Distilled, Diffusers, 音声付き)
推奨GPU: L40S 48GB (Serverless) - 19B/22B級のため RTX 4090 24GB は
  CPUオフロードで動く場合もあるが遅く、VRAM不足になりやすい
  - 5秒・768x512で 1本 約30-60秒 (L40S, 蒸留版 8ステップ)
  - 音声付き (ビデオとオーディオを同時生成)
  - モデル: diffusers/LTX-2.3-Distilled-Diffusers (LTX2Pipeline)
  - 注意: 旧 Lightricks/LTX-2.3 には model_index.json がなく
    from_pretrained で読めなかったが、diffusers公式が
    diffusers/LTX-2.3-Diffusers / -Distilled-Diffusers として
    Diffusers形式で再配布しており、最新の diffusers (git main)
    の LTX2Pipeline で読める。requirements は git 版 diffusers を使うこと。
環境変数:
  HF_TOKEN - Hugging Faceトークン (任意。公開モデルのため基本不要)
  MODEL_ID - デフォルト diffusers/LTX-2.3-Distilled-Diffusers
入力 (job['input']):
  prompt: str (必須)
  negative_prompt: str (任意)
  image_base64: str (任意, I2Vの場合)
  width: int (デフォルト 768, 32の倍数)
  height: int (デフォルト 512, 32の倍数)
  num_frames: int (デフォルト 121 = 5秒@24fps, 8k+1 に丸める)
  num_inference_steps: int (デフォルト 8, 蒸留版は 8固定)
  guidance_scale: float (デフォルト 1.0, 蒸留版は 1.0固定)
  seed: int (任意, -1でランダム)
出力:
  {"video_base64": "...mp4 (音声付き)...", "width": 768, "height": 512,
   "fps": 24, "model": "...", "has_audio": true}
注意:
  - ライセンス: LTX Video 2 Open Source License。Wan の Apache 2.0 とは
    違い、性的コンテンツに明示的な制限がある点に注意。
    NSFW用途で使う場合は、ライセンス条文を自分で確認すること。
  - 動画は run + ポーリング方式 (status確認) を使うこと。runsync は避ける。
"""
import os
import base64
import random
import tempfile

import runpod
import torch
from diffusers import LTX2Pipeline
from diffusers.pipelines.ltx2.export_utils import encode_video
from diffusers.pipelines.ltx2.utils import DEFAULT_NEGATIVE_PROMPT, DISTILLED_SIGMA_VALUES

MODEL_ID = os.environ.get("MODEL_ID", "diffusers/LTX-2.3-Distilled-Diffusers")

print(f"Loading {MODEL_ID} ...", flush=True)
pipe = LTX2Pipeline.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.bfloat16,
    token=os.environ.get("HF_TOKEN") or None,
)
pipe.enable_model_cpu_offload()
try:
    pipe.vae.enable_tiling()
except Exception:
    pass
print("Model loaded.", flush=True)

_I2V_PIPE = None


def get_i2v_pipe():
    global _I2V_PIPE
    if _I2V_PIPE is None:
        from diffusers import LTX2ImageToVideoPipeline

        print("Loading I2V pipeline (same checkpoint) ...", flush=True)
        _I2V_PIPE = LTX2ImageToVideoPipeline.from_pretrained(
            MODEL_ID,
            torch_dtype=torch.bfloat16,
            token=os.environ.get("HF_TOKEN") or None,
        )
        _I2V_PIPE.enable_model_cpu_offload()
        try:
            _I2V_PIPE.vae.enable_tiling()
        except Exception:
            pass
    return _I2V_PIPE


def handler(job):
    inp = job.get("input", {}) or {}
    prompt = inp.get("prompt", "")
    if not prompt:
        return {"error": "prompt is required"}

    width = int(inp.get("width", 768))
    height = int(inp.get("height", 512))
    # LTXは 32の倍数
    width = (width // 32) * 32
    height = (height // 32) * 32
    num_frames = int(inp.get("num_frames", 121))
    # LTXは 8k+1 フレーム
    num_frames = ((num_frames - 1) // 8) * 8 + 1
    fps = float(inp.get("frame_rate", 24.0))
    # 蒸留版は 8ステップ / CFG=1 固定
    steps = int(inp.get("num_inference_steps", 8))
    guidance = float(inp.get("guidance_scale", 1.0))
    negative_prompt = inp.get("negative_prompt") or DEFAULT_NEGATIVE_PROMPT
    seed = inp.get("seed", -1)
    if seed is None or int(seed) < 0:
        seed = random.randint(0, 2**32 - 1)
    seed = int(seed)
    generator = torch.Generator(device="cpu").manual_seed(seed)

    kwargs = dict(
        prompt=prompt,
        negative_prompt=negative_prompt,
        width=width,
        height=height,
        num_frames=num_frames,
        frame_rate=fps,
        num_inference_steps=steps,
        sigmas=DISTILLED_SIGMA_VALUES,
        guidance_scale=guidance,
        output_type="np",
        generator=generator,
        return_dict=False,
    )

    image_b64 = inp.get("image_base64")
    if image_b64:
        import io

        from PIL import Image

        img = Image.open(io.BytesIO(base64.b64decode(image_b64))).convert("RGB")
        video, audio = get_i2v_pipe()(image=img, **kwargs)
    else:
        video, audio = pipe(**kwargs)

    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
        out_path = f.name

    audio_tensor = None
    audio_sr = None
    try:
        if audio is not None:
            audio_tensor = audio[0].float().cpu()
            audio_sr = pipe.vocoder.config.output_sampling_rate
    except Exception:
        audio_tensor = None

    encode_video(
        video[0],
        fps=fps,
        audio=audio_tensor,
        audio_sample_rate=audio_sr,
        output_path=out_path,
    )

    with open(out_path, "rb") as f:
        video_b64 = base64.b64encode(f.read()).decode("utf-8")
    os.remove(out_path)

    return {
        "video_base64": video_b64,
        "width": width,
        "height": height,
        "fps": fps,
        "num_frames": num_frames,
        "seed": seed,
        "model": MODEL_ID,
        "has_audio": audio_tensor is not None,
    }


runpod.serverless.start({"handler": handler})
