
# Flux.1をPythonで動かす方法

間違いなどがあったら指摘していただけるとうれしいです。質問もお待ちしています。

Windowsの場合はRAM32GB以上推奨

## YouTubeの使い方動画
[![](https://img.youtube.com/vi/QWgKL8-F2Bw/0.jpg)](https://www.youtube.com/watch?v=QWgKL8-F2Bw)



## Nvidia関係のコマンド

```bash
nvidia-smi
```

```bash
nvcc -V
```

## コマンド

```bash
python -m venv venv
```

```bash
.\venv\Scripts\activate
```

```bash
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
```

```bash
pip install --upgrade transformers diffusers
```

```bash
pip install transformers accelerate optimum-quanto sentencepiece protobuf
```

## プログラム

```python
import torch
from diffusers import FluxPipeline

pipe = FluxPipeline.from_pretrained("black-forest-labs/FLUX.1-schnell", torch_dtype=torch.bfloat16)
pipe.enable_sequential_cpu_offload()
pipe.vae.enable_slicing()
pipe.vae.enable_tiling()

pipe.to(torch.float16)

prompt = "A cat holding a sign that says hello world"
out = pipe(
    prompt=prompt,
    guidance_scale=0.,
    height=768,
    width=1360,
    num_inference_steps=4,
    max_sequence_length=256,
).images[0]
out.save("image.png")
```

## プログラム改良版

```python
import torch
import datetime
from diffusers import FluxPipeline

now = datetime.datetime.now()
formatted_now = now.strftime("%Y%m%d_%H%M%S")

pipe = FluxPipeline.from_pretrained("black-forest-labs/FLUX.1-schnell", torch_dtype=torch.bfloat16)
pipe.enable_sequential_cpu_offload()
pipe.vae.enable_slicing()
pipe.vae.enable_tiling()

pipe.to(torch.float16)

prompt = input("prompt:")
out = pipe(
    prompt=prompt,
    guidance_scale=0.,
    height=768,
    width=1360,
    num_inference_steps=4,
    max_sequence_length=256,
).images[0]
out.save(f"{formatted_now}.png")
```

## Gradio(おまけ)

```bash
pip install gradio
```

[schnell-Download](./app-gra.py)  
[dev-Download](./app-gra-dev.py)  
※dev版の実行にはアクセストークンが必要です

## リンクなど

* [Python](https://www.python.org/)
* [PyTorch](https://pytorch.org/)



## 制作
**sskrc**

---

今後の改良に関する提案やバグ報告は、お気軽にIssueを通してご連絡ください。
