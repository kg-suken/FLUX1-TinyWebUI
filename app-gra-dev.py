import torch
import datetime
from diffusers import FluxPipeline
import gradio as gr
from huggingface_hub import login
import os
# 現在のスクリプトのディレクトリパスを取得
script_dir = os.path.dirname(os.path.abspath(__file__))
# 保存先ディレクトリを指定（path/images）
save_dir = os.path.join(script_dir, 'images')

# ディレクトリが存在しない場合は作成
os.makedirs(save_dir, exist_ok=True)

login(token="")

# FluxPipelineの初期化
pipe = FluxPipeline.from_pretrained('black-forest-labs/FLUX.1-dev', torch_dtype=torch.bfloat16)
pipe.enable_sequential_cpu_offload()
pipe.vae.enable_slicing()
pipe.vae.enable_tiling()
pipe.to(torch.float16)


# 画像生成関数
def generate_image(prompt, prompt_2, height, width, num_inference_steps, guidance_scale, num_images_per_prompt):
    now = datetime.datetime.now()
    formatted_now = now.strftime('%Y%m%d_%H%M%S')

    # 画像生成
    out_images = pipe(
        prompt=prompt,
        prompt_2=prompt_2 if prompt_2 else prompt,  # prompt_2が指定されていない場合はpromptを使用
        guidance_scale=guidance_scale,
        height=height,
        width=width,
        num_inference_steps=num_inference_steps,
        num_images_per_prompt=num_images_per_prompt,
        max_sequence_length=512,
    ).images
    
    # 画像の保存
    file_paths = []
    for i, image in enumerate(out_images):
        file_path = os.path.join(save_dir, f'{formatted_now}-{i+1}.png') 
        image.save(file_path)
        file_paths.append(file_path)  # 生成した画像ファイルのパスをリストに追加
    
    return file_paths  # 生成したすべての画像ファイルのパスを返す

# Gradio UIの設定
with gr.Blocks(title="Flux1") as demo:
    with gr.Row():
        # 左側の設定カラム
        with gr.Column():
            gr.Markdown("### 画像生成パラメータの設定")
            prompt_input = gr.Textbox(
                label="プロンプトを入力", 
                placeholder="画像生成のためのテキストを入力してください",
                info="メインの画像生成テキストを指定します。"
            )
            prompt_2_input = gr.Textbox(
                label="セカンドプロンプトを入力（オプション）", 
                placeholder="空欄の場合はプロンプトが使用されます",
                info="オプションのテキストを指定します。"
            )
            height_input = gr.Slider(
                label="画像の高さ (ピクセル)", 
                minimum=256, maximum=2048, step=64, value=1024, 
                info="生成される画像の高さを指定します"
            )
            width_input = gr.Slider(
                label="画像の幅 (ピクセル)", 
                minimum=256, maximum=2048, step=64, value=1024, 
                info="生成される画像の幅を指定します"
            )
            steps_input = gr.Slider(
                label="推論ステップ数", 
                minimum=1, maximum=100, step=1, value=50, 
                info="ステップ数を増やすと高品質な画像が得られます"
            )
            guidance_scale_input = gr.Slider(
                label="ガイダンススケール", 
                minimum=0.0, maximum=20.0, step=0.1, value=3.5, 
                info="テキストにどれだけ忠実な画像を生成するかを指定します"
            )
            num_images_input = gr.Slider(
                label="生成する画像の枚数", 
                minimum=1, maximum=12, step=1, value=1, 
                info="プロンプトごとに生成する画像の数を指定します"
            )

        # 右側の出力カラム
        with gr.Column():
            gr.Markdown("### 生成された画像")
            generate_button = gr.Button("画像を生成")
            output_images = gr.Gallery(
                label="生成された画像", 
                show_label=True,
                scale=1,
            )

    # ボタン押下時に画像生成を実行
    generate_button.click(
        fn=generate_image, 
        inputs=[
            prompt_input, prompt_2_input, height_input, width_input, 
            steps_input, guidance_scale_input, num_images_input
        ], 
        outputs=output_images
    )

# アプリの起動
demo.launch(server_name="0.0.0.0", server_port=8080)
