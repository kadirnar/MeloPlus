import os

os.system('python -m unidic download')

import nltk

nltk.download('averaged_perceptron_tagger_eng')

import gradio as gr
from meloplus.api import TTS as MeloInference
from huggingface_hub import hf_hub_download
import json
import requests
import zipfile
import shutil
from pathlib import Path

# Cache for loaded models
model_cache = {}


def get_model_paths(model_id, version):
    base_dir = os.path.dirname(os.path.abspath(__file__))
    cache_dir = base_dir  # Cache dosyalarını proje dizinine kaydet

    try:
        # HuggingFace'den model dosyalarını indir
        g_path = hf_hub_download(repo_id=model_id, filename=f"G_{version}.pth", cache_dir=cache_dir)
        d_path = hf_hub_download(repo_id=model_id, filename=f"D_{version}.pth", cache_dir=cache_dir)
        dur_path = hf_hub_download(repo_id=model_id, filename=f"DUR_{version}.pth", cache_dir=cache_dir)
        config_path = hf_hub_download(repo_id=model_id, filename="config.json", cache_dir=cache_dir)
    except Exception as e:
        raise FileNotFoundError(f"Model dosyaları indirilemedi: {str(e)}")

    return g_path, d_path, dur_path, config_path


def get_model(model_id, version):
    cache_key = f"{model_id}_{version}"
    if cache_key not in model_cache:
        g_path, d_path, dur_path, config_path = get_model_paths(model_id, version)
        model = MeloInference(
            language="EN",  # Default language
            device="auto",
            use_hf=False,  # Local dosyaları kullanacağız
            config_path=config_path,
            ckpt_path=g_path)
        model_cache[cache_key] = model
    return model_cache[cache_key]


def text_to_speech(text, model_id, version):
    try:
        model = get_model(model_id, version)
        if model is None:
            return None, "Error: Failed to initialize model"

        # Ses dosyasını oluştur
        output_path = "temp_audio.wav"
        model.tts_to_file(
            text=text,
            speaker_id=0,  # Varsayılan speaker ID
            output_path=output_path,
            quiet=True  # Gereksiz çıktıları gizle
        )

        # Dosya yolunu döndür
        return output_path, None
    except Exception as e:
        return None, f"Error: {str(e)}"


def create_interface():
    with gr.Blocks() as demo:
        gr.Markdown("# MeloPlus Text-to-Speech Demo")

        with gr.Row():
            with gr.Column():
                model_id_input = gr.Textbox(
                    label="Model ID",
                    placeholder="Enter model ID (e.g., Vyvo/MeloTTS-Ljspeech)",
                    lines=1,
                    value="Vyvo/MeloTTS-Ljspeech"  # Varsayılan model
                )

                version_input = gr.Textbox(
                    label="Model Version",
                    placeholder="Enter version number (e.g., 152000)",
                    lines=1,
                    value="152000"  # Varsayılan versiyon
                )

                text_input = gr.Textbox(
                    label="Text Input", placeholder="Enter the text you want to convert to speech", lines=3)

                generate_btn = gr.Button("Generate Speech")

        with gr.Row():
            error_output = gr.Textbox(label="Error (if any)")
            audio_output = gr.Audio(label="Generated Speech", type="filepath")

        # Enter tuşu ile çalıştırma
        text_input.submit(
            fn=text_to_speech,
            inputs=[text_input, model_id_input, version_input],
            outputs=[audio_output, error_output])

        # Generate butonu ile çalıştırma
        generate_btn.click(
            fn=text_to_speech,
            inputs=[text_input, model_id_input, version_input],
            outputs=[audio_output, error_output])

        gr.Markdown(
            """
        ### Usage Notes:
        - Enter the HuggingFace model ID (e.g., Vyvo/MeloTTS-Ljspeech)
        - Enter the model version number (e.g., 152000)
        - Type the text you want to convert to speech
        - Click Generate Speech button or press Enter to create the audio
        """)

    return demo


if __name__ == "__main__":
    demo = create_interface()
    demo.launch(debug=True)
