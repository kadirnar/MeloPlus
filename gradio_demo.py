import os
import nltk

os.system('python -m unidic download')
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


def get_model_paths(model_id, version, hf_token=None):
    base_dir = os.path.dirname(os.path.abspath(__file__))
    cache_dir = base_dir  # Save cache files in project directory

    try:
        # Download model files from HuggingFace with token if provided
        g_path = hf_hub_download(
            repo_id=model_id, filename=f"G_{version}.pth", cache_dir=cache_dir, token=hf_token)
        d_path = hf_hub_download(
            repo_id=model_id, filename=f"D_{version}.pth", cache_dir=cache_dir, token=hf_token)
        dur_path = hf_hub_download(
            repo_id=model_id, filename=f"DUR_{version}.pth", cache_dir=cache_dir, token=hf_token)
        config_path = hf_hub_download(
            repo_id=model_id, filename="config.json", cache_dir=cache_dir, token=hf_token)
    except Exception as e:
        raise FileNotFoundError(f"Could not download model files: {str(e)}")

    return g_path, d_path, dur_path, config_path


def get_model(model_id, version, hf_token=None):
    cache_key = f"{model_id}_{version}"
    if cache_key not in model_cache:
        g_path, d_path, dur_path, config_path = get_model_paths(model_id, version, hf_token)
        model = MeloInference(
            language="EN",  # Default language
            device="auto",
            use_hf=False,  # Using local files
            config_path=config_path,
            ckpt_path=g_path)
        model_cache[cache_key] = model
    return model_cache[cache_key]


def text_to_speech(text, model_id, version, hf_token=None):
    try:
        model = get_model(model_id, version, hf_token)
        if model is None:
            return None, "Error: Failed to initialize model"

        # Create audio file
        output_path = "temp_audio.wav"
        model.tts_to_file(
            text=text,
            speaker_id=0,  # Default speaker ID
            output_path=output_path,
            quiet=True  # Hide unnecessary output
        )

        # Return file path
        return output_path, None
    except Exception as e:
        return None, f"Error: {str(e)}"


def create_interface():
    # Custom CSS for better styling
    custom_css = """
        :root {
            --bg-dark: #1a1b2e;
            --bg-darker: #141625;
            --accent-primary: #7aa2f7;
            --accent-secondary: #89ddff;
            --text-primary: #c0caf5;
            --text-secondary: #9aa5ce;
            --border-color: #1d2033;
        }

        .gradio-container {
            background: linear-gradient(135deg, var(--bg-dark) 0%, #1a1b35 50%, var(--bg-darker) 100%) !important;
        }

        .container {
            max-width: 1100px !important;
            margin: auto;
            padding: 1rem;
        }

        h1.main-title {
            font-size: 3.5rem !important;
            font-weight: 600 !important;
            text-align: center !important;
            margin: 1rem auto 2rem auto !important;
            padding: 2rem !important;
            background: linear-gradient(90deg, var(--accent-primary), var(--accent-secondary));
            -webkit-background-clip: text !important;
            -webkit-text-fill-color: transparent !important;
            width: 100% !important;
            display: flex !important;
            justify-content: center !important;
            align-items: center !important;
        }

        /* Force center alignment for markdown content */
        h1.main-title > :first-child {
            text-align: center !important;
            width: 100% !important;
            display: flex !important;
            justify-content: center !important;
            align-items: center !important;
        }

        /* Center the emoji with the text */
        h1.main-title span {
            display: inline-flex !important;
            align-items: center !important;
            justify-content: center !important;
            gap: 0.5rem !important;
        }

        .header-subtitle {
            font-size: 1.5rem;
            color: var(--text-secondary);
            font-weight: 400;
            letter-spacing: 0.5px;
        }
        .input-group {
            background: var(--bg-darker);
            padding: 1rem;
            border-radius: 0.5rem;
            border: 1px solid var(--border-color);
            margin-bottom: 0.75rem;
        }
        .output-group {
            background: var(--bg-darker);
            padding: 1rem;
            border-radius: 0.5rem;
            border: 1px solid var(--border-color);
            margin-bottom: 0.75rem;
        }
        .info-box {
            background: var(--bg-darker);
            border: 1px solid var(--border-color);
            padding: 0.75rem;
            border-radius: 0.5rem;
            font-size: 0.9rem;
        }
        .info-box ol {
            color: var(--text-secondary);
            margin-bottom: 0;
        }
        .section-title {
            font-size: 1.1rem;
            font-weight: 600;
            margin-bottom: 0.75rem;
            color: var(--accent-primary);
        }
        .custom-button {
            background: var(--accent-primary) !important;
            border: none !important;
            color: var(--bg-darker) !important;
            padding: 0.5rem 1rem !important;
            border-radius: 0.3rem !important;
            font-weight: 600 !important;
            transition: all 0.2s ease !important;
        }
        .custom-button:hover {
            transform: translateY(-1px) !important;
            filter: brightness(110%) !important;
            box-shadow: 0 4px 12px rgba(122, 162, 247, 0.2) !important;
        }
        .compact-input {
            margin-bottom: 0.5rem !important;
        }
    """

    with gr.Blocks(theme=gr.themes.Soft(
            primary_hue=gr.themes.colors.blue,
            secondary_hue=gr.themes.colors.purple,
            neutral_hue=gr.themes.colors.gray,
    ), css=custom_css) as demo:

        # Header Section
        gr.Markdown("# 🎵 MeloPlus Text-to-Speech Demo", elem_classes=["main-title"])

        # Main Content
        with gr.Row(equal_height=True):
            # Left Column - Inputs
            with gr.Column(scale=1):
                with gr.Group(elem_classes="input-group"):
                    gr.HTML('<div class="section-title">⚙️ Configuration</div>')

                    model_id_input = gr.Textbox(
                        label="Model ID",
                        placeholder="e.g., Vyvo/MeloTTS-Ljspeech",
                        value="Vyvo/MeloTTS-Ljspeech",
                        container=True,
                        elem_classes="compact-input")

                    version_input = gr.Textbox(
                        label="Version",
                        placeholder="e.g., 152000",
                        value="152000",
                        container=True,
                        elem_classes="compact-input")

                    hf_token_input = gr.Textbox(
                        label="🔑 HF Token",
                        placeholder="For private models",
                        type="password",
                        container=True,
                        elem_classes="compact-input")

                    text_input = gr.Textbox(
                        label="Text Input",
                        placeholder="Enter text to convert...",
                        lines=3,
                        container=True,
                        elem_classes="compact-input")

                    generate_btn = gr.Button("🔊 Generate", elem_classes="custom-button")

            # Right Column - Outputs
            with gr.Column(scale=1):
                # Audio Output Section
                with gr.Group(elem_classes="output-group"):
                    gr.HTML('<div class="section-title">🎧 Output</div>')

                    audio_output = gr.Audio(label="", type="filepath", container=True)

                    error_output = gr.Textbox(label="Status", container=True)

                # Guide Section
                with gr.Group(elem_classes="info-box"):
                    gr.HTML(
                        """
                        <div class="section-title">💡 Quick Tips</div>
                        <ol style="margin: 0; padding-left: 1.2rem;">
                            <li>Enter model ID or use default</li>
                            <li>Set version number</li>
                            <li>Add HF token for private models</li>
                            <li>Enter text and click Generate</li>
                        </ol>
                        """)

        # Event Handlers
        text_input.submit(
            fn=text_to_speech,
            inputs=[text_input, model_id_input, version_input, hf_token_input],
            outputs=[audio_output, error_output])

        generate_btn.click(
            fn=text_to_speech,
            inputs=[text_input, model_id_input, version_input, hf_token_input],
            outputs=[audio_output, error_output])

    return demo


if __name__ == "__main__":
    demo = create_interface()
    demo.launch(debug=True)
