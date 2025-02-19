from meloplus import MeloInference
import nltk

nltk.download('averaged_perceptron_tagger_eng')

# Speed is adjustable
speed = 1.0

# English
text = "Did you ever hear a folk tale about a giant turtle?"
ckpt_path = "G_100000.pth"
model = MeloInference(language="EN", device="auto", ckpt_path=ckpt_path)
speaker_ids = model.hps.data.spk2id

# American accent
output_path = 'en-us.wav'
model.tts_to_file(text, speaker_ids['EN-US'], output_path, speed=speed)
