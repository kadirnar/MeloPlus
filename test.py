from meloplus import MeloInference

# Speed is adjustable
speed = 1.0

# English
text = "Did you ever hear a folk tale about a giant turtle?"
model = MeloInference(language="EN", device="auto")
speaker_ids = model.hps.data.spk2id

# American accent
output_path = 'en-us.wav'
model.tts_to_file(text, speaker_ids['EN-US'], output_path, speed=speed)
