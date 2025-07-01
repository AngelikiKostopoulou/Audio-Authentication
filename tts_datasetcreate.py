from pathlib import Path
import random
import re
import torch
from TTS.api import TTS
import time

NUM_PHRASES = 1000  # Number of phrases to generate per model
# Define phrase components for more randomness
SUBJECTS = ["I", "You", "We", "They", "He", "She"]

ACTIONS_BY_SUBJECT = {
    "I":    ["am going to", "was", "have been", "might be", "love", "hate", "enjoy", "prefer", "like", "dislike"],
    "You":  ["are going to", "were", "have been", "might be", "love", "hate", "enjoy", "prefer", "like", "dislike"],
    "We":   ["are going to", "were", "have been", "might be", "love", "hate", "enjoy", "prefer", "like", "dislike"],
    "They": ["are going to", "were", "have been", "might be", "love", "hate", "enjoy", "prefer", "like", "dislike"],
    "He":   ["is going to", "was", "has been", "might be", "loves", "hates", "enjoys", "prefers", "likes", "dislikes"],
    "She":  ["is going to", "was", "has been", "might be", "loves", "hates", "enjoys", "prefers", "likes", "dislikes"],
}

VERBS = ["walking", "running", "eating", "sleeping", "working", "studying", "reading", "writing", "talking", "listening"]
PLACES = ["there", "here", "at home", "at school", "in the park", "at the office", "outside", "inside", "in the car", "on the street"]

# List of TTS model names to iterate over
MODELS = [
    "tts_models/en/ljspeech/tacotron2-DDC",
    "tts_models/en/ljspeech/glow-tts",
    # "tts_models/multilingual/multi-dataset/xtts_v2",
    # "tts_models/multilingual/multi-dataset/xtts_v1.1",
    # "tts_models/multilingual/multi-dataset/bark",

    # "tts_models/en/vctk/vits",
    # Add more model names as needed
]

def generate_random_phrase():
    for i in range(NUM_PHRASES):
        subject = random.choice(SUBJECTS)
        action = random.choice(ACTIONS_BY_SUBJECT[subject])
        verb = random.choice(VERBS)
        place = random.choice(PLACES)
        # return f"{subject} {action} {verb} {place}."
        yield f"{subject} {action} {verb} {place}."

# def generate_permutations(n=10):
#     return [generate_random_phrase() for _ in range(n)]

def simplify_model_name(model_name):
    return re.sub(r'[^a-zA-Z0-9]', '_', model_name.split('/')[-1])

def generate_wavs(model_name, output_dir, num_samples=10):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Cuda is {'available' if torch.cuda.is_available() else 'not available'}")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tts = TTS(model_name).to(device=device)
    simple_name = simplify_model_name(model_name)
    # phrases = generate_permutations(num_samples)
    time_0 = time.time()
    for idx, text in enumerate(generate_random_phrase()):
        wav_path = output_dir / f"{simple_name}_{idx+1}.wav"
        tts.tts_to_file(text=text, file_path=str(wav_path))
        print(f"Saved: {wav_path}")
    print(f"Time taken for CUDA: {time.time() - time_0:.2f} seconds")

if __name__ == "__main__":
    output_base = Path("output_wavs")
    num_samples = 10
    for model in MODELS:
        print(model)
        output_dir = output_base / simplify_model_name(model)
        generate_wavs(model, output_dir, num_samples=num_samples)