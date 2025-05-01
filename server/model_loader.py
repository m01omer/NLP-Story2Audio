# model_loader.py
import torch
from parler_tts import ParlerTTSForConditionalGeneration
from transformers import AutoTokenizer

def load_model():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    MODEL_NAME = "parler-tts/parler_tts_mini_v0.1"
    model = ParlerTTSForConditionalGeneration.from_pretrained(MODEL_NAME).to(device)
    prompt_tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    description_tokenizer = AutoTokenizer.from_pretrained(model.config.text_encoder._name_or_path)
    return model, prompt_tokenizer, description_tokenizer

def load_urdu_model():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    MODEL_NAME = "facebook/mms-tts-urd-script_arabic"
    model = ParlerTTSForConditionalGeneration.from_pretrained(MODEL_NAME).to(device)
    prompt_tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    description_tokenizer = AutoTokenizer.from_pretrained(model.config.text_encoder._name_or_path)
    return model, prompt_tokenizer, description_tokenizer