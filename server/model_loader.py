# model_loader.py
import torch
from transformers import SpeechT5ForTextToSpeech, SpeechT5Tokenizer, SpeechT5Processor, SpeechT5HifiGan

def load_model(model_name="microsoft/speecht5_tts"):
    """
    Load the SpeechT5 model and related components.
    
    Args:
        model_name (str): Name of the model to load
        
    Returns:
        tuple: (model, tokenizer, processor, vocoder, device)
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Load all components with explicit float32 dtype to avoid BFloat16 issues
    tts_model = SpeechT5ForTextToSpeech.from_pretrained(
        model_name, 
        torch_dtype=torch.float32
    ).to(device)
    
    tokenizer = SpeechT5Tokenizer.from_pretrained(model_name)
    processor = SpeechT5Processor.from_pretrained(model_name)
    
    vocoder = SpeechT5HifiGan.from_pretrained(
        "microsoft/speecht5_hifigan", 
        torch_dtype=torch.float32
    ).to(device)
    
    return tts_model, tokenizer, processor, vocoder, device