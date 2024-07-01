import os
import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline, logging, AutoTokenizer


WHISPER_MODEL_PATH = 'N:\\models\\voice\\model\\'
WHISPER_TOKENIZER_PATH = 'N:\\models\\voice\\tokenizer\\'

def get_whisper_model(model_id='distil-whisper/distil-small.en'):
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    model_name = model_id.split('/')[1]
    if WHISPER_MODEL_PATH:
        model_cache_path = WHISPER_MODEL_PATH+model_name
    if WHISPER_TOKENIZER_PATH:
        tokenizer_cache_path = WHISPER_TOKENIZER_PATH+model_name

    model = AutoModelForSpeechSeq2Seq.from_pretrained(
    model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True, cache_dir = model_cache_path) if WHISPER_MODEL_PATH else \
    AutoModelForSpeechSeq2Seq.from_pretrained(
    model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True)
    model.to(device)

    processor = AutoProcessor.from_pretrained(model_id, cache_dir=tokenizer_cache_path) if WHISPER_TOKENIZER_PATH else \
    AutoProcessor.from_pretrained(model_id)

        
    whisper = pipeline(
            "automatic-speech-recognition",model=model,tokenizer=processor.tokenizer,
            feature_extractor=processor.feature_extractor,max_new_tokens=128,
            torch_dtype=torch_dtype,device=device)
    if not os.path.exists(tokenizer_cache_path):
        processor.save_pretrained(tokenizer_cache_path)
    if not os.path.exists(model_cache_path):
        model.save_pretrained(model_cache_path)
    return whisper