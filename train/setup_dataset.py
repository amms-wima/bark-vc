# %% [markdown]
# # Imports

# %%
import torch
import torch.nn as nn
import os
import re
import gc
import json
import math
import hashlib
import numpy as np
import logging
import torchaudio
from tqdm.auto import tqdm
import torch.nn.functional as F
from encodec.utils import convert_audio
from accelerate import Accelerator
from accelerate.utils import set_seed
from transformers import BertTokenizer
from huggingface_hub import hf_hub_download
from packaging import version
from diffusers.optimization import get_scheduler

from utils.bitsandbytes import BitsAndBytesConfig, importlib_metadata, get_keys_to_not_convert, replace_with_bnb_linear, set_module_quantized_tensor_to_device
from utils.lora import convert_linear_layer_to_lora, only_optimize_lora_parameters, convert_lora_to_linear_layer
from bark.model import GPTConfig, GPT
from bark.model_fine import FineGPT, FineGPTConfig

# %% [markdown]
# # Training Args

# %%
train_batch_size = 8
eval_batch_size = 8
grad_accum = 2
ckpt_path = 'models/text_2.pt'
model_type = "text"
dataset_path = 'datasets/joe_biden_state_of_union/'
logging_dir = 'logs/'
log_with = 'wandb'
hubert_path = 'data/models/hubert/hubert.pt'
hubert_tokenizer_path = 'data/models/hubert/tokenizer.pth'

output_dir = 'semantic_output/'
resume_from_checkpoint = None

checkpointing_steps = 1000

mixed_precision = 'bf16'
bits = 16 #4 4 and 8 bit are a work in progress
compute_dtype = torch.bfloat16
double_quant = True
quant_type = 'nf4'

lora_dim = 64
lora_scaling = 1
lora_dropout = 0.1
lora_module_name = 'transformer.h'
optimize_lora_params_only = False

learning_rate = 1e-4
scale_lr = False
use_8bit_adam = False
adam_beta1 = 0.9
adam_beta2 = 0.999
adam_epsilon = 1e-8
weight_decay = 0.01

llm_int8_skip_modules = None
keep_in_fp32_modules = ['lm_head']

lr_scheduler_type = 'linear'
lr_warmup_steps = 60
num_train_epochs = 5
max_train_steps = None
max_grad_norm = 1.0

seed = 741

# %% [markdown]
# # Define Functions

# %%
CONTEXT_WINDOW_SIZE = 1024

MAX_TEXT_LEN = 256

SEMANTIC_RATE_HZ = 49.9
SEMANTIC_VOCAB_SIZE = 10_000

TEXT_ENCODING_OFFSET = 10_048
SEMANTIC_PAD_TOKEN = 10_000
TEXT_PAD_TOKEN = 129_595
SEMANTIC_INFER_TOKEN = 129_599

MAX_SEMANTIC_LEN = 511

SAMPLE_RATE = 24_000
CHANNELS = 1

logger = logging.getLogger(__name__)


USE_SMALL_MODELS = os.environ.get("SERP_USE_SMALL_MODELS", False)

default_cache_dir = os.path.join(os.path.expanduser("~"), ".cache")
CACHE_DIR = os.path.join(os.getenv("XDG_CACHE_HOME", default_cache_dir), "serp", "bark_v0")


def _clear_cuda_cache():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def _md5(fname):
    hash_md5 = hashlib.md5()
    with open(fname, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()


def _download(from_hf_path, file_name, to_local_path):
    to_local_path = to_local_path.replace("\\", "/")
    path = '/'.join(to_local_path.split("/")[:-1])
    os.makedirs(path, exist_ok=True)
    hf_hub_download(repo_id=from_hf_path, filename=file_name, local_dir=path)
    os.replace(os.path.join(path, file_name), to_local_path)


def _tokenize(tokenizer, text):
    return tokenizer.encode(text, add_special_tokens=False)


def _detokenize(tokenizer, enc_text):
    return tokenizer.decode(enc_text)


def _normalize_whitespace(text):
    return re.sub(r"\s+", " ", text).strip()


REMOTE_MODEL_PATHS = {
    "text_small": {
        "repo_id": "suno/bark",
        "file_name": "text.pt",
        "checksum": "b3e42bcbab23b688355cd44128c4cdd3",
    },
    "coarse_small": {
        "repo_id": "suno/bark",
        "file_name": "coarse.pt",
        "checksum": "5fe964825e3b0321f9d5f3857b89194d",
    },
    "fine_small": {
        "repo_id": "suno/bark",
        "file_name": "fine.pt",
        "checksum": "5428d1befe05be2ba32195496e58dc90",
    },
    "text": {
        "repo_id": "suno/bark",
        "file_name": "text_2.pt",
        "checksum": "54afa89d65e318d4f5f80e8e8799026a",
    },
    "coarse": {
        "repo_id": "suno/bark",
        "file_name": "coarse_2.pt",
        "checksum": "8a98094e5e3a255a5c9c0ab7efe8fd28",
    },
    "fine": {
        "repo_id": "suno/bark",
        "file_name": "fine_2.pt",
        "checksum": "59d184ed44e3650774a2f0503a48a97b",
    },
}


def _load_model(ckpt_path, device, use_small=False, model_type="text"):
    if model_type == "text":
        ConfigClass = GPTConfig
        ModelClass = GPT
    elif model_type == "coarse":
        ConfigClass = GPTConfig
        ModelClass = GPT
    elif model_type == "fine":
        ConfigClass = FineGPTConfig
        ModelClass = FineGPT
    else:
        raise NotImplementedError()
    model_key = f"{model_type}_small" if use_small or USE_SMALL_MODELS else model_type
    model_info = REMOTE_MODEL_PATHS[model_key]
    if ckpt_path in [None, '']:
        ckpt_path = os.path.join(CACHE_DIR, model_info["file_name"])
    if not os.path.exists(ckpt_path):
        logger.info(f"{model_type} model not found, downloading into `{CACHE_DIR}`.")
        _download(model_info["repo_id"], model_info["file_name"], ckpt_path)
    checkpoint = torch.load(ckpt_path, map_location=device)
    # this is a hack
    model_args = checkpoint["model_args"]
    if "input_vocab_size" not in model_args:
        model_args["input_vocab_size"] = model_args["vocab_size"]
        model_args["output_vocab_size"] = model_args["vocab_size"]
        del model_args["vocab_size"]
    gptconf = ConfigClass(**checkpoint["model_args"])
    model = ModelClass(gptconf)
    state_dict = checkpoint["model"]
    # fixup checkpoint
    unwanted_prefix = "_orig_mod."
    for k, v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix) :]] = state_dict.pop(k)
    extra_keys = set(state_dict.keys()) - set(model.state_dict().keys())
    extra_keys = set([k for k in extra_keys if not k.endswith(".attn.bias")])
    missing_keys = set(model.state_dict().keys()) - set(state_dict.keys())
    missing_keys = set([k for k in missing_keys if not k.endswith(".attn.bias")])
    if len(extra_keys) != 0:
        raise ValueError(f"extra keys found: {extra_keys}")
    if len(missing_keys) != 0:
        raise ValueError(f"missing keys: {missing_keys}")
    model.load_state_dict(state_dict, strict=False)
    n_params = model.get_num_params()
    val_loss = checkpoint["best_val_loss"].item()
    print(f"Loaded {model_type} model with {n_params} params, val_loss={val_loss:.4f}.")
    del checkpoint, state_dict
    _clear_cuda_cache()
    if model_type == "text":
        tokenizer = BertTokenizer.from_pretrained("bert-base-multilingual-cased")
        return model, tokenizer
    return model


def load_filepaths_and_text(filename, split="|"):
    with open(filename, encoding='utf-8', errors='ignore') as f:
        filepaths_and_text = [line.strip().split(split) for line in f]
        base = os.path.dirname(filename)
        for j in range(len(filepaths_and_text)):
            filepaths_and_text[j][0] = os.path.join(base, filepaths_and_text[j][0])
    return filepaths_and_text

class TtsDataset(torch.utils.data.Dataset):
    def __init__(self, opt):
        self.path = os.path.dirname(opt['path'])
        self.mode = opt['mode']
        self.audiopaths_and_text = load_filepaths_and_text(os.path.join(opt['path'] , opt['mode'] + '_valid.txt'))
        self.tokenizer = opt['tokenizer']

    def __getitem__(self, index):
        audiopath_and_text = self.audiopaths_and_text[index]
        audiopath, text = audiopath_and_text[0], audiopath_and_text[1]

        input_ids = np.array(_tokenize(self.tokenizer, text)) + TEXT_ENCODING_OFFSET
        input_ids = torch.from_numpy(input_ids).long()
        tokens = np.load(audiopath.replace('.wav', '.npz').replace('wavs', 'tokens'))
        semantic_tokens = tokens['semantic']
        semantic_tokens = torch.from_numpy(semantic_tokens).long()

        return input_ids, semantic_tokens

    def __len__(self):
        return len(self.audiopaths_and_text)


class TtsCollater():
    def __init__(self):
        pass
    def __call__(self, batch):
        max_text_len = MAX_TEXT_LEN
        max_semantic_tokens_len = MAX_SEMANTIC_LEN
        texts = []
        semantic_tokens = []

        for b in batch:
            text, semantic_tokens_ = b
            text = F.pad(text, (0, max_text_len-len(text)), value=TEXT_PAD_TOKEN)
            semantic_history = torch.from_numpy(np.array([SEMANTIC_PAD_TOKEN] * 256))
            text = torch.cat([text, semantic_history, torch.tensor([SEMANTIC_INFER_TOKEN])])
            texts.append(text)
            semantic_tokens_ = semantic_tokens_[:max_semantic_tokens_len]
            semantic_tokens.append(F.pad(semantic_tokens_, (0, max_semantic_tokens_len-len(semantic_tokens_)), value=SEMANTIC_PAD_TOKEN))

        return {
            'input_ids': torch.stack(texts).contiguous(),
            'semantic_tokens': torch.stack(semantic_tokens).contiguous()
        }
    

accelerator = Accelerator(
    gradient_accumulation_steps=grad_accum,
    mixed_precision=mixed_precision,
    log_with=log_with,
    logging_dir=logging_dir,
)
device = accelerator.device

os.makedirs(output_dir, exist_ok=True)

set_seed(seed)

# %% [markdown]
# # Setup Dataset (only need to do this once)

# %%
max_duration_sec = 15.12 # the maximum allowed duration in seconds

path = dataset_path

# From https://github.com/gitmylo/bark-voice-cloning-HuBERT-quantizer
from hubert.hubert_manager import HuBERTManager
hubert_manager = HuBERTManager()
from hubert.pre_kmeans_hubert import CustomHubert
from hubert.customtokenizer import CustomTokenizer
hubert_manager.make_sure_hubert_installed()
hubert_manager.make_sure_tokenizer_installed()

# Load the HuBERT model
hubert_model = CustomHubert(checkpoint_path=hubert_path).to(device)
hubert_model.eval()
for param in hubert_model.parameters():
    param.requires_grad = False

# Load the CustomTokenizer model
hubert_tokenizer = CustomTokenizer.load_from_checkpoint(hubert_tokenizer_path).to(device)  # Automatically uses the right layers

from bark.generation import load_codec_model
codec_model = load_codec_model(use_gpu=True)
codec_model.eval()
for param in codec_model.parameters():
    param.requires_grad = False


def get_duration(wav, sr):
    return wav.shape[1] / sr

valid_lines_train = []
# convert wavs to semantic tokens
for wav_path, txt in load_filepaths_and_text(path + 'train.txt'):
    wav, sr = torchaudio.load(wav_path)
    if not get_duration(wav, sr) > max_duration_sec:
        valid_lines_train.append((wav_path, txt))
    wav = convert_audio(wav, sr, SAMPLE_RATE, CHANNELS).to(device)

    semantic_vectors = hubert_model.forward(wav, input_sample_hz=SAMPLE_RATE)
    semantic_tokens = hubert_tokenizer.get_token(semantic_vectors)

    # save semantic tokens
    os.makedirs(os.path.join(path, 'tokens'), exist_ok=True)
    semantic_tokens = semantic_tokens.cpu().numpy()

    # Extract discrete codes from EnCodec
    with torch.no_grad():
        encoded_frames = codec_model.encode(wav.unsqueeze(0))
    codes = torch.cat([encoded[0] for encoded in encoded_frames], dim=-1).squeeze()  # [n_q, T]

    # move codes to cpu
    codes = codes.cpu().numpy()

    # save tokens
    np.savez_compressed(os.path.join(path, 'tokens', os.path.basename(wav_path).replace('.wav', '.npz')), fine=codes, coarse=codes[:2, :], semantic=semantic_tokens)

# rewrite train.txt with valid lines
with open(path + 'train_valid.txt', 'w', encoding='utf-8') as f:
    for wav_path, txt in valid_lines_train:
        wav_path = os.path.relpath(wav_path, dataset_path).replace('\\', '/')
        f.write(f'{wav_path}|{txt}\n')

valid_lines_valid = []
for wav_path, txt in load_filepaths_and_text(path + 'valid.txt'):
    wav, sr = torchaudio.load(wav_path)
    if not get_duration(wav, sr) > max_duration_sec:
        valid_lines_valid.append((wav_path, txt))
    wav = convert_audio(wav, sr, SAMPLE_RATE, CHANNELS).to(device)

    semantic_vectors = hubert_model.forward(wav, input_sample_hz=SAMPLE_RATE)
    semantic_tokens = hubert_tokenizer.get_token(semantic_vectors)

    # save semantic tokens
    os.makedirs(os.path.join(path, 'tokens'), exist_ok=True)
    semantic_tokens = semantic_tokens.cpu().numpy()
    
    # Extract discrete codes from EnCodec
    with torch.no_grad():
        encoded_frames = codec_model.encode(wav.unsqueeze(0))
    codes = torch.cat([encoded[0] for encoded in encoded_frames], dim=-1).squeeze()  # [n_q, T]

    # move codes to cpu
    codes = codes.cpu().numpy()

    # save tokens
    np.savez_compressed(os.path.join(path, 'tokens', os.path.basename(wav_path).replace('.wav', '.npz')), fine=codes, coarse=codes[:2, :], semantic=semantic_tokens)

# rewrite valid.txt with valid lines
with open(path + 'valid_valid.txt', 'w', encoding='utf-8') as f:
    for wav_path, txt in valid_lines_valid:
        wav_path = os.path.relpath(wav_path, dataset_path).replace('\\', '/')
        f.write(f'{wav_path}|{txt}\n')

del hubert_model
del hubert_tokenizer
del codec_model
gc.collect()
torch.cuda.empty_cache()


logger.info(f"setup_dataset completed!")