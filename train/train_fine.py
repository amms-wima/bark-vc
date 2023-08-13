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
ckpt_path = 'models/fine_2.pt'
model_type = "fine"
dataset_path = 'datasets/joe_biden_state_of_union/'
logging_dir = 'logs/'
log_with = 'wandb'
hubert_path = 'data/models/hubert/hubert.pt'
hubert_tokenizer_path = 'data/models/hubert/tokenizer.pth'

output_dir = 'fine_output/'
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

MAX_SEMANTIC_LEN = 256

SEMANTIC_RATE_HZ = 49.9
SEMANTIC_VOCAB_SIZE = 10_000

TEXT_ENCODING_OFFSET = 10_048
SEMANTIC_PAD_TOKEN = 10_000
TEXT_PAD_TOKEN = 129_595
SEMANTIC_INFER_TOKEN = 129_599

MAX_COARSE_LEN = 768

SAMPLE_RATE = 24_000
CHANNELS = 1

COARSE_SEMANTIC_PAD_TOKEN = 12_048
COARSE_INFER_TOKEN = 12_050

CODEBOOK_SIZE = 1024
N_COARSE_CODEBOOKS = 2
N_FINE_CODEBOOKS = 8
COARSE_RATE_HZ = 75

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


def _flatten_codebooks(arr, offset_size=CODEBOOK_SIZE):
    assert len(arr.shape) == 2
    arr = arr.copy()
    if offset_size is not None:
        for n in range(1, arr.shape[0]):
            arr[n, :] += offset_size * n
    flat_arr = arr.ravel("F")
    return flat_arr


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
        self.audiopaths_and_text = load_filepaths_and_text(os.path.join(opt['path'] , opt['mode'] + '.txt'))

    def __getitem__(self, index):
        audiopath_and_text = self.audiopaths_and_text[index]
        audiopath = audiopath_and_text[0]

        tokens = np.load(audiopath.replace('.wav', '.npz').replace('wavs', 'tokens'))
        fine_tokens = tokens['fine']

        return torch.from_numpy(fine_tokens)

    def __len__(self):
        return len(self.audiopaths_and_text)


class TtsCollater():
    def __init__(self):
        pass
    def __call__(self, batch):
        max_len = 1024
        fine_tokens = []

        for fine_tokens_ in batch:
            if fine_tokens_.shape[1] > max_len:
                start_idx = np.random.randint(0, fine_tokens_.shape[1] - max_len + 1)
                fine_tokens_ = fine_tokens_[:, start_idx : start_idx + max_len]

            pad_size = max_len - fine_tokens_.shape[1]
            fine_tokens_ = F.pad(fine_tokens_, (0, pad_size), value=CODEBOOK_SIZE)

            fine_tokens_ = fine_tokens_.T

            fine_tokens.append(fine_tokens_)

        return {'fine_tokens': torch.stack(fine_tokens).contiguous()}
    

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
# # Setup

# %%
model = _load_model(ckpt_path, device, use_small=False, model_type=model_type)

# %%
if scale_lr:
    learning_rate = (
        learning_rate * grad_accum * train_batch_size * accelerator.num_processes
    )

if use_8bit_adam:
    try:
        import bitsandbytes as bnb
    except ImportError:
        raise ImportError(
            "To use 8-bit Adam, please install the bitsandbytes library: `pip install bitsandbytes`."
        )

    optimizer_class = bnb.optim.AdamW8bit
else:
    optimizer_class = torch.optim.AdamW

# %%
quantization_config=BitsAndBytesConfig(
    load_in_4bit=bits == 4,
    load_in_8bit=bits == 8,
    llm_int8_threshold=6.0,
    llm_int8_has_fp16_weight=False,
    bnb_4bit_compute_dtype=compute_dtype,
    bnb_4bit_use_double_quant=double_quant,
    bnb_4bit_quant_type=quant_type # {'fp4', 'nf4'}
)

# if quantization_config.load_in_8bit or quantization_config.load_in_4bit:
#     if quantization_config.load_in_8bit:
#         logger.info("Detected 8-bit loading: activating 8-bit loading for this model")
#     elif quantization_config.load_in_4bit:
#         logger.info("Detected 4-bit loading: activating 4-bit loading for this model")

#     # We keep some modules such as the lm_head in their original dtype for numerical stability reasons
#     if llm_int8_skip_modules is None or len(llm_int8_skip_modules) == 0:
#         modules_to_not_convert = [] # get_keys_to_not_convert(model)
#     else:
#         modules_to_not_convert = llm_int8_skip_modules

#     if not isinstance(modules_to_not_convert, list):
#         modules_to_not_convert = [modules_to_not_convert]

#     modules_to_not_convert.extend(keep_in_fp32_modules)

#     supports_4bit = version.parse(importlib_metadata.version("bitsandbytes")) >= version.parse("0.39.0")

#     if quantization_config.load_in_4bit and not supports_4bit:
#         raise ValueError(
#             "You have a version of `bitsandbytes` that is not compatible with 4bit inference and training"
#             " make sure you have the latest version of `bitsandbytes` installed"
#         )
    
#     if len(modules_to_not_convert) == 0:
#         modules_to_not_convert = None

#     model = replace_with_bnb_linear(
#         model, modules_to_not_convert=modules_to_not_convert, quantization_config=quantization_config
#     )

#     # training in 8-bit is only available in 0.37.0+
#     model._is_kbit_training_enabled = version.parse(
#         importlib_metadata.version("bitsandbytes")
#     ) >= version.parse("0.37.0")

#     model.config.quantization_config = quantization_config

# %%
if bits == 4:
    from accelerate.utils import CustomDtype
    target_dtype = CustomDtype.INT4
elif bits == 8:
    target_dtype = torch.int8

if lora_dim > 0:
    for param in model.parameters():
        if param.ndim == 1:
            # cast the small parameters (e.g. layernorm) to fp32 for stability
            param.data = param.data.to(torch.float32)
            
    class CastOutputToFloat(nn.Sequential):
        def forward(self, x):
            return super().forward(x).to(torch.float32)

    # model.lm_head = CastOutputToFloat(model.lm_head)
    for i, lm_head in enumerate(model.lm_heads):
        model.lm_heads[i] = CastOutputToFloat(lm_head)

    model = convert_linear_layer_to_lora(model, lora_module_name,
                                            lora_dim=lora_dim, lora_scaling=lora_scaling,
                                            lora_dropout=lora_dropout)
    if optimize_lora_params_only:
        model = only_optimize_lora_parameters(model)

# %%
params_to_optimize = (
        param for param in model.parameters() if param.requires_grad
    )

optimizer = optimizer_class(
    params_to_optimize,
    lr=learning_rate,
    betas=(adam_beta1, adam_beta2),
    weight_decay=weight_decay,
    eps=adam_epsilon,
)

# %%
opt_train = {
    'path': dataset_path,
    'mode': 'train',
}

opt_val = {
    'path': dataset_path,
    'mode': 'valid',
}

train_dataset = TtsDataset(opt_train)
validation_dataset = TtsDataset(opt_val)

train_dataloader = torch.utils.data.DataLoader(
    train_dataset,
    batch_size=train_batch_size,
    collate_fn=TtsCollater(),
)

validation_dataloader = torch.utils.data.DataLoader(
    validation_dataset,
    batch_size=eval_batch_size,
    collate_fn=TtsCollater(),
)

criterion = torch.nn.CrossEntropyLoss(ignore_index=COARSE_SEMANTIC_PAD_TOKEN)

# Scheduler and math around the number of training steps.
overrode_max_train_steps = False
num_update_steps_per_epoch = math.ceil(len(train_dataloader) / grad_accum)
if max_train_steps is None:
    max_train_steps = num_train_epochs * num_update_steps_per_epoch
    overrode_max_train_steps = True

lr_scheduler = get_scheduler(
    lr_scheduler_type,
    optimizer=optimizer,
    num_warmup_steps=lr_warmup_steps * grad_accum,
    num_training_steps=max_train_steps * grad_accum,
)

# %%
model, optimizer, train_dataloader, validation_dataloader, lr_scheduler = accelerator.prepare(
    model, optimizer, train_dataloader, validation_dataloader, lr_scheduler
)
accelerator.register_for_checkpointing(lr_scheduler)

weight_dtype = torch.float32
if accelerator.mixed_precision == "fp16":
    weight_dtype = torch.float16
elif accelerator.mixed_precision == "bf16":
    weight_dtype = torch.bfloat16

# %%
# We need to recalculate our total training steps as the size of the training dataloader may have changed.
num_update_steps_per_epoch = math.ceil(len(train_dataloader) / grad_accum)
if overrode_max_train_steps:
    max_train_steps = num_train_epochs * num_update_steps_per_epoch
# Afterwards we recalculate our number of training epochs
num_train_epochs = math.ceil(max_train_steps / num_update_steps_per_epoch)

# We need to initialize the trackers we use, and also store our configuration.
# The trackers initializes automatically on the main process.
if accelerator.is_main_process:
    accelerator.init_trackers("bark_coarse", config={})

total_batch_size = train_batch_size * accelerator.num_processes * grad_accum
logger.info("***** Running training *****")
logger.info(f"  Num examples = {len(train_dataset)}")
logger.info(f"  Num batches each epoch = {len(train_dataloader)}")
logger.info(f"  Num Epochs = {num_train_epochs}")
logger.info(f"  Instantaneous batch size per device = {train_batch_size}")
logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
logger.info(f"  Gradient Accumulation steps = {grad_accum}")
logger.info(f"  Total optimization steps = {max_train_steps}")
global_step = 0
first_epoch = 0

if resume_from_checkpoint:
    if resume_from_checkpoint != "latest":
        path = os.path.basename(resume_from_checkpoint)
    else:
        # Get the most recent checkpoint
        dirs = os.listdir(output_dir)
        dirs = [d for d in dirs if d.startswith("checkpoint")]
        dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
        path = dirs[-1]
    accelerator.print(f"Resuming from checkpoint {path}")
    accelerator.load_state(os.path.join(output_dir, path))
    global_step = int(path.split("-")[1])

    resume_global_step = global_step * grad_accum
    first_epoch = resume_global_step // num_update_steps_per_epoch
    resume_step = resume_global_step % num_update_steps_per_epoch


# %%
if accelerator.is_main_process:
    model.eval()
    validation_loss = 0.0
    num_batches = 0
    num_samples = 0
    with torch.no_grad():
        for val_step, val_batch in enumerate(validation_dataloader):
            # Similar to training, process the validation batch
            fine_targets_7 = val_batch['fine_tokens'][:, :, 6]
            fine_tokens_input_7 = torch.cat([val_batch['fine_tokens'][:, :, :6], torch.zeros_like(val_batch['fine_tokens'][:, :, 6:])], dim=2)
            fine_targets_8 = val_batch['fine_tokens'][:, :, 7]
            fine_tokens_input_8 = torch.cat([val_batch['fine_tokens'][:, :, :7], torch.zeros_like(val_batch['fine_tokens'][:, :, 7:])], dim=2)

            # Forward pass for validation
            logits_7 = model(6, fine_tokens_input_7)
            logits_8 = model(7, fine_tokens_input_8)

            # Calculate the validation loss
            loss_7 = criterion(logits_7.view(-1, model.config.output_vocab_size), fine_targets_7.view(-1))
            loss_8 = criterion(logits_8.view(-1, model.config.output_vocab_size), fine_targets_8.view(-1))

            loss = (loss_7 + loss_8) / 2
            validation_loss += loss.item()
            num_batches += 1
            num_samples += val_batch['fine_tokens'].size(0)

    average_validation_loss = validation_loss / num_batches
    logger.info(f"Validation Loss: {average_validation_loss} over {num_samples} samples and {num_batches} batches.")
    print(f"Validation Loss: {average_validation_loss} over {num_samples} samples and {num_batches} batches.")

# %% [markdown]
# #  Training

# %%
# Only show the progress bar once on each machine.
progress_bar = tqdm(range(global_step, max_train_steps), disable=not accelerator.is_local_main_process)
progress_bar.set_description("Steps")

for epoch in range(first_epoch, num_train_epochs):
    model.train()
    for step, batch in enumerate(train_dataloader):
        # Skip steps until we reach the resumed step
        if resume_from_checkpoint and epoch == first_epoch and step < resume_step:
            if step % grad_accum == 0:
                progress_bar.update(1)
            continue

        with accelerator.accumulate(model):
            fine_targets_7 = batch['fine_tokens'][:, :, 6]
            fine_tokens_input_7 = torch.cat([batch['fine_tokens'][:, :, :6], torch.zeros_like(batch['fine_tokens'][:, :, 6:])], dim=2)
            fine_targets_8 = batch['fine_tokens'][:, :, 7]
            fine_tokens_input_8 = torch.cat([batch['fine_tokens'][:, :, :7], torch.zeros_like(batch['fine_tokens'][:, :, 7:])], dim=2)

            # Forward pass
            logits_7 = model(6, fine_tokens_input_7)
            logits_8 = model(7, fine_tokens_input_8)

            # Calculate the loss
            loss_7 = criterion(logits_7.view(-1, model.config.output_vocab_size), fine_targets_7.view(-1))
            loss_8 = criterion(logits_8.view(-1, model.config.output_vocab_size), fine_targets_8.view(-1))

            loss = (loss_7 + loss_8) / 2

            accelerator.backward(loss)
            if accelerator.sync_gradients:
                params_to_clip = (
                    param for param in model.parameters() if param.requires_grad
                )
                accelerator.clip_grad_norm_(params_to_clip, max_grad_norm)
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()

        # Checks if the accelerator has performed an optimization step behind the scenes
        if accelerator.sync_gradients:
            progress_bar.update(1)
            global_step += 1

            if global_step % checkpointing_steps == 0:
                if accelerator.is_main_process:
                    save_path = os.path.join(output_dir, f"checkpoint-{global_step}")
                    accelerator.save_state(save_path)
                    logger.info(f"Saved state to {save_path}")

        logs = {"loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
        progress_bar.set_postfix(**logs)
        accelerator.log(logs, step=global_step)

        if global_step >= max_train_steps:
            break
    
    accelerator.wait_for_everyone()

if accelerator.is_main_process:
    if lora_dim > 0:
        model = convert_lora_to_linear_layer(model)
    # save model
    accelerator.save(model.state_dict(), os.path.join(output_dir, "pytorch_model.bin"))
    
    config = model.config.__dict__
    # save config
    with open(os.path.join(output_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=2)

accelerator.end_training()

# %% [markdown]
# # Validation

# %%
if accelerator.is_main_process:
    model.eval()
    validation_loss = 0.0
    num_batches = 0
    num_samples = 0
    with torch.no_grad():
        for val_step, val_batch in enumerate(validation_dataloader):
            # Similar to training, process the validation batch
            fine_targets_7 = val_batch['fine_tokens'][:, :, 6]
            fine_tokens_input_7 = torch.cat([val_batch['fine_tokens'][:, :, :6], torch.zeros_like(val_batch['fine_tokens'][:, :, 6:])], dim=2)
            fine_targets_8 = val_batch['fine_tokens'][:, :, 7]
            fine_tokens_input_8 = torch.cat([val_batch['fine_tokens'][:, :, :7], torch.zeros_like(val_batch['fine_tokens'][:, :, 7:])], dim=2)

            # Forward pass for validation
            logits_7 = model(6, fine_tokens_input_7)
            logits_8 = model(7, fine_tokens_input_8)

            # Calculate the validation loss
            loss_7 = criterion(logits_7.view(-1, model.config.output_vocab_size), fine_targets_7.view(-1))
            loss_8 = criterion(logits_8.view(-1, model.config.output_vocab_size), fine_targets_8.view(-1))

            loss = (loss_7 + loss_8) / 2
            validation_loss += loss.item()
            num_batches += 1
            num_samples += val_batch['fine_tokens'].size(0)

    average_validation_loss = validation_loss / num_batches
    logger.info(f"Validation Loss: {average_validation_loss} over {num_samples} samples and {num_batches} batches.")
    print(f"Validation Loss: {average_validation_loss} over {num_samples} samples and {num_batches} batches.")


