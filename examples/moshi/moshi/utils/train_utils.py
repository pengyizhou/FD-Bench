import torch
from torch import nn

import ipdb
# from moshi.models.loaders import _lm_kwargs, PAD_ID
PAD_ID=3
EPAD_ID=0
TEXT_CARD=32000
AUDIO_CARD = 2048
def get_loss_function_text(device):
    vocab_size_text = TEXT_CARD  # 32000
    # As padding tokens are predominant for audio batches, we reduce their weight by 50% in the cross-entropy loss.
    padding_token = PAD_ID
    weights = torch.ones(vocab_size_text)
    weights[padding_token] = 0.5
    padding_token = EPAD_ID
    weights[padding_token] = 0.5
    return nn.CrossEntropyLoss(weight=weights.to(device), ignore_index=vocab_size_text)

def get_loss_function_audio(device):
    vocab_size_audio = AUDIO_CARD  # 2048
    weights = torch.ones(vocab_size_audio)
    return nn.CrossEntropyLoss(weight=weights.to(device), ignore_index=vocab_size_audio)

def get_optimizer_transformer(model, lr_llm, lr_dep, lr_base):
    param_groups = [
        {"params": model.transformer.parameters(), "lr": lr_llm},
        {"params": model.depformer.parameters(), "lr": lr_dep},
        {"params": [p for n, p in model.named_parameters() if not (n.startswith("transformer.layers") or n.startswith("depformer.layers"))], "lr": lr_base},
    ]
    return torch.optim.AdamW(param_groups, weight_decay=0.1)
