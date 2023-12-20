import torch
import torch.nn as nn
import torch.nn.functional as F

import math
from typing import Dict, List, Tuple
from contextlib import contextmanager
from dataclasses import dataclass

# from lit_llama.lora import MergedLinear
from lora_with_w import MergedLinear
import model_spatial as llama


@dataclass
class LoRAConfig:
    r: float = 0.0
    alpha: float = 1.0
    dropout: float = 0.0


class SpatialCausalSelfAttention(llama.SpatialCausalSelfAttention):
    lora_config = None

    def __init__(self, config: llama.LLaMASpatialConfig) -> None:
        """Causal self-attention with calculating qkv matrices with a single matrix* and Low Ranking Adaptation for
        parameter-efficient fine-tuning.

        *Instead of creating multiple heads and concatenating the result (in addition to creating separate matrices for
        query, key and value for each head) we can do this in a single pass with a single weight matrix.

        Args:
            config:
                ``"block_size"``: size of the context of the model,
                ``"vocab_size"``: number of unique tokens,
                ``"padded_vocab_size"``: padded size of the vocabulary to the nearest multiple of 64 (leads to a greater performance),
                ``"n_layer"``: number of transformer blocks (self-attention + MLP),
                ``"n_head"``: number of heads in multi-head attention mechanism,
                ``"n_embd"``: size of the embedding: vector representation of each token.
        """
        # Skip the parent class __init__ altogether and replace it to avoid
        # useless allocations
        nn.Module.__init__(self)
        assert config.n_embd % config.n_head == 0

        # key, query, value projections for all heads, but in a batch
        # self.c_attn = MergedLinear(
        #     in_features=config.n_embd,
        #     out_features=3 * config.n_embd,
        #     pill_dim=2,
        #     enable_pill=[True, True, True],
        #     r=self.lora_config.r,
        #     lora_alpha=self.lora_config.alpha,
        #     lora_dropout=self.lora_config.dropout,
        #     enable_lora=[True, False, True],
        #     fan_in_fan_out=False,
        #     merge_weights=True,
        #     bias=False)

        # self.c_attn = MergedLinear(
        #     in_features=config.n_embd,
        #     out_features=3 * config.n_embd,
        #     r=self.lora_config.r,
        #     lora_alpha=self.lora_config.alpha,
        #     lora_dropout=self.lora_config.dropout,
        #     enable_lora=[True, False, True],
        #     fan_in_fan_out = False,
        #     merge_weights=True,
        #     bias=False)
        self.c_attn = MergedLinear(
            in_features=config.n_embd,
            out_features=3 * config.n_embd,
            r=self.lora_config.r,
            lora_alpha=self.lora_config.alpha,
            lora_dropout=self.lora_config.dropout,
            enable_lora=[True, False, True],
            fan_in_fan_out=False,
            merge_weights=True,
            bias=False)
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=False)
        self.spatial_linear = nn.Linear(config.s_embd, config.s_embd * 3)
        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=False)
        self.spatial_pill_proj = nn.Linear(config.s_embd * config.n_head,
                                           config.s_embd, bias=False)
        # regularization
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.s_embd = config.s_embd
        self.block_size = config.block_size
        self.rope_cache = None
        self.config = config


class SpatialMergedLinear(nn.Module):
    def __init__(self, in_features, out_features, pill_dim, r, lora_alpha, lora_dropout, enable_lora, fan_in_fan_out,
                 merge_weights, bias, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.n_linear = MergedLinear(
            in_features=in_features,
            out_features=out_features,
            r=r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            enable_lora=enable_lora,
            fan_in_fan_out=fan_in_fan_out,
            merge_weights=merge_weights,
            bias=bias)
        self.n_linear = nn.Linear(in_features, out_features, bias=False)  # Im Mr Meeseeks!
        self.spatial_linear = nn.Linear(pill_dim, pill_dim * 3)

    def forward(self, x: torch.Tensor, s: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.n_linear(x), self.spatial_linear(s)


def mark_only_lora_and_spatial_as_trainable(model: nn.Module, bias: str = 'none') -> None:
    """Freeze all modules except LoRA's and depending on 'bias' value unfreezes bias weights.

    Args:
        model: model with LoRA layers
        bias:
            ``"none"``: all bias weights will be frozen,
            ``"lora_only"``: only bias weight for LoRA layers will be unfrozen,
            ``"all"``: all bias weights will be unfrozen.

    Raises:
        NotImplementedError: if `bias` not in ["none", "lora_only", "all"]
    """
    # freeze all layers except LoRA's
    for n, p in model.named_parameters():
        if 'lora_' not in n and 'spatial_' not in n:
            p.requires_grad = False

    # depending on the `bias` value unfreeze bias weights
    if bias == 'none':
        return
    # elif bias == 'all':
    #     for n, p in model.named_parameters():
    #         if 'bias' in n:
    #             p.requires_grad = True
    # elif bias == 'lora_only':
    #     for m in model.modules():
    #         if isinstance(m, LoRALayer) and \
    #             hasattr(m, 'bias') and \
    #             m.bias is not None:
    #                 m.bias.requires_grad = True
    else:
        raise NotImplementedError


def lora_spatial_state_dict(model: nn.Module, bias: str = 'none') -> Dict[str, torch.Tensor]:
    """Return state_dict with weights of LoRA's A and B matrices and with biases depending on the `bias` value.

    Args:
        model: model with LoRA layers
        bias:
            ``"none"``: state dict will not store bias weights,
            ``"lora_only"``: state dict will store bias weights only from LoRA layers,
            ``"all"``: state dict will store all bias weights.

    Returns:
        Weights and biases of LoRA layers

    Raises:
        NotImplementedError: if `bias` not in ["none", "lora_only", "all"]
    """
    my_state_dict = model.state_dict()
    if bias == 'none':
        return {k: my_state_dict[k] for k in my_state_dict if 'lora_' in k or 'spatial_' in k}
    # elif bias == 'all':
    #     return {k: my_state_dict[k] for k in my_state_dict if 'lora_' in k or 'bias' in k}
    # elif bias == 'lora_only':
    #     to_return = {}
    #     for k in my_state_dict:
    #         if 'lora_' in k:
    #             to_return[k] = my_state_dict[k]
    #             bias_name = k.split('lora_')[0]+'bias'
    #             if bias_name in my_state_dict:
    #                 to_return[bias_name] = my_state_dict[bias_name]
    #     return to_return
    else:
        raise NotImplementedError


@contextmanager
def lora_pill(r, alpha, dropout, enabled: bool = True, n_pill_s=False, s_pill_n=False):
    """Apply context manager under which you can instantiate the model with LoRA.

    In a nutshell the code inside this function forces to use LoRA variant of causal self-attention
    instead of the original one (without LoRA).

    Args:
        r: rank of the weight update matrices. To make sense of using LoRA the rank should be smaller than the rank of
            the weights of the model.  The rank can be as low as 1: https://arxiv.org/pdf/2106.09685.pdf (section 7.2)
        alpha: alpha is needed for scaling updates as alpha/r
            "This scaling helps to reduce the need to retune hyperparameters when we vary r"
            https://arxiv.org/pdf/2106.09685.pdf (section 4.1)
        dropout: dropout that is applied on the input in the LoRA branch (before multiplying by matrix A)
        enabled: enables/disables LoRA
    """
    if not enabled:
        yield
        return

    # CausalSelfAttention.lora_config = LoRAConfig(r=r, alpha=alpha, dropout=dropout)

    SpatialCausalSelfAttention.lora_config = llama.LLaMASpatialConfig(r=r, alpha=alpha, dropout=dropout,
                                                                      n_pill_s=n_pill_s, s_pill_n=s_pill_n)
    # when entering context manager replace link to causal self-attention class from original
    # to a variant with LoRA
    # causal_self_attention = llama.CausalSelfAttention
    # llama.CausalSelfAttention = CausalSelfAttention

    spatial_causal_self_attention = llama.SpatialCausalSelfAttention
    llama.SpatialCausalSelfAttention = SpatialCausalSelfAttention
    yield
    # when exiting context manager - restore link to original causal self-attention class
    llama.SpatialCausalSelfAttention = spatial_causal_self_attention

    SpatialCausalSelfAttention.lora_config = None
