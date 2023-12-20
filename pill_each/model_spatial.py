"""Full definition of a LLaMA Language Model, all of it in this single file.

Based on the nanoGPT implementation: https://github.com/karpathy/nanoGPT.
"""
# mypy: ignore-errors
import sys
from pathlib import Path
import math
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
from torch.nn import functional as F
from typing_extensions import Self

# support running without installing as a package
wd = Path(__file__).parent.parent.resolve()
sys.path.append(str(wd))

from lit_llama.utils import find_multiple

MaskCache = torch.Tensor
RoPECache = torch.Tensor
KVCache = Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]


@dataclass
class LLaMASpatialConfig:
    block_size: int = 2048
    vocab_size: int = 32000
    max_seq_length: int = 256
    padded_vocab_size: Optional[int] = None
    n_layer: int = 32
    n_head: int = 32
    n_embd: int = 4096
    s_embd: int = 4
    s_head: int = 1
    n_pill_s: float = 0.0
    s_pill_n: float = 0.0
    r: float = 0.0
    alpha: float = 1.0
    dropout: float = 0.0
    s_code_layer = 3
    start_poi_idx = None
    max_poi_len = None
    bbox = {"lon_min": 0.0, "lat_min": 0.0, "lon_max": 0.0, "lat_max": 0.0}

    use_repeat = False
    use_rnn = False

    def __post_init__(self):
        if self.padded_vocab_size is None:
            self.padded_vocab_size = find_multiple(self.vocab_size, 64)

    @classmethod
    def from_name(cls, name: str) -> Self:
        return cls(**llama_configs[name])


llama_configs = {
    "7B": dict(n_layer=32, n_head=32, n_embd=4096),
    "13B": dict(n_layer=40, n_head=40, n_embd=5120),
    "30B": dict(n_layer=60, n_head=52, n_embd=6656),
    "65B": dict(n_layer=80, n_head=64, n_embd=8192),
}


class SpatialBlock(nn.Module):
    def __init__(self, config: LLaMASpatialConfig) -> None:
        super().__init__()
        self.rms_1 = SpatialRMSNorm(config.n_embd, config.s_embd)
        self.attn = SpatialCausalSelfAttention(config)
        self.rms_2 = SpatialRMSNorm(config.n_embd, config.s_embd)
        self.mlp = SpatialMLP(config)

    def forward(
            self,
            x: torch.Tensor,
            s: torch.Tensor,
            rope: RoPECache,
            mask: MaskCache,
            max_seq_length: int,
            input_pos: Optional[torch.Tensor] = None,
            kv_cache: Optional[KVCache] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[KVCache]]:
        x, s = self.rms_1(x, s)
        hx, hs, new_kv_cache = self.attn(x, s, rope, mask, max_seq_length, input_pos, kv_cache)
        x = x + hx
        s = s + hs
        mx, ms = self.mlp(*self.rms_2(x, s))
        x, s = x + mx, s + ms
        return x, s, new_kv_cache


# class Num_Coder(nn.Module):  # this version is for extracting an entire patch
#     def __init__(self, config: LLaMASpatialConfig) -> None:
#         super().__init__()
#         self.config = config
#         if config.patch_len is not None:
#             if type(config.patch_len) == int:
#                 self.fixed_len = config.patch_len
#                 self.num_encoder = nn.Linear(2, config.patch_len * config.s_embd)
#                 self.num_decoder = nn.Linear(config.patch_len * config.s_embd, 2)
#             else:
#                 raise Exception("patche_len not support")
#         else:
#             self.fixed_len = None
#             self.num_encoder = nn.GRU(config.s_embd, config.s_embd, config.s_code_layer)
#             self.num_decoder = nn.GRU(config.s_embd, config.s_embd, config.s_code_layer)
#         self.start_poi_idx = config.start_poi_idx
#
#     def encode(self, raw_spatial, spatial_tensor, spatial_scope=None):
#         if self.fixed_len is not None and spatial_scope is None:
#             B, L, _ = raw_spatial.size()
#             spa = self.num_encoder(raw_spatial).view(B, self.fixed_len * L, self.config.s_embd)
#             spatial_tensor[:, self.start_poi_idx:, :] = spa[:, :-1, :]
#         elif spatial_scope is not None:
#             ...
#         else:
#             # TODO: implement can enlarge patching
#             raise Exception("not support GRU")
#         return spatial_tensor
#
#     def decode(self, coord, spatial_scope=None):
#         if self.fixed_len is not None and spatial_scope is None:
#             spa = coord[:, self.start_poi_idx - 1:, :]
#             B, L, _ = spa.size()
#             spa = self.num_decoder(spa.view(B, -1, self.config.s_embd * self.fixed_len))
#         elif spatial_scope is not None:
#             ...
#         else:
#             # TODO: implement can enlarge patching
#             raise Exception("not support GRU")
#         return spa


class SpatialCoder(nn.Module):  # jn: this version is for extracting last output of a patch
    def __init__(self, config: LLaMASpatialConfig) -> None:
        super().__init__()
        self.config = config
        self.start_poi_idx = config.start_poi_idx
        self.max_word_len = config.max_poi_len
        self.s_embd = config.s_embd
        self.use_repeat = config.use_repeat
        self.use_rnn = config.use_rnn

        # init encoder
        if self.use_repeat:
            self.spatial_encoder = nn.Linear(2, self.s_embd)
        else:
            self.spatial_encoder = nn.Linear(2, self.max_word_len * self.s_embd)

        if self.use_rnn:
            self.spatial_rnn = nn.LSTM(self.s_embd, self.s_embd, self.config.s_code_layer, batch_first=True)
        else:
            self.spatial_rnn = None

        # init decoder
        self.spatial_decoder = nn.Linear(self.s_embd, 2)

    def form_spatial_sentence(self, spatial_tensor, poi_mask, spatial_scope):
        B, L, _, _ = spatial_tensor.size()
        word_lens = spatial_scope[:, :, 1] - spatial_scope[:, :, 0]  # calculate POI word length
        index_matrix = torch.arange(self.max_word_len, device=spatial_tensor.device).repeat(B, L, 1)
        mask = index_matrix < word_lens.unsqueeze(-1)  # cutoff mask
        extracted = spatial_tensor[mask]  # cutoff words
        assert not (poi_mask.sum(1) != word_lens.sum(1)).any()  # sentence length equals word length's sum

        spatial_sentence = torch.zeros(*poi_mask.shape, extracted.shape[-1],
                                       device=spatial_tensor.device,
                                       dtype=spatial_tensor.dtype)
        spatial_sentence[poi_mask] = extracted

        return spatial_sentence

    def encode(self, raw_spatial, poi_mask, spatial_scope=None):
        B, L, _ = raw_spatial.shape
        if self.use_repeat:  # repeat method
            spatial_tensor = self.spatial_encoder(raw_spatial).unsqueeze(1).repeat(1, 1, self.max_word_len, 1)
        else:  # generate a max length
            spatial_tensor = self.spatial_encoder(raw_spatial).view(B, L, self.max_word_len, self.s_embd)

        if self.use_rnn:
            spatial_tensor = spatial_tensor.view(-1, self.max_word_len, self.s_embd)
            spatial_tensor, _ = self.spatial_rnn(spatial_tensor)
            spatial_tensor = spatial_tensor.view(B, L, self.max_word_len, self.s_embd)

        spatial_tensor = self.form_spatial_sentence(spatial_tensor, poi_mask, spatial_scope)
        # spatial_tensor = torch.cat([torch.zeros([B, self.start_poi_idx, self.s_embd], device=spatial_tensor.device),
        #                             spatial_tensor], dim=1)
        # need to remove the last node and add instruction paddings
        return spatial_tensor

    def decode(self, spatial_tensor, spatial_scope=None):
        if spatial_scope is not None:  # length not fixed
            index = torch.arange(spatial_tensor.shape[0]).unsqueeze(1)  # batch size index
            last_nodes = spatial_scope[:, :, 1] - 1  # get the last node of words
            last_nodes[last_nodes < 0] = 0  # get rid of -1
            spatial_tensor = spatial_tensor[index, last_nodes]
            coord = self.spatial_decoder(spatial_tensor)
        else:  # length fixed
            spatial_tensor = spatial_tensor[:, self.start_poi_idx - 1:, :]
            spatial_tensor = spatial_tensor[:, ::self.max_word_len, :]  # sample by fixed length
            coord = self.spatial_decoder(spatial_tensor)
        return coord


class LLaMA_spatial(nn.Module):
    def __init__(self, config: LLaMASpatialConfig) -> None:
        super().__init__()
        assert config.padded_vocab_size is not None
        self.config = config

        self.lm_head = nn.Linear(config.n_embd, config.padded_vocab_size, bias=False)
        self.spatial_head = nn.Linear(config.s_embd, config.s_embd, bias=False)
        self.transformer = nn.ModuleDict(
            dict(
                wte=nn.Embedding(config.padded_vocab_size, config.n_embd),
                h=nn.ModuleList(SpatialBlock(config) for _ in range(config.n_layer)),
                ln_f=SpatialRMSNorm(config.n_embd, config.s_embd),
            )
        )

        self.rope_cache: Optional[RoPECache] = None
        self.mask_cache: Optional[MaskCache] = None
        self.kv_caches: List[KVCache] = []
        self.spatial_coder = SpatialCoder(config=config)

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02 / math.sqrt(2 * self.config.n_layer))
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02 / math.sqrt(2 * self.config.n_layer))

    def forward(
            self, idx: torch.Tensor, poi_mask: torch.Tensor, raw_spatial: torch.Tensor, spatial_scope: torch.Tensor,
            max_seq_length: Optional[int] = None, input_pos: Optional[torch.Tensor] = None
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, List[KVCache]]]:
        B, T = idx.size()
        s = None
        block_size = self.config.block_size
        if max_seq_length is None:
            max_seq_length = block_size
        assert T <= max_seq_length, f"Cannot forward sequence of length {T}, max seq length is only {max_seq_length}"
        assert max_seq_length <= block_size, f"Cannot attend to {max_seq_length}, block size is only {block_size}"
        assert T <= block_size, f"Cannot forward sequence of length {T}, block size is only {block_size}"

        if self.rope_cache is None:
            self.rope_cache = self.build_rope_cache(idx)
        if self.mask_cache is None:
            self.mask_cache = self.build_mask_cache(idx)

        spatial_addition = self.spatial_coder.encode(raw_spatial, poi_mask, spatial_scope)
        assert idx.shape == spatial_addition.shape[:2]

        if input_pos is not None:
            rope = self.rope_cache.index_select(0, input_pos)
            mask = self.mask_cache.index_select(2, input_pos)
            mask = mask[:, :, :, :max_seq_length]
        else:
            rope = self.rope_cache[:T]
            mask = self.mask_cache[:, :, :T, :T]

        # forward the model itself
        x = self.transformer.wte(idx)  # token embeddings of shape (b, t, n_embd)

        if spatial_addition is not None:
            # x = torch.cat([x, spatial_addition], dim=-1)
            s = spatial_addition

        if input_pos is None:  # proxy for use_cache=False
            for block in self.transformer.h:
                x, s, _ = block(x, s, rope, mask, max_seq_length)
        else:
            if not self.kv_caches:
                head_size = self.config.n_embd // self.config.n_head
                s_head_size = self.config.s_embd // self.config.s_head
                cache_shape = (B, self.config.n_head, max_seq_length, head_size)
                s_cache_shape = (B, self.config.s_head, max_seq_length, s_head_size)
                self.kv_caches = [
                    (torch.zeros(cache_shape, device=x.device, dtype=x.dtype),
                     torch.zeros(cache_shape, device=x.device, dtype=x.dtype),
                     torch.zeros(s_cache_shape, device=s.device, dtype=s.dtype),
                     torch.zeros(s_cache_shape, device=s.device, dtype=s.dtype))
                    for _ in range(self.config.n_layer)
                ]
            for i, block in enumerate(self.transformer.h):
                x, s, self.kv_caches[i] = block(x, s, rope, mask, max_seq_length, input_pos, self.kv_caches[i])

        x, s = self.transformer.ln_f(x, s)
        if spatial_addition is not None:
            w = x
            # s = x[..., self.config.n_embd:]
            logits = self.lm_head(w)  # (b, t, vocab_size)
            coord = self.spatial_head(s)
            coord = self.spatial_coder.decode(coord, spatial_scope)
            return logits, coord
        else:
            logits = self.lm_head(x)  # (b, t, vocab_size)
            return logits

    @classmethod
    def from_name(cls, name: str) -> Self:
        return cls(LLaMASpatialConfig.from_name(name))

    def build_rope_cache(self, idx: torch.Tensor) -> RoPECache:
        return build_rope_cache(
            seq_len=self.config.block_size,
            n_elem=self.config.n_embd // self.config.n_head,
            dtype=idx.dtype,
            device=idx.device,
        )

    def build_mask_cache(self, idx: torch.Tensor) -> MaskCache:
        ones = torch.ones((self.config.block_size, self.config.block_size), device=idx.device, dtype=torch.bool)
        return torch.tril(ones).unsqueeze(0).unsqueeze(0)

    def reset_cache(self) -> None:
        self.kv_caches.clear()
        if self.mask_cache.device.type == "xla":
            # https://github.com/Lightning-AI/lit-parrot/pull/83#issuecomment-1558150179
            self.rope_cache = None
            self.mask_cache = None


class SpatialCausalSelfAttention(nn.Module):
    def __init__(self, config: LLaMASpatialConfig) -> None:
        super().__init__()
        assert config.n_embd % config.n_head == 0

        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=False)
        self.spatial_linear = nn.Linear(config.s_embd, config.s_embd * 3)
        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=False)
        self.spatial_pill_proj = nn.Linear(config.s_embd * config.n_head,
                                           config.s_embd, bias=False)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.s_embd = config.s_embd
        self.block_size = config.block_size
        self.config = config

    def forward(
            self,
            x: torch.Tensor,
            s: torch.Tensor,
            rope: RoPECache,
            mask: MaskCache,
            max_seq_length: int,
            input_pos: Optional[torch.Tensor] = None,
            kv_cache: Optional[KVCache] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[KVCache]]:
        B, T, C = x.size()  # batch size, sequence length, embedding dimensionality (n_embd)
        C = self.n_embd

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        origin = self.c_attn(x)
        pill = self.spatial_linear(s)
        q_orig, k_orig, v_orig = origin.split(self.n_embd, dim=2)
        q_pill, k_pill, v_pill = pill.split(self.s_embd, dim=2)

        head_size = C // self.n_head
        k = k_orig.view(B, T, self.n_head, head_size)
        q = q_orig.view(B, T, self.n_head, head_size)
        v = v_orig.view(B, T, self.n_head, head_size)

        q = apply_rope(q, rope)
        k = apply_rope(k, rope)

        # q = self.take_pill(q, q_pill)
        # k = self.take_pill(k, k_pill)
        # v = self.take_pill(v, v_pill)
        # q_orig, k_orig = self.take_pill(q.flatten(2), q_pill), self.take_pill(k.flatten(2), k_pill)
        # q_orig, k_orig = q_orig.unsqueeze(2), k_orig.unsqueeze(2)

        q_pill = q_pill.view(B, T, 1, self.s_embd)
        k_pill = k_pill.view(B, T, 1, self.s_embd)
        v_pill = v_pill.view(B, T, 1, self.s_embd)

        k = k.transpose(1, 2)  # (B, nh, T, hs)
        q = q.transpose(1, 2)  # (B, nh, T, hs)
        v = v.transpose(1, 2)  # (B, nh, T, hs)

        q_pill = q_pill.transpose(1, 2)  # (B, nh, T, hs)
        k_pill = k_pill.transpose(1, 2)  # (B, nh, T, hs)
        v_pill = v_pill.transpose(1, 2)  # (B, nh, T, hs)

        if kv_cache is not None:
            cache_k, cache_v, cache_k_s, cache_v_s = kv_cache
            # check if reached token limit
            if input_pos[-1] >= max_seq_length:
                input_pos = torch.tensor(max_seq_length - 1, device=input_pos.device)
                # shift 1 position to the left
                cache_k = torch.roll(cache_k, -1, dims=2)
                cache_v = torch.roll(cache_v, -1, dims=2)
                cache_k_s = torch.roll(cache_k_s, -1, dims=2)
                cache_v_s = torch.roll(cache_v_s, -1, dims=2)
            k = cache_k.index_copy(2, input_pos, k)
            k_pill = cache_k_s.index_copy(2, input_pos, k_pill)
            v = cache_v.index_copy(2, input_pos, v)
            v_pill = cache_v_s.index_copy(2, input_pos, v_pill)
            kv_cache = k, v, k_pill, v_pill

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        #  att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        #  att = att.masked_fill(mask[:,:,:T,:T] == 0, float('-inf'))
        #  att = F.softmax(att, dim=-1)
        #  y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)

        # efficient attention using Flash Attention CUDA kernels
        # y_out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, dropout_p=0.0)
        attn_mask = torch.zeros_like(mask, dtype=q.dtype).masked_fill(~mask, -float(
            'inf')) if mask.dtype == torch.bool else mask
        n_attn_weight = q @ k.transpose(-2, -1) / math.sqrt(q.size(-1))
        s_attn_weight = q_pill @ k_pill.transpose(-2, -1) / math.sqrt(q_pill.size(-1))
        s_attn_mask = attn_mask
        n_attn_mask = attn_mask
        if self.config.n_pill_s is not None:
            s_attn_mask = s_attn_mask + n_attn_weight * self.config.n_pill_s
        if self.config.s_pill_n is not None:
            n_attn_mask = n_attn_mask + s_attn_weight * self.config.n_pill_s
        n_attn_weight = torch.dropout(torch.softmax(n_attn_weight + n_attn_mask, dim=-1), p=0.0, train=True)
        s_attn_weight = torch.dropout(torch.softmax(s_attn_weight + s_attn_mask, dim=-1), p=0.0, train=True)

        y_out = n_attn_weight @ v
        y_pill = s_attn_weight @ v_pill

        y_out = y_out.transpose(1, 2)
        y_pill = y_pill.transpose(1, 2)

        # y = y_out[..., :head_size]
        y = y_out.contiguous().view(B, T, C)  # re-assemble all head outputs side by side
        # y_pill = y_out[..., head_size:]
        y_pill = y_pill.contiguous().view(B, T, -1)

        # output projection
        y = self.c_proj(y)
        y_pill = self.spatial_pill_proj(y_pill)

        return y, y_pill, kv_cache
        # return y, kv_cache


class SpatialMLP(nn.Module):
    def __init__(self, config: LLaMASpatialConfig) -> None:
        super().__init__()
        hidden_dim = 4 * config.n_embd
        n_hidden = int(2 * hidden_dim / 3)
        n_hidden = find_multiple(n_hidden, 256)
        self.n_embd = config.n_embd

        self.c_fc1 = nn.Linear(config.n_embd, n_hidden, bias=False)
        self.c_fc2 = nn.Linear(config.n_embd, n_hidden, bias=False)
        self.c_proj = nn.Linear(n_hidden, config.n_embd, bias=False)

        self.spatial_pill_fc1 = nn.Linear(config.s_embd, config.s_embd * 4, bias=False)
        self.spatial_pill_fc2 = nn.Linear(config.s_embd, config.s_embd * 4, bias=False)
        self.spatial_pill_proj = nn.Linear(config.s_embd * 4, config.s_embd, bias=False)

    def forward(self, x: torch.Tensor, s: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        w = x
        # s = x[..., self.n_embd:]

        w = F.silu(self.c_fc1(w)) * self.c_fc2(w)
        w = self.c_proj(w)
        s = F.silu(self.spatial_pill_fc1(s)) * self.spatial_pill_fc2(s)
        s = self.spatial_pill_proj(s)

        # x = torch.cat([w, s], dim=-1)
        return w, s


class SpatialRMSNorm(nn.Module):
    """Root Mean Square Layer Normalization.

    Derived from https://github.com/bzhangGo/rmsnorm/blob/master/rmsnorm_torch.py. BSD 3-Clause License:
    https://github.com/bzhangGo/rmsnorm/blob/master/LICENSE.
    """

    def __init__(self, size: int, pill_size: int, dim: int = -1, eps: float = 1e-5) -> None:
        super().__init__()
        self.scale = nn.Parameter(torch.ones(size))
        self.spatial_pill_scale = nn.Parameter(torch.ones(pill_size))
        self.eps = eps
        self.dim = dim
        self.size = size

    def forward(self, x: torch.Tensor, s: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # NOTE: the original RMSNorm paper implementation is not equivalent
        # norm_x = x.norm(2, dim=self.dim, keepdim=True)
        # rms_x = norm_x * d_x ** (-1. / 2)
        # x_normed = x / (rms_x + self.eps)
        w = x
        norm_w = torch.mean(w * w, dim=self.dim, keepdim=True)
        w_normed = w * torch.rsqrt(norm_w + self.eps)
        norm_s = torch.mean(s * s, dim=self.dim, keepdim=True)
        s_normed = s * torch.rsqrt(norm_s + self.eps)
        # return torch.cat([self.scale * w_normed, self.pill_scale * s_normed], dim=-1)
        return self.scale * w_normed, self.spatial_pill_scale * s_normed


def build_rope_cache(
        seq_len: int, n_elem: int, dtype: torch.dtype, device: torch.device, base: int = 10000
) -> RoPECache:
    """Enhanced Transformer with Rotary Position Embedding.

    Derived from: https://github.com/labmlai/annotated_deep_learning_paper_implementations/blob/master/labml_nn/
    transformers/rope/__init__.py. MIT License:
    https://github.com/labmlai/annotated_deep_learning_paper_implementations/blob/master/license.
    """
    # $\Theta = {\theta_i = 10000^{\frac{2(i-1)}{d}}, i \in [1, 2, ..., \frac{d}{2}]}$
    theta = 1.0 / (base ** (torch.arange(0, n_elem, 2, dtype=dtype, device=device) / n_elem))

    # Create position indexes `[0, 1, ..., seq_len - 1]`
    seq_idx = torch.arange(seq_len, dtype=dtype, device=device)

    # Calculate the product of position index and $\theta_i$
    idx_theta = torch.outer(seq_idx, theta).float()

    cache = torch.stack([torch.cos(idx_theta), torch.sin(idx_theta)], dim=-1)

    # this is to mimic the behaviour of complex32, else we will get different results
    if dtype in (torch.float16, torch.bfloat16, torch.int8):
        cache = cache.half()
    return cache


def apply_rope(x: torch.Tensor, rope_cache: RoPECache) -> torch.Tensor:
    # truncate to support variable sizes
    T = x.size(1)
    rope_cache = rope_cache[:T]

    # cast because the reference does
    xshaped = x.float().reshape(*x.shape[:-1], -1, 2)
    rope_cache = rope_cache.view(1, xshaped.size(1), 1, xshaped.size(3), 2)
    x_out2 = torch.stack(
        [
            xshaped[..., 0] * rope_cache[..., 0] - xshaped[..., 1] * rope_cache[..., 1],
            xshaped[..., 1] * rope_cache[..., 0] + xshaped[..., 0] * rope_cache[..., 1],
        ],
        -1,
    )

    x_out2 = x_out2.flatten(3)
    return x_out2.type_as(x)
