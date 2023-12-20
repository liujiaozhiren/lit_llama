import os
import sys
import time
import warnings
from pathlib import Path
from typing import Optional

import lightning as L
import torch
import random
import ahocorasick

from pill_each.spatial_lora import lora_pill

# support running without installing as a package
wd = Path(__file__).parent.parent.resolve()
sys.path.append(str(wd))

from model_spatial import LLaMA_spatial, LLaMASpatialConfig
from lit_llama import LLaMA
from lit_llama.utils import EmptyInitOnDevice, lazy_load, llama_model_lookup, quantization
from lit_llama.tokenizer import Tokenizer
from scripts.prepare_alpaca import generate_prompt
from prepare import gen_sentence, prepare, POI_Find_Dict

# from datasets import load_dataset

lora_r = 8
lora_alpha = 16
lora_dropout = 0.05
max_seq_length = 256
which_devices = [2]

@torch.no_grad()
def not_spatial_generate(
    model: LLaMA,
    idx: torch.Tensor,
    max_new_tokens: int,
    *,
    max_seq_length: Optional[int] = None,
    temperature: float = 1.0,
    top_k: Optional[int] = None,
    eos_id: Optional[int] = None,
) -> torch.Tensor:
    """Takes a conditioning sequence (prompt) as input and continues to generate as many tokens as requested.

    The implementation of this function is modified from A. Karpathy's nanoGPT.

    Args:
        model: The model to use.
        idx: Tensor of shape (T) with indices of the prompt sequence.
        max_new_tokens: The number of new tokens to generate.
        max_seq_length: The maximum sequence length allowed.
        temperature: Scales the predicted logits by 1 / temperature
        top_k: If specified, only sample among the tokens with the k highest probabilities
        eos_id: If specified, stop generating any more token once the <eos> token is triggered
    """
    # create an empty tensor of the expected final shape and fill in the current tokens
    T = idx.size(0)
    T_new = T + max_new_tokens
    if max_seq_length is None:
        max_seq_length = min(T_new, model.config.block_size)

    device, dtype = idx.device, idx.dtype
    # create an empty tensor of the expected final shape and fill in the current tokens
    empty = torch.empty(T_new, dtype=dtype, device=device)
    empty[:T] = idx
    idx = empty
    input_pos = torch.arange(0, T, device=device)

    if idx.device.type == "xla":
        import torch_xla.core.xla_model as xm

        xm.mark_step()

    # generate max_new_tokens tokens
    for _ in range(max_new_tokens):
        x = idx.index_select(0, input_pos).view(1, -1)

        # forward
        logits = model(x, max_seq_length, input_pos)
        logits = logits[0, -1] / temperature

        # optionally crop the logits to only the top k options
        if top_k is not None:
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits = torch.where(logits < v[[-1]], -float("Inf"), logits)

        probs = torch.nn.functional.softmax(logits, dim=-1)
        idx_next = torch.multinomial(probs, num_samples=1).to(dtype=dtype)

        # advance
        input_pos = input_pos[-1:] + 1

        if idx.device.type == "xla":
            xm.mark_step()

        # concatenate the new generation
        idx = idx.index_copy(0, input_pos, idx_next)

        # if <eos> token is triggered, return the output (stop generation)
        if idx_next == eos_id:
            return idx[:input_pos]  # include the EOS token

    return idx


@torch.no_grad()
def generate(
        model: LLaMA_spatial,
        idx: torch.Tensor,
        poi_mask: torch.Tensor,
        spatial_pill: torch.Tensor,
        spatial_scope: torch.Tensor,
        max_new_tokens: int,
        *,
        max_seq_length: Optional[int] = None,
        temperature: float = 1.05,
        top_k: Optional[int] = None,
        eos_id: Optional[int] = None,
) -> (torch.Tensor, torch.Tensor):
    """Takes a conditioning sequence (prompt) as input and continues to generate as many tokens as requested.

    The implementation of this function is modified from A. Karpathy's nanoGPT.

    Args:
        model: The model to use.
        idx: Tensor of shape (T) with indices of the prompt sequence.
        max_new_tokens: The number of new tokens to generate.
        max_seq_length: The maximum sequence length allowed.
        temperature: Scales the predicted logits by 1 / temperature
        top_k: If specified, only sample among the tokens with the k highest probabilities
        eos_id: If specified, stop generating any more token once the <eos> token is triggered
    """
    # create an empty tensor of the expected final shape and fill in the current tokens
    T = idx.size(0)
    T_new = T + max_new_tokens
    if max_seq_length is None:
        max_seq_length = min(T_new, model.config.block_size)
    max_seq_length = max_seq_length + max_new_tokens

    device = idx.device
    # create an empty tensor of the expected final shape and fill in the current tokens
    empty_1 = torch.empty(T_new, dtype=idx.dtype, device=device)
    empty_2 = torch.zeros(T_new, dtype=poi_mask.dtype, device=device)
    empty_1[:T] = idx
    empty_2[:T] = poi_mask
    idx = empty_1
    poi_mask = empty_2
    input_pos = torch.arange(0, T, device=device)
    # spatial_pill_ = spatial_pill
    # spatial_pill = torch.zeros((T_new, model.config.s_embd), dtype=spatial_pill_.dtype, device=device)
    # spatial_pill[:T] = spatial_pill_
    if idx.device.type == "xla":
        import torch_xla.core.xla_model as xm

        xm.mark_step()

    # generate max_new_tokens tokens
    for _ in range(max_new_tokens):
        x = idx.index_select(0, input_pos).view(1, -1)
        pm = poi_mask.index_select(0, input_pos).view(1, -1)
        # forward
        logits, coord = model(x, pm, spatial_pill.unsqueeze(0),
                              spatial_scope.unsqueeze(0), max_seq_length)
        logits = logits[0, -1] / temperature

        # optionally crop the logits to only the top k options
        if top_k is not None:
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits = torch.where(logits < v[[-1]], -float("Inf"), logits)

        probs = torch.nn.functional.softmax(logits, dim=-1)
        idx_next = torch.multinomial(probs, num_samples=1).to(dtype=idx.dtype)

        # advance
        input_pos = torch.arange(0, input_pos.shape[0]+1, device=device)

        if idx.device.type == "xla":
            xm.mark_step()

        # concatenate the new generation
        idx = idx.index_copy(0, input_pos[-1], idx_next)
        # spatial_pill = spatial_pill.index_copy(0, input_pos, coord[0, -1:, :])

        # if <eos> token is triggered, return the output (stop generation)
        if idx_next == eos_id:
            return idx[:input_pos[-1]]  # include the EOS token

    return idx, spatial_pill


def generate_sentence(instruction, poi_token, poi_list, tokenizer, device):
    places = [poi_list[token][1] for token in poi_token]
    places = " , ".join(places)
    prompt = instruction + places
    encoded = tokenizer.encode(prompt, bos=True, eos=False, device=device)
    return encoded


def generate_main(
        tokenizer_path: str = "../checkpoints/lit-llama/tokenizer.model",
        out_dir: str = "../out/pill/poi/",
        trainset_dir: str = "../data/spatial_dataset/poi_train.pt",
        validset_dir: str = "../data/spatial_dataset/poi_test.pt",
        poi_dir: str = "../data/spatial_dataset/poi_list.pt",
        lora_path: Path = Path("../out/lora/alpaca/lit-llama-lora-finetuned.pth"),
        checkpoint_path: Path = Path("../checkpoints/lit-llama/7B/lit-llama.pth"),
        spatial=True
):
    lora_path = None
    fabric = L.Fabric(accelerator="cuda", devices=which_devices, precision="bf16-true")
    fabric.launch()
    # fabric.seed_everything(1337 + fabric.global_rank)

    if fabric.global_rank == 0:
        os.makedirs(out_dir, exist_ok=True)

    # train_data, val_data = load_datasets(data_dir=data_dir)
    data = torch.load(trainset_dir)
    data.extend(torch.load(validset_dir))
    poi_list = torch.load(poi_dir)
    poi_finder = POI_Find_Dict(poi_list)
    llama_config = LLaMASpatialConfig.from_name("7B")
    tokenizer = Tokenizer(Path(tokenizer_path))
    data, tmp = prepare(data, poi_list, tokenizer,
                        max_seq_length=max_seq_length,
                        stage="generate")
    #valid_data, _ = prepare(valid_data, poi_list, tokenizer, padded_vocab_size=llama_config.padded_vocab_size)
    llama_config.max_poi_len, llama_config.bbox = tmp

    print("Loading model ...", file=sys.stderr)
    t0 = time.time()
    if spatial:
        with fabric.init_module(), lora_pill(r=lora_r, alpha=lora_alpha, dropout=lora_dropout, enabled=True):
            model = LLaMA_spatial(llama_config)
            # strict=False because missing keys due to LoRA weights not contained in checkpoint state
            if checkpoint_path is not None:
                with lazy_load(checkpoint_path) as pretrained_checkpoint:
                    model.load_state_dict(pretrained_checkpoint, strict=False)
                    print("Load from checkpoint:{}.".format(checkpoint_path))
            if lora_path is not None:
                with lazy_load(lora_path) as lora_checkpoint:
                    model.load_state_dict(lora_checkpoint, strict=False)
                    print("Load from lora:{}.".format(lora_path))
    else:
        with lazy_load(checkpoint_path) as checkpoint:
            name = llama_model_lookup(checkpoint)
            with fabric.init_module(empty_init=True):
                model = LLaMA.from_name(name)
            model.load_state_dict(checkpoint)
    print(f"Time to load model: {time.time() - t0:.02f} seconds.", file=sys.stderr)

    device = fabric.device
    model.eval()

    command = ""
    while command != "quit":
        try:
            ix = int(command)
        except:
            ix = random.randint(0, len(data) - 1)
            print("random index: {}".format(ix))

        lang_input = data[ix]["language_inputs"].type(torch.int64).to(device)
        lang_label = data[ix]["language_labels"].type(torch.int64).to(device)
        poi_mask = data[ix]["poi_mask"].type(torch.bool).to(device)
        raw_spatial = data[ix]["raw_spatial"].type(torch.cuda.BFloat16Tensor).to(device)
        spatial_scope = data[ix]["spatial_scopes"].type(torch.int64).to(device)

        sentence = tokenizer.decode(lang_label.view(-1))
        print("correct answer: \n{}".format(sentence))
        if spatial:
            output, coords = generate(
                model,
                idx=lang_input,
                poi_mask=poi_mask,
                spatial_pill=raw_spatial,
                spatial_scope=spatial_scope,
                max_seq_length=256,
                max_new_tokens=12,
            )
        else:
            output = not_spatial_generate(model, lang_input, max_new_tokens=12)
            model.reset_cache()
        output = tokenizer.decode(output.view(-1))
        print("predicted answer: \n{}".format(output))
        command = input()


class WordExtract:
    def __init__(self, keywords):
        self.keywords = keywords  # type(list)  list id equals to keyword id
        self.automat = self.build_automat()

    def build_automat(self):
        auto = ahocorasick.Automaton()
        for index, keyword in enumerate(self.keywords):
            auto.add_word(keyword, (index, keyword))
        auto.make_automaton()
        return auto

    def extract_keywords(self, text):
        keyword_matches = set()
        for end_index, (keyword_index, original_keyword) in self.automat.iter(text):
            start_index = end_index - len(original_keyword) + 1
            keyword_matches.add((keyword_index, start_index, end_index))
        return keyword_matches


def extract_sample():
    sample_text = "这是一个示例文本，其中包含一些关键词，如Python、关键字、文本处理等如Python。"
    sample_keywords = ["关键词", "Python", "文本处理"]

    WE = WordExtract(sample_keywords)
    result = WE.extract_keywords(sample_text)  # extract word
    print("文本中的关键词:", result)


if __name__ == "__main__":
    from jsonargparse import CLI

    torch.set_float32_matmul_precision("high")
    warnings.filterwarnings(
        # Triggered internally at ../aten/src/ATen/EmptyTensor.cpp:31
        "ignore",
        message="ComplexHalf support is experimental and many operators don't support it yet"
    )
    warnings.filterwarnings(
        # Triggered in bitsandbytes/autograd/_functions.py:298
        "ignore",
        message="MatMul8bitLt: inputs will be cast from torch.bfloat16 to float16 during quantization",
    )
    CLI(generate_main)

