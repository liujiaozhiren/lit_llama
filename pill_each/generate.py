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

from model_spatial import LLaMA_spatial

from lit_llama.utils import EmptyInitOnDevice, lazy_load, llama_model_lookup
from lit_llama.tokenizer import Tokenizer
from scripts.prepare_alpaca import generate_prompt

# from datasets import load_dataset

lora_r = 8
lora_alpha = 16
lora_dropout = 0.05


@torch.no_grad()
def generate(
        model: LLaMA_spatial,
        idx: torch.Tensor,
        spatial_pill: torch.Tensor,
        max_new_tokens: int,
        *,
        max_seq_length: Optional[int] = None,
        temperature: float = 1.0,
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

    device, dtype = idx.device, idx.dtype
    # create an empty tensor of the expected final shape and fill in the current tokens
    empty = torch.empty(T_new, dtype=dtype, device=device)
    empty[:T] = idx
    idx = empty
    input_pos = torch.arange(0, T, device=device)
    spatial_pill_ = spatial_pill
    spatial_pill = torch.zeros((T_new, model.config.s_embd), dtype=spatial_pill_.dtype, device=device)
    spatial_pill[:T] = spatial_pill_
    if idx.device.type == "xla":
        import torch_xla.core.xla_model as xm

        xm.mark_step()

    # generate max_new_tokens tokens
    for _ in range(max_new_tokens):
        x = idx.index_select(0, input_pos).view(1, -1)
        spad = spatial_pill.index_select(0, input_pos).view(1, -1, 2)

        # forward
        logits, space = model(x, spad, max_seq_length, input_pos)
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
        spatial_pill = spatial_pill.index_copy(0, input_pos, space[0, -1:, :])

        # if <eos> token is triggered, return the output (stop generation)
        if idx_next == eos_id:
            return idx[:input_pos]  # include the EOS token

    return idx, spatial_pill


def generate_sentence(instruction, poi_token, poi_list, tokenizer, device):
    places = [poi_list[token][1] for token in poi_token]
    places = " , ".join(places)
    prompt = instruction + places
    encoded = tokenizer.encode(prompt, bos=True, eos=False, device=device)
    return encoded


def generate_response(model, prompt_sentence, spatial_addition, tokenizer):
    output, coords = generate(
        model,
        idx=prompt_sentence,
        spatial_pill=spatial_addition,
        max_seq_length=256,
        max_new_tokens=50,
    )
    output = tokenizer.decode(output)
    return output, coords  # output.split("### Response:")[1].strip()


def generate_main(
        dataset_dir: str = "../data/spatial_dataset/poi_test.pt",
        poi_dir: str = "../data/spatial_dataset/poi_list.pt",
        *,
        # compilation fails as it does not support torch.complex64 for RoPE
        # compile: bool = False,
        accelerator: str = "auto",
        lora_path: Path = Path("../out/pill/poi/llama-7b-lora-pill-finetuned.pth"),
        checkpoint_path: Path = Path("../checkpoints/lit-llama/7B/lit-llama.pth"),
        tokenizer_path: Path = Path("../checkpoints/lit-llama/tokenizer.model"),
        dtype: torch.float32 = torch.float32,
        quantize: Optional[str] = None,
) -> None:
    assert lora_path.is_file()
    assert checkpoint_path.is_file()
    assert tokenizer_path.is_file()

    print("Loading model ...", file=sys.stderr)
    t0 = time.time()
    fabric = L.Fabric(accelerator=accelerator, devices=1)
    with lazy_load(checkpoint_path) as pretrained_checkpoint, lazy_load(lora_path) as lora_checkpoint:
        name = llama_model_lookup(pretrained_checkpoint)

        with EmptyInitOnDevice(
                device=fabric.device, dtype=dtype, quantization_mode=quantize
        ), lora_pill(r=lora_r, alpha=lora_alpha, dropout=lora_dropout, enabled=True):
            model = LLaMA_spatial.from_name(name)
            # 1. Load the pretrained weights
            model.load_state_dict(pretrained_checkpoint, strict=False)
            # 2. Load the fine-tuned lora weights
            model.load_state_dict(lora_checkpoint, strict=False)

    device = fabric.device
    tokenizer = Tokenizer(tokenizer_path)
    model.eval()
    print(f"Time to load model: {time.time() - t0:.02f} seconds.", file=sys.stderr)

    test_data = torch.load(dataset_dir)
    poi_list = torch.load(poi_dir)
    command = ""
    while command != "quit":
        try:
            ix = int(command)
        except:
            ix = random.randint(0, len(test_data) - 1)
            print("random index: {}".format(ix))
        data = test_data[ix]
        sentence = data["sentence"]
        print("correct answer: \n{}".format(sentence))
        spatial_addition = data["spatial_addition"]
        poi_token = data["poi_token"]
        instruction = "Given a user's visited places sequence as follows, " \
                      "predict which place the user will visit next: "

        prompt_sentence = generate_sentence(instruction, poi_token[:len(poi_token) // 2],
                                            poi_list, tokenizer, device)
        spatial_addition = torch.stack(spatial_addition[:len(prompt_sentence)]).to(device)
        output, coords = generate_response(model, prompt_sentence, spatial_addition, tokenizer)
        print("predicted answer:".format(output))
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

