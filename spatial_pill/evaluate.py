# This mimics GPTQ's evaluation metrics: https://github.com/IST-DASLab/gptq/
# Thanks to E. Frantar et al GPTQ: Accurate Post-training Compression for GPT, arXiv:2210.17323
import random

import math
import sys
import time
from pathlib import Path
from typing import Optional

import lightning as L
import torch
from tqdm import tqdm
from collections import OrderedDict

# support running without installing as a package
wd = Path(__file__).parent.parent.resolve()
sys.path.append(str(wd))

from model_extra import LLaMA_extra, LLaMAExtraConfig
from lora_pill import lora_pill
from generate import generate
from lit_llama.utils import EmptyInitOnDevice, lazy_load, llama_model_lookup
from lit_llama.tokenizer import Tokenizer
from scripts.prepare_alpaca import generate_prompt

from datasets import load_dataset

instruction_tuning = True
lora_r = 8
lora_alpha = 16
lora_dropout = 0.05


def main(
        dataset_dir: str = "../data/spatial_dataset/poi_test.pt",
        *,
        # compilation fails as it does not support torch.complex64 for RoPE
        # compile: bool = False,
        accelerator: str = "auto",
        lora_path: Path = Path("../out/pill/poi/llama-7b-lora-pill-finetuned.pth"),
        checkpoint_path: Path = Path("../checkpoints/lit-llama/7B/lit-llama.pth"),
        tokenizer_path: Path = Path("../checkpoints/lit-llama/tokenizer.model"),
        dtype: str = "float32",
        quantize: Optional[str] = None,
) -> None:
    """Generates text samples based on a pre-trained LLaMA_extra model and tokenizer
       finetuned with LoRA.

    Args:
        datasets: The datasets to use as a comma separated string
        # compile: Whether to compile the model.
        accelerator: The hardware to run on. Possible choices are:
            ``"cpu"``, ``"cuda"``, ``"mps"``, ``"gpu"``, ``"tpu"``, ``"auto"``.
        lora_path: Path to the checkpoint with trained LoRA weights, which are the output of
            `finetune_lora.py`.
        checkpoint_path: The checkpoint path to load.
        tokenizer_path: The tokenizer path to load.
        dtype: The tensor dtype for choosing the floating-point precision
        quantize: Whether to quantize the model and using which method:
            ``"llm.int8"``: LLM.int8() mode,
            ``"gptq.int4"``: GPTQ 4-bit mode.
    """
    assert lora_path.is_file()
    assert checkpoint_path.is_file()
    assert tokenizer_path.is_file()

    if quantize is not None:
        raise NotImplementedError("Quantization in LoRA is not supported yet")

    fabric = L.Fabric(accelerator=accelerator, devices=1)

    dt = getattr(torch, dtype, None)
    if not isinstance(dt, torch.dtype):
        raise ValueError(f"{dtype} is not a valid dtype.")
    dtype = dt

    print("Loading model ...", file=sys.stderr)
    t0 = time.time()

    with lazy_load(checkpoint_path) as pretrained_checkpoint, lazy_load(lora_path) as lora_checkpoint:
        name = llama_model_lookup(pretrained_checkpoint)

        with EmptyInitOnDevice(
                device=fabric.device, dtype=dtype, quantization_mode=quantize
        ), lora_pill(r=lora_r, alpha=lora_alpha, dropout=lora_dropout, enabled=True):
            model = LLaMA_extra.from_name(name)

            # 1. Load the pretrained weights
            model.load_state_dict(pretrained_checkpoint, strict=False)
            # 2. Load the fine-tuned lora weights
            model.load_state_dict(lora_checkpoint, strict=False)

    print(f"Time to load model: {time.time() - t0:.02f} seconds.", file=sys.stderr)

    model.eval()

    # if compile:
    #     model = torch.compile(model)

    total_toks = 0
    model = fabric.setup_module(model)

    tokenizer = Tokenizer(tokenizer_path)

    test_data = torch.load(dataset_dir)

    command = input()
    while command != "quit":
        ix = random.randint(0, len(test_data)-1)
        data = test_data[ix]
        sentence_token = data["sentence_token"]
        spatial_addition = data["spatial_addition"]

        encoded_text = tokenizer.encode(
            sentence, bos=True, eos=False, device=fabric.device
        )
        encoded_text = encoded_text[
                       None, : 256 * model.config.block_size
                       ]  # add batch dimension, trim like gptq implementation
        t0 = time.perf_counter()

        nlls = 0
        toks = 0
        with torch.inference_mode():
            block_size = 2048  # this is for compat with gptq, and indeed we get much worse beyond this (https://github.com/facebookresearch/llama/blob/57b0eb62de0636e75af471e49e2f1862d908d9d8/llama/model.py#L30)
            for i in tqdm(range(0, encoded_text.shape[1], block_size)):
                inp = encoded_text[:, i: i + block_size]
                logits = model(inp)[0]
                nll = torch.nn.functional.cross_entropy(
                    logits[:-1], inp[0, 1:].to(dtype=torch.long), reduction="sum"
                )
                toks += inp.size(1) - 1
                nlls += nll.item()

        print(encoded_text.shape, logits.shape)
        ppl = math.exp(nlls / toks)
        print(f"Perplexity on {dsname}: {ppl:.2f}")
        total_toks += toks

        t = time.perf_counter() - t0
        print(
            f"\n\nTime for inference: {t:.02f} sec total, {total_toks / t:.02f} tokens/sec",
            file=sys.stderr,
        )
        print(
            f"Memory used: {torch.cuda.max_memory_reserved() / 1e9:.02f} GB",
            file=sys.stderr,
        )
        command = input()


def evaluation(
        model: torch.nn.Module,
        test_data: list,
) -> None:
    """The training loop.

    Loosely based on the nanoGPT implementation: https://github.com/karpathy/nanoGPT.
    """
    step_count = 0
    pbar = tqdm(total=len(test_data), ncols=100)
    for data_num in range(len(test_data)):

        t0 = time.time()

        input_ids, targets, spatial_addition = fake_batch(test_data[data_num])

        logits, space = model(input_ids, spatial_addition=spatial_addition)
        place_acc, space_acc =

        dt = time.time() - t0
        postfix = OrderedDict([
            ('loss', f'{loss.item():.4f}'),
            ('time', f'{dt * 1000:.2f}ms'),
        ])
        pbar.set_postfix(postfix)
        pbar.update(1)
        # if iter_num % log_interval == 0:
        #     fabric.print(f"iter {iter_num}: loss {loss.item():.4f}, time: {dt * 1000:.2f}ms")
    pbar.close()


def fake_batch(batch_data):
    input_ids = [batch_data["sentence_token"]]
    targets = [batch_data["sentence_token"]]
    spatial_addition = [batch_data["spatial_addition"]]
    return torch.stack(input_ids), torch.stack(targets), torch.stack(spatial_addition)


def cal_acc(logits, space, targets, spatial_target):
    return


if __name__ == "__main__":
    from jsonargparse import CLI

    torch.set_float32_matmul_precision("high")
    CLI(main)
