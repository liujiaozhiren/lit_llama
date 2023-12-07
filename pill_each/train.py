"""
Instruction-tuning with LoRA on the Alpaca dataset.

Note: If you run into a CUDA error "Expected is_sm80 to be true, but got false", uncomment the line
`torch.backends.cuda.enable_flash_sdp(False)` in the script below (see https://github.com/Lightning-AI/lit-llama/issues/101).
"""
import sys, math
from pathlib import Path
import os
import time
from typing import List

import lightning as L
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from lit_llama.utils import lazy_load
# from pill_each.model_spatial import LLaMA_spatial
from model_spatial import LLaMA_spatial, LLaMASpatialConfig
from pill_each.prepare import prepare, POI_Find_Dict
from pill_each.spatial_lora import lora_pill, mark_only_lora_and_spatial_as_trainable, lora_spatial_state_dict

# support running without installing as a package
wd = Path(__file__).parent.parent.resolve()
sys.path.append(str(wd))

from generate import generate
from lit_llama.tokenizer import Tokenizer
from scripts.prepare_alpaca import generate_prompt

instruction_tuning = True
eval_interval = 100
save_interval = 100
eval_iters = 10
log_interval = 1

# Hyperparameters
learning_rate = 3e-3  # 3e-4
batch_size = 128
micro_batch_size = 6
gradient_accumulation_iters = batch_size // micro_batch_size
assert gradient_accumulation_iters > 0
max_iters = 50000 * 3 // micro_batch_size
weight_decay = 0.0
max_seq_length = 256  # see scripts/prepare_alpaca.py
lora_r = 8
lora_alpha = 16
lora_dropout = 0.05
warmup_iters = 100


def main(
        # data_dir: str = "../data/alpaca",
        pretrained_path: str = "../checkpoints/lit-llama/7B/lit-llama.pth",
        tokenizer_path: str = "../checkpoints/lit-llama/tokenizer.model",
        out_dir: str = "../out/lora/alpaca",
        trainset_dir: str = "../data/spatial_dataset/poi_train.pt",
        validset_dir: str = "../data/spatial_dataset/poi_test.pt",
        poi_dir: str = "../data/spatial_dataset/poi_list.pt",
        accelerator: str = "auto",
        lora_path: Path = Path("../out/pill/poi/llama-7b-lora-pill-finetuned.pth"),
        checkpoint_path: Path = Path("../checkpoints/lit-llama/7B/lit-llama.pth"),
        # dtype: torch.float32 = torch.float32,
        # quantize: Optional[str] = None,
):
    fabric = L.Fabric(accelerator="cuda", devices=1, precision="bf16-true")
    fabric.launch()
    fabric.seed_everything(1337 + fabric.global_rank)

    if fabric.global_rank == 0:
        os.makedirs(out_dir, exist_ok=True)

    # train_data, val_data = load_datasets(data_dir=data_dir)
    data = torch.load(trainset_dir)
    data.extend(torch.load(validset_dir))
    poi_list = torch.load(poi_dir)
    poi_finder = POI_Find_Dict(poi_list)
    # config = LLaMAConfig.from_name("7B")
    # config.block_size = max_seq_length
    # pretrained_checkpoint = lazy_load(checkpoint_path)
    # lora_checkpoint = lazy_load(lora_path)
    llama_config = LLaMASpatialConfig.from_name("7B")
    tokenizer = Tokenizer(Path(tokenizer_path))
    data, tmp = prepare(data, poi_list, tokenizer, padded_vocab_size=llama_config.padded_vocab_size)
    #valid_data, _ = prepare(valid_data, poi_list, tokenizer, padded_vocab_size=llama_config.padded_vocab_size)
    train_data, valid_data = data[:int(len(data)*0.8)], data[int(len(data)*0.8):]
    llama_config.start_poi_idx, llama_config.patch_len = tmp
    with lazy_load(checkpoint_path) as pretrained_checkpoint, lazy_load(lora_path) as lora_checkpoint:
        with fabric.init_module(), lora_pill(r=lora_r, alpha=lora_alpha, dropout=lora_dropout, enabled=True):
            model = LLaMA_spatial(llama_config)
            # strict=False because missing keys due to LoRA weights not contained in checkpoint state
            model.load_state_dict(pretrained_checkpoint, strict=False)
            model.load_state_dict(lora_checkpoint, strict=False)
    mark_only_lora_and_spatial_as_trainable(model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    model, optimizer = fabric.setup(model, optimizer)
    train(fabric, model, optimizer, train_data, valid_data, tokenizer, out_dir, poi_finder)

    # Save the final LoRA checkpoint at the end of training
    checkpoint = lora_spatial_state_dict(model)
    fabric.save(os.path.join(out_dir, "lit-llama-lora-finetuned.pth"), checkpoint)


def train(
        fabric: L.Fabric,
        model: LLaMA_spatial,
        optimizer: torch.optim.Optimizer,
        train_data: List,
        valid_data: List,
        tokenizer: Tokenizer,
        out_dir: str,
        poi_finder: POI_Find_Dict,
) -> None:
    """The training loop.

    Loosely based on the nanoGPT implementation: https://github.com/karpathy/nanoGPT.
    """
    step_count = 0
    with tqdm(range(max_iters), f"Initial Training...", mininterval=2) as tq:
        for iter_num in tq:
            if step_count <= warmup_iters:
                # linear warmup
                lr = learning_rate * step_count / warmup_iters
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr
            t0 = time.time()

            lang_input, lang_label, spa_input, spa_label, spa_mask, trainable_mask,\
                raw_spatial, poi_idx = get_batch(fabric, train_data)
            with fabric.no_backward_sync(model, enabled=((iter_num + 1) % gradient_accumulation_iters != 0)):
                logits, coord = model(lang_input, spa_input, raw_spatial, poi_idx, 256)
                loss_lang, loss_spa = merged_loss_fn(logits, coord, lang_label, raw_spatial, spa_mask, trainable_mask)
                loss = loss_lang + loss_spa
                fabric.backward(loss / gradient_accumulation_iters)
                optimizer.step()
                optimizer.zero_grad()
                # val_loss, hr10, hr50 = validate(fabric, model, valid_data, tokenizer, poi_finder)
            if (iter_num + 1) % gradient_accumulation_iters == 0:
                optimizer.step()
                optimizer.zero_grad()

                step_count += 1

                if step_count % eval_interval == 0:
                    val_loss, hr10, hr50 = validate(fabric, model, valid_data, tokenizer, poi_finder)
                    fabric.print(f"step {iter_num}: val loss {val_loss:.4f}, hr10 {hr10}, hr50 {hr50}")
                    fabric.barrier()

                if step_count % save_interval == 0:
                    print(f"Saving LoRA weights to {out_dir}")
                    # We are only saving the LoRA weights
                    # TODO: Provide a function/script to merge the LoRA weights with pretrained weights
                    checkpoint = lora_spatial_state_dict(model)
                    fabric.save(os.path.join(out_dir, f"iter-{iter_num:06d}-ckpt.pth"), checkpoint)

            dt = time.time() - t0
            if iter_num % log_interval == 0:
                tq.set_description(
                    f"loss {loss.item():.4f}:({loss_lang},{loss_spa}), time: {dt * 1000:.2f}ms")


def generate_response(model, instruction, tokenizer):
    sample = {"instruction": instruction, "input": ""}
    prompt = instruction
    if instruction_tuning:
        prompt = generate_prompt(sample)
    encoded = tokenizer.encode(prompt, bos=True, eos=False, device=model.device)

    output = generate(
        model,
        idx=encoded,
        max_seq_length=max_seq_length,
        max_new_tokens=100,
    )
    output = tokenizer.decode(output)
    return output  # output.split("### Response:")[1].strip()


@torch.no_grad()
def validate(fabric: L.Fabric, model: torch.nn.Module, val_data: List, tokenizer: Tokenizer,
             poi_finder: POI_Find_Dict) -> torch.Tensor:
    fabric.print("Validating ...")
    model.eval()
    infer_batch_size = micro_batch_size
    eval_iters = math.ceil(len(val_data) / infer_batch_size)
    losses = torch.zeros(eval_iters)
    total_base, hr_10, hr_50 = 0, 0, 0
    with tqdm(range(eval_iters), "in validating...", mininterval=2) as tq:
        for k in tq:
            iter = range(k * infer_batch_size, min((k + 1) * infer_batch_size, len(val_data)))
            lang_input, lang_label, spa_input, spa_label, spa_mask, \
                trainable_mask, valid_data, raw_spatial,poi_idx = get_batch(fabric, val_data,train=False,iter=iter)
            logits, coord = model(lang_input, spa_input, raw_spatial,poi_idx, 256)
            hr10, hr50, total = valid_accuracy(valid_data, logits, coord, tokenizer, poi_finder)
            total_base += total
            hr_10 += hr10
            hr_50 += hr50
            loss_lang, loss_spa = merged_loss_fn(logits, coord, lang_label, spa_label, spa_mask, trainable_mask)
            loss = loss_lang + loss_spa
            losses[k] = loss.item()
    out = losses.mean()

    # produce an example:
    # instruction = "Recommend a movie for me to watch during the weekend and explain the reason."
    #
    # output = generate_response(model, instruction, tokenizer)
    # fabric.print(instruction)
    # fabric.print(output)
    #
    # model.train()
    return out.item(), hr_10 / total_base, hr_50 / total_base


def lon_lat_decode(input: torch.Tensor):
    return input[:, 0].mean(), input[:, 1].mean()


def valid_accuracy(val_data, logits, coords, tokenizer: Tokenizer, poi_finder: POI_Find_Dict, temperature=1.0,
                   top_k=1):
    infer_pois = val_data['infer_poi']
    pos_start_idxs = val_data['pos_start_idx']
    patch_lens = val_data['patch_len']
    poi_nums = val_data['poi_num']
    total = 0
    hr10 = 0
    hr50 = 0
    for i in range(len(logits)):
        logit = logits[i]
        coord = coords[i]
        poi_list = infer_pois[i]
        pos_start_idx = pos_start_idxs[i]
        patch_len = patch_lens[i]
        poi_num = poi_nums[i]

        # optionally crop the logits to only the top k options
        if top_k is not None:
            v, _ = torch.topk(logit, min(top_k, logit.size(-1)))
            logit = torch.where(logit < v[:, -1].unsqueeze(1), -float("Inf"), logit)

        probs = torch.nn.functional.softmax(logit, dim=-1)
        idx_next = torch.multinomial(probs, num_samples=1).to(dtype=torch.int64)

        for start_poi in range(poi_num // 2, poi_num):
            start_position = pos_start_idx + start_poi * patch_len - 1
            end_position = pos_start_idx + start_poi * patch_len - 1 + patch_len
            infer_cat_name = tokenizer.decode(idx_next[start_position:end_position].view(-1))
            poi = poi_list[start_poi]
            lon, lat = lon_lat_decode(coord[start_position:end_position])
            pos = poi_finder.find_cat_pos(infer_cat_name, lon, lat, poi[0])
            total += 1
            if pos >= 0 and pos < 10:
                hr10 += 1
            if pos >= 0 and pos < 50:
                hr50 += 1
    return hr10, hr50, total


def merged_loss_fn(lang_output, spa_output, lang_label, spa_label, spa_mask, trainable_mask):
    # shift the targets such that output n predicts token n+1
    loss_lang = language_loss_fn(lang_output, lang_label, trainable_mask)
    loss_spa = spatial_loss_fn(spa_output, spa_label, spa_mask, trainable_mask)
    loss = loss_lang + loss_spa
    return loss_lang, loss_spa


def spatial_loss_fn(logits, target, spa_mask, trainable_mask):
    mask = spa_mask | trainable_mask
    mse_loss = F.mse_loss(logits[~mask], target[~mask], reduction='mean')
    return mse_loss


def language_loss_fn(logits, targets, mask):
    # shift the targets such that output n predicts token n+1
    logits = logits.contiguous()
    targets = targets.contiguous()
    ignore_value = -1
    if mask is not None:
        targets[mask] = ignore_value
    loss = torch.nn.functional.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1),
                                             ignore_index=ignore_value)
    return loss


def get_batch(fabric: L.Fabric, data: list, train=True, iter=None):
    # data = [(poi_id, cat, token_len, lon, lat, timestamp),...]
    ix = torch.randint(len(data), (micro_batch_size,))
    # ix = torch.tensor([0, 1, 2, 3])
    if not train:
        ix = iter
    language_input = [data[i]["language_inputs"].type(torch.int64) for i in ix]
    language_label = [data[i]["language_labels"].type(torch.int64) for i in ix]
    spatial_input = [data[i]["spatial_inputs"].type(torch.float32) for i in ix]
    spatial_label = [data[i]["spatial_labels"].type(torch.float32) for i in ix]
    spatial_mask = [data[i]["spatial_masks"].type(torch.bool) for i in ix]
    trainable_mask = [data[i]["trainable_masks"].type(torch.bool) for i in ix]
    raw_spatial = [data[i]["raw_spatial"].type(torch.float32) for i in ix]
    poi_idx = [data[i]["spatial_start_idx"].type(torch.int64) for i in ix]


    if not train:
        infer_poi = [data[i]["infer_poi"] for i in ix]
        # pos_start_idx = [data[i]["pos_start_idx"] for i in ix]
        # patch_len = [data[i]["patch_len"] for i in ix]
        poi_num = [data[i]["poi_num"] for i in ix]

    max_len = max(len(s) for s in language_input)

    def pad_right(x, pad_value):
        # pad right based on the longest sequence
        n = max_len - len(x)
        if type(pad_value) != torch.Tensor:
            return torch.cat((x, torch.full((n,), pad_value, dtype=x.dtype)))
        else:
            return torch.cat((x, pad_value.expand(n, -1)))

    li = torch.stack([pad_right(x, pad_value=0) for x in language_input])
    ll = torch.stack([pad_right(x, pad_value=-1) for x in language_label])
    # y = torch.stack([pad_right(x, pad_id=-1) for x in labels])
    si = torch.stack([pad_right(x, pad_value=torch.zeros(2, dtype=x.dtype, device=x.device)) for x in spatial_input])
    sl = torch.stack(
        [pad_right(x, pad_value=torch.zeros(2, dtype=x.dtype, device=x.device)) for x in spatial_label])
    sm = torch.stack([pad_right(x, pad_value=True) for x in spatial_mask])
    tm = torch.stack([pad_right(x, pad_value=True) for x in trainable_mask])
    rs = torch.stack(raw_spatial)
    language_input, language_label, spatial_input, spatial_label, spatial_mask, trainable_mask, raw_spatial = \
        fabric.to_device((li.pin_memory(), ll.pin_memory(), si.pin_memory(), sl.pin_memory(), sm.pin_memory(),
                          tm.pin_memory(), rs.pin_memory()))
    if train:
        return language_input, language_label, spatial_input, spatial_label, spatial_mask, trainable_mask, raw_spatial, poi_idx
    valid_data = {}
    valid_data['infer_poi'] = infer_poi
    # valid_data['pos_start_idx'] = pos_start_idx
    # valid_data['patch_len'] = patch_len
    valid_data['poi_num'] = poi_num
    return language_input, language_label, spatial_input, spatial_label, spatial_mask, trainable_mask, valid_data, raw_spatial, poi_idx


if __name__ == "__main__":
    # Uncomment this line if you see an error: "Expected is_sm80 to be true, but got false"
    # torch.backends.cuda.enable_flash_sdp(False)
    torch.set_float32_matmul_precision("high")

    from jsonargparse import CLI

    CLI(main)
