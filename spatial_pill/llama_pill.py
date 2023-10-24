"""
Instruction-tuning with LoRA on the Alpaca dataset.

Note: If you run into a CUDA error "Expected is_sm80 to be true, but got false", uncomment the line
`torch.backends.cuda.enable_flash_sdp(False)` in the script below (see https://github.com/Lightning-AI/lit-llama/issues/101).
"""
import random
import sys
from pathlib import Path
import os
import time

import lightning as L
import numpy as np
import torch

# support running without installing as a package
from tqdm import tqdm
from collections import OrderedDict

wd = Path(__file__).parent.parent.resolve()
sys.path.append(str(wd))

from lora_pill import mark_only_lora_pill_as_trainable, lora_pill, lora_pill_state_dict
from model_extra import LLaMA_extra, LLaMAExtraConfig

instruction_tuning = True
eval_interval = 100
save_interval = 100
eval_iters = 100
log_interval = 1

# Hyperparameters
learning_rate = 3e-4
batch_size = 128
micro_batch_size = 4
gradient_accumulation_iters = batch_size // micro_batch_size
assert gradient_accumulation_iters > 0
max_iters = 20000 * 3 // micro_batch_size
weight_decay = 0.0
max_seq_length = 256  # see scripts/prepare_alpaca.py
lora_r = 8
lora_alpha = 16
lora_dropout = 0.05
warmup_iters = 100


def main(
        data_dir: str = "../data/spatial_dataset",
        pretrained_path: str = "../checkpoints/lit-llama/7B/lit-llama.pth",
        tokenizer_path: str = "../checkpoints/lit-llama/tokenizer.model",
        out_dir: str = "../out/pill/poi",
):
    fabric = L.Fabric(accelerator="cuda", devices=3, precision="bf16-true")
    fabric.launch()
    fabric.seed_everything(1337 + fabric.global_rank)

    if fabric.global_rank == 0:
        os.makedirs(out_dir, exist_ok=True)

    train_data, test_data = load_datasets(data_dir=data_dir, cal_max_len=False)

    config = LLaMAExtraConfig.from_name("7B")
    config.block_size = max_seq_length

    checkpoint = torch.load(pretrained_path)

    with fabric.init_module(), lora_pill(r=lora_r, alpha=lora_alpha, dropout=lora_dropout, enabled=True):
        model = LLaMA_extra(config)
        # strict=False because missing keys due to LoRA weights not contained in checkpoint state
        model.load_state_dict(checkpoint, strict=False)

    mark_only_lora_pill_as_trainable(model)

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    model, optimizer = fabric.setup(model, optimizer)
    train(fabric, model, optimizer, train_data, test_data, tokenizer_path, out_dir)

    # Save the final LoRA checkpoint at the end of training
    checkpoint = lora_pill_state_dict(model)
    fabric.save(os.path.join(out_dir, "llama-7b-lora-pill-finetuned.pth"), checkpoint)


def train(
        fabric: L.Fabric,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        train_data: list,
        val_data: list,
        tokenizer_path: str,
        out_dir: str,
) -> None:
    """The training loop.

    Loosely based on the nanoGPT implementation: https://github.com/karpathy/nanoGPT.
    """
    step_count = 0
    BH = BatchHandler(train_data)
    pbar = tqdm(total=max_iters, ncols=100)
    for iter_num in range(max_iters):

        if step_count <= warmup_iters:
            # linear warmup
            lr = learning_rate * step_count / warmup_iters
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr

        t0 = time.time()

        input_ids, targets, spatial_addition = BH.get_batch(fabric)
        with fabric.no_backward_sync(model, enabled=((iter_num + 1) % gradient_accumulation_iters != 0)):
            logits, space = model(input_ids, spatial_addition=spatial_addition)
            loss_1 = loss_fn(logits, targets)
            loss_2 = loss_sp(space, spatial_addition)
            loss = loss_1 + loss_2
            fabric.backward(loss / gradient_accumulation_iters)

        if (iter_num + 1) % gradient_accumulation_iters == 0:
            optimizer.step()
            optimizer.zero_grad()
            step_count += 1

            if step_count % eval_interval == 0:
                val_loss = validate(fabric, model, val_data, tokenizer_path)
                fabric.print(f"step {iter_num}: val loss {val_loss:.4f}")
                fabric.barrier()

            if step_count % save_interval == 0:
                print(f"Saving LoRA weights to {out_dir}")
                # We are only saving the LoRA weights
                # TODO: Provide a function/script to merge the LoRA weights with pretrained weights
                checkpoint = lora_pill_state_dict(model)
                fabric.save(os.path.join(out_dir, f"iter-{iter_num:06d}-ckpt.pth"), checkpoint)

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


@torch.no_grad()
def validate(fabric: L.Fabric, model: torch.nn.Module, val_data: list, tokenizer_path: str) -> torch.Tensor:
    fabric.print("Validating ...")
    model.eval()
    losses = torch.zeros(eval_iters)
    BH = BatchHandler(val_data)
    for k in range(eval_iters):
        input_ids, targets, spatial_addition = BH.get_batch(fabric)
        logits, space = model(input_ids, spatial_addition=spatial_addition)
        loss_1 = loss_fn(logits, targets)
        loss_2 = loss_sp(space, spatial_addition)
        loss = loss_1 + loss_2
        losses[k] = loss.item()
    out = losses.mean()

    # produce an example:
    # instruction = "predict which place the user will visit: "
    #
    # output = generate_response(model, instruction, tokenizer_path)
    # fabric.print(instruction)
    # fabric.print(output)

    model.train()
    return out.item()


def loss_fn(logits, targets):
    # shift the targets such that output n predicts token n+1
    logits = logits[..., :-1, :].contiguous()
    targets = targets[..., 1:].contiguous()
    loss = torch.nn.functional.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
    return loss


def loss_sp(coord, targets):
    # shift the targets such that output n predicts token n+1
    coord = coord[..., :-1, :].contiguous()
    targets = targets[..., 1:, :].contiguous()
    loss = torch.nn.functional.mse_loss(coord, targets)
    return loss


class BatchHandler:
    def __init__(self, data: list):
        self.index = list(range(len(data)))
        self.data = data

    def get_batch(self, fabric: L.Fabric):
        if len(self.index) < micro_batch_size:
            self.__init_ix()
        ix = [self.__get_ix() for _ in range(micro_batch_size)]

        input_ids = [self.data[i]["sentence_token"].type(torch.int64) for i in ix]
        labels = [self.data[i]["sentence_token"].type(torch.int64) for i in ix]
        spatial_addition = [self.data[i]["spatial_addition"] for i in ix]

        max_len = max(len(s) for s in input_ids)

        x = torch.stack([self.pad_right(x, pad_id=0, max_len=max_len) for x in input_ids])
        y = torch.stack([self.pad_right(x, pad_id=-1, max_len=max_len) for x in labels])
        s = torch.stack([self.space_pad_right(s, pad_id=0, max_len=max_len) for s in spatial_addition])
        x, y, s = fabric.to_device((x.pin_memory(), y.pin_memory(), s.pin_memory()))
        return x, y, s

    @staticmethod
    def pad_right(x, pad_id, max_len):
        # pad right based on the longest sequence
        n = max_len - len(x)
        return torch.cat((x, torch.full((n,), pad_id, dtype=x.dtype)))

    @staticmethod
    def space_pad_right(s, pad_id, max_len):
        # pad right based on the longest sequence
        s = torch.stack(s)
        n = max_len - len(s)
        return torch.cat((s, torch.full((n, s.shape[-1],), pad_id, dtype=s.dtype)))

    def __get_ix(self):
        i = random.randint(0, len(self.index)-1)
        ix = self.index[i]
        self.index.remove(ix)
        return ix

    def __init_ix(self):
        self.index = list(range(len(self.data)))


def load_datasets(data_dir, cal_max_len=False):
    train_data = torch.load(os.path.join(data_dir, "poi_train.pt"))
    test_data = torch.load(os.path.join(data_dir, "poi_test.pt"))
    if cal_max_len:
        l1 = calMaxLength(train_data)
        l2 = calMaxLength(test_data)
        global max_seq_length
        max_seq_length = l1 if l1 > l2 else l2
        print(max_seq_length)
    return train_data, test_data


def calMaxLength(data):
    max_length = 0
    for tuple in data:
        length = len(tuple["sentence_token"])
        if length > max_length:
            max_length = length
    return max_length


if __name__ == "__main__":
    # Uncomment this line if you see an error: "Expected is_sm80 to be true, but got false"
    # torch.backends.cuda.enable_flash_sdp(False)
    torch.set_float32_matmul_precision("high")

    from jsonargparse.cli import CLI

    CLI(main)
