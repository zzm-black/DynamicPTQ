# coding=utf-8
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import datetime
import json
import os
import sys
from contextlib import contextmanager
from logging import Logger
from pathlib import Path

import torch
import torch.distributed as dist
from transformers import LlamaTokenizerFast
import transformers
from eval_utils.main import ptq_model
from eval_utils.modeling_llama import LlamaForCausalLM
from utils import utils
from utils.process_args import process_args_ptq

log: Logger = utils.get_logger("spinquant")


def _prepare_flatquant_import() -> None:
    workspace_root = Path(__file__).resolve().parents[1]
    flatquant_root = str((workspace_root / "FlatQuant").resolve())
    if flatquant_root not in sys.path:
        sys.path.insert(0, flatquant_root)


@contextmanager
def _flatquant_cwd():
    current_dir = os.getcwd()
    flatquant_dir = str((Path(__file__).resolve().parents[1] / "FlatQuant").resolve())
    try:
        os.chdir(flatquant_dir)
        yield
    finally:
        os.chdir(current_dir)


def train() -> None:
    dist.init_process_group(backend="nccl", timeout=datetime.timedelta(hours=8))
    model_args, training_args, ptq_args = process_args_ptq()
    local_rank = utils.get_local_rank()

    log.info("the rank is {}".format(local_rank))
    torch.distributed.barrier()

    config = transformers.AutoConfig.from_pretrained(
        model_args.input_model, token=model_args.access_token
    )
    # Llama v3.2 specific: Spinquant is not compatiable with tie_word_embeddings, clone lm_head from embed_tokens
    process_word_embeddings = False
    if config.tie_word_embeddings:
        config.tie_word_embeddings = False
        process_word_embeddings = True
    dtype = torch.bfloat16 if training_args.bf16 else torch.float16
    model = LlamaForCausalLM.from_pretrained(
        pretrained_model_name_or_path=model_args.input_model,
        config=config,
        torch_dtype=dtype,
        token=model_args.access_token,
    )
    if process_word_embeddings:
        model.lm_head.weight.data = model.model.embed_tokens.weight.data.clone()
    if ptq_args.distribute:
        log.info("Distribute mode enabled: skip eager full-model cuda() to avoid single-GPU OOM.")
    else:
        model.to(utils.DEV)

    model = ptq_model(ptq_args, model, model_args)
    model.seqlen = training_args.model_max_length
    if local_rank == 0:
        log.info("Model PTQ completed {}".format(model))
        log.info("Start to load tokenizer...")
    tokenizer = LlamaTokenizerFast.from_pretrained(
        pretrained_model_name_or_path=model_args.input_model,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side="right",
        use_fast=True,
        add_eos_token=False,
        add_bos_token=False,
        token=model_args.access_token,
    )
    log.info("Complete tokenizer loading...")
    model.config.use_cache = False

    _prepare_flatquant_import()
    import flatquant.data_utils as fq_data_utils
    import flatquant.eval_utils as fq_eval_utils

    if ptq_args.distribute:
        utils.distribute_model(model)
    else:
        model.to(utils.DEV)
    final_metrics = {}
    for eval_dataset in ptq_args.eval_datasets:
        with _flatquant_cwd():
            testloader = fq_data_utils.get_loaders(
                None,
                eval_dataset,
                tokenizer,
                seqlen=model.seqlen,
                eval_mode=True,
            )
        dataset_ppl = fq_eval_utils.ppl_eval(model, testloader)
        final_metrics[f"ppl/{eval_dataset}"] = float(dataset_ppl)
        log.info("%s ppl is: %s", eval_dataset, dataset_ppl)

    if ptq_args.lm_eval:
        import lm_eval
        from lm_eval import tasks as lm_eval_tasks
        from lm_eval import utils as lm_eval_utils
        from lm_eval.models.huggingface import HFLM

        hflm = HFLM(
            pretrained=model, tokenizer=tokenizer, batch_size=ptq_args.lm_eval_batch_size
        )
        task_manager = (
            lm_eval_tasks.TaskManager() if hasattr(lm_eval_tasks, "TaskManager") else None
        )
        all_tasks = (
            list(getattr(task_manager, "all_tasks", [])) if task_manager is not None else []
        )
        if all_tasks:
            task_names = lm_eval_utils.pattern_match(ptq_args.lm_eval_tasks, all_tasks)
        else:
            task_names = list(ptq_args.lm_eval_tasks)
        if not task_names:
            raise ValueError(
                f"No lm_eval tasks resolved from: {ptq_args.lm_eval_tasks}"
            )
        metric_vals = {}
        for task_name in task_names:
            task_result = lm_eval.simple_evaluate(
                hflm,
                tasks=[task_name],
                batch_size=ptq_args.lm_eval_batch_size,
                task_manager=task_manager,
            )["results"][task_name]
            metric_vals[task_name] = round(
                task_result.get("acc_norm,none", task_result.get("acc,none", 0.0)), 4
            )
        metric_vals["acc_avg"] = round(
            sum(metric_vals.values()) / len(metric_vals.values()), 4
        )
        for task, val in metric_vals.items():
            key = "lm_eval/acc_avg" if task == "acc_avg" else f"lm_eval/{task}"
            final_metrics[key] = float(round(val * 100, 2))

    if local_rank == 0:
        print(json.dumps(final_metrics, indent=2, ensure_ascii=False))
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    train()
