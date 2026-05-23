import utils
import torch
import json
import logging
import os
import sys
import re
from contextlib import contextmanager
from pathlib import Path
import model_utils
import data_utils
import transformers
import quant_utils
import rotation_utils
import gptq_utils
import eval_utils
import hadamard_utils


def _prepare_flatquant_import() -> None:
    workspace_root = Path(__file__).resolve().parents[2]
    flatquant_root = str((workspace_root / "FlatQuant").resolve())
    if flatquant_root not in sys.path:
        sys.path.insert(0, flatquant_root)


@contextmanager
def _flatquant_cwd():
    current_dir = os.getcwd()
    flatquant_dir = str((Path(__file__).resolve().parents[2] / "FlatQuant").resolve())
    try:
        os.chdir(flatquant_dir)
        yield
    finally:
        os.chdir(current_dir)


def _build_tokenizer(model_name: str, hf_token: str | None):
    try:
        return transformers.AutoTokenizer.from_pretrained(
            model_name, use_fast=False, token=hf_token
        )
    except TypeError:
        return transformers.AutoTokenizer.from_pretrained(
            model_name, use_fast=False, use_auth_token=hf_token
        )


def _parse_deep_layer_ids(raw: str) -> set[int]:
    layer_ids: set[int] = set()
    for token in raw.split(","):
        token = token.strip()
        if not token:
            continue
        layer_id = int(token)
        if layer_id < 0:
            raise ValueError(f"deep layer id must be non-negative (0-based), got: {layer_id}")
        layer_ids.add(layer_id)
    return layer_ids


def _layer_id_0based_from_name(module_name: str) -> int | None:
    match = re.search(r"layers\.(\d+)\.", module_name)
    if match is None:
        return None
    return int(match.group(1))


def main():
    args = utils.parser_gen()
    deep_layers = _parse_deep_layer_ids(args.deep_layers) if args.enable_deep else set()
    if args.wandb:
        import wandb
        wandb.init(project=args.wandb_project, entity=args.wandb_id)
        wandb.config.update(args)
        
    transformers.set_seed(args.seed)
    model = model_utils.get_model(args.model, args.hf_token)
    model.eval()
    
    
    # Rotate the weights
    if args.rotate:
        rotation_utils.fuse_layer_norms(model)
        rotation_utils.rotate_model(model, args)
        utils.cleanup_memory(verbos=True)
            
        quant_utils.add_actquant(model) #Add Activation Wrapper to the model
        qlayers = quant_utils.find_qlayers(model)
        for name in qlayers:
            if 'down_proj' in name:
                had_K, K = hadamard_utils.get_hadK(model.config.intermediate_size)
                qlayers[name].online_full_had = True
                qlayers[name].had_K = had_K
                qlayers[name].K = K
                qlayers[name].fp32_had = args.fp32_had
            if 'o_proj' in name:
                had_K, K = hadamard_utils.get_hadK(model.config.num_attention_heads)
                qlayers[name].online_partial_had = True
                qlayers[name].had_K = had_K
                qlayers[name].K = K
                qlayers[name].had_dim = model.config.hidden_size//model.config.num_attention_heads
                qlayers[name].fp32_had = args.fp32_had
    else:
        quant_utils.add_actquant(model) #Add Activation Wrapper to the model as the rest of the code assumes it is present
                
    if args.w_bits < 16:
        save_dict = {}
        if args.load_qmodel_path: # Load Quantized Rotated Model
            assert args.rotate, "Model should be rotated to load a quantized model!"
            assert not args.save_qmodel_path, "Cannot save a quantized model if it is already loaded!"
            print("Load quantized model from ", args.load_qmodel_path)
            save_dict = torch.load(args.load_qmodel_path)
            model.load_state_dict(save_dict["model"])
            
        elif not args.w_rtn: # GPTQ Weight Quantization
            assert "llama" in args.model, "Only llama is supported for GPTQ!"

            _prepare_flatquant_import()
            import flatquant.data_utils as fq_data_utils

            tokenizer = _build_tokenizer(args.model, args.hf_token)
            with _flatquant_cwd():
                trainloader = fq_data_utils.get_loaders(
                    None,
                    args.cal_dataset,
                    tokenizer,
                    nsamples=args.nsamples,
                    seqlen=model.seqlen,
                    eval_mode=False,
                )
            quantizers = gptq_utils.gptq_fwrd(model, trainloader, utils.DEV, args)
            save_dict["w_quantizers"] = quantizers
        else: # RTN Weight Quantization
            quantizers = gptq_utils.rtn_fwrd(model, utils.DEV, args)
            save_dict["w_quantizers"] = quantizers
            
        if args.save_qmodel_path:
            save_dict["model"] = model.state_dict()
            torch.save(save_dict, args.save_qmodel_path)


    # Add Input Quantization
    if args.a_bits < 16 or args.v_bits < 16:
        qlayers = quant_utils.find_qlayers(model, layers=[quant_utils.ActQuantWrapper])
        down_proj_groupsize = -1
        if args.a_groupsize > 0 and "llama" in args.model:
            down_proj_groupsize = utils.llama_down_proj_groupsize(model, args.a_groupsize)
        
        for name in qlayers:            
            layer_input_bits = args.a_bits
            layer_groupsize = args.a_groupsize
            layer_a_sym = not(args.a_asym)
            layer_a_clip = args.a_clip_ratio

            layer_id = _layer_id_0based_from_name(name)
            if args.enable_deep and layer_id in deep_layers:
                layer_input_bits = args.deep_act_bits
            
            if 'v_proj' in name and args.v_bits < 16: #Set the v_proj precision
                qlayers[name].out_quantizer.configure(bits=args.v_bits,
                                              groupsize=args.v_groupsize,
                                              sym=not(args.v_asym),
                                              clip_ratio=args.v_clip_ratio)
            
            if 'lm_head' in name: #Skip lm_head quantization   
                layer_input_bits = 16
            
            if 'down_proj' in name: #Set the down_proj precision
                if args.int8_down_proj:
                    layer_input_bits = 8
                layer_groupsize = down_proj_groupsize

                
            qlayers[name].quantizer.configure(bits=layer_input_bits,
                                              groupsize=layer_groupsize,
                                              sym=layer_a_sym,
                                              clip_ratio=layer_a_clip)

    if args.k_bits < 16:
        if args.k_pre_rope:
            raise NotImplementedError("Pre-RoPE quantization is not supported yet!")
        else:
            rope_function_name = model_utils.get_rope_function_name(model)
            layers = model_utils.get_layers(model)
            k_quant_config = {'k_bits':args.k_bits, "k_groupsize": args.k_groupsize,
                                          "k_sym": not(args.k_asym), "k_clip_ratio": args.k_clip_ratio}
            for layer in layers:
                rotation_utils.add_qk_rotation_wrapper_after_function_call_in_forward(
                            layer.self_attn, 
                            rope_function_name, 
                            config=model.config,
                            **k_quant_config)
        
    _prepare_flatquant_import()
    import flatquant.data_utils as fq_data_utils
    import flatquant.eval_utils as fq_eval_utils

    eval_tokenizer = _build_tokenizer(args.model, args.hf_token)
    if args.distribute:
        utils.distribute_model(model)
    else:
        # Align with FlatQuant evaluation flow: run PPL on GPU to avoid CPU-bound slowdown.
        model.to(utils.DEV)
    final_metrics = {}
    for eval_dataset in args.eval_datasets:
        with _flatquant_cwd():
            testloader = fq_data_utils.get_loaders(
                None,
                eval_dataset,
                eval_tokenizer,
                seqlen=model.seqlen,
                eval_mode=True,
            )
        dataset_ppl = fq_eval_utils.ppl_eval(model, testloader)
        final_metrics[f"ppl/{eval_dataset}"] = float(dataset_ppl)
        if args.wandb:
            wandb.log({f"ppl/{eval_dataset.upper()}": dataset_ppl})

    if not args.lm_eval:
        print(json.dumps(final_metrics, indent=2, ensure_ascii=False))
        return
    else:
        # Import lm_eval utils
        import lm_eval
        from lm_eval import utils as lm_eval_utils
        from lm_eval import tasks as lm_eval_tasks
        from lm_eval.models.huggingface import HFLM

        
    
    hflm = HFLM(pretrained=model, tokenizer=eval_tokenizer, batch_size=args.lm_eval_batch_size)

    task_manager = lm_eval_tasks.TaskManager() if hasattr(lm_eval_tasks, "TaskManager") else None
    all_tasks = list(getattr(task_manager, "all_tasks", [])) if task_manager is not None else []
    if all_tasks:
        task_names = lm_eval_utils.pattern_match(args.tasks, all_tasks)
    else:
        task_names = list(args.tasks)
    if not task_names:
        raise ValueError(f"No lm_eval tasks resolved from: {args.tasks}")
    metric_vals = {}
    for task_name in task_names:
        task_result = lm_eval.simple_evaluate(
            hflm,
            tasks=[task_name],
            batch_size=args.lm_eval_batch_size,
            task_manager=task_manager,
        )["results"][task_name]
        metric_vals[task_name] = round(task_result.get("acc_norm,none", task_result.get("acc,none", 0.0)), 4)

    metric_vals["acc_avg"] = round(sum(metric_vals.values()) / len(metric_vals.values()), 4)
    for task, val in metric_vals.items():
        key = "lm_eval/acc_avg" if task == "acc_avg" else f"lm_eval/{task}"
        final_metrics[key] = float(round(val * 100, 2))
    print(json.dumps(final_metrics, indent=2, ensure_ascii=False))

    if args.wandb:
        wandb.log(metric_vals)


if __name__ == '__main__':
    main()
