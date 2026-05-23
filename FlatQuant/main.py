import transformers
import torch

import flatquant.utils as utils
import flatquant.args_utils as args_utils
import flatquant.model_utils as model_utils
import flatquant.data_utils as data_utils
import flatquant.eval_utils as eval_utils
import flatquant.train_utils as train_utils
import flatquant.flat_utils as flat_utils
import gptq_utils


def _parse_deep_layers(deep_layers: str, num_layers: int):
    value = deep_layers.strip().lower()
    if value == "last5":
        start = max(0, num_layers - 5)
        return tuple(range(start, num_layers))

    layers = []
    for token in value.split(","):
        token = token.strip()
        if not token:
            continue
        layer_id = int(token)
        if layer_id < 0:
            raise ValueError(f"deep layer id must be >= 0, got: {layer_id}")
        if layer_id >= num_layers:
            raise ValueError(f"deep layer id out of range [0, {num_layers - 1}], got: {layer_id}")
        layers.append(layer_id)
    if not layers:
        raise ValueError("deep_layers must be 'last5' or comma-separated 0-based layer ids.")
    return tuple(sorted(set(layers)))


def _get_qmin_qmax(bits: int, sym: bool):
    if sym:
        q_max = torch.tensor(2 ** (bits - 1) - 1)
        q_min = -q_max - 1
    else:
        q_max = torch.tensor(2**bits - 1)
        q_min = torch.tensor(0)
    return q_max, q_min


def _apply_deep_activation_bits(model, args, logger):
    if not args.enable_deep:
        return

    num_layers = len(model.model.layers)
    deep_layers = set(_parse_deep_layers(args.deep_layers, num_layers))
    logger.info(
        "Applying DEEP activation bits: default_a_bits=%d, deep_act_bits=%d, deep_layers=%s",
        args.a_bits,
        args.deep_act_bits,
        sorted(deep_layers),
    )
    for layer_idx, layer in enumerate(model.model.layers):
        target_bits = args.deep_act_bits if layer_idx in deep_layers else args.a_bits
        for module in layer.modules():
            act_q = getattr(module, "act_quantizer", None)
            if act_q is not None and hasattr(act_q, "bits"):
                act_q.bits = int(target_bits)
                if hasattr(act_q, "q_max") and hasattr(act_q, "q_min"):
                    qmax, qmin = _get_qmin_qmax(target_bits, bool(getattr(act_q, "sym", True)))
                    qmax = qmax.to(act_q.q_max.device)
                    qmin = qmin.to(act_q.q_min.device)
                    act_q.q_max = qmax
                    act_q.q_min = qmin


def main():
    args, logger = args_utils.parser_gen()
    utils.seed_everything(seed=args.seed)

    model, apply_flatquant_to_model = model_utils.get_model(args.model, args.hf_token)
    model.eval()
    tokenizer = transformers.AutoTokenizer.from_pretrained(args.model, use_fast=False, use_auth_token=args.hf_token)

    # get calibration data
    trainloader = data_utils.get_loaders(
        args, args.cali_dataset, tokenizer, 
        nsamples=args.nsamples, seqlen=model.seqlen, eval_mode=False, 
    )
    logger.info("Finished loading training data.")

    if args.quantize:
        model = apply_flatquant_to_model(args, model)
        logger.info("Finished applying FlatQuant to model.")
        _apply_deep_activation_bits(model, args, logger)
        if args.resume:
            flat_utils.load_flat_parameters(args, model)
        elif args.reload_matrix:
            flat_utils.load_flat_matrices(args, model, path=args.matrix_path)
        elif (args.cali_trans or args.add_diag or args.lwc or args.lac):
            train_utils.cali_flat_quant(args, model, trainloader, utils.DEV, logger=logger)
        if args.save_matrix and not args.reload_matrix:
            flat_utils.save_flat_matrices(args, model)
        flat_utils.reparameterize_model(model)
        logger.info("Finished reparameterize model.")

    if args.w_bits < 16:
        save_dict = {}
        if args.gptq: # GPTQ Weight Quantization
            quantizers = gptq_utils.gptq_fwrd(model, trainloader, utils.DEV, args)
        else: # RTN Weight Quantization
            quantizers = gptq_utils.rtn_fwrd(model, utils.DEV, args)
        save_dict["w_quantizers"] = quantizers

    ## save quantized weight
    if args.quantized_save:
        flat_utils.save_quantized_weights_with_safetensors(args, model, quantizers)

    if args.distribute_model:
        utils.distribute_model(model)
    else:
        model.to(utils.DEV)
    
    # Evaluating PPL
    for eval_dataset in args.eval_datasets:
        logger.info(eval_dataset)
        testloader = data_utils.get_loaders(
                args,
                eval_dataset,
                tokenizer,
                seqlen=model.seqlen,
                eval_mode=True
            )
        dataset_ppl = eval_utils.ppl_eval(model, testloader)
        logger.info(dataset_ppl)


    if args.lm_eval:
        import lm_eval
        from lm_eval import utils as lm_eval_utils
        from lm_eval.models.huggingface import HFLM

        hflm = HFLM(pretrained=model, tokenizer=tokenizer, batch_size=args.lm_eval_batch_size)

        task_names = args.tasks

        results = {}
        for task_name in task_names:
            logger.info(f"Evaluating {task_name}...")
            result = lm_eval.simple_evaluate(hflm, tasks=[task_name], batch_size=args.lm_eval_batch_size)['results']
            result = result[task_name]
            acc = round(result.get('acc_norm,none', result['acc,none']) * 100, 2)
            results[task_name] = acc
            logger.info(f"acc: {acc}%")
        metric_vals = {task: result for task, result in results.items()}
        metric_vals['acc_avg'] = round(sum(metric_vals.values()) / len(metric_vals.values()), 2)
        logger.info(metric_vals)


if __name__ == '__main__':
    main()