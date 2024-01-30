import argparse
import datetime
import json
import logging
import os
import socket
import subprocess
from itertools import product
from pathlib import Path

import pandas as pd
import torch
import transformers
from tqdm.auto import tqdm

from specdec import SpecInfer, SpecExec2, utils

device = torch.device("cuda:0")
_DEFAULT_DEVICE_SIZE = 2


def create_spec_generator(model_name_0, model_name_1, gen_type="SX2", offload=False, device_size=_DEFAULT_DEVICE_SIZE, check_tokenizer=False):
    """Creates a SpecGenerator object for different generation types.

    This function loads draft and target pre-trained language models specified by their names
    and creates a SpecBase subclass object based on the provided generation type.
    It also handles several configuration options like device placement and tokenizer verification.

    Args:
        model_name_0 (str): Name of the draft model.
        model_name_1 (str): Name of the target model.
        gen_type (str, optional): Generation type. Defaults to "SX2" (SpecExec2).
            Valid options include:
                - "SX2", "X2", "spec_exec_2", "specexec2": SpecExec2 generator
                - "SI", "spec_infer", "specinfer": SpecInfer generator
        offload (bool, optional): Whether to offload model 1 using offloading library. Defaults to False.
        device_size (int, optional): Device size for offloading. Defaults to `_DEFAULT_DEVICE_SIZE`.
        check_tokenizer (bool, optional): Whether to verify if both models have the same tokenizer. Defaults to False.

    Returns:
        SpecGenerator: An instance of a SpecBase subclass object based on the provided parameters.

    Raises:
        ValueError: If an invalid `gen_type` is provided.
    """

    if len(model_name_0.split("::")) == 2:
        model_name_0, rev_0 = model_name_0.split("::")
    else:
        rev_0 = "main"  # default in `from_pretrained()`

    if len(model_name_1.split("::")) == 2:
        model_name_1, rev_1 = model_name_1.split("::")
    else:
        rev_1 = "main"  # default in `from_pretrained()`

    tokenizer = transformers.AutoTokenizer.from_pretrained(model_name_0, legacy=False)

    if check_tokenizer:
        # verify that the two models have the same tokenizer
        tokenizer_1 = transformers.AutoTokenizer.from_pretrained(model_name_1, legacy=False)
        vv0 = tokenizer.get_vocab()
        vv1 = tokenizer_1.get_vocab()

        ignored_tokens = ["[PAD]"]  # disregard these tokens when comparing the cokonizers' vocabs
        assert set(vv0.keys()).difference(ignored_tokens) == set(vv1.keys()).difference(ignored_tokens)
        for k in set(vv0.keys()).difference(ignored_tokens):
            assert vv0[k] == vv1[k]
        del tokenizer_1, vv0, vv1

    logger.info(f"Loading Model 0: `{model_name_0}`")
    model_0 = transformers.AutoModelForCausalLM.from_pretrained(model_name_0, device_map=device, torch_dtype=torch.float16, revision=rev_0)
    logger.info(f"Loading Model 1: `{model_name_1}`")
    if offload:
        from offloading.offload_model import load_gptq_offloaded_model

        model_1 = load_gptq_offloaded_model(model_name_1, device_size=device_size, main_device=device)
    else:
        model_1 = transformers.AutoModelForCausalLM.from_pretrained(model_name_1, device_map=device, torch_dtype=torch.float16, revision=rev_1)

    try:
        from auto_gptq import exllama_set_max_input_length

        try:
            model_1 = exllama_set_max_input_length(model_1, max_input_length=8192)
            print("set `exllama_set_max_input_length` OK")
        except (AttributeError, ValueError):
            # AttributeError may happen if GPTQ-quantized model has no attribute 'device_to_buffers'
            # could be fixed by using code from post_init()
            logger.warning("Failed to set `exllama_set_max_input_length`")
    except AttributeError:
        pass

    if gen_type.lower() in ("sx2", "x2", "spec_exec_2", "specexec2"):
        spec_generator = SpecExec2(model_0, model_1, tokenizer)
    elif gen_type.lower() in ("si", "spec_infer", "specinfer"):
        spec_generator = SpecInfer(model_0, model_1, tokenizer)
    else:
        raise ValueError(f"unknown {gen_type=}")

    logger.info(f"Created spec_generator of type {gen_type}; Models: {model_name_0}, {model_name_1}")
    return spec_generator


def run_tests(
    spec_generator,
    dataset,
    args,
    max_budget=None,
    max_n_beams=None,
    max_beam_len=None,
    max_branch_width=None,
    min_log_prob=None,
    **kwargs,
):
    """runs uniform experiments from dataset using same set of parameters"""

    test_logs = []

    for i in range(args.n_tests):
        prompt = dataset[i]
        with utils.Timing() as t_single:
            _ = spec_generator.generate(
                prompt,
                max_n_beams=max_n_beams,
                max_beam_len=max_beam_len,
                max_new_tokens=args.max_new_tokens,
                branching=args.branching,
                max_budget=max_budget,
                max_branch_width=max_branch_width,
                replacement=args.replacement,
                verbose=args.verbose,
                temperature=args.temperature,
                draft_temperature=args.draft_temperature,
                top_p=args.top_p,
                min_log_prob=min_log_prob,
                seed=args.seed,
                **kwargs,
            )
        test_time = t_single.elapsed
        spec_generator.summary["speed"] = round(spec_generator.summary["new_tokens"] / test_time, 2)
        test_logs.append(spec_generator.summary)
        generated_text = spec_generator.tokenizer.decode(spec_generator.prefix_tokens[spec_generator.original_num_tokens :]).__repr__().strip("'")

        excl_keys = ["ver", "model_name_0", "model_name_1"]
        log1 = {k: v for k, v in spec_generator.summary.items() if k not in excl_keys}
        log1 = {"run": i, **log1, "text": generated_text[:32]}
        log1["prompt_text"] = log1["prompt_text"].replace(r" [\INST] ", "")[-32:]  # last 32 prompt chars

        log_one_line(log1, save_dir=args.save_dir, exp_name=args.exp_name, verbose=args.verbose, msg_type="")

    df = pd.DataFrame(test_logs)

    exp_summary = dict(
        max_n_beams=max_n_beams,
        max_beam_len=max_beam_len,
        min_log_prob=min_log_prob,
        max_budget=max_budget,
        max_branch_width=max_branch_width,
        gen_rate=(df.new_tokens.sum() / df.iters.sum()).round(2),
        gen_speed=(df.new_tokens.sum() / (df.new_tokens / df.speed).sum()).round(2),
        t0=df.t0.mean().round(2),
        t1=df.t1.mean().round(2),
        input_0=df.input_0.mean().round(1),
        input_1=df.input_1.mean().round(1),
        tree_size=df.tree_size.mean().round(1),
        tree_w=df.tree_w.mean().round(1),
        tree_h=df.tree_h.mean().round(1),
        prompt_len=df.prompt_len.mean().round(1),
        lowest_cum_log_prob=df.lowest_cum_log_prob.mean().round(4),
    )
    torch.cuda.empty_cache()
    return exp_summary, test_logs


def log_one_line(data_dict, verbose, save_dir, exp_name, msg_type=None):
    """
    Logs key-value pairs from a dictionary to both the console (as a single line) and a JSONL file,
    with optional filtering for certain keys and conditional logging based on verbosity.

    Args:
        data_dict (dict): A dictionary containing the data to be logged.
        verbose (bool): If True, logs to stdout regardless of logger level.
        save_dir (str): Path to the directory where the log file will be saved.
        exp_name (str): Name of the experiment, used for the log file name.
        msg_type (str, optional): A message type to be included in the log file. Defaults to None.
    """
    stdout_blacklist = ["prompt_text", "text"]

    log_line = "  ".join([f"{k}:{v}" for k, v in data_dict.items() if k not in stdout_blacklist and v is not None])
    if msg_type == "":
        log_line = "    " + log_line
    if verbose or (logger.level >= logging.INFO):
        print(log_line)

    # logging to file
    if msg_type is not None:
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)
        log_filename = save_path / f"{exp_name}.jsonl"
        with log_filename.open("a") as file:
            file.write(json.dumps({"msg_type": msg_type, **data_dict}) + "\n")


def arg_to_list(args, arg):
    """
    Converts a command-line argument value to a list of appropriate types.
    Handles different value formats (single value, comma-separated values, "None"),
    converts to integers or floats as needed, and returns a list of parsed values.

    Args:
        args: An object containing command-line arguments (e.g., argparse.Namespace).
        arg (str): The name of the argument to retrieve and convert.

    Returns:
        list: A list of parsed values from the argument.
    """
    arg_value = getattr(args, arg)
    float_args = ["min_log_prob"]
    if arg_value is None:
        return [None]

    def from_str(s):
        """
        Parses a string value into an integer, float, or None.
        Args:  s (str): The string to parse.
        Returns: int, float, or None: The parsed value.
        """
        s = s.strip()
        if s.lower() == "none":
            return None
        elif arg in float_args:
            return float(s)
        else:
            return int(s)

    return [from_str(s) for s in arg_value.split(",")]


def main(args):
    logger.warning(f"Starting test with models {args.model_0}, {args.model_1}")
    spec_generator = create_spec_generator(
        model_name_0=args.model_0,
        model_name_1=args.model_1,
        gen_type=args.gen_type,
        offload=args.offload,
        device_size=args.device_size,
        check_tokenizer=True,
    )
    logger.debug(f"mem use {0}")  # TODO log memory usage

    if args.dataset.lower().startswith("oasst"):
        logger.warning("loading OASST-based prompts set")
        dataset = utils.get_dataset("oasst_prompts")
    elif args.dataset.lower().startswith("wiki"):
        logger.warning("loading Wikitext2-based prompts set")
        dataset = utils.get_dataset("wikitext_prompts")
    else:
        raise ValueError(f"Unknown args.dataset value: `{args.dataset}`")

    if args.device_size != _DEFAULT_DEVICE_SIZE and not args.offload:
        logger.warning(f"Passed --device_size of {args.device_size}, but offloading is disabled")

    logs = []
    summaries = []

    config_dict = dict(
        gen_type=args.gen_type,
        model_0=args.model_0,
        model_1=args.model_1,
        temperature=args.temperature,
        max_n_beams=args.max_n_beams,
        max_beam_len=args.max_beam_len,
        top_p=args.top_p,
        max_new_tokens=args.max_new_tokens,
        max_budget=args.max_budget,
        max_branch_width=args.max_branch_width,
        branching=args.branching,
        min_log_prob=args.min_log_prob,
        replacement=args.replacement,
        n_tests=args.n_tests,
        seed=args.seed,
        dataset=args.dataset,
        timestamp=datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        date=datetime.datetime.today().strftime("%y%m%d"),
        hostname=socket.gethostname(),
        commit=subprocess.check_output(["git", "rev-parse", "HEAD"]).strip().decode("utf-8"),
        offload=args.offload,
        device=torch.cuda.get_device_name(device).replace("NVIDIA ", ""),
    )
    if args.offload:
        config_dict["device_size"] = args.device_size
    log_one_line(config_dict, save_dir=args.save_dir, exp_name=args.exp_name, verbose=args.verbose, msg_type="config")

    with torch.inference_mode():
        if args.zero:
            log_one_line({"mode": "zero"}, save_dir=args.save_dir, exp_name=args.exp_name, verbose=args.verbose, msg_type="zero")
            spec_generator.tokenizer.pad_token_id = spec_generator.tokenizer.eos_token_id
            total_time = 0

            gene_config = transformers.GenerationConfig(
                max_new_tokens=32,
                do_sample=True,  # Use sampling
                temperature=0.6,  # Sampling temperature
                top_p=0.9,
                bos_token_id=1,
                eos_token_id=2,
                pad_token_id=2,
            )

            for i in range(args.n_tests):
                prompt = dataset[i]
                inputs = spec_generator.tokenizer(prompt, return_tensors="pt").to(device)
                with utils.Timing() as t:
                    spec_generator.target_model.generate(**inputs, generation_config=gene_config)
                log_one_line({"prompt": i, "elapsed": round(t.elapsed, 3)}, save_dir=args.save_dir, exp_name=args.exp_name, verbose=args.verbose, msg_type="zero")
                total_time += t.elapsed

            log_dict_zero = {"total_time": round(total_time, 3), "speed": round(args.max_new_tokens * args.n_tests / total_time, 3)}
            log_one_line(
                log_dict_zero,
                save_dir=args.save_dir,
                exp_name=args.exp_name,
                verbose=args.verbose,
                msg_type="zero",
            )
            print("-" * 120 + "\n   S U M M A R Y  (run without speculative decoding) \n" + "-" * 120)
            print(log_dict_zero)
            print("-" * 120)

            return None, None

    budget_classes = ["SpecExec2"]  # classes driven by token budgets
    if spec_generator.__class__.__name__ not in budget_classes:
        args.max_budget = "0"
        args.max_branch_width = "0"

    # Convert string arguments to lists of integers
    sweep_args_present = []
    args_can_sweep = ["max_n_beams", "max_beam_len", "max_budget", "max_branch_width", "min_log_prob"]
    arg_lists = []
    for arg in args_can_sweep:
        arg_list = arg_to_list(args, arg)
        arg_lists.append(arg_list)
        if len(arg_list) > 1:
            sweep_args_present.append(arg)

    if len(sweep_args_present) > 2:
        logger.warning(f"More than two sweep arguments detected: {sweep_args_present}.")

    combinations = product(*arg_lists)

    for max_n_beams, max_beam_len, max_budget, max_branch_width, min_log_prob in tqdm(combinations):  # align with `args_can_sweep`
        # preventing large model overloading
        # if (max_n_beams * max_beam_len > 4096) and (args.gen_type not in ["SDA", "SX"]):
        #     continue

        with utils.Timing() as t:
            summary, test_logs = run_tests(
                spec_generator=spec_generator,
                dataset=dataset,
                args=args,
                max_n_beams=max_n_beams,
                max_beam_len=max_beam_len,
                max_budget=max_budget,
                max_branch_width=max_branch_width,
                min_log_prob=min_log_prob,
            )
        summary["exp_time"] = round(t.elapsed, 2)
        summaries.append(summary)
        logs.extend(test_logs)
        log_one_line(summary, save_dir=args.save_dir, exp_name=args.exp_name, verbose=args.verbose, msg_type="exp")

        if args.wandb:
            wandb.init(project=args.wandb_project, name=f"{args.exp_name}__b{max_n_beams}x{max_beam_len}")
            wandb.log({**config_dict, **summary})
            wandb.finish()

        torch.cuda.empty_cache()

    # printing the summary table
    df = pd.DataFrame(summaries)
    pd.set_option("display.width", 160)
    print("-" * 120 + "\n       A R G U M E N T S \n" + "-" * 120)
    print(args)
    print("-" * 120 + "\n       S U M M A R Y   R E S U L T S \n" + "-" * 120)
    print(df[[*args_can_sweep, "gen_rate", "gen_speed"]])
    print("-" * 120)

    return summaries, logs


if __name__ == "__main__":
    if "logger" not in globals():
        logger = utils.get_logger()

    os.environ["TOKENIZERS_PARALLELISM"] = "false"  # avoiding warnings

    # DEFAULT MODEL NAMES
    model_name_0 = "TheBloke/Llama-2-7B-Chat-GPTQ"
    model_name_1 = "TheBloke/Llama-2-70B-chat-GPTQ"

    parser = argparse.ArgumentParser()

    parser.add_argument("--exp_name", help="Experiment name", default="experiment")
    parser.add_argument("--save_dir", help="Experiments directory", default="logs")
    parser.add_argument("--model_0", help="Model 0 name", default=model_name_0)
    parser.add_argument("--model_1", help="Model 1 name", default=model_name_1)
    parser.add_argument("-d", "--dataset", help="Datastet for testing. oasst or wikitext only for now", default="oasst")
    parser.add_argument("-g", "--gen_type", help="SI or SX2 or other class", default="SX2")
    parser.add_argument("--temperature", help="Sampling temperature", default=1.0, type=float)  # 0 for greedy
    parser.add_argument("--top_p", help="Sampling top_p", default=1.0, type=float)
    parser.add_argument("-t", "--temp", help="Sampling temperature and top_p as 4 digit string. '0609'-> 0.6, 0.9", default=None)
    parser.add_argument("--n_tests", help="Num of tests in each config", default=10, type=int)
    parser.add_argument("-b", "--max_n_beams", help="Num of beams in each exp; CAN SWEEP", default="128")
    parser.add_argument("-m", "--max_beam_len", help="max beam len; CAN SWEEP", default="32")
    parser.add_argument("--branching", help="tree styles for fixed trees", default=None)
    parser.add_argument("--max_budget", help="speculation token budget for fixed trees; CAN SWEEP", default=None)
    parser.add_argument("--max_branch_width", help="max_branch_width for fixed trees; CAN SWEEP", default="128")
    parser.add_argument("--replacement", help="draft model sampling with replacement", action="store_true")
    parser.add_argument("--repack", help="repack draft tree by combining identical node paths", action="store_true")
    parser.add_argument("--max_new_tokens", default=32, type=int)
    parser.add_argument("--min_log_prob", help="min log proba threshold for added leafs; CAN SWEEP", default=None)
    parser.add_argument("-v", "--verbose", action="store_true")
    parser.add_argument("--loglevel", default="WARNING")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("-o", "--offload", action="store_true")
    parser.add_argument("--device_size", type=int, default=_DEFAULT_DEVICE_SIZE)
    parser.add_argument("--wandb", help="Wandb enabled", action="store_true")
    parser.add_argument("--draft_temperature", default=None, type=float),
    parser.add_argument("--wandb_project", help="Wandb project name", default="spec_trees")
    parser.add_argument("--zero", help="zero speculation", action="store_true")

    args = parser.parse_args()

    logger.setLevel(getattr(logging, args.loglevel.upper(), logging.INFO))

    if args.wandb:
        import wandb

    if args.branching:
        # trying to converting string argument to int (except non-numerical strings)
        try:
            args.branching = int(args.branching)
        except ValueError:
            pass

    if args.temp is not None:
        # overriding args.temperature and args.top_p with decoded args.temp
        assert len(args.temp) == 4, f"args.temp should be a 4-digit string, received {args.temp}."
        args.temperature = float(f"{args.temp[0]}.{args.temp[1]}")
        args.top_p = float(f"{args.temp[2]}.{args.temp[3]}")

    with utils.Timing() as t:
        summaries, logs = main(args)
    logging.info(f"tests completed in {t.elapsed:.1f} s.")
