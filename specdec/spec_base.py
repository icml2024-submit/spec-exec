"""
Abstract class for SpecDec, SpecExec and whatever comes next
"""
import copy
from abc import ABC, abstractmethod

import numpy as np
import torch

from . import utils

if "logger" not in globals():
    logger = utils.get_logger()


class SpecBase(ABC):
    def __init__(self, draft_model, target_model, tokenizer):
        self.draft_model = draft_model
        self.target_model = target_model
        self.tokenizer = tokenizer
        self.device = self.draft_model.device

    def generate(self, *args, **kwargs):
        """wrapper around generator"""
        for _ in self.generate_and_yield(*args, **kwargs):
            pass
        return self.prefix_tokens

    @torch.inference_mode()
    def generate_and_yield(
        self,
        prompt,
        max_new_tokens,
        seed=0,
        keep_history=False,
        **kwargs,
    ):
        if kwargs:
            logger.debug(f"Found unused {kwargs=}")

        self.prefix_tokens = self.tokenizer.encode(prompt)
        self.original_num_tokens = len(self.prefix_tokens)

        logger.info(f"{self.__class__.__name__} starting generation.")
        logger.debug(f"Prompt: '{prompt}'")

        # models' KV cache to be persistent between iterations.
        self.target_model_outputs = None
        self.draft_model_outputs = None

        self.history = []
        self.log = []
        self.summary = {
            **kwargs,
            "draft_model_name": self.draft_model.config._name_or_path,
            "target_model_name": self.target_model.config._name_or_path,
            "prompt_len": len(self.prefix_tokens),
            "prompt_text": prompt,
            "seed": seed,
            "max_new_tokens": max_new_tokens,
        }

        self.levels = self.get_tree_levels(**kwargs)  # in case the child class works with fixed trees

        # main generation cycle
        iter = 1
        while len(self.prefix_tokens) < max_new_tokens + self.original_num_tokens:
            logger.debug(f"=====  I T E R  {iter}  ========")

            utils.set_seed(seed)
            with utils.Timing(synchronize=True) as t0:
                stats0, tree = self.grow_tree(prefix_tokens=self.prefix_tokens, **kwargs)
            if keep_history:
                self.history.append((copy.deepcopy(tree), copy.deepcopy(self.target_model_outputs)))  # DEBUG

            utils.set_seed(seed)
            with utils.Timing(synchronize=True) as t1:
                stats1, fresh_tokens = self.validate_tree(tree, **kwargs)
            logger.info(
                f"{iter:>3}.  "
                + f"Draft: {t0.elapsed:.3f}s, {stats0['tree_w']}w/{stats0['tree_h']}h/{stats0['tree_size']}size;  "
                + f"Target: {t1.elapsed:.3f}s, +{len(fresh_tokens)} tokens: {repr(self.tokenizer.decode(fresh_tokens))};  inp1:{stats1['input_len_1']}"
            )

            self.prefix_tokens.extend(fresh_tokens)
            log1 = {
                "iter": iter,
                "t0": round(t0.elapsed, 2),
                "t1": round(t1.elapsed, 2),
                "new_tokens": len(fresh_tokens),
            }
            self.log.append({**log1, **stats0, **stats1})
            iter += 1

            yield fresh_tokens

        self.summary.update(
            {
                "iters": len(self.log),
                "new_tokens": len(self.prefix_tokens) - self.original_num_tokens,
                "tree_h": round(np.mean([x.get("tree_h") for x in self.log]), 2),
                "tree_w": round(np.mean([x.get("tree_w") for x in self.log]), 2),
                "tree_size": round(np.mean([x.get("tree_size") for x in self.log]), 2),
                "t0": round(sum([x.get("t0", 0) for x in self.log]) / len(self.log), 4),
                "t1": round(sum([x.get("t1", 0) for x in self.log]) / len(self.log), 4),
                "input_0": round(sum([x.get("input_len_0", 0) for x in self.log]) / len(self.log), 1),
                "input_1": round(sum([x.get("input_len_1", 0) for x in self.log]) / len(self.log), 1),
                "lowest_cum_log_prob": round(np.mean([x.get("lowest_cum_log_prob", 0) for x in self.log]), 4),
                "draft_iters": round(np.mean([x.get("draft_iters", 0) for x in self.log]), 2),
            }
        )
        self.summary["gen_rate"] = round(self.summary["new_tokens"] / self.summary["iters"], 2)
        logger.debug(f"\nResult tokens: {self.prefix_tokens}\n string:  {repr(self.tokenizer.decode(self.prefix_tokens))}")

    @abstractmethod
    def grow_tree(self, tree, **kwargs):
        pass

    @abstractmethod
    def validate_tree(self, **kwargs):
        pass

    def get_tree_levels(self, **kwargs):
        # sets self.levels to None unless a child class overrides this method
        pass
