"""
SpecExec, version 2
"""
import logging

import numpy as np
import torch
import torch.nn.functional as F

from . import utils
from .spec_X import SpecExec
from .trees import TreeBase, TopKHeap
from .utils import kv_cache_mask_apply

if "logger" not in globals():
    logger = utils.get_logger()


class SpecExec2(SpecExec):
    @torch.inference_mode()
    def grow_tree(self, prefix_tokens, max_budget, max_beam_len, max_n_beams=32, max_branch_width=16, **kwargs):
        """grows speculative tree

        Args:
            prefix_tokens (_type_): incoming tokenized prefix
            max_tokens (_type_): maximum tree size in tokens
            max_beam_length (_type_): number of growth iterations
            max_n_beams (int, optional): number of tokens considered at each iteration
            max_branch_width (int, optional): max number of children per branch.

        Returns:
            statistics and tree
        """

        logger.debug(f"=================  G R O W  {self.__class__.__name__}  ==================================================")

        logger.debug(f"prefix text: {repr(self.tokenizer.decode(prefix_tokens[-32:]))}")
        logger.trace(f"prefix tokens: {prefix_tokens}")
        input_tokens_count = []  # for logging
        n_beams = 1

        prefix_len = len(prefix_tokens)
        past_kv_size = 0 if self.draft_model_outputs is None else self.draft_model_outputs.past_key_values[0][0].shape[2]

        tree = TreeBase.from_token_ids(prefix_tokens, device=self.device)

        heap = TopKHeap(max_budget=max_budget, tree=tree)

        # TODO: remove this extra draft_model run, make it like in SD.
        self.draft_model_outputs = self.draft_model.forward(
            input_ids=tree.token_ids[past_kv_size:].unsqueeze(0),
            past_key_values=self.draft_model_outputs.past_key_values if self.draft_model_outputs is not None else None,
        )

        amask_draft = torch.ones((1, 1, n_beams, prefix_len), dtype=torch.int64, device=self.device)
        draft_logits = self.draft_model_outputs.logits[:, -1, :]  # handled differently in init phase

        for draft_iter in range(max_beam_len):  # practically unlimited cycle
            logger.debug(f"Grow iteration: {draft_iter} --------------------------")
            parent_indices = torch.arange(tree.size - n_beams, tree.size, device=self.device)  # last n_beams tokens of the tree

            logprobs = torch.log_softmax(draft_logits, dim=-1)  # shape: [n_beams, voc_size]

            heap_complete_flag = heap.update(parent_indices=parent_indices, logprobs=logprobs, max_branch_width=max_budget)

            if heap_complete_flag:
                max_n_beams = max_budget - (tree.size - prefix_len)

            if max_n_beams > 0:
                best_child_token_ids, best_parent_indices, cum_beam_scores = heap.get_top_k(k=max_n_beams)
                n_beams = best_child_token_ids.shape[-1]
            else:
                n_beams = 0

            if n_beams == 0:
                logging.debug(f"beams exhausted after {draft_iter} steps")
                break

            if logger.level <= logging.DEBUG:
                logger.trace(
                    f"{draft_iter=}:  best_parents:{best_parent_indices.tolist()}, tokens:{best_child_token_ids.tolist()} "
                    f"{[self.tokenizer.decode(t) for t in best_child_token_ids]}"
                )

            # extending tree tensors
            tree.token_ids = torch.cat((tree.token_ids, best_child_token_ids))
            tree.positions = torch.cat((tree.positions, tree.positions[best_parent_indices] + 1))
            tree.parent_indices = torch.cat((tree.parent_indices, best_parent_indices))
            tree.cum_log_probs = torch.cat((tree.cum_log_probs, cum_beam_scores))

            if heap_complete_flag:
                # only reachable if max_n_beams > 0, picking remaining tokens from the heap.
                logging.debug(f"Early stopping: heap and tree complete after {draft_iter + 1} iterations")
                break

            amask_addition = amask_draft[:, :, best_parent_indices - prefix_len + 1, :]  # reshuffle mask rows to match hypos with new token_ids
            new_amask_eye = (
                torch.eye(best_child_token_ids.shape[-1], device=self.device, dtype=torch.int64).unsqueeze(0).unsqueeze(0)
            )  # create eye mask part for the new token_ids
            amask_addition = torch.cat((amask_addition, new_amask_eye), dim=3)  # build new amask rows
            amask_draft = F.pad(amask_draft, (0, n_beams))  # extend the original rows to match the new rows' length

            amask_draft = torch.cat([amask_draft, amask_addition], dim=2)  # concat with the new rows

            # generating next set of candidates with draft_model
            position_ids = amask_addition.squeeze(0).sum(dim=-1) - 1
            if (position_ids is not None) and not torch.equal(amask_addition.sum(dim=-1).flatten() - 1, position_ids.flatten()):
                logger.warn(f"positions mismatch! {amask_draft.sum(dim=-1).flatten() - 1} != {position_ids.flatten()}")

            pos_counts = torch.unique(position_ids, return_counts=True)
            logger.debug(f"heap_size:{heap.size}  positions: {(pos_counts[0] - prefix_len).tolist()}, {pos_counts[1].tolist()}")  # token positions counts
            past_key_values = None if self.draft_model_outputs is None else self.draft_model_outputs.past_key_values

            self.draft_model_outputs = self.draft_model.forward(
                input_ids=best_child_token_ids.reshape(1, -1),
                attention_mask=amask_addition,
                past_key_values=past_key_values,
                position_ids=position_ids,
                use_cache=True,
            )
            draft_logits = self.draft_model_outputs.logits[0, :, :]
            input_tokens_count.append(best_child_token_ids.numel())

        if heap.size > 0:
            # heap.best_min_prob = heap.get_best_min_prob()
            # heap.smart_trim()
            tree.token_ids = torch.cat((tree.token_ids, heap.token_ids[: heap.size]))
            # tree.positions = torch.cat((tree.positions, heap.positions[best_parent_indices] + 1))
            tree.parent_indices = torch.cat((tree.parent_indices, heap.parent_indices[: heap.size]))
            tree.cum_log_probs = torch.cat((tree.cum_log_probs, heap.cum_log_probs[: heap.size]))

        logging.debug(f"before trim: {tree.size=}")
        if tree.size - prefix_len >= max_budget:
            tree = tree.trim_budget(max_budget, prefix_len)
        logging.debug(f"after trim {tree.size}")

        if logger.level <= logging.TRACE:
            tree.draw(tokenizer=self.tokenizer)  # drawing beams tree
        # truncate draft model's KV cache to just prompt size  # smarter truncation possible with negligible effect
        self.draft_model_outputs["past_key_values"] = kv_cache_mask_apply(self.draft_model_outputs["past_key_values"], truncate=prefix_len)

        logger.debug(f"generated: n_beams={n_beams}, n_tokens={tree.size - prefix_len}")

        stats = {
            "tree_w": np.unique(tree.positions.tolist(), return_counts=True)[1].max(),
            "tree_h": tree.positions.max().item() - prefix_len + 1,
            "tree_size": tree.size - prefix_len,  # tree size net of prefix len
            "input_len_0": sum(input_tokens_count),
            "draft_iters": draft_iter + 1,
            "lowest_cum_log_prob": round(tree.cum_log_probs.min().item(), 4),
        }
        logger.debug(f"input_tokens_count: {sum(input_tokens_count)}, {input_tokens_count}")
        logger.debug(f"tree layer sizes: {torch.unique(tree.positions[prefix_len:], return_counts=True)[1].tolist()}")  # Tree nodes counts by level
        logger.info(f"{stats}")

        return stats, tree
