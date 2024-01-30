""""
SpecInfer reproduction.
n_beams sequences are generated starting form the root, with no subsequent branching
"""

import logging

import numpy as np
import torch

from . import utils
from .spec_base import SpecBase
from .trees import TreeBase
from .utils import kv_cache_mask_apply

if "logger" not in globals():
    logger = utils.get_logger()


class SpecInfer(SpecBase):
    @torch.inference_mode()
    def grow_tree(
        self,
        prefix_tokens,
        max_n_beams,
        max_beam_len,
        replacement=False,
        repack=False,
        min_log_prob=None,
        max_budget=None,
        temperature=1.0,
        top_p=1.0,
        draft_temperature=None,
        **kwargs,
    ):
        """Builds up newly created sampling tree."""

        if repack and (replacement is not True):
            raise ValueError("Non-False option `repack` requires `replacement=True`")

        draft_temperature = temperature if draft_temperature is None else draft_temperature

        logger.debug(f"=================  G R O W  {self.__class__.__name__}  ==================================================")
        logger.debug(f"prefix text: {repr(self.tokenizer.decode(prefix_tokens[-32:]))}")
        logger.trace(f"prefix tokens: {prefix_tokens}")

        input_tokens_count = 0  # for logging
        best_hypo_ids = torch.tensor([0], device=self.device)
        n_beams = 1  # = best_hypo_ids.shape[0]
        prefix_len = len(prefix_tokens)

        tree = TreeBase.from_token_ids(prefix_tokens, device=self.device)

        # for next_position in range(prefix_len, prefix_len + max_beam_len):
        next_position = prefix_len
        while True:
            logger.trace(f"Grow position: {next_position} --------------------------")
            past_kv_size = 0 if self.draft_model_outputs is None else self.draft_model_outputs.past_key_values[0][0].shape[2]

            edge_token_indices = torch.arange(tree.size - n_beams, tree.size, device=self.device)

            if next_position == prefix_len:
                # first run of the tree
                input_ids = tree.token_ids[past_kv_size:].reshape(1, -1)
                position_ids = None
                amask_draft = None
            else:
                # subsequent runs of the tree, require 4D mask and positions, assumes past_key_values present
                input_ids = tree.token_ids[edge_token_indices].reshape(1, -1)
                position_ids = torch.ones(1, n_beams, dtype=torch.long, device=self.device) * (next_position - 1)
                if amask_draft is None:
                    amask_draft = torch.ones(1, 1, 1, prefix_len, device=self.device, dtype=torch.int64)
                amask_draft = amask_draft[:, :, best_hypo_ids, :]  # reshuffle mask rows to match hypos with new token_ids
                new_amask_eye = (
                    torch.eye(n_beams, device=self.device, dtype=torch.int64).unsqueeze(0).unsqueeze(0)
                )  # create eye mask part for the new token_ids
                amask_draft = torch.cat((amask_draft, new_amask_eye), dim=3)
                # logging.debug(f"{amask_draft.shape, input_ids.shape, position_ids}")
            if (position_ids is not None) and not torch.equal(amask_draft.sum(dim=-1).flatten() - 1, position_ids.flatten()):
                logger.warn(f"positions mismatch! {amask_draft.sum(dim=-1).flatten() - 1} != {position_ids.flatten()}")

            self.draft_model_outputs = self.draft_model.forward(
                input_ids=input_ids,
                attention_mask=amask_draft,
                past_key_values=None if self.draft_model_outputs is None else self.draft_model_outputs.past_key_values,
                position_ids=position_ids,
                use_cache=True,
            )
            input_tokens_count += input_ids.shape[1]

            draft_logits = self.draft_model_outputs.logits[0, -edge_token_indices.shape[0] :, :]  # new logits appear in the end of output
            draft_logits = draft_logits / draft_temperature  # Apply temperature scaling
            draft_probs = torch.softmax(draft_logits, dim=1)

            best_child_token_ids, best_hypo_ids, best_child_probs = self.select_best_children(
                probs=draft_probs,
                beams_cum_log_probas=tree.cum_log_probs[edge_token_indices],  # incoming cumulative probabilities, shaped (cohort_size)
                max_n_beams=max_n_beams,
                replacement=replacement,
                min_log_prob=min_log_prob,
                step=next_position - prefix_len,
                **kwargs,
            )

            n_beams = best_hypo_ids.shape[0]

            # early cycle termination check; relevant with min_log_prob parameter or fixed tree structure
            if n_beams == 0:
                logging.debug(f"beams exhausted after {next_position - prefix_len} steps")
                break

            child_cum_log_probs = (tree.cum_log_probs[edge_token_indices][best_hypo_ids] + torch.log(best_child_probs)).flatten()
            best_parents_indices = edge_token_indices[best_hypo_ids]

            # Logging:
            if logger.level <= logging.DEBUG:  # the next line takes ~17-25 ms
                decoded_best_token_ids = [self.tokenizer.decode(t) for t in best_child_token_ids.flatten()]  # only used for logging
                logger.debug(
                    f"new tokens:{best_child_token_ids.tolist()}, {decoded_best_token_ids}; "
                    f"cum_log_probs:{[round(clp.item(), 1) for clp in child_cum_log_probs]}"
                )
                logger.trace(f"Tokens:{decoded_best_token_ids}; cum_log_probs:{[round(p.item(), 2) for p in child_cum_log_probs]}")

            # storing parents' q_probs
            for idx, prob in zip(range(tree.size - edge_token_indices.shape[0], tree.size), draft_probs):
                tree.q_probs[idx] = prob
            # extending tree tensors
            tree.token_ids = torch.cat((tree.token_ids, best_child_token_ids.view(-1)))
            tree.positions = torch.cat((tree.positions, torch.ones(n_beams, device=self.device, dtype=torch.int64) * next_position))
            tree.parent_indices = torch.cat((tree.parent_indices, best_parents_indices))
            tree.cum_log_probs = torch.cat((tree.cum_log_probs, child_cum_log_probs))

            next_position += 1

            if self.levels is not None:
                # limiting number of delivered tokens to the set budget for the classes with fixed trees
                if tree.size - prefix_len >= max_budget:
                    tree = tree.trimmed(prefix_len + max_budget)
                    break
            else:
                if next_position - prefix_len >= max_beam_len:
                    break

        if logger.level <= logging.DEBUG:
            # drawing beams tree
            tree.draw(tokenizer=self.tokenizer)

        # truncate Draft model's KV cache to just prompt size  # smarter truncation possible with negligible effect
        self.draft_model_outputs["past_key_values"] = kv_cache_mask_apply(self.draft_model_outputs["past_key_values"], truncate=prefix_len)

        logger.debug(f"generated: n_beams={n_beams}, n_tokens={tree.size - prefix_len}")

        if repack:
            # trees with replacement are always repacked to combine stems, matching on prefixes, into brenched trees
            tree = tree.repacked()
            logger.debug(f"repacked: n_beams={n_beams}, n_tokens={tree.size - prefix_len}")

        stats = {
            "tree_w": np.unique(tree.positions.tolist(), return_counts=True)[1].max(),
            "tree_h": tree.positions.max().item() - prefix_len + 1,
            "tree_size": tree.size - prefix_len,  # tree size net of prefix len
            "input_len_0": input_tokens_count,
        }

        return stats, tree

    @torch.inference_mode()
    def select_best_children(
        self,
        probs,
        beams_cum_log_probas,
        max_n_beams,
        replacement,
        **kwargs,
    ):
        prev_n_beams = beams_cum_log_probas.shape[-1]
        # assert new_n_beams % prev_n_beams == 0, f"n_beams expansion is possible only by whole multiple. Got {new_n_beams} -> {prev_n_beams}"
        samples_per_beam = max_n_beams // prev_n_beams

        best_hypo_ids = torch.repeat_interleave(
            torch.arange(prev_n_beams, device=probs.device, dtype=torch.int64),
            samples_per_beam,
        )
        best_child_token_ids = torch.multinomial(probs, num_samples=samples_per_beam, replacement=replacement).reshape(1, -1)

        best_child_probs = torch.gather(probs, dim=-1, index=best_child_token_ids)
        return best_child_token_ids, best_hypo_ids, best_child_probs

    @torch.inference_mode()
    def validate_tree(self, tree, temperature, top_p, **kwargs):
        """validation of the generated sequences with Target model"""
        logger.debug(f"=================  V A L I D A T E   {self.__class__.__name__}   ============================")

        initial_kv_len = self.target_model_outputs["past_key_values"][0][0].shape[2] if self.target_model_outputs is not None else 0
        prefix_len = tree.prefix_len
        amask_target = tree.build_amask_from_parent_ids(initial_kv_len)
        num_unprocessed_tokens = prefix_len - initial_kv_len

        logits_offset = prefix_len - num_unprocessed_tokens  # offset between tree index and model logits index

        input_ids = tree.token_ids[logits_offset:].unsqueeze(0)

        logger.debug(f"Target fwd inputs: {input_ids.shape=}, {initial_kv_len=}, {amask_target.shape=}")

        self.target_model_outputs = self.target_model.forward(
            input_ids=input_ids,
            attention_mask=amask_target,
            position_ids=tree.positions[initial_kv_len:].unsqueeze(0),
            past_key_values=None if self.target_model_outputs is None else self.target_model_outputs["past_key_values"],
            use_cache=True,
        )

        target_logits = self.target_model_outputs.logits.squeeze(0)  # LOGITS_COORDS
        logger.debug(f"{target_logits.shape=}, {num_unprocessed_tokens=}")

        target_softmax_probas = utils.get_sampling_probas(target_logits, temperature, top_p)  # LOGITS_COORDS

        # GOING DOWN THE TREE TO DECODE
        current_parent_index = prefix_len - 1
        fresh_token_ids = []
        cache_indices = []  # for pruning target model cache after the iteration

        extra_token_allowed = True

        for position in tree.positions[prefix_len:].unique():
            logger.trace(f"Verify position: {position} --------------------------")

            candidates_mask = torch.logical_and(tree.parent_indices == current_parent_index, tree.positions == position)  # ABS_COORDS for parent ids
            if not torch.any(candidates_mask):
                logger.debug(f"End of beam at {position=}, need to sample extra token.")
                break
            candidates_indices = torch.where(candidates_mask)[0]
            debug_line = f"{position}, candidates:{str(candidates_indices.tolist()):<12}  "

            p = target_softmax_probas[tree.parent_indices[candidates_indices[0]] - logits_offset, :]
            q = tree.q_probs[tree.parent_indices[candidates_indices[0]].item()]

            chosen_candidate_index, chosen_token_id = self._inference_step(p, q, candidates_indices, candidate_token_ids=tree.token_ids[candidates_indices])

            if chosen_candidate_index is not None:  # accepted
                fresh_token_ids.append(chosen_token_id)
                p_accept = (p[chosen_token_id] / q[chosen_token_id]).clamp_max(1.0)
                debug_line += f"accepted, p={p_accept:.3f}  "
                logger.debug(debug_line + f"chosen:{chosen_candidate_index}, token:{chosen_token_id} ({repr(self.tokenizer.decode(chosen_token_id))})")
                cache_indices.append(chosen_candidate_index.item())
                current_parent_index = chosen_candidate_index

            else:  # rejected
                assert chosen_token_id not in tree.token_ids[candidates_indices]
                fresh_token_ids.append(chosen_token_id)
                debug_line += f"sampled from p_adj: token={chosen_token_id} {repr(self.tokenizer.decode(fresh_token_ids[-1]))}; "
                logger.debug(debug_line + "It was from outside of the tree.")
                extra_token_allowed = False
                break  # exit the "for position_id" loop

        if extra_token_allowed:
            # adding extra token in the end if previous was sampled from the tree (otherwise see break above)
            extra_token_pos_in_target = chosen_candidate_index - logits_offset
            sampled_token_id = torch.multinomial(target_softmax_probas[extra_token_pos_in_target], num_samples=1).item()

            logger.debug(f"Extra token after complete beam: {sampled_token_id} ({repr(self.tokenizer.decode(sampled_token_id))})")
            fresh_token_ids.append(sampled_token_id)

        self._prune_target_model_kv_cache(cache_indices, prefix_len)

        logger.debug(f"sampled {len(fresh_token_ids)} tokens: {fresh_token_ids} {repr(self.tokenizer.decode(fresh_token_ids))}")
        stats = {"input_len_1": input_ids.shape[-1], "cache_len_1": logits_offset, "accepted_tokens": len(fresh_token_ids)}

        return stats, fresh_token_ids

    @staticmethod
    @torch.no_grad()
    def _inference_step(p, q, candidate_idxs, candidate_token_ids) -> (int, int):
        """return chosen candidate_id, chosen candidate_token_id. If chosen_candidate_id is None, all cands were rejected"""
        p_adj = p.clone()
        q = q.clone()

        chosen_token_id = None
        for candidate_idx, candidate_token_id in zip(candidate_idxs, candidate_token_ids):
            p_adj_i = p_adj[candidate_token_id]
            q_i = q[candidate_token_id]

            r = torch.rand(1, device=p.device)
            p_accept = (p_adj_i / q_i).clamp_max(1.0)
            if r <= p_accept:
                chosen_token_id = candidate_token_id
                return candidate_idx, chosen_token_id.item()
            else:
                p_adj = (p_adj - q).clip(min=0)
                p_adj = torch.nn.functional.normalize(p_adj, dim=0, p=1)

                q[candidate_token_id] = 0
                q = torch.nn.functional.normalize(q, dim=0, p=1)

        assert chosen_token_id is None  # we did not accept any token => sample from p_adj
        try:
            chosen_token_id = torch.multinomial(p_adj, num_samples=1)
        except RuntimeError:
            raise RuntimeError("problem with p_adj!" f"{sum(p_adj > 0).item()}, {candidate_idx, candidate_idxs, candidate_token_ids}")
        # note that at this point, chosen_token_idx can never be in tree
        return None, chosen_token_id.item()

    @torch.no_grad()
    def _prune_target_model_kv_cache(self, cache_indices, prefix_len):
        """retains on KV cache only elements related to the prefix and to the selected tokens"""
        target_cache_keep_mask = torch.cat((torch.arange(prefix_len), torch.tensor(cache_indices))).int().to(self.device)
        self.target_model_outputs["past_key_values"] = kv_cache_mask_apply(self.target_model_outputs["past_key_values"], mask=target_cache_keep_mask)
        kv_len = self.target_model_outputs["past_key_values"][0][0].shape[2]
        logger.trace(f"Pruned target KV cache len={kv_len}, mask={target_cache_keep_mask.int().tolist()}")
