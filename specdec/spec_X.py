"""
SpecExec, version 2
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


class SpecExec(SpecBase):
    @torch.inference_mode()
    def get_next_beams(self, logprobs, beam_scores, num_beams=None, min_log_prob=None, **kwargs):
        """
        produces up to num_beams top beams by cumulative log_prob
        with log_prob >= min_log_prob limit

        """
        flat_log_probs = (beam_scores.unsqueeze(-1) + logprobs).flatten()
        flat_best = flat_log_probs.topk(k=num_beams, largest=True)

        if min_log_prob is not None:
            flat_best_mask = torch.where(flat_best.values > min_log_prob)[0]
            flat_best_values = flat_best.values[flat_best_mask]
            flat_best_indices = flat_best.indices[flat_best_mask]
        else:
            flat_best_values = flat_best.values
            flat_best_indices = flat_best.indices

        best_hypo_ids = flat_best_indices // len(self.tokenizer)
        best_child_token_ids = flat_best_indices % len(self.tokenizer)

        return best_child_token_ids, best_hypo_ids, flat_best_values

    @torch.inference_mode()
    def grow_tree(self, prefix_tokens, max_n_beams, max_beam_len, min_log_prob=None, **kwargs):
        """
        Creates and grows tree.
        input: prefix_tokens
        """
        logger.debug(f"=================  G R O W  {self.__class__.__name__}  ==================================================")

        logger.debug(f"prefix text: {repr(self.tokenizer.decode(prefix_tokens[-32:]))}")
        logger.trace(f"prefix tokens: {prefix_tokens}")
        input_tokens_count = 0  # for logging
        n_beams = 1

        prefix_len = len(prefix_tokens)
        past_kv_size = 0 if self.draft_model_outputs is None else self.draft_model_outputs.past_key_values[0][0].shape[2]

        tree = TreeBase.from_token_ids(prefix_tokens, device=self.device)

        # TODO: remove this extra draft_model run, make it like in SD.
        self.draft_model_outputs = self.draft_model.forward(
            input_ids=tree.token_ids[past_kv_size:].unsqueeze(0),
            past_key_values=self.draft_model_outputs.past_key_values if self.draft_model_outputs is not None else None,
        )

        amask_draft = torch.ones((1, 1, n_beams, prefix_len), dtype=torch.int64, device=self.device)
        draft_logits = self.draft_model_outputs.logits[:, -1, :]  # handled differently in init phase

        for next_position in range(prefix_len, prefix_len + max_beam_len):
            logger.trace(f"Grow position: {next_position} --------------------------")
            edge_token_indices = torch.arange(tree.size - n_beams, tree.size, device=self.device)
            logprobs = torch.log_softmax(draft_logits, dim=-1)  # shape: [n_beams, voc_size]
            best_child_token_ids, best_hypo_ids, cum_beam_scores = self.get_next_beams(
                logprobs, beam_scores=tree.cum_log_probs[-n_beams:], num_beams=max_n_beams, min_log_prob=min_log_prob
            )
            n_beams = best_hypo_ids.shape[-1]
            if n_beams == 0:
                logging.info(f"beams exhausted after {next_position - prefix_len} steps")
                break

            if logger.level <= logging.DEBUG:
                logger.debug(
                    f"pos {next_position}:  best_hypos:{best_hypo_ids.tolist()}, tokens:{best_child_token_ids.tolist()} "
                    f"{[self.tokenizer.decode(t) for t in best_child_token_ids]}"
                )

            # extending tree tensors
            tree.token_ids = torch.cat((tree.token_ids, best_child_token_ids))
            tree.positions = torch.cat((tree.positions, torch.ones(n_beams, device=self.device, dtype=torch.int64) * next_position))
            tree.parent_indices = torch.cat((tree.parent_indices, edge_token_indices[best_hypo_ids]))
            tree.cum_log_probs = torch.cat((tree.cum_log_probs, cum_beam_scores))

            amask_draft = amask_draft[:, :, best_hypo_ids, :]  # reshuffle mask rows to match hypos with new token_ids
            new_amask_eye = torch.eye(best_hypo_ids.shape[-1], device=self.device).unsqueeze(0).unsqueeze(0)  # create eye mask part for the new token_ids
            amask_draft = torch.cat((amask_draft, new_amask_eye), dim=3)

            # generating next set of candidates with draft_model
            # logging.debug(f"{amask_draft.shape, best_child_token_ids.reshape(1, -1).shape, n_beams, next_position}")
            position_ids = torch.ones(1, n_beams, dtype=torch.long, device=self.device) * next_position

            if (position_ids is not None) and not torch.equal(amask_draft.sum(dim=-1).flatten() - 1, position_ids.flatten()):
                logger.warn(f"positions mismatch! {amask_draft.sum(dim=-1).flatten() - 1} != {position_ids.flatten()}")

            self.draft_model_outputs = self.draft_model.forward(
                input_ids=best_child_token_ids.reshape(1, -1),
                attention_mask=amask_draft,
                past_key_values=None if self.draft_model_outputs is None else self.draft_model_outputs.past_key_values,
                position_ids=position_ids,
                use_cache=True,
            )
            draft_logits = self.draft_model_outputs.logits[0, :, :]
            input_tokens_count += best_child_token_ids.numel()

        if logger.level <= logging.DEBUG:
            tree.draw(tokenizer=self.tokenizer)

        # truncate draft model's KV cache to just prompt size  # smarter truncation possible with negligible effect
        self.draft_model_outputs["past_key_values"] = kv_cache_mask_apply(self.draft_model_outputs["past_key_values"], truncate=prefix_len)

        logger.debug(f"generated: n_beams={n_beams}, n_tokens={tree.size - prefix_len}")

        stats = {
            "tree_w": np.unique(tree.positions.tolist(), return_counts=True)[1].max(),
            "tree_h": tree.positions.max().item() - prefix_len + 1,
            "tree_size": tree.size - prefix_len,  # tree size net of prefix len
            "input_len_0": input_tokens_count,
            "draft_iters": next_position - prefix_len + 1,
            "lowest_cum_log_prob": round(tree.cum_log_probs.min().item(), 4),
        }

        return stats, tree

    @torch.inference_mode()
    def validate_tree(self, tree: TreeBase, temperature=1.0, top_p=1.0, **kwargs):
        """validation of the generated sequences with Target model"""
        logger.debug(f"=================  V A L I D A T E   {self.__class__.__name__}   ============================")

        initial_kv_len = self.target_model_outputs["past_key_values"][0][0].shape[2] if self.target_model_outputs is not None else 0
        prefix_len = len(self.prefix_tokens)
        amask_target = tree.build_amask_from_parent_ids(initial_kv_len)
        num_unprocessed_tokens = prefix_len - initial_kv_len

        logits_offset = prefix_len - num_unprocessed_tokens  # offset between tree index and model logits index

        input_ids = tree.token_ids[logits_offset:].unsqueeze(0)

        logger.debug(f"target fwd inputs: {input_ids.shape=}, {initial_kv_len=}, {amask_target.shape=}")

        self.target_model_outputs = self.target_model.forward(
            input_ids=input_ids,
            attention_mask=amask_target,
            position_ids=tree.positions[initial_kv_len:].unsqueeze(0),
            past_key_values=None if self.target_model_outputs is None else self.target_model_outputs["past_key_values"],
            use_cache=True,
        )

        target_logits = self.target_model_outputs.logits.squeeze(0)  # LOGITS_COORDS
        logger.debug(f"{target_logits.shape=}, {num_unprocessed_tokens=}")

        all_target_token_choices, _ = self.sampler_from_logits(logits=target_logits, temperature=temperature, top_p=top_p)

        # building accept flags based on matching tree proposls with target model choices
        logits_offset = prefix_len - num_unprocessed_tokens
        new_token_positions = torch.arange(prefix_len, tree.size)
        target_old_pos_ids = tree.parent_indices[new_token_positions] - logits_offset

        draft_token_choices = tree.token_ids[prefix_len:]
        target_token_choices = all_target_token_choices[target_old_pos_ids]
        accept_flags = torch.cat(
            (
                torch.ones(prefix_len, device=self.device, dtype=torch.bool),
                (draft_token_choices == target_token_choices),
            )
        )

        # TODO: Consider using torch with product of target_mask and accept_flags (to replace the next block)
        # ALT TORCH CODE  - NEEDS DEBUG, ERRORS IN DETECTING GAPS IN ACCEPTED SEQUENCES !!!
        # masked_target1 = target_mask[0, 0, :, :].cpu() * accept_flags.unsqueeze(0)
        # seq_lengths = masked_target1.sum(axis=1)
        # best_sequence_index = seq_lengths.argmax()
        # best_sequence_length = seq_lengths.max()
        # best_sequence_mask = masked_target1[best_sequence_index].bool()

        # using masked array with map of accepted tokens to extract winning sequences.
        masked_target = np.ma.masked_array(
            data=torch.tile(accept_flags, (amask_target.shape[-2], 1)).cpu().numpy().astype(int),
            mask=(amask_target[0, 0, :, :].cpu().numpy() == 0),
        )
        accepted_cumprod = np.cumprod(masked_target, axis=1)
        num_accepts = accepted_cumprod.sum(axis=1).data
        best_sequence_index = num_accepts.argmax()
        best_sequence_length = num_accepts[best_sequence_index]
        logger.debug(f"accepted {best_sequence_length - prefix_len} tokens. seq {best_sequence_index}")

        # preparing mask for pruning
        best_sequence_mask = np.logical_not(masked_target.mask[best_sequence_index])
        last_accepted_token_position = np.where(best_sequence_mask)[0][-1]
        best_sequence_mask[last_accepted_token_position + 1 :] = 0

        fresh_token_ids = tree.token_ids[best_sequence_mask][prefix_len:].tolist()

        # Generate one last token based on target model logits
        if logger.level < logging.DEBUG:  # trace
            logger.trace(f"{[(i, self.tokenizer.decode(t)) for i, t in enumerate(all_target_token_choices)]}")
            logger.trace(f"{last_accepted_token_position=}, {logits_offset=}")

        next_token_id = all_target_token_choices[last_accepted_token_position - logits_offset].item()
        logger.debug(f"{next_token_id=}, '{self.tokenizer.decode(next_token_id)}'")
        fresh_token_ids.append(next_token_id)

        self._prune_target_model_kv_cache_with_mask(best_sequence_mask)

        logger.debug(f"sampled {len(fresh_token_ids)} tokens: {fresh_token_ids} {repr(self.tokenizer.decode(fresh_token_ids))}")
        stats = {"input_len_1": input_ids.shape[-1], "cache_len_1": logits_offset, "accepted_tokens": len(fresh_token_ids)}

        return stats, fresh_token_ids

    @staticmethod
    @torch.inference_mode()
    def sampler_from_logits(logits, temperature=1.0, top_p=0.9, min_tokens_to_keep=1):
        """
        Performs token sampling from logits using top-p (nucleus) sampling or deterministic selection.
        Args:
            logits (torch.Tensor): Logits from a language model.
            temperature (float): Adjusts distribution sharpness (higher = more random);  0 for greedy.
            top_p (float): Cumulative probability threshold for top-p sampling.
            min_tokens_to_keep (int): Minimum tokens to keep regardless of top_p.
        Returns: Tuple[torch.Tensor, torch.Tensor]: Indices and log probabilities of selected tokens.
        """
        scores = logits / temperature  # Apply temperature scaling

        if temperature > 0:
            # Sort scores in descending order for top-p sampling
            sorted_logits, sorted_indices = torch.sort(scores, descending=True)
            cumulative_probs = torch.softmax(sorted_logits, dim=-1).cumsum(dim=-1)

            # Create a mask to remove logits not in the top-p
            sorted_indices_to_remove = cumulative_probs <= (1 - top_p)
            sorted_indices_to_remove[..., -min_tokens_to_keep:] = 0  # Keep at least min_tokens_to_keep tokens

            # Scatter the indices to the original order and mask the logits
            indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
            scores.masked_fill_(indices_to_remove, -float("inf"))

            # Sampling from the filtered logits
            probs = torch.softmax(scores, dim=-1)
            selection = torch.multinomial(probs, 1)[:, 0]
        else:
            # Greedy selection
            selection = torch.argmax(scores, dim=-1)

        # Compute log probabilities
        logprobs = torch.log_softmax(scores, dim=-1)
        logprobs_1 = torch.index_select(logprobs, 1, selection).diag()

        return selection.to(logits.device), logprobs_1.to(logits.device)

    def _prune_target_model_kv_cache_with_mask(self, best_sequence_mask):
        self.target_model_outputs["past_key_values"] = kv_cache_mask_apply(
            self.target_model_outputs["past_key_values"], mask=torch.from_numpy(best_sequence_mask)
        )
        kv_len = self.target_model_outputs["past_key_values"][0][0].shape[2]
        logger.debug(f"Pruned target KV cache len={kv_len}")
