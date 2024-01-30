"""
Classes for trees. Version 2.1
"""
from dataclasses import dataclass
from typing import Dict

import torch

HIGH_NEGATIVE = -100500


@dataclass
class TreeBase:
    token_ids: torch.tensor  # int
    positions: torch.tensor  # int
    parent_indices: torch.tensor  # int
    cum_log_probs: torch.tensor  # float
    log_probs: torch.tensor  # float
    q_probs: Dict[int, torch.tensor]  # used by SpecInfer and related methods

    @property
    def size(self):
        return self.token_ids.shape[-1]

    def get_beam(self, idx, min_idx=-1):
        # get token id sequence from tree, ending with idx
        beam = []
        while idx > min_idx:
            beam.append(self.token_ids[idx].item())
            idx = self.parent_indices[idx]
        return beam[::-1]

    def draw(self, tokenizer=None, start=None):
        from anytree import Node, RenderTree

        if self.size == 0:
            return "[ ! empty_tree ! ]"
        if start is None:
            if self.parent_indices.max() == self.positions.max() - 1:  # tree is just a trunk
                start = 0
            else:
                start = self.parent_indices[self.parent_indices != torch.arange(self.parent_indices.shape[-1], device=self.parent_indices.device) - 1][0]
        parents = self.parent_indices[start:] - start
        tokens = self.token_ids[start:]

        if tokenizer is not None:
            tokens = [repr(tokenizer.decode(t)).strip("'") for t in tokens]
        else:
            tokens = [str(t.item()) for t in tokens]

        nodes = [None] * len(tokens)

        # Iterate through the tokens and parents to build the tree
        for i, (token, parent_idx) in enumerate(zip(tokens, parents)):
            if parent_idx == -1:
                # Root node
                nodes[i] = Node(token)
            else:
                # Non-root node
                nodes[i] = Node(token, parent=nodes[parent_idx])

        # Print the tree
        for pre, fill, node in RenderTree(nodes[0]):
            print(f"{pre}{node.name}")

    def trimmed(self, lim):
        """
        return copy of self trimmed to size_limit size
        """
        return self.__class__(
            token_ids=self.token_ids[:lim],
            positions=self.positions[:lim],
            parent_indices=self.parent_indices[:lim],
            cum_log_probs=self.cum_log_probs[:lim],
            log_probs=self.log_probs[:lim],
            q_probs={k: v for k, v in self.q_probs.items() if k < lim},
        )

    @classmethod
    def from_token_ids(cls, prefix_tokens, device):
        prefix_len = len(prefix_tokens)
        tree = cls(
            token_ids=torch.tensor(prefix_tokens, device=device),
            positions=torch.arange(prefix_len, device=device),
            parent_indices=torch.arange(prefix_len, device=device) - 1,
            cum_log_probs=torch.zeros(prefix_len, device=device, dtype=torch.float32),
            log_probs=torch.zeros(prefix_len, device=device, dtype=torch.float32),
            q_probs={},
        )
        tree.prefix_len = prefix_len
        return tree

    @classmethod
    def from_stems(cls, stems, device):
        node_indices = {}
        combined_values = []
        combined_parents = []
        positions = []

        current_index = 0

        for stem in stems:
            parent_index = -1  # start with no parent
            node_path = ""  # Track the path to handle repeated values
            depth = 0  # depth of the current node

            for node in stem:
                node_path += str(node) + ","

                if node_path not in node_indices:
                    # Add new node
                    node_indices[node_path] = current_index
                    combined_values.append(node)
                    combined_parents.append(parent_index)
                    positions.append(depth)

                    parent_index = current_index
                    current_index += 1
                else:
                    # Node already exists, update the parent_index
                    parent_index = node_indices[node_path]
                depth += 1

        # Convert lists to tensors
        token_ids = torch.tensor(combined_values, device=device)
        parent_indices = torch.tensor(combined_parents, device=device)
        positions = torch.tensor(positions, device=device, dtype=torch.int64)
        cum_log_probs = torch.zeros(len(combined_values), device=device, dtype=torch.float32)
        log_probs = torch.zeros(len(combined_values), device=device, dtype=torch.float32)
        q_probs = {}
        return cls(token_ids, positions, parent_indices, cum_log_probs, log_probs, q_probs)

    def repacked(self):
        """
        combines coinciding nodes
        returns new tree
        not memory-efficient
        """

        prefix_len = self.get_prefix_len()

        # storage for the new tree components
        node_indices = {}
        token_ids = [*self.token_ids[:prefix_len].tolist()]
        parent_indices = [*self.parent_indices[:prefix_len].tolist()]
        positions = [*self.positions[:prefix_len].tolist()]
        cum_log_probs = [*self.cum_log_probs[:prefix_len].tolist()]
        log_probs = [*self.log_probs[:prefix_len].tolist()]
        q_probs = {k: v for k, v in self.q_probs.items() if k < prefix_len}

        self.root_index = prefix_len - 1
        current_index = self.root_index + 1

        for i in range(self.root_index + 1, self.size):
            node_path = self.get_beam(i, min_idx=self.root_index)
            node_path = tuple(node_path.tolist())
            # trace_line = (f"{i}, {node_path}, {tokenizer.decode(node_path)}")

            if node_path not in node_indices:
                # Add new node
                node_indices[node_path] = current_index
                token_ids.append(self.token_ids[i])

                parent_path = node_path[:-1]
                parent_index = node_indices.get(parent_path, self.root_index)
                parent_indices.append(parent_index)

                # copy other fields
                positions.append(self.positions[i])
                cum_log_probs.append(self.cum_log_probs[i])
                try:
                    log_probs.append(self.log_probs[i])
                except IndexError:
                    # may not exist for SI-related self.
                    pass
                q_probs[current_index] = self.q_probs.get(i)

                # trace_line += (f"  :{current_index}  par={parent_path}   {parent_index}")
                current_index += 1
            else:
                pass

        new_tree = self.__class__(
            token_ids=torch.tensor(token_ids, device=self.token_ids.device),
            positions=torch.tensor(positions, device=self.token_ids.device),
            parent_indices=torch.tensor(parent_indices, device=self.token_ids.device),
            cum_log_probs=torch.tensor(cum_log_probs, device=self.token_ids.device),
            log_probs=torch.tensor(log_probs, device=self.token_ids.device),
            q_probs=q_probs,
        )

        return new_tree

    def trim_budget(self, budget, prefix_len):
        if budget + prefix_len >= self.size:
            return self  # consider rerturning copy

        device = self.token_ids.device
        topk_indices = torch.topk(self.cum_log_probs[prefix_len:], budget).indices
        topk_indices = torch.sort(topk_indices).values

        # helper indices
        tree_index = torch.arange(self.size, device=device)
        old_index = torch.cat([torch.arange(prefix_len, device=device), tree_index[topk_indices + prefix_len]])
        new_index = torch.arange(old_index.shape[-1], device=device)

        # helper index for parents indices conversion
        interim_index = torch.ones_like(self.token_ids, device=device) * -99
        interim_index[old_index] = new_index
        new_parents = interim_index[self.parent_indices][old_index]
        new_parents[torch.where(self.parent_indices == -1)[0]] = -1

        tree2 = self.__class__(
            token_ids=self.token_ids[old_index],
            positions=self.positions[old_index],
            parent_indices=new_parents,
            cum_log_probs=self.cum_log_probs[old_index],
            log_probs=self.log_probs[old_index] if len(self.log_probs) > max(old_index) else self.log_probs,
            q_probs={i: self.q_probs[oi] for i, oi in enumerate(old_index)} if self.q_probs else {},
        )

        return tree2

    @torch.no_grad()
    def build_amask_from_parent_ids(self, start=0):
        """
        building 4D mask
        parent_indices parents indices.
        start: first input_ids token index
        assumes that all elements before start are ones.
        """

        # build the rectangular part filled with ones:
        start_mask = torch.ones(self.parent_indices.shape[-1] - start, start, dtype=torch.int64, device=self.parent_indices.device)

        # shift the parents indices to the left
        parent_idxs = (self.parent_indices - start)[start:]

        square_mask = torch.eye(parent_idxs.shape[0], dtype=torch.int64, device=parent_idxs.device)

        for i in range(parent_idxs.shape[-1] - 1, -1, -1):
            parent_idx = parent_idxs[i]
            if parent_idx >= 0:
                related_rows = torch.where(square_mask[:, i] > 0)[0]
                square_mask[related_rows, parent_idx] = 1

        return torch.cat((start_mask, square_mask), dim=-1).unsqueeze(0).unsqueeze(0)


@dataclass
class TopKHeap:
    def __init__(self, max_budget, tree, device=torch.device(0)):
        self.max_budget = max_budget
        self.tree = tree
        self.size = 0  # cursor position
        self.token_ids = torch.empty(max_budget * 2, dtype=torch.int64, device=device)
        self.parent_indices = torch.empty(max_budget * 2, dtype=torch.int64, device=device)
        self.cum_log_probs = torch.ones(max_budget * 2, dtype=torch.float32, device=device) * -100500
        self.best_min_prob = HIGH_NEGATIVE

    @torch.no_grad()
    def update(self, parent_indices, logprobs, max_branch_width=4, **kwargs):
        """
        adds to heap max_branch_width elements from each parent based on the greatest logprobs.
        """
        top_children = logprobs.topk(k=max_branch_width, dim=-1, sorted=False)  # incoming token logprobs
        parent_probs = self.tree.cum_log_probs[parent_indices]  # parents' cum log probs
        new_probs = (parent_probs.expand(max_branch_width, -1).T + top_children.values).flatten()  # resulting cum_log_probs for candidates

        # assumes self.best_min_prob updated at previous iteration
        if new_probs.max() < self.best_min_prob:  # tree+heap have over max_budget tokens better than any of new ones
            return True

        k = min(self.max_budget, new_probs.numel())

        new_token_ids = top_children.indices.view(-1)  # all new tokens before filtering

        new_top = new_probs.topk(k, sorted=False)  # k best cum_log_probs
        new_token_ids = new_token_ids[new_top.indices]  # shape k
        new_parent_indices = parent_indices[new_top.indices // max_branch_width]  # shape k
        new_cum_log_probs = new_top.values

        self.token_ids[self.size : self.size + k] = new_token_ids
        self.parent_indices[self.size : self.size + k] = new_parent_indices
        self.cum_log_probs[self.size : self.size + k] = new_cum_log_probs
        self.size += k

        self.best_min_prob = self.get_best_min_prob()
        self.smart_trim()

        return False

    def get_best_min_prob(self):
        # gets prob limit based on tree and heap together. returns HIGH_NEGATIVE if budget not reached.
        if self.size + self.tree.size - self.tree.prefix_len < self.max_budget:
            return HIGH_NEGATIVE
        else:
            all_probs = torch.cat((self.tree.cum_log_probs[self.tree.prefix_len :], self.cum_log_probs[: self.size]))
            min_prob = all_probs.topk(self.max_budget).values.min()
            return min_prob

    def smart_trim(self):
        # trims heap to keep self.max_budget best tokens (counting together wiht the tree)
        # assumes fresh self.best_min_prob value

        if self.best_min_prob > HIGH_NEGATIVE:
            keep_index = self.cum_log_probs[: self.size] >= self.best_min_prob
            new_size = keep_index.sum()
            self.token_ids[:new_size] = self.token_ids[: self.size][keep_index]
            self.parent_indices[:new_size] = self.parent_indices[: self.size][keep_index]
            self.cum_log_probs[:new_size] = self.cum_log_probs[: self.size][keep_index]
            self.size = new_size

    @torch.no_grad()
    def get_top_k(self, k):
        # returns k best (token_ids, parents, cum probas)
        k = min(k, self.size)

        if k == 0:
            return None, None, None

        best_index = self.cum_log_probs[: self.size].topk(k=k, sorted=False).indices

        best_token_ids = self.token_ids[best_index]
        best_parent_indices = self.parent_indices[best_index]
        best_cum_log_probs = self.cum_log_probs[best_index]

        keep_index = torch.ones(self.size, device=self.token_ids.device, dtype=torch.bool)
        keep_index[best_index] = False
        old_size = self.size
        self.size = self.size - k
        if self.size:
            self.token_ids[: self.size] = self.token_ids[:old_size][keep_index]
            self.parent_indices[: self.size] = self.parent_indices[:old_size][keep_index]
            self.cum_log_probs[: self.size] = self.cum_log_probs[:old_size][keep_index]

        return best_token_ids, best_parent_indices, best_cum_log_probs
