from __future__ import annotations

import itertools
from typing import Optional

import networkx as nx


class CausalDAG:
    def __init__(self) -> None:
        self.graph: nx.DiGraph = nx.DiGraph()

    def add_variable(self, name: str) -> "CausalDAG":
        self.graph.add_node(name)
        return self

    def add_edge(self, cause: str, effect: str) -> "CausalDAG":
        self.graph.add_edge(cause, effect)
        if not nx.is_directed_acyclic_graph(self.graph):
            self.graph.remove_edge(cause, effect)
            raise ValueError(f"Adding {cause} -> {effect} creates a cycle.")
        return self


    def parents(self, node: str) -> set[str]:
        return set(self.graph.predecessors(node))

    def children(self, node: str) -> set[str]:
        return set(self.graph.successors(node))

    def ancestors(self, node: str) -> set[str]:
        return nx.ancestors(self.graph, node)

    def descendants(self, node: str) -> set[str]:
        return nx.descendants(self.graph, node)

    def nodes(self) -> list[str]:
        return list(self.graph.nodes)

    def edges(self) -> list[tuple[str, str]]:
        return list(self.graph.edges)


    def is_d_separated(self, x: str, y: str, given: set[str]) -> bool:
        return nx.is_d_separator(self.graph, {x}, {y}, given)

    def _blocks_all_backdoor_paths(self, treatment: str, outcome: str,
                                   adjustment_set: set[str]) -> bool:
        desc_x = self.descendants(treatment)
        if adjustment_set & desc_x:
            return False
        mutilated = self.graph.copy()
        for child in list(mutilated.successors(treatment)):
            mutilated.remove_edge(treatment, child)

        return nx.is_d_separator(mutilated, {treatment}, {outcome}, adjustment_set)

    def find_backdoor_adjustment_sets(
        self,
        treatment: str,
        outcome: str,
        max_set_size: Optional[int] = None,
    ) -> list[set[str]]:

        candidates = [
            v for v in self.nodes()
            if v not in {treatment, outcome}
        ]

        if max_set_size is None:
            max_set_size = len(candidates)

        valid_sets: list[set[str]] = []
        for size in range(0, max_set_size + 1):
            for combo in itertools.combinations(candidates, size):
                z = set(combo)
                if self._blocks_all_backdoor_paths(treatment, outcome, z):
                    valid_sets.append(z)

        return valid_sets

    def do(self, treatment: str) -> "CausalDAG":
        mutilated = CausalDAG()
        mutilated.graph = self.graph.copy()
        parents_of_treatment = list(mutilated.graph.predecessors(treatment))
        for p in parents_of_treatment:
            mutilated.graph.remove_edge(p, treatment)
        return mutilated

    def __repr__(self) -> str:
        lines = ["CausalDAG:", f"  Nodes : {self.nodes()}", "  Edges :"]
        for u, v in self.edges():
            lines.append(f"    {u} -> {v}")
        return "\n".join(lines)
