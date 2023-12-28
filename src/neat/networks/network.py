from __future__ import annotations
from abc import ABC, abstractmethod
from neat.genes import Link, Node
from neat.genomes.genome import Genome
from neat.parameters import Parameters
from neat.types import LinkID, NodeID

from .graph import Graph


class Network(ABC):
    def __init__(self, nodes: dict[NodeID, Node], links: dict[LinkID, Link]):
        self.nodes = nodes
        self.graph = Graph()
        self.values = {node: 0.0 for node in nodes.keys()}

        for node_id in nodes.keys():
            self.graph.add_node(node_id)

        for link in links.values():
            self.graph.add_link(link.in_node.id, link.out_node.id, link.weight)

    @staticmethod
    @abstractmethod
    def create(genome: Genome, params: Parameters) -> Network:
        """Receives a genome and returns its phenotype (Network)."""

    @abstractmethod
    def activate(self, input: list[float]) -> list[float]:
        """Passes the given input to the network."""
