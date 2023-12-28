from __future__ import annotations
from neat.genes import NodeType
from neat.genomes.genome import Genome
from neat.parameters import Parameters
from .network import Network


class FeedForwardNetwork(Network):
    @staticmethod
    def create(genome: Genome, params: Parameters) -> FeedForwardNetwork:
        return FeedForwardNetwork(genome.nodes, genome.links)

    def activate(self, input: list[float]) -> list[float]:
        input_nodes = [n for n in self.nodes.values() if n.node_type == NodeType.INPUT]
        output_nodes = set([k for k, n in self.nodes.items() if n.node_type == NodeType.OUTPUT])
        for node, value in zip(input_nodes, input):
            node.add_value(value)

        for node_id in self.graph.toposort():
            self.values[node_id] = self.nodes[node_id].activate()
            for neighbor, weight in self.graph.get_neighbors(node_id).items():
                self.nodes[neighbor].add_value(self.values[node_id] * weight)

        return [value for node, value in self.values.items() if node in output_nodes]
