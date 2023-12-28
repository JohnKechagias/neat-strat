from __future__ import annotations

import copy
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Optional
from neat.types import LinkID, NodeID

import utils
from genes import Link, LinkTraits, Node, NodeTraits, NodeType
from parameters import Parameters

from neat.innovation import InnovationRecord


@dataclass
class Genome:
    params: Parameters

    @classmethod
    def initialize_configuration(cls, params: Parameters):
        cls.params = params

    @classmethod
    def from_file(cls, file: Path) -> Genome:
        with file.open() as _:
            ...

        innovation_record = InnovationRecord()
        return Genome(0, innovation_record)

    def __init__(
        self,
        id: int,
        innovation_record: InnovationRecord,
        nodes: Optional[dict[int, Node]] = None,
        links: Optional[dict[int, Link]] = None,
    ):
        self.id: int = id
        self.fitness = 0.0
        self.innov_record = innovation_record

        if nodes is None and links is None:
            self.initialize_default_genome()

    def initialize_default_genome(self):
        self.nodes: dict[NodeID, Node] = {}
        self.links: dict[LinkID, Link] = {}

        self._node_id = -1
        self._link_id = -1

        input_nodes: dict[int, Node] = {}
        output_nodes: dict[int, Node] = {}
        for _ in range(self.params.number_of_inputs):
            node = self.create_new_node(NodeType.INPUT)
            input_nodes[node.id] = node

        for _ in range(self.params.number_of_outputs):
            node = self.create_new_node(NodeType.OUTPUT)
        for input in input_nodes.values():
            for output in output_nodes.values():
                link = self.create_new_link(input, output)
                self.links[link.id] = link

        self.nodes = {**input_nodes, **output_nodes}

    def crossover(self, other: Genome) -> Genome:
        """Configure a new genome by crossover from two parent genomes."""

        primary_parent, secondary_parent = self, other
        if self.fitness < other.fitness:
            primary_parent, secondary_parent = other, self

        links = {}
        nodes = {}

        for key, link1 in primary_parent.links.items():
            if link2 := secondary_parent.links.get(key):
                # Homologous gene: combine genes from both parents.
                links[key] = link1.crossover(link2)
            else:
                # Excess or disjoint gene: copy from the fittest parent.
                links[key] = copy.copy(link1)

        for key, node1 in primary_parent.nodes.items():
            if node2 := secondary_parent.nodes.get(key):
                # Homologous gene: combine genes from both parents.
                nodes[key] = node1.crossover(node2)
            else:
                # Extra gene: copy from the fittest parent
                nodes[key] = copy.copy(node1)

        id = utils.get_genome_id()

        genome = Genome(id, self.innov_record, nodes, links)
        genome._node_id = max(primary_parent._node_id, secondary_parent._node_id)
        genome._link_id = max(primary_parent._link_id, secondary_parent._link_id)
        return genome

    def get_distance(self, other: Genome) -> float:
        node_distance = 0.0
        disjoint_nodes = 0
        num_of_common_node_genes = 0

        if self.nodes or other.nodes:
            for node_id, node in self.nodes.items():
                if node2 := other.nodes.get(node_id):
                    num_of_common_node_genes += 1
                    node_distance += node.distance(node2)
                else:
                    disjoint_nodes += 1

        disjoint_nodes += len(other.nodes) - num_of_common_node_genes
        max_nodes = max(len(self.nodes), len(other.nodes))
        node_distance = (
            node_distance
            + (
                self.params.speciation.compatibility_disjoint_coefficient
                * disjoint_nodes
            )
        ) / max_nodes

        link_distance = 0.0
        disjoint_links = 0
        num_of_common_link_genes = 0

        if self.links or other.links:
            for link_id, link in self.links.items():
                if link2 := other.links.get(link_id):
                    num_of_common_link_genes += 1
                    link_distance += link.distance(link2)
                else:
                    disjoint_links += 1

        disjoint_links += len(other.links) - num_of_common_link_genes
        max_links = max(len(self.links), len(other.links))
        link_distance = (
            link_distance
            + (
                self.params.speciation.compatibility_disjoint_coefficient
                * disjoint_links
            )
        ) / max_links

        return node_distance + link_distance

    def get_input_nodes(self) -> list[Node]:
        return [n for n in self.nodes.values() if n.node_type == NodeType.INPUT]

    def is_compatible(self, genome: Genome) -> bool:
        compatibility_threshold = self.params.speciation.compatibility_threshold
        return self.get_distance(genome) < compatibility_threshold

    def get_node_id(self) -> int:
        return self.innov_record.get_node_id()

    def get_link_id(self, in_node: Node, out_node: Node) -> int:
        return self.innov_record.get_link_id(in_node.id, out_node.id)

    def mutate(self):
        if random.random() < self.params.node_addition_chance:
            self.mutate_add_node()

        if random.random() < self.params.node_deletion_chance:
            self.mutate_delete_node()

        if random.random() < self.params.link_addition_chance:
            self.mutate_add_link()

        if random.random() < self.params.link_deletion_chance:
            self.mutate_delete_link()

        if random.random() < self.params.link_toggle_enable_chance:
            self.mutate_toggle_enable()

        for node in self.nodes.values():
            self.mutate_node(node)

        for link in self.links.values():
            self.mutate_link(link)

    def mutate_add_node(self):
        link_to_split = random.choice(self.links)
        link_to_split.disable()

        node = self.create_new_node()
        self.nodes[node.id] = node

        first_link = self.create_new_link(link_to_split.in_node, node)
        self.links[first_link.id] = first_link

        weight = link_to_split.weight
        second_link = self.create_new_link(node, link_to_split.out_node, weight)
        self.links[second_link.id] = second_link

    def mutate_delete_node(self):
        hidden_nodes = [
            n for n in self.nodes.values() if n.node_type == NodeType.HIDDEN
        ]

        if not hidden_nodes:
            return

        node = random.choice(hidden_nodes)

        self.links = {
            k: l
            for k, l in self.links.items()
            if l.in_node != node and l.out_node != node
        }
        self.nodes.pop(node.id)

    def mutate_add_link(self):
        in_nodes = [n for n in self.nodes.values() if n.node_type != NodeType.OUTPUT]
        in_node = random.choice(in_nodes)
        out_nodes = [n for n in self.nodes.values() if n.node_type != NodeType.INPUT]
        out_node = random.choice(out_nodes)

        if in_node == out_node:
            return

        for link in self.links.values():
            if link.in_node == in_node and link.out_node == out_node:
                return

        link = self.create_new_link(in_node, out_node)
        self.links[link.id] = link

    def mutate_delete_link(self):
        link = random.choice(self.links)
        self.links.pop(link.id)

    def mutate_toggle_enable(self):
        link_to_toggle_enable = random.choice(self.links)

        if not link_to_toggle_enable.enabled:
            link_to_toggle_enable.enable()
            return

        # We need to make sure that another gene connects out of the in-node,
        # because if not, a section of network will break off and become isolated.
        in_node_id = link_to_toggle_enable.in_node.id
        sum_of_links_with_the_specific_in_node = 0

        for link in self.links.values():
            if link.in_node.id == in_node_id:
                sum_of_links_with_the_specific_in_node += 1

            if sum_of_links_with_the_specific_in_node >= 2:
                link_to_toggle_enable.disable()
                return

    def mutate_node(self, node: Node):
        if random.random() < self.params.bias_mutation_chance:
            node.traits.bias = utils.clamp(
                node.traits.bias + random.gauss(0.0, self.params.bias_mutate_power),
                self.params.bias_min_value,
                self.params.bias_max_value,
            )
        # TODO add mutation of aggregation and activation functions.

    def mutate_link(self, link: Link):
        if link.frozen:
            return

        if not random.random() < self.params.weight_mutation_chance:
            return

        mutation_power = self.params.weight_mutate_power
        if random.random() < self.params.weight_severe_mutation_chance:
            mutation_power *= 2

        link.traits.weight
        link.traits.weight = utils.clamp(
            link.weight + utils.randon_sign() * random.random() * mutation_power,
            self.params.weight_min_value,
            self.params.weight_max_value,
        )

    def create_new_node(self, node_type: NodeType = NodeType.HIDDEN) -> Node:
        id = self.get_node_id()
        traits = NodeTraits(
            self.params.bias_default_value,
            self.params.response_default_value,
            self.params.aggregation_default_value,
            self.params.activation_default_value,
        )
        return Node(id, traits, node_type)

    def create_new_link(self, in_node: Node, out_node: Node, weight: float = 1) -> Link:
        id = self.get_link_id(in_node, out_node)
        traits = LinkTraits()
        traits.weight = weight
        return Link(id, in_node, out_node, traits)

    def reenable_link(self):
        for link in self.links.values():
            if not link.enabled:
                link.enable()
                break

    def reset_weights(self):
        for link in self.links.values():
            link.reset_weight()
