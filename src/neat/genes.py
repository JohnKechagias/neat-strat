from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto
from inspect import get_annotations
from random import random

from neat.types import LinkID, NodeID
from neat.activations import ActivationFuncs, get_activation_func
from neat.aggregations import AggregationFuncs, get_aggregation_func


class NodeType(Enum):
    INPUT = auto()
    HIDDEN = auto()
    OUTPUT = auto()


@dataclass
class NodeTraits:
    bias: float
    response: float
    aggregator: AggregationFuncs
    activator: ActivationFuncs

    def distance(self, other: NodeTraits) -> float:
        distance = abs(self.bias - other.bias)
        distance += abs(self.response - other.response)

        if self.activator != other.activator:
            distance += 1.0

        if self.aggregator != other.aggregator:
            distance += 1.0

        return distance

    def crossover(self, other: NodeTraits) -> NodeTraits:
        attrs = {}
        for attr in get_annotations(NodeTraits).values():
            if random() < 0.5:
                attrs[attr] = getattr(self, attr)
            else:
                attrs[attr] = getattr(other, attr)

        return NodeTraits(**attrs)


@dataclass
class Node:
    id: NodeID
    traits: NodeTraits
    node_type: NodeType
    input: list[float] = []

    def distance(self, other: Node) -> float:
        return self.traits.distance(other.traits)

    def crossover(self, other: Node) -> Node:
        assert self.id == other.id
        new_traits = self.traits.crossover(other.traits)
        return Node(self.id, new_traits, self.node_type)

    def add_value(self, value: float):
        self.input.append(value)

    def reset_value(self):
        self.input = []

    def activate(self) -> float:
        aggregator = get_aggregation_func(self.traits.aggregator)
        activator = get_activation_func(self.traits.activator)
        aggregated_input = aggregator(self.input)

        if self.node_type in (NodeType.INPUT, NodeType.OUTPUT):
            return aggregated_input
        elif self.node_type == NodeType.HIDDEN:
            return activator(aggregated_input * self.traits.response + self.traits.bias)
        else:
            raise NotImplementedError(
                f"Activation of {self.node_type} is not currently implemented."
            )


@dataclass
class LinkTraits:
    weight: float = 1
    enabled: bool = True
    frozen: bool = False

    def distance(self, other: LinkTraits) -> float:
        distance = abs(self.weight - other.weight)

        if self.enabled != other.enabled:
            distance += 1.0

        if self.frozen != other.frozen:
            distance += 1.0

        return distance

    def crossover(self, other: LinkTraits) -> LinkTraits:
        attrs = {}
        for attr in get_annotations(LinkTraits).values():
            if random() < 0.5:
                attrs[attr] = getattr(self, attr)
            else:
                attrs[attr] = getattr(other, attr)

        return LinkTraits(**attrs)


@dataclass
class Link:
    id: LinkID
    in_node: Node
    out_node: Node
    traits: LinkTraits

    @property
    def enabled(self) -> bool:
        return self.traits.enabled

    @property
    def frozen(self) -> bool:
        return self.traits.frozen

    @property
    def weight(self) -> float:
        return self.traits.weight

    @property
    def simple_link(self) -> tuple[NodeID, NodeID]:
        return (self.in_node.id, self.out_node.id)

    def enable(self):
        self.traits.enabled = True

    def disable(self):
        self.traits.enabled = False

    def reset_weight(self):
        self.traits.weight = 1

    def distance(self, other: Link) -> float:
        return self.traits.distance(other.traits)

    def crossover(self, other: Link) -> Link:
        assert self.id == other.id
        new_traits = self.traits.crossover(other.traits)
        return Link(
            self.id,
            self.in_node,
            self.out_node,
            new_traits,
        )
