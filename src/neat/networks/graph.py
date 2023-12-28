from dataclasses import dataclass
from neat.types import NodeID
from neat.logging import LOGGER
from typing import Optional


class CycleFoundError(Exception):
    """Raised whenever a graph cycle is found when it should not.
    For example when trying to topologically sort the graph."""


class NodeNotFoundError(Exception):
    """Raised when trying to access a graph node that does not exist."""


class LinkNotFoundError(Exception):
    """Raised when trying to access a graph link that does not exist."""


@dataclass
class Node:
    id: NodeID
    neighbors: dict[NodeID, float] = {}

    def add_neighbor(self, neighbor: NodeID, weight: float):
        self.neighbors[neighbor] = weight

    def remove_neighbor(self, neighbor: NodeID):
        self.neighbors.pop(neighbor)


class Graph:
    def __init__(self):
        self.graph: dict[NodeID, Node] = {}
        self.in_degrees: dict[NodeID, int] = {}
    
    def get_neighbors(self, node: NodeID) -> dict[NodeID, float]:
        return self.graph[node].neighbors

    def add_node(self, node: NodeID):
        if node in self.graph:
            LOGGER.info(f"Attempted to add already existing node '{node}'.")
            return

        self.graph[node] = Node(node)
        self.in_degrees[node] = 0

    def remove_node(self, node: NodeID):
        if node not in self.graph:
            raise NodeNotFoundError(f"Node '{node}' does not exist.")

        for neighbor in self.graph[node].neighbors:
            self.in_degrees[neighbor] -= 1

        self.graph.pop(node)
        self.in_degrees.pop(node)

    def add_link(self, input: NodeID, output: NodeID, weight: float):
        self.graph[input].add_neighbor(output, weight)
        self.in_degrees[output] += 1

    def remove_link(self, input: NodeID, output: NodeID):
        if output not in self.graph[input].neighbors:
            raise LinkNotFoundError(f"Link '{input} -> {output}' does not exist.")

        self.graph[input].remove_neighbor(output)
        self.in_degrees[output] -= 1

    def edit_link_weight(self, input: NodeID, output: NodeID, weight: float):
        self.graph[input].add_neighbor(output, weight)

    def toposort(self, stack: Optional[list[NodeID]] = None) -> list[NodeID]:
        """Sorts the nodes in topological order. That means that, for any given
        node, nodes that need to be activated before it are on the left of it
        and nodes that need to be activate after it are on the right of it.

        Based on Kahn's algorithm.
        
        Returns:
            The list of sorted node layers (input to output layer).
        """
        if stack is None:
            stack = []

        in_degree: dict[NodeID, int] = self.in_degrees.copy()
        ordered: list[NodeID] = []

        for node, in_degrees in in_degree.items():
            if in_degrees == 0:
                stack.append(node)

        num_of_visited_nodes = 0
        while stack:
            node = stack.pop()
            ordered.append(node)

            for neighbor in self.graph[node].neighbors:
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    stack.append(neighbor)

            num_of_visited_nodes += 1

        if num_of_visited_nodes != len(self.graph):
            LOGGER.warning("Tried to topologically sort a cyclic graph!")

        return ordered
