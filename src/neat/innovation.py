from dataclasses import dataclass, field


@dataclass
class InnovationRecord:
    nodes_counter: int = 0
    links_counter: int = 0
    links_record: dict[tuple[int, int], int] = field(default_factory=dict)
    species_counter: int = 0

    def get_node_id(self) -> int:
        node_id = self.nodes_counter
        self.nodes_counter += 1
        return node_id

    def get_link_id(self, in_node_id: int, out_node_id: int) -> int:
        link = (in_node_id, out_node_id)
        if link_id := self.links_record.get(link):
            return link_id

        link_id = self.links_counter
        self.links_record[link] = link_id
        self.links_counter += 1
        return link_id

    def new_species(self) -> int:
        species_id = self.species_counter
        self.species_counter += 1
        return species_id
