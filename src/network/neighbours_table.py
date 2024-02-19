from .constants import BOARD_SIZE, Coords

NEIGHBOURS_TABLE: list[list[Coords]] = [
    [(1, 0), (0, 1)],
    [(1, 1), (0, 2), (0, 0), (1, 2), (1, 0)],
    [(1, 2), (0, 3), (0, 1)],
    [(1, 3), (0, 4), (0, 2), (1, 4), (1, 2)],
    [(1, 4), (0, 3)],
    [(2, 0), (0, 0), (1, 1), (0, 1)],
    [(2, 1), (0, 1), (1, 2), (1, 0), (2, 2), (2, 0)],
    [(2, 2), (0, 2), (1, 3), (1, 1), (0, 3), (0, 1)],
    [(2, 3), (0, 3), (1, 4), (1, 2), (2, 4), (2, 2)],
    [(2, 4), (0, 4), (1, 3), (0, 3)],
    [(3, 0), (1, 0), (2, 1), (1, 1)],
    [(3, 1), (1, 1), (2, 2), (2, 0), (3, 2), (3, 0)],
    [(3, 2), (1, 2), (2, 3), (2, 1), (1, 3), (1, 1)],
    [(3, 3), (1, 3), (2, 4), (2, 2), (3, 4), (3, 2)],
    [(3, 4), (1, 4), (2, 3), (1, 3)],
    [(4, 0), (2, 0), (3, 1), (2, 1)],
    [(4, 1), (2, 1), (3, 2), (3, 0), (4, 2), (4, 0)],
    [(4, 2), (2, 2), (3, 3), (3, 1), (2, 3), (2, 1)],
    [(4, 3), (2, 3), (3, 4), (3, 2), (4, 4), (4, 2)],
    [(4, 4), (2, 4), (3, 3), (2, 3)],
    [(3, 0), (4, 1), (3, 1)],
    [(3, 1), (4, 2), (4, 0)],
    [(3, 2), (4, 3), (4, 1), (3, 3), (3, 1)],
    [(3, 3), (4, 4), (4, 2)],
    [(3, 4), (4, 3), (3, 3)],
]


def get_coord_hash(x: int, y: int) -> int:
    return x * BOARD_SIZE + y


def lookup_neighbours(x: int, y: int) -> list[Coords]:
    return NEIGHBOURS_TABLE[get_coord_hash(x, y)]


def compute_neighbors(x: int, y: int) -> list[Coords]:
    neighbors = [(x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)]

    if not y % 2:
        neighbors.append((x - 1, y + 1))
        neighbors.append((x - 1, y - 1))
    else:
        neighbors.append((x + 1, y + 1))
        neighbors.append((x + 1, y - 1))

    def is_neighbor_valid(coords: Coords) -> bool:
        x, y = coords
        return x >= 0 and x < BOARD_SIZE and y >= 0 and y < BOARD_SIZE

    return list(filter(is_neighbor_valid, neighbors))
