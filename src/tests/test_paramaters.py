from tests.data import CONFIG_FILEPATH
from neat.parameters import Parameters


def test_read_parameters():
    params = Parameters(CONFIG_FILEPATH)
