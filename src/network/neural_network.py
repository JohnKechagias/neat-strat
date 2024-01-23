import multiprocessing
import pickle
from pathlib import Path

import neat
import visualize

runs_per_net = 5
simulation_seconds = 60.0
time_const = 0.01


def eval_genome(genome, config):
    net = neat.ctrnn.CTRNN.create(genome, config, time_const)

    fitnesses = []
    for _ in range(runs_per_net):
        sim = cart_pole.CartPole()
        net.reset()

        # Run the given simulation for up to num_steps time steps.
        fitness = 0.0
        while sim.t < simulation_seconds:
            inputs = sim.get_scaled_state()
            action = net.advance(inputs, time_const, time_const)

            # Apply action to the simulated cart-pole
            force = cart_pole.discrete_actuator_force(action)
            sim.step(force)

            # Stop if the network fails to keep the cart within the position or angle limits.
            # The per-run fitness is the number of time steps the network can balance the pole
            # without exceeding these limits.
            if (
                abs(sim.x) >= sim.position_limit
                or abs(sim.theta) >= sim.angle_limit_radians
            ):
                break

            fitness = sim.t

        fitnesses.append(fitness)

        # print("{0} fitness {1}".format(net, fitness))

    # The genome's fitness is its worst performance across all runs.
    return min(fitnesses)


def run():
    local_path = Path(__file__)
    config_path = local_path / "config.ini"

    config = neat.Config(
        neat.DefaultGenome,
        neat.DefaultReproduction,
        neat.DefaultSpeciesSet,
        neat.DefaultStagnation,
        config_path,
    )

    pop = neat.Population(config)
    stats = neat.StatisticsReporter()
    pop.add_reporter(stats)
    pop.add_reporter(neat.StdOutReporter(True))

    evaluator = neat.ParallelEvaluator(multiprocessing.cpu_count(), eval_genome)
    winner = pop.run(evaluator.evaluate)

    with open("winner", "wb") as f:
        pickle.dump(winner, f)

    print(winner)

    visualize.plot_stats(stats, ylog=True, view=True, filename="ctrnn-fitness.svg")
    visualize.plot_species(stats, view=True, filename="ctrnn-speciation.svg")

    node_names = {-1: "x", -2: "dx", -3: "theta", -4: "dtheta", 0: "control"}
    visualize.draw_net(config, winner, True, node_names=node_names)

    visualize.draw_net(
        config, winner, view=True, node_names=node_names, filename="winner.gv"
    )
    visualize.draw_net(
        config,
        winner,
        view=True,
        node_names=node_names,
        filename="winner-ctrnn-pruned.gv",
        prune_unused=True,
    )


if __name__ == "__main__":
    run()
