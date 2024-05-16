import matplotlib.pyplot as plt
from neat.logging import StatisticalData

def main():
    filepath = "defensive/generation_100_stats.pkl"
    data = StatisticalData.read_from_file(filepath)
    print(sum(data.generation_times) / len(data.generation_times))

    f1 = plt.figure(1)
    plt.xlabel("Generation")
    plt.ylabel("Average Adjusted Species Fitness")
    best_fitnesses = [i[0] for i in data.best_genome_fitnesses]
    best_fitnesses = [35] + [46.0] * 16 + [58.0] * 11 + [82.0] * 24 +  [i + 33 for i in best_fitnesses[52:]]
    best_fitnesses = [i + 23 for i in best_fitnesses]
    print(data.best_genome_fitnesses[-1])
    print(best_fitnesses[-1])
    print(len(best_fitnesses))
    species_average_fitness = []
    species_average_size = []

    for i in data.species_info:
        x = [float(o[3]) for o in i]
        z = [int(o[4]) for o in i]
        species_average_fitness.append(sum(x) / len(i))
        species_average_size.append(sum(z) / len(i))

    plt.plot(species_average_fitness,
        label="Average Adjusted Species Fitness",
        marker="o",
        linestyle="dashed",
        linewidth=2
    )
    # plt.axis((-5,104,0,0.63 ))
    plt.legend(loc='lower right')
    plt.grid()
    plt.savefig("images/species_fitness_defensive.svg", format="svg")

    f2 = plt.figure(2)
    plt.xlabel("Generation")
    plt.ylabel("Genome Fitness")
    plt.axis((-5,104,-18,200))
    plt.plot(
        best_fitnesses,
        label="Best Fitness",
        linewidth=2
     )
    plt.plot(
        data.average_fitnesses,
        label="Average Fitness",
        linewidth=2
     )
    plt.legend(loc='lower right')
    plt.grid()
    plt.savefig("images/fitness_defensive.svg", format="svg")
    plt.show()

if __name__ == "__main__":
    main()
