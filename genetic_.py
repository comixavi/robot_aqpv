import time
from enum import Enum
from mapstate_ import MapState
from random import randint
import multiprocessing
import numpy as np


class PlayRet(Enum):
    MISS = 0
    SCORE = 1


def create_population(population_size, move_length):
    return [[(randint(-1, 1), randint(-1, 1)) for _ in range(move_length)] for _ in range(population_size)]


def crossover(parent1, parent2):
    split_index = len(parent1) // 2
    child = parent1[:split_index] + parent2[split_index:]
    return child


def mutate(individual, mutation_rate):
    for i in range(len(individual)):
        if np.random.rand() < mutation_rate:
            individual[i] = (randint(-1, 1), randint(-1, 1))
    return individual


def parallel_fitness_evaluation(individual, initial_grid):
    _, fitness = play_move(individual, np.copy(initial_grid))
    return fitness


def select_parents(population, fitness_scores):
    total_fitness = sum(fitness_scores)
    probabilities = [score / total_fitness for score in fitness_scores]

    # Check if probabilities are non-negative
    if any(prob < 0 for prob in probabilities):
        # If any probability is negative, adjust all probabilities to be non-negative
        min_prob = min(probabilities)
        probabilities = [prob - min_prob for prob in probabilities]

    # Normalize probabilities
    total_prob = sum(probabilities)
    if total_prob != 0:
        probabilities = [prob / total_prob for prob in probabilities]

    indices = np.arange(len(population))
    selected_indices = np.random.choice(indices, size=2, replace=False, p=probabilities)
    parent1 = population[selected_indices[0]]
    parent2 = population[selected_indices[1]]
    return parent1, parent2


def genetic_algorithm(population_size, move_length, generations, mutation_rate, grid):
    population = create_population(population_size, move_length)
    initial_grid = np.copy(grid)  # Make a copy of the initial grid

    best_moves = None
    best_fitness = float('-inf')  # Initialize with negative infinity

    for generation in range(generations):
        fitness_scores = []
        new_population = []

        # Evaluate fitness scores in parallel
        with multiprocessing.Pool() as pool:
            fitness_scores = pool.starmap(parallel_fitness_evaluation, [(individual, initial_grid) for individual in population])

        # Find the best solution in the current population
        max_fitness_index = np.argmax(fitness_scores)
        if fitness_scores[max_fitness_index] > best_fitness:
            best_moves = population[max_fitness_index]
            best_fitness = fitness_scores[max_fitness_index]

            if best_fitness > 2500:
                return best_moves, best_fitness

        # Create a new population using selection, crossover, and mutation
        for _ in range(population_size // 2):
            parent1, parent2 = select_parents(population, fitness_scores)
            child1 = crossover(parent1, parent2)
            child2 = crossover(parent2, parent1)
            new_population.extend([mutate(child1, mutation_rate), mutate(child2, mutation_rate)])

        population = new_population

    return best_moves, best_fitness


def play_move(moves, grid, draw=False):
    fitness = 0

    robot_x = None
    robot_y = None

    goal_x = None
    goal_y = None

    rows = len(grid)
    cols = len(grid[0])

    for i in range(rows):
        for j in range(cols):
            if grid[i][j] == MapState.ROBOT.value:
                robot_x = j
                robot_y = i

    for i in range(rows):
        for j in range(cols):
            if grid[i][j] == MapState.GOAL.value:
                goal_x = j
                goal_y = i

    if robot_x is None or robot_y is None:
        return PlayRet.MISS, float('-inf')

    last_move = (0, 0)

    for move in moves:
        dx, dy = move

        if dx == 0 and dy == 0:
            continue

        if robot_y + dy >= rows or robot_y + dy < 0 or robot_x + dx >= cols or robot_x + dx < 0:
            fitness -= 2500
            return PlayRet.MISS, fitness

        if grid[robot_y + dy][robot_x + dx] == MapState.GOAL.value:
            fitness += 2500
            return PlayRet.SCORE, fitness

        if grid[robot_y + dy][robot_x + dx] != MapState.FREE.value:
            fitness -= 2500
            return PlayRet.MISS, fitness

        fitness += 1

        if last_move == move:
            fitness += 1

        grid[robot_y][robot_x] = MapState.FREE.value

        robot_y = robot_y + dy
        robot_x = robot_x + dx

        grid[robot_y][robot_x] = MapState.ROBOT.value

        if draw:
            print_map(grid)

    fitness -= 5000
    return PlayRet.MISS, fitness


def print_map(matrix):
    state_symbols = {
        MapState.ROBOT.value: '🤖',
        MapState.FREE.value: '⬜',
        MapState.OBSTACLE.value: '⬛',
        MapState.EXTENDED_OBSTACLE.value: '⬛',
        MapState.GOAL.value: '🎯'
    }

    for row in matrix:
        for cell in row:
            print(state_symbols.get(cell, ' '), end=' ')
        print()
    print("----------------")


def def_100x100():
    matrix_100x100 = np.full((100, 100), MapState.FREE.value)

    # Set different MapState values for specific cells
    matrix_100x100[0, 0] = MapState.ROBOT.value
    matrix_100x100[20, 20] = MapState.OBSTACLE.value
    matrix_100x100[30, 40] = MapState.OBSTACLE.value
    matrix_100x100[50, 60] = MapState.OBSTACLE.value
    matrix_100x100[70, 80] = MapState.OBSTACLE.value
    matrix_100x100[99, 99] = MapState.GOAL.value

    # Add more obstacles
    matrix_100x100[40, 30] = MapState.OBSTACLE.value
    matrix_100x100[60, 50] = MapState.OBSTACLE.value
    matrix_100x100[80, 70] = MapState.OBSTACLE.value

    # Add additional obstacles (2nd time)
    matrix_100x100[10, 70] = MapState.OBSTACLE.value
    matrix_100x100[20, 80] = MapState.OBSTACLE.value
    matrix_100x100[30, 90] = MapState.OBSTACLE.value

    # Add additional obstacles (3rd time)
    matrix_100x100[70, 10] = MapState.OBSTACLE.value
    matrix_100x100[80, 20] = MapState.OBSTACLE.value
    matrix_100x100[90, 30] = MapState.OBSTACLE.value

    return matrix_100x100


def def_5x5():
    matrix_5x5 = np.full((5, 5), MapState.FREE.value)
    matrix_5x5[0, 0] = MapState.ROBOT.value
    matrix_5x5[2, 2] = MapState.OBSTACLE.value
    matrix_5x5[3, 2] = MapState.OBSTACLE.value
    matrix_5x5[2, 3] = MapState.OBSTACLE.value
    matrix_5x5[3, 1] = MapState.OBSTACLE.value
    matrix_5x5[4, 4] = MapState.GOAL.value

    return  matrix_5x5


if __name__ == "__main__":
    start_time = time.time()
    matrix = def_5x5()

    print_map(matrix)

    best_moves_, best_fitness_ = genetic_algorithm(population_size=1000, move_length=10, generations=1000,
                                                   mutation_rate=0.1, grid=matrix)
    print("Best moves:", best_moves_)
    print("Best fitness:", best_fitness_)

    print_map(matrix)
    play_move(best_moves_, grid=matrix, draw=True)
    print_map(matrix)

    end_time = time.time()
    execution_time = end_time - start_time
    print("Execution time:", execution_time, "seconds")
