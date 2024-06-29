from enum import Enum
from mapstate_ import MapState
from random import randint

import time
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
    state, fitness = play_move(individual, np.copy(initial_grid))
    return state, fitness


def select_parents(population, fitness_scores):
    total_fitness = sum(fitness_scores)
    probabilities = [score / total_fitness for score in fitness_scores]

    if any(prob < 0 for prob in probabilities):
        min_prob = min(probabilities)
        probabilities = [prob - min_prob for prob in probabilities]

    total_prob = sum(probabilities)
    if total_prob != 0:
        probabilities = [prob / total_prob for prob in probabilities]

    indices = np.arange(len(population))
    selected_indices = np.random.choice(indices, size=2, replace=False, p=probabilities)
    parent1 = population[selected_indices[0]]
    parent2 = population[selected_indices[1]]

    return parent1, parent2


def distance(node1, node2):
    return np.sqrt((node1[0] - node2[0]) ** 2 + (node1[1] - node2[1]) ** 2)


def total_distance(points):
    total = 0
    for i in range(len(points) - 1):
        total += distance(points[i], points[i + 1])
    return total


def genetic_algorithm(population_size, move_length, generations, mutation_rate, grid, max_time):
    population = create_population(population_size, move_length)
    initial_grid = np.copy(grid)

    best_moves = None
    best_fitness = float('-inf')
    t0 = time.time_ns()

    for generation in range(generations):
        if time.time_ns() - t0 > max_time:
            return None

        fitness_scores = []
        states = []
        new_population = []

        for individual in population:
            res = parallel_fitness_evaluation(individual, initial_grid)
            fitness_scores.append(res[1])
            states.append(res[0])

        max_fitness_index = np.argmax(fitness_scores)
        if fitness_scores[max_fitness_index] > best_fitness:
            best_moves = population[max_fitness_index]
            best_fitness = fitness_scores[max_fitness_index]

            if best_fitness > 2500 and states[max_fitness_index] == PlayRet.SCORE:
                return generations, best_moves, time.time_ns() - t0, total_distance(best_moves)

        for _ in range(population_size // 2):
            parent1, parent2 = select_parents(population, fitness_scores)
            child1 = crossover(parent1, parent2)
            child2 = crossover(parent2, parent1)
            new_population.extend([mutate(child1, mutation_rate), mutate(child2, mutation_rate)])

        population = new_population

    return None if best_fitness < 2500 else generations, best_moves, time.time_ns() - t0, total_distance(best_moves)


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

        if (grid[robot_y + dy][robot_x + dx] == MapState.GOAL.value or
                (goal_x == robot_x + dx and goal_y == robot_y + dy)):
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


def print_map(mtx):
    state_symbols = {
        MapState.ROBOT.value: 'R',
        MapState.FREE.value: 'F',
        MapState.OBSTACLE.value: 'O',
        MapState.EXTENDED_OBSTACLE.value: 'E',
        MapState.GOAL.value: 'G'
    }

    for row in mtx:
        for cell in row:
            print(state_symbols.get(cell, ' '), end=' ')
        print()
    print("----------------")


def def_100x100():
    matrix_100x100 = np.full((100, 100), MapState.FREE.value)

    matrix_100x100[0, 0] = MapState.ROBOT.value
    matrix_100x100[20, 20] = MapState.OBSTACLE.value
    matrix_100x100[30, 40] = MapState.OBSTACLE.value
    matrix_100x100[50, 60] = MapState.OBSTACLE.value
    matrix_100x100[70, 80] = MapState.OBSTACLE.value
    matrix_100x100[99, 99] = MapState.GOAL.value

    matrix_100x100[40, 30] = MapState.OBSTACLE.value
    matrix_100x100[60, 50] = MapState.OBSTACLE.value
    matrix_100x100[80, 70] = MapState.OBSTACLE.value

    matrix_100x100[10, 70] = MapState.OBSTACLE.value
    matrix_100x100[20, 80] = MapState.OBSTACLE.value
    matrix_100x100[30, 90] = MapState.OBSTACLE.value

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

    return matrix_5x5


if __name__ == "__main__":
    matrix = def_5x5()

    print_map(matrix)

    best_fitness_, best_moves_, _, _ = genetic_algorithm(population_size=10000, move_length=15, generations=100,
                                                         mutation_rate=0.25, grid=matrix, max_time=10_000_000_000_000)
    print("Best moves:", best_moves_)
    print("Best fitness:", best_fitness_)

    print_map(matrix)
    play_move(best_moves_, grid=matrix, draw=True)
    print_map(matrix)
