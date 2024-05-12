import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
import random

def create_neural_network(input_size, output_size):
    model = Sequential([
        Dense(64, activation='relu', input_dim=input_size),
        Dense(64, activation='relu'),
        Dense(output_size, activation='linear')
    ])
    model.compile(optimizer=Adam(), loss='mse')
    return model

def generate_initial_weights(population_size, model):
    initial_population = []
    for _ in range(population_size):
        weights = model.get_weights()
        new_weights = [np.random.permutation(w.flat).reshape(w.shape) for w in weights]
        initial_population.append(new_weights)
    return initial_population

def decode_weights(individual, model):
    model.set_weights(individual)
    return model

def evaluate_model(model, grid):
    # Simplified evaluation: Convert grid to input, predict output, calculate fitness
    input_grid = grid.flatten()
    predicted_path = model.predict(input_grid.reshape(1, -1))
    fitness = calculate_fitness(predicted_path, grid)
    return fitness

def calculate_fitness(predicted_path, grid):
    # Placeholder function to evaluate the path's quality
    return -np.sum(predicted_path)  # Dummy example

# Main GA loop to update weights
def genetic_algorithm(population_size, generations, mutation_rate, grid):
    input_size = grid.size
    output_size = 4  # Assuming 4 possible moves (up, down, left, right)
    model = create_neural_network(input_size, output_size)
    population = generate_initial_weights(population_size, model)

    for _ in range(generations):
        fitness_scores = [evaluate_model(decode_weights(ind, model), grid) for ind in population]
        # Selection, Crossover, and Mutation operations follow...

    # Return the best model
    best_index = np.argmax(fitness_scores)
    best_model = decode_weights(population[best_index], model)
    return best_model

# Example usage
grid = np.zeros((10, 10))
grid[9, 9] = 1  # Goal at the bottom-right corner
trained_model = genetic_algorithm(100, 50, 0.1, grid)
import numpy as np

def print_map(grid, path=None):
    state_symbols = {
        'robot': '🤖',
        'free': '⬜',
        'obstacle': '⬛',
        'goal': '🎯',
        'path': '🔵'
    }

    # If a path is provided, mark it on the grid
    if path is not None:
        for x, y in path:
            if grid[y, x] == 0:  # Only mark the path if it's on free space
                grid[y, x] = 'path'

    for row in grid:
        print(' '.join(state_symbols.get(cell, cell) for cell in row))
    print("----------------")

def simulate_path(grid, start, model):
    """ Simulate the path on the grid using the model's output moves """
    current_position = start
    path = [start]
    steps = 30  # Define maximum number of steps to avoid infinite loops

    for _ in range(steps):
        input_grid = grid.flatten()
        move_probabilities = model.predict(input_grid.reshape(1, -1))[0]
        move_index = np.argmax(move_probabilities)
        dx, dy = get_move_from_index(move_index)
        new_position = (current_position[0] + dx, current_position[1] + dy)

        if not is_valid_position(grid, new_position):
            break
        path.append(new_position)
        current_position = new_position

        if grid[current_position[1], current_position[0]] == 'goal':
            print("Goal reached!")
            break

    return path

# Example usage
grid = np.zeros((10, 10))
grid[9, 9] = 'goal'  # Goal at the bottom-right corner
grid[0, 0] = 'robot' # Starting point
trained_model = genetic_algorithm(100, 50, 0.1, grid)  # Assuming this returns a trained model
path = simulate_path(grid, (0, 0), trained_model)
print_map(grid, path)
