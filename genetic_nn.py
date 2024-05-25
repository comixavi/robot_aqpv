import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
from heapq import heappush, heappop
from random import randint
from enum import Enum
from torch.utils.data.dataloader import default_collate
from genetic_ import genetic_algorithm, play_move


class MapState(Enum):
    ROBOT = 1.0
    FREE = 2.0
    OBSTACLE = 3.0
    EXTENDED_OBSTACLE = 4.0
    GOAL = 5.0


class GAParameterPredictor(nn.Module):
    def __init__(self, input_dimension):
        super(GAParameterPredictor, self).__init__()
        self.fc1 = nn.Linear(input_dimension, 64)
        self.fc2 = nn.Linear(64, 64)
        self.output_layer = nn.Linear(64, 4)  # Now outputting 4 values

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.output_layer(x)
        return x


class GridDataset(Dataset):
    def __init__(self, maps, solutions):
        self.maps = maps
        self.solutions = solutions

    def __len__(self):
        return len(self.maps)

    def __getitem__(self, idx):
        return self.maps[idx], self.solutions[idx]


def transform_back(scaled_values):
    if scaled_values.numel() != 4:
        raise ValueError("Expected scaled_values tensor with exactly four elements")

    # Define the min and max for each parameter including mutation rate
    min_population_size, max_population_size = 50, 1000
    min_move_length, max_move_length = 10, 200
    min_generations, max_generations = 10, 100
    min_mutation_rate, max_mutation_rate = 0.01, 0.25  # example ranges for mutation rate

    # Extract scalar values from tensor
    population_size = int(scaled_values[0].item() * (max_population_size - min_population_size) + min_population_size)
    move_length = int(scaled_values[1].item() * (max_move_length - min_move_length) + min_move_length)
    generations = int(scaled_values[2].item() * (max_generations - min_generations) + min_generations)
    mutation_rate = scaled_values[3].item() * (max_mutation_rate - min_mutation_rate) + min_mutation_rate

    return np.array([population_size, move_length, generations, mutation_rate])


def pad_grid(grid, target_dim):
    """ Pad the grid to the target dimensions with EXTENDED_OBSTACLE. """
    pad_height = target_dim[0] - grid.shape[0]
    pad_width = target_dim[1] - grid.shape[1]

    # Pad bottom and right with extended obstacles if necessary
    padded_grid = np.pad(grid,
                         ((0, pad_height), (0, pad_width)),
                         'constant', constant_values=MapState.EXTENDED_OBSTACLE.value)
    return padded_grid


def manhattan_distance(start, goal):
    return abs(goal[0] - start[0]) + abs(goal[1] - start[1])


def a_star_search(grid, start, goal):
    heuristic = manhattan_distance
    neighbors = [(0, 1, 'right'), (1, 0, 'down'), (0, -1, 'left'), (-1, 0, 'up')]
    close_set = set()
    came_from = {}
    gscore = {start: 0}
    fscore = {start: heuristic(start, goal)}
    oheap = []

    heappush(oheap, (fscore[start], start))

    while oheap:
        current = heappop(oheap)[1]

        if current == goal:
            path = []
            while current in came_from:
                prev = came_from[current]
                direction = next(d for (dx, dy, d) in neighbors if (prev[0] + dx, prev[1] + dy) == current)
                path.append(direction)
                current = prev
            return path[::-1]

        close_set.add(current)
        for i, j, dir in neighbors:
            neighbor = current[0] + i, current[1] + j
            tentative_g_score = gscore[current] + 1
            if 0 <= neighbor[0] < grid.shape[0] and 0 <= neighbor[1] < grid.shape[1] and grid[neighbor] != MapState.OBSTACLE.value:
                if neighbor in close_set and tentative_g_score >= gscore.get(neighbor, 0):
                    continue

                if tentative_g_score < gscore.get(neighbor, float('inf')) or neighbor not in [i[1] for i in oheap]:
                    came_from[neighbor] = current
                    gscore[neighbor] = tentative_g_score
                    fscore[neighbor] = tentative_g_score + heuristic(neighbor, goal)
                    heappush(oheap, (fscore[neighbor], neighbor))

    return False  # Return False if no path is found


def find_positions(grid):
    start_pos = None
    goal_pos = None

    for i in range(grid.shape[0]):
        for j in range(grid.shape[1]):
            if grid[i][j] == MapState.ROBOT.value:
                start_pos = (i, j)
            elif grid[i][j] == MapState.GOAL.value:
                goal_pos = (i, j)
            if start_pos is not None and goal_pos is not None:
                return start_pos, goal_pos
    return start_pos, goal_pos


def generate_map(width, height):
    grid = np.full((height, width), MapState.FREE.value)
    robot_pos = (randint(0, height - 1), randint(0, width - 1))
    goal_pos = robot_pos
    while goal_pos == robot_pos:
        goal_pos = (randint(0, height - 1), randint(0, width - 1))
    grid[robot_pos] = MapState.ROBOT.value
    grid[goal_pos] = MapState.GOAL.value

    num_obstacles = randint(1, int(0.2 * width * height))
    for _ in range(num_obstacles):
        obstacle_pos = robot_pos
        while obstacle_pos == robot_pos or obstacle_pos == goal_pos:
            obstacle_pos = (randint(0, height - 1), randint(0, width - 1))
        grid[obstacle_pos] = MapState.OBSTACLE.value

    return grid


def create_dataset(num_samples, min_size=10, max_size=50, target_dim=(25, 25)):
    maps = []
    solutions = []

    for _ in range(num_samples):
        width = randint(min_size, max_size)
        height = randint(min_size, max_size)
        grid = generate_map(width, height)
        start, goal = find_positions(grid)
        solution = a_star_search(grid, start, goal)
        padded_grid = pad_grid(grid, target_dim)
        maps.append(padded_grid)
        solutions.append(solution)

    return maps, solutions


def predict_parameters(model, input_features):
    model.eval()
    with torch.no_grad():
        input_tensor = torch.tensor(input_features).float()
        predicted_scaled_params = model(input_tensor.unsqueeze(0))
        predicted_params = transform_back(predicted_scaled_params.squeeze(0))
    return predicted_params


def train_model(model, train_loader, criterion, optimizer, epochs=1):
    model.train()
    for epoch in range(epochs):
        for inputs, targets in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs.float())
            loss = criterion(outputs, torch.tensor(targets, dtype=torch.float32).float())
            loss.backward()
            optimizer.step()


def print_map(mtx):
    state_symbols = {
        MapState.ROBOT.value: '🤖',
        MapState.FREE.value: '⬜',
        MapState.OBSTACLE.value: '⬛',
        MapState.EXTENDED_OBSTACLE.value: '⬛',
        MapState.GOAL.value: '🎯'
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


def custom_collate(batch):
    def pad_grid(grid, target_shape):
        padded_grid = np.pad(grid,
                             ((0, target_shape[0] - grid.shape[0]),
                              (0, target_shape[1] - grid.shape[1])),
                             mode='constant', constant_values=MapState.EXTENDED_OBSTACLE.value)
        return padded_grid

    max_height = max([item[0].shape[0] for item in batch])
    max_width = max([item[0].shape[1] for item in batch])

    padded_batch = []
    for grid, solution in batch:
        padded_grid = pad_grid(grid, (max_height, max_width))
        padded_batch.append((torch.tensor(padded_grid, dtype=torch.float32), solution))

    return default_collate(padded_batch)


if __name__ == "__main__":
    # Generate dataset
    num_samples = 100
    target_dim = (50, 50)
    maps, solutions = create_dataset(num_samples, min_size=5, max_size=10, target_dim=target_dim)
    dataset = GridDataset(maps, solutions)
    train_loader = DataLoader(dataset, batch_size=1, shuffle=True, collate_fn=custom_collate)

    # Initialize the model
    model = GAParameterPredictor(input_dimension=target_dim[0] * target_dim[1])
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    # Train the model
    train_model(model, train_loader, criterion, optimizer, epochs=5)

    # Example test grid
    test_matrix = def_5x5()
    padded_test_matrix = pad_grid(test_matrix, target_dim)
    test_input_features = np.array(padded_test_matrix).flatten()  # Flatten the matrix to create a feature vector

    # Predict parameters
    predicted_params = predict_parameters(model, test_input_features)
    population_size, move_length, generations, mutation_rate = map(int, predicted_params[:-1])
    mutation_rate = predicted_params[-1]  # Keep mutation rate as a float

    # Print predicted parameters
    print("Predicted Parameters:", predicted_params)

    # Run the genetic algorithm using predicted parameters
    best_fitness_, best_moves_, _, _ = genetic_algorithm(population_size=population_size, move_length=move_length,
                                                         generations=generations, mutation_rate=mutation_rate,
                                                         grid=test_matrix, max_time=10_000_000_000_000)
    print("Best moves:", best_moves_)
    print("Best fitness:", best_fitness_)

    # Display the initial and final states of the grid
    print("Initial Grid:")
    print_map(test_matrix)

    # Assuming play_move modifies the grid in-place, let's clone the grid first if needed
    final_grid = np.copy(test_matrix)
    play_move(best_moves_, grid=final_grid, draw=True)

    print("Final Grid After GA Moves:")
    print_map(final_grid)
