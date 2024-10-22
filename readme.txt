Important:
	-> rrt_.py: This file implements and tests (with all test scenarios except the real-time one, which is in main.py) all versions of the RRT algorithms.
	-> genetic_.py: Implementation of the genetic algorithm.
	-> main.py: Functions for reading from files, constructing the map, all adjacent operations for visualization, calling functions, and visualizing solutions are implemented here.
	-> astar.py: Implementation of the A* algorithm.
	-> mapstate_.py: Contains the enum with possible states and their values on the map.
Adjacent Files:
	-> util_.py: Implementation of the Bresenham line algorithm.
	-> test_plot_state.py: Used for visualizing different scenarios during development.
	-> rrt_test.py: A file initially used to familiarize myself with the RRT algorithm.
	-> rrt_star.py: An empty file because I concluded that it was more convenient to develop methods in the same file to have access to common resources. Now that the work is complete, such division could help with long-term modularity.
	-> random_test.py: Used to test different inputs and outputs for Python functions.
	-> nn_test.py: Used to familiarize myself with working with neural networks in Python.
	-> merge_excels.py: An attempt to centralize results, currently non-functional.
	-> genetic_nn.py: An attempt to combine the neural network approach with genetic algorithms.
	-> DBSCAN_test.py: To visualize the results of DBSCAN filtering in a fully controlled environment.
Excel Files:
	-> Files in result/:* Results from the last testing performed, their average.
	-> result:* Partial results during development.
	-> model_params.xlsx: Parameters obtained for the neural network during testing.
	-> *.bag: Files obtained from LiDAR.
	-> *.txt: Conversion of .bag files for processing.
Exceptions:
	-> run_all.txt: ROS configuration file from the robot.
	-> readme.txt: This readme file itself.
