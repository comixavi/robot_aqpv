from enum import Enum


class MapState(Enum):
    ROBOT = 1.0
    FREE = 2.0
    OBSTACLE = 3.0
    EXTENDED_OBSTACLE = 4.0
    GOAL = 5.0
