from typing import List
from dataclasses import dataclass
import math
import random

@dataclass
class Sphere:
    center: tuple
    radius: float

class APFRRTPlanner:
    def plan(self, start: List[float], goal: List[float], obstacles: List[Sphere]):
        path = [start]
        current = list(start)
        for _ in range(20):
            step = [g - c for g, c in zip(goal, current)]
            norm = math.sqrt(sum(s * s for s in step)) + 1e-6
            current = [c + 0.1 * s / norm for c, s in zip(current, step)]
            path.append(list(current))
        path.append(goal)
        return path
