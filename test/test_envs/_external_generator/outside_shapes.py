
from envs.environment_initializers import RandomPolygon

def make_initializer(target_angle=90, low=6, high=9, **kwargs):
    """Stands in for a generator maintained elsewhere."""
    return RandomPolygon(list(range(low, high + 1)), target_angle)
