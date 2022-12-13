from functools import partial
from typing import Tuple, Iterable, Sequence, List, Dict, DefaultDict
from random import sample
from math import fsum, sqrt
from collections import defaultdict




def mean(data: Iterable[float]) -> float:
    'Accurate arithmetic mean'
    data = list(data)
    return fsum(data) / len(data)

def transpose(matrix: Iterable[Iterable]) -> Iterable[tuple]:
    'Swap rows with columns for a 2-D array'
    return zip(*matrix)

Point = Tuple[float, ...]
Centroid = Point

def dist(p: Point, q: Point, sqrt=sqrt, fsum=fsum, zip=zip) -> float:
    'Multi-dimensional euclidean distance'
    return sqrt(fsum((x1 - x2) ** 2.0 for x1, x2 in zip(p, q)))

def assign_data(centroids: Sequence[Centroid], data: Iterable[Point]) -> Dict[Centroid, Sequence[Point]]:
    'Assign data the closest centroid'
    d : DefaultDict[Point, List[Point]] = defaultdict(list)
    for point in data:
        centroid: Point = min(centroids, key=partial(dist, point))
        d[centroid].append(point)
    return dict(d)

def compute_centroids(groups: Iterable[Sequence[Point]]) -> List[Centroid]:
    'Compute the centroid of each group'
    return [tuple(map(mean, transpose(group))) for group in groups]

def k_means(data: Iterable[Point], k:int=2, iterations:int=10) -> List[Point]:
    'Return k-centroids for the data'
    data = list(data)
    centroids = sample(data, k)
    for i in range(iterations):
        labeled = assign_data(centroids, data)
        centroids = compute_centroids(labeled.values())
    return centroids
