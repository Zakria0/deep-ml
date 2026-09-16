import numpy as np
import heapq

def k_nearest_neighbors(points, query_point, k):
    """
    Find k nearest neighbors to a query point
    
    Args:
        points: List of tuples representing points [(x1, y1), (x2, y2), ...]
        query_point: Tuple representing query point (x, y)
        k: Number of nearest neighbors to return
    
    Returns:
        List of k nearest neighbor points as tuples
        When distances are tied, points appearing earlier in the input list come first.
    """
    x, y = query_point
    min_heap = []

    for xi, yi in points:
        dist = (xi - x) ** 2 + (yi - y) ** 2
        min_heap.append([dist, xi, yi])
    
    heapq.heapify(min_heap)

    res = []
    while k > 0:
        dist, xi, yi = heapq.heappop(min_heap)
        res.append((xi, yi))
        k -= 1
    
    return res









