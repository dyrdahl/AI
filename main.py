# CS 471 AI - Assignment 1
# Detect if circles are in clusters (Overlapping or Tangent)
# By: Shane Dyrdahl
# 9/12/2024

import math

def compare_adjacent_circles(circle_a, circle_b):
    x1, y1, r1 = circle_a
    x2, y2, r2 = circle_b
    
    # Also, we can convert all coordinates and radii to Decimal using string representation
    # for exact precision due to Floating Point Arithmetic inaccurate
    # x1, y1, r1 = Decimal(str(x1)), Decimal(str(y1)), Decimal(str(r1))
    # x2, y2, r2 = Decimal(str(x2)), Decimal(str(y2)), Decimal(str(r2))
    
    # Distance between the two circles centers
    distance = math.sqrt(((x2 - x1) ** 2) + ((y2 - y1) ** 2))
    
    # absolute difference
    abs_diff = abs(r1 - r2)
    
    # Sum of radii
    rad_sum = r1 + r2
    
    # Check if circles overlap
    if abs_diff < distance <= rad_sum:
        return True
    
    return False

def dfs(node, graph, visited):
    visited.add(node)
    
    for adjacent in graph[node]:
        if adjacent not in visited:
            dfs(adjacent, graph, visited)
            
def circle_struct(circles):
    n = len(circles)
    graph = {x: [] for x in range(n)}  # Create an empty adjacency list
    
    # Build the graph: connect nodes (circles) if they overlap
    for k in range(n):
        for j in range(k + 1, n):
            if compare_adjacent_circles(circles[k], circles[j]):
                graph[k].append(j)
                graph[j].append(k)
    
    # Create a set of all visited nodes to check for non-clustered circles or disconnected clusters
    visited = set()

    # Begin a depth first search to check for clusters
    dfs(0, graph, visited)
    
    # Less nodes visited than total nodes = some circles not clustered (or multiple disconnected clusters)
    if len(visited) < n:
        return False
    
    return True


if __name__ == '__main__':
    # Test Case - 1 [TRUE]
    case1 = [(1, 3, 0.7), (2, 3, 0.4), (3, 3, 0.9)]
    sol_case1 = circle_struct(case1)
    print(f"Test Case 1:")
    print(f"  Circles: {case1}")
    print(f"  Output: {sol_case1}")

    # Test Case - 2 [FALSE]
    case2 = [(1.5, 1.5, 1.3), (4, 4, 0.7)]
    sol_case2 = circle_struct(case2)
    print(f"\nTest Case 2:")
    print(f"  Circles: {case2}")
    print(f"  Output: {sol_case2}")
    
    # Test Case - 3 [FALSE]
    case3 = [(0.5, 0.5, 0.5), (1.5, 1.5, 1.1), (0.7, 0.7, 0.4), (4, 4, 0.7)]
    sol_case3 = circle_struct(case3)
    print(f"\nTest Case 3:")
    print(f"  Circles: {case3}")
    print(f"  Output: {sol_case3}")
    
    # Test Case - 4 [TRUE]
    case4 = [(0.5, 0.5, 0.5), (1.5, 1.5, 1.1), (0.7, 0.7, 0.4), (4, 4, 0.7), (3, 5, 0.8), (3.7, 2, 1.6)]
    sol_case4 = circle_struct(case4)
    print(f"\nTest Case 4:")
    print(f"  Circles: {case4}")
    print(f"  Output: {sol_case4}")
    