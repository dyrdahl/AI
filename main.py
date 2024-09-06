import math
from decimal import *
import time   # DEBUG

def compare_adjacent_circles(circle_a, circle_b):
    x1, y1, r1 = circle_a
    x2, y2, r2 = circle_b
    # print(f"x1: {x1}, y1: {y1}, r1: {r1}  || x2: {x2}, y2: {y2}, r2: {r2}")  # Debug
    
    # # Convert all coordinates and radii to Decimal using string representation for exact precision due to Floating Point Arithmetic inaccurate
    # x1, y1, r1 = Decimal(str(x1)), Decimal(str(y1)), Decimal(str(r1))
    # x2, y2, r2 = Decimal(str(x2)), Decimal(str(y2)), Decimal(str(r2))
    
    # Distance between the two circles centers
    distance = math.sqrt(((x2 - x1) ** 2) + ((y2 - y1) ** 2))

    # absolute difference
    abs_diff = abs(r1 - r2)
    
    # print(f"abs_diff: {abs_diff}")
    # Sum of radii
    rad_sum = r1 + r2
    
    # Check if circles overlap
    if abs_diff < distance < rad_sum:
        return True
    
    return False

def circle_struct(circles):
    n = len(circles)
    graph = {i: [] for i in range(n)}  # Create an empty adjacency list

    # Build the graph: connect nodes (circles) if they overlap
    for i in range(n):
        for j in range(i + 1, n):
            if compare_adjacent_circles(circles[i], circles[j]):
                graph[i].append(j)
                graph[j].append(i)
    
    # Check if any circle has no adjacent nodes, return False if found
    for circle, adjacent_nodes in graph.items():
        if not adjacent_nodes:
            # print(f"Circle {circle} has no adjacent nodes.")
            return False
        
    return True

if __name__ == '__main__':
    tic = time.perf_counter()
    circle1 = [(1, 3, 0.7), (2, 3, 0.4), (3, 3, 0.9)]  # True
    circle2 = [(1.5, 1.5, 1.3), (4, 4, 0.7)]  # False
    circle3 = [(0.5, 0.5, 0.5), (1.5, 1.5, 1.1), (0.7, 0.7, 0.4), (4, 4, 0.7)]  # False
    circle4 = [(0.5, 0.5, 0.5), (1.5, 1.5, 1.1), (0.7, 0.7, 0.4), (4, 4, 0.7), (3, 5, 0.8), (3.7, 2, 1.6)]  # True
    circle5 = [(3.5, 3.5, 1), (4, 4, 3), (3, 4, 2), (4, 4, 2)]  # False
    circle6 = [(2, 2.5, 1.8), (2, 2.5, 1), (3.4, 2.3, 0.2)]  # False

    for i in range(1, 2):
        print(i)
        print(f"Circle 1: {circle_struct(circle1)}")     # True
        print(f"Circle 2: {circle_struct(circle2)}")     # False
        print(f"Circle 3: {circle_struct(circle3)}")     # False
        print(f"Circle 4: {circle_struct(circle4)}")     # True
        print(f"Circle 5: {circle_struct(circle5)}")     # False
        print(f"Circle 6: {circle_struct(circle6)}")     # False
        

    
    toc = time.perf_counter()
    print(f"Program executed in: {toc - tic:0.7f} seconds")
    