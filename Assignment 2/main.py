# CS 471 AI - Assignment 2
# Implement Algorithms: Hill-Climbing and Random-Restart Hill-Climbing
# By: Shane Dyrdahl
# 9/24/2024

import random

# This function evaluates the given formula (equation) at three points: left (l), middle (m), and right (r)
# It determines the direction to "climb" by comparing the values of the formula at these points
def check_neighbors(equation, l, m, r):
    left = equation(l)
    middle = equation(m)
    right = equation(r)

    # Choose direction to climb
    if left == right and middle < left:  # If both left & right are equal & higher than middle, climb random direction
        climb_start = random.choice(["right", "left"])
    elif left > middle:  # Climb Left
        climb_start = "left"
    elif middle < right:  # Climb Right
        climb_start = "right"
    else:
        climb_start = "middle"  # If no direction is clear, stay in the middle

    return climb_start

def round_nearest(num, step_size):
    match step_size:
        case 1:
            return round(num)               # Round to nearest whole number
        case 0.5:
            return round(num * 2) / 2       # Round to nearest 0.5
        case 0.01:
            return round(num * 100) / 100   # Round to nearest 0.01

# Random-restart hill-climbing algorithm that tries a specified number of restarts to find the global maximum
def random_restart(restarts, formula, step_size, lowerBound, upperBound):
    randomNums = []           # List to store unique randomly generated starting positions
    global_max = 0.0          # Store the maximum y-value found so far
    global_max_xpos = 0.0     # Store the x-position corresponding to the global maximum
    
    for n in range(1, restarts + 1):
        while True:     # Loop until a unique x_pos is generated
            restart_x_pos = round_nearest(random.uniform(lowerBound, upperBound), step_size)
            if restart_x_pos not in randomNums:      # Check for uniqueness
                randomNums.append(restart_x_pos)     # Store the unique x_pos
                
                # Perform hill-climbing from the randomly generated x_pos
                restart_max_y, restart_x_pos = climb(formula, step_size, restart_x_pos, lowerBound, upperBound)
                
                # Update global max
                if restart_max_y > global_max:
                    global_max = restart_max_y
                    global_max_xpos = restart_x_pos
                break             # Break the loop once a unique position is found and hill-climbing is performed
    return global_max, global_max_xpos           # Return the global maximum value and its corresponding x position

# This function performs the hill-climbing algorithm
def climb(formula, stepSize, start, lowerB, upperB, tolerance=1e-10):
    current = start     # Set the starting x position
    if start == lowerB:
        direction = 'right'
    elif start == upperB:
        direction = 'left'
    else:
        # Evaluate the neighbors to determine the climbing direction
        direction = check_neighbors(formula, (start - stepSize), start, (start + stepSize))

    upperBound = upperB
    climb_max_y = formula(current)   # Calculate the initial y value at the start position
    match direction:
        case 'right':
            upperBound = upperB             # Set upper bound for climbing right
        case 'left':
            upperBound = lowerB             # Set lower bound for climbing left
            stepSize = -stepSize            # Reverse step size for left direction
        case 'middle':
            return climb_max_y, current     # If staying in the middle, return the current maximum

    while (current <= upperBound) if direction == 'right' else (current >= upperBound):
        y = formula(current)  # Calculate y value at the current x position

        # To handle floating-point precision errors
        if abs(current) < tolerance:
            current = 0.0
        
        # If a local maximum is found, return the previous max
        if climb_max_y > y:
            return climb_max_y, current - stepSize
        
        current += stepSize     # Update the current x position
        climb_max_y = y         # Update the maximum y value
        
    return climb_max_y, current  # Return the found maximum y value and its corresponding x position

def f(x1):
    return 2 - (x1**2)

def g(x2):
    return (0.0051 * x2**5) - (0.1367 * x2**4) + (1.24 * x2**3) - (4.456 * x2**2) + (5.66 * x2) - 0.287


if __name__ == '__main__':
    # Question 1
    # a) step-size 0.5
    print("Question 1:")
    step_A = 0.5
    lowerBound_A = -5.0
    upperBound_A = 5.0
    starting = -2.0
    # print(f"\n\n    Global Maxima: {random_restart(20, f, step_A, lowerBound_A, upperBound_A)}")
    max_y, x_pos = climb(f, step_A, starting, lowerBound_A, upperBound_A)
    print(f"  a) With step-size of 0.5: \n       Global Max: {max_y} at x = {x_pos}")

    # Question 1
    # b) step-size 0.01
    step_A = 0.01
    max_y, x_pos = climb(f, step_A, starting, lowerBound_A, upperBound_A)
    print(f"\n  b) With step-size of 0.01: \n       Global Max: {max_y} at x = {x_pos}")

    # ---------------------------------------------------------------------------------------------
    # Question 2
    # a) Random-restart, 20 restarts
    print("\nQuestion 2:")
    step_B = 0.5
    lowerBound_B = 0
    upperBound_B = 10.0
    max_y, x_pos = random_restart(20, g, step_B, lowerBound_B, upperBound_B)
    print(f"  a) Random-Restart: \n       Global Max: {max_y} at x = {x_pos}")
    # max_y, x_pos = climb(g, step_B, starting, lowerBound_B, upperBound_B)
    # print(f"\nMax: {max_y} at x = {x_pos}")

    starting = 0
    # print(f"\n\n    Global Maxima: {random_restart(20, g, step_B, lowerBound_B, upperBound_B)}")
    max_y, x_pos = climb(g, step_B, starting, lowerBound_B, upperBound_B)
    print(f"\n  b) Hill-Climb: \n       Global Max: {max_y} at x = {x_pos}")



