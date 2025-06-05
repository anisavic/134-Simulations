import matplotlib.pyplot as plt
import numpy as np
import math
from collections import Counter, defaultdict
import random


def main():
    # Take user input for n and M
    n = int(input("Enter the value of n: "))
    M = int(input("Enter number of trials: "))
    c = float(input("Enter c:"))
    #k is fn of c
    k = int(math.ceil((n/2) * math.log(n, 10) + c * n))
    # Create the initial list
    initial_list = list(range(1, n + 1))  # Creates a list [1, 2, ..., n]

    # Define range of c values
    c_values = [-.1 + i * 0.125 for i in range(45)]
    var_distances = []
    #Plot variation distances
    #plot_variation_dist(initial_list, n, M, 'u')


    #plot uniform
    k = int((n/2) * math.log(n, 10) + c * n)  # nlogn + c*n
    plot_vs_uniform(n, k, c, frequency_of_transpositions(k, M, initial_list))


# Example transpositions function
def random_transpositions(lst, k):
    n = len(lst)
    result = lst.copy()
    frequency = [0.0] * n
    for _ in range(k):
        i = random.randrange(n)
        j = random.randrange(n)
        result[i], result[j] = result[j], result[i]

    return result

#given # of permutations(k), the number of trials(M), and the initial list (1,...,n),
# goal is to output the frequency average. We do multiple trials to get a more accurate feel for how close it is to the uniform dist
def frequency_of_transpositions(k, M, lst): #k: number of permutations
    n = len(lst)
    listOfFrequencies = [[0.0] * n for _ in range(M)]
    sumFrequencies = [0.0] * n
    for m in range(M): # Repeat for M trials
        lst = lst.copy()
        for _ in range(k): # Perform k random transpositions
            i = random.randrange(n)
            j = random.randrange(n)
            lst[i], lst[j] = lst[j], lst[i]
            for index in range(n): #update frequencies
                if index + 1 == lst[index]:
                    listOfFrequencies[m][index] += 1
        for i in range(n):
            listOfFrequencies[m][i] /= k


        for index in range(n):
            sumFrequencies[index] += listOfFrequencies[m][index]
    for index in range(n):
        sumFrequencies[index] /= M
    total_sum = sum(sumFrequencies)

    sumFrequencies = [freq / total_sum for freq in sumFrequencies]
    return sumFrequencies


def plot_vs_uniform(n, k, c , empirical_distribution):
    """
    Plots the uniform distribution and the distribution from the frequency_of_transpositions function.

    Args:
        n (int): Size of the list (1, ..., n).
        k (int): Number of permutations performed.
        M (int): Number of trials.
        frequency data (frequency_of_transpositions).
    """
    # Generate the original list
    original_list = list(range(1, n + 1))

    # Create the uniform distribution
    uniform_distribution = [1 / n] * n

    # Plot both distributions
    positions = np.arange(1, n + 1)  # Positions 1, 2, ..., n

    # Calculate total variation distance
    var_dist = get_var_dist(n, empirical_distribution)

    plt.figure(figsize=(10, 6))
    plt.bar(positions - 0.2, uniform_distribution, width=0.4, label="Uniform Distribution", color="blue", alpha=0.7)
    plt.bar(positions + 0.2, empirical_distribution, width=0.4, label="Output Distribution", color="orange", alpha=0.7)

    # Styling the plot
    plt.title(
        f"Comparison of Uniform Distribution vs. Empirical Distribution\n n = {n}, c = {c}(k = {k}), total variation distance = {var_dist:.6f}")
    plt.xlabel("Permutation Index")
    plt.ylabel("Frequency")
    plt.xticks(positions)  # Ensure proper labeling of x-axis
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.show()


def plot_variation_distances(c_values, var_distances, n, M, lower_bound):
    """
    Plots the total variation distances against c values.

    Args:
        c_values (list): List of c values used
        var_distances (list): Corresponding total variation distances
        upper_bound (bool): Whether to plot the upper bound curve
    """
    plt.figure(figsize=(10, 6))
    plt.plot(c_values, var_distances, 'bo-', label='Variation Distance')

    if lower_bound:
        upper_bound_vals = [2 * (1 - math.exp(-math.exp(-2 * c))) for c in c_values]
        plt.plot(c_values, upper_bound_vals, 'r--', label='Upper Bound')
        plt.fill_between(c_values, var_distances, upper_bound_vals, alpha=0.2, color='red')

    plt.title(f"Total Variation Distance vs. c Values\n  M = {M}, n = {n}")
    plt.xlabel("c values")
    plt.ylabel("Total Variation Distance")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_variation_dist(initial_list, n, M, upperOrLower):
    """
    Plots the total variation distances against c values, with either an upper or lower bound curve.

    Args:
        c_vals (list): List of c values used.
        var_dists (list): Corresponding total variation distances.
        n (int): Size of the list.
        M (int): Number of trials.
        upperOrLower (str):
            'u' -> Fit the curve b * e^(-2c) such that all points are below the curve.
            'l' -> Plot the lower bound curve [2 * (1 - math.exp(-math.exp(-2 * c)))].
    """
    var_dists = []
    c_vals = []
    if upperOrLower == 'u':
        c_vals = [-.6 + i * 0.125 for i in range(45)]
    else:
        c_vals = [-2 + i * 0.125 for i in range(24)]

    for cval in c_vals:
        firstK = int(math.ceil((n / 2) * math.log(n, 10) + cval * n))  # nlogn + c*n
        frequencies = frequency_of_transpositions(firstK, M, initial_list)
        total_var_dist = get_var_dist(n, frequencies)
        var_dists.append(total_var_dist)

    plt.figure(figsize=(10, 6))
    plt.plot(c_vals, var_dists, 'bo-', label='Variation Distance')


    if upperOrLower == 'u':
        # Find b such that all points are below b * e^(-2c)
        c_vals_np = np.array(c_vals)
        var_dists_np = np.array(var_dists)

        # Define the curve function
        def curve_below(c, b):
            return b * np.exp(-2 * c)

        # Optimize to find the smallest b such that all points are below the curve
        from scipy.optimize import minimize_scalar

        def obj_function(b):
            # Objective: Make the minimum difference between the curve and the points positive
            differences = [curve_below(c, b) - d for c, d in zip(c_vals_np, var_dists_np) if curve_below(c, b) > d]
            
            if not differences:  # Handle the case where the list is empty
                return float('inf')  # Return a large value to penalize this case during optimization
            
            return max(differences)

        # Start the optimization
        result = minimize_scalar(obj_function, bounds=(0, 10), method='bounded')  # Adjust bounds if needed
        b_opt = result.x

        # Plot the curve b*e^(-2c)
        fitted_curve = [curve_below(c, b_opt) for c in c_vals]
        plt.plot(c_vals, fitted_curve, 'g--', label=f'Theoretical Bound: {b_opt:.4f}*e^(-2c)')

    elif upperOrLower == 'l':
        # Plot the lower bound curve
        lower_bound_curve = [0.5 * (1 - math.exp(-math.exp(-2 * c))) for c in c_vals]
        plt.plot(c_vals, lower_bound_curve, 'r--', label='Lower Bound: 1/2 * (1 - exp(-exp(-2c)))')


    if upperOrLower == 'u':
        plt.title(f"Total Variation Distance vs. c Values: Upper Bound\n  n = {n}")
    else:
        plt.title(f"Total Variation Distance vs. c Values: Lower Bound\n  n = {n}")

    plt.xlabel("$c$ values")


def get_var_dist(n, empirical_dist):
    """
    Calculate total variation distance between frequency data and uniform distribution.
    
    Args:
        n (int): Size of the list
        empirical_dist (list): List of frequencies
        
    Returns:
        float: Total variation distance
    """
    uniform_prob = float(1 / n)
    total_diff = 0.0

    # Go over all observed permutations
    for prob in empirical_dist:
        total_diff += abs(prob - uniform_prob)

    # Account for permutations that were not observed (implicitly have prob = 0)
    num_missing = n - len(empirical_dist)
    total_diff += num_missing * uniform_prob
    return total_diff





# Example usage
if __name__ == "__main__":
    main()