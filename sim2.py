from collections import Counter
import itertools
import matplotlib.pyplot as plt
import numpy as np
import random
import math
"""
This code is more focused on emulating exactly what is from the paper. That is, from T*k(x0), we permute some k times
and then we compare the frequency of how often permutations occur to the uniform disribution of all permutations. Choosing N=10n!
for histogram accuracy so it touches on same permutations multiple times. I will compare how close it gets to the uniform distribution for varying values of c and see 
how accurate the paper is.
"""
def random_transposition_shuffle(n, k):
    perm = list(range(n))  # Start with identity permutation
    for _ in range(k):
        i, j = random.randint(0, n-1), random.randint(0, n-1)
        perm[i], perm[j] = perm[j], perm[i]
    return tuple(perm)

def total_variation_distance(empirical_dist, n):
    """
    Compute total variation distance between the empirical distribution and uniform over S_n.

    Parameters:
    - empirical_dist: dict of {permutation tuple: frequency / N}
    - n: number of elements in the permutation (size of S_n)

    Returns:
    - float: total variation distance
    """
    uniform_prob = 1 / math.factorial(n)
    total_diff = 0.0

    # Go over all observed permutations
    for prob in empirical_dist.values():
        total_diff += abs(prob - uniform_prob)

    # Account for permutations that were not observed (implicitly have prob = 0)
    num_missing = math.factorial(n) - len(empirical_dist)
    total_diff += num_missing * uniform_prob

    return total_diff #they dont use 1/2


def plot_variation_dist_l(n, N):
    """
    Plots the total variation distances against c values, with either an upper or lower bound curve.

    Args:
        n (int): Size of the list.
        N (int): Number of trials.
        upperOrLower (str): 'u' for upper bound, 'l' for lower bound.
    """
    var_dists = []  # To store total variation distances
    c_vals = [-2 + i * 0.125 for i in range(24)]  # Generate c values

    for cval in c_vals:
        samples = Counter()  # Reset samples for each cval
        firstK = int(math.ceil((n / 2) * math.log(n, 10) + cval * n))  # nlogn + c*n

        # Collect sampled permutations
        for _ in range(N):
            perm = random_transposition_shuffle(n, firstK)  # One sample
            samples[perm] += 1

        # Compute the empirical distribution
        empirical_distribution = {perm: count / N for perm, count in samples.items()}

        # Calculate total variation distance and store it
        var_dists.append(total_variation_distance(empirical_distribution, n))  # Append scalar

    # Plot the variation distances
    plt.figure(figsize=(10, 6))
    plt.plot(c_vals, var_dists, 'bo-', label='Variation Distance')


    # Plot the lower bound curve
    lower_bound_curve = [2 * (1 - math.exp(-math.exp(-2 * c))) for c in c_vals]
    plt.plot(c_vals, lower_bound_curve, 'r--', label='Lower Bound: 1/2 * (1 - exp(-exp(-2c)))')

    plt.title(f"Total Variation Distance vs. c Values: Lower Bound\n  n = {n}")
    plt.xlabel("c values")
    plt.ylabel("Total Variation Distance")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_variation_dist_u(n, N):
    """
        Plots the total variation distances against c values, with either an upper or lower bound curve.
        Args:
            n (int): Size of the list.
            N (int): Number of trials.
            upperOrLower (str): 'u' for upper bound, 'l' for lower bound.
        """
    var_dists = []  # To store total variation distances 
    c_vals = [-.25 + i * 0.25 for i in range(10)]  # Generate c values
    addtl = [2 + i * .75 for i in range(5)]
    c_vals.extend(addtl)

    for cval in c_vals:
        samples = Counter()  # Reset samples for each cval
        firstK = int(math.ceil((n / 2) * math.log(n, 10) + cval * n))  # nlogn + c*n

        # Collect sampled permutations
        for _ in range(N):
            perm = random_transposition_shuffle(n, firstK)  # One sample
            samples[perm] += 1

        # Compute the empirical distribution
        empirical_distribution = {perm: count / N for perm, count in samples.items()}

        # Calculate total variation distance and store it
        var_dists.append(total_variation_distance(empirical_distribution, n))  # Append scalar

    plt.figure(figsize=(10, 6))
    plt.plot(c_vals, var_dists, 'bo-', label='Variation Distance')

    # Find b such that all points are below b * e^(-2c)
    c_vals_np = np.array(c_vals)
    var_dists_np = np.array(var_dists)

    # Define the curve function
    def curve_below(c, b):
        return b * np.exp(-2 * c)

    # Optimize to find the smallest b such that all points lie below the curve
    from scipy.optimize import minimize_scalar

    def obj_function(b):
        # Objective: minimize b while keeping all points below curve
        if b <= 0:
            return float('inf')
        differences = curve_below(c_vals_np, b) - var_dists_np
        if np.any(differences < 0):  # If any point is above the curve
            return float('inf')
        return b

    result = minimize_scalar(obj_function, bounds=(1e-10, 2), method='bounded')
    b_opt = result.x

    # Plot the curve b*e^(-2c)  
    fitted_curve = [curve_below(c, b_opt) for c in c_vals]
    plt.plot(c_vals, fitted_curve, 'g--', label=f'Upper Bound: {b_opt:.4f}*e^(-2c)')
    plt.title(f"Total Variation Distance vs. c Values: Upper Bound\n  n = {n}")
    plt.xlabel("c values")
    plt.ylabel("Total Variation Distance")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()
def plot_vs_uniform(n, k, c, empirical_distribution):
    """
    Plots the uniform distribution and the distribution from the frequency_of_transpositions function.

    Args:
        n (int): Size of the list (1, ..., n).
        k (int): Number of permutations performed.
        empirical_distribution: Dictionary with permutation frequencies.
    """
    # Generate the full list of all possible permutations in lexicographical order
    all_permutations = list(itertools.permutations(range(n)))
    num_permutations = len(all_permutations)  # This equals math.factorial(n)

    # Fill data with zeros for missing permutations
    full_data = [empirical_distribution.get(perm, 0) for perm in all_permutations]

    # Create the uniform distribution
    uniform_distribution = [1 / num_permutations] * num_permutations

    # Calculate total variation distance
    var_dist = total_variation_distance(empirical_distribution, n)

    # Plot both distributions
    positions = np.arange(1, num_permutations + 1)

    plt.figure(figsize=(10, 6))
    plt.bar(positions - 0.2, uniform_distribution, width=0.4, label="Uniform Distribution", color="blue", alpha=0.7)
    plt.bar(positions + 0.2, full_data, width=0.4, label="Empirical Distribution", color="orange", alpha=0.7)

    # Styling the plot
    plt.title(
        f"Comparison of Uniform Distribution vs. Empirical Distribution\n n = {n}, c = {c}(k = {k}), total variation distance = {var_dist:.6f}")
    plt.xlabel("Permutation Index")
    plt.ylabel("Frequency")
    # Set 10 evenly spaced ticks from 0 to n!
    tick_positions = np.linspace(0, num_permutations, 10)
    plt.xticks(tick_positions, [int(x) for x in tick_positions])
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.show()


def main():
    # Simulate many samples from T^{*k}
    n = int(input("Enter the value of n:"))
    N= 15*math.factorial(n)
    c = float(input("Enter c:"))
    #k is fn of c
    k = int(math.ceil((n/2) * math.log(n, 10) + c * n))

    #Simulation!
    """samples = Counter()
    for _ in range(N):
        perm = random_transposition_shuffle(n, k)  # one sample
        samples[perm] += 1"""


    # This approximates the row of T^{*k} corresponding to the identity
    #empirical_distribution = {perm: count / N for perm, count in samples.items()}


    #Uniform
    #plot_vs_uniform(n, k, c, empirical_distribution)
    plot_variation_dist_l(n, N)
    plot_variation_dist_u(n, N)


# Example usage
if __name__ == "__main__":
    main()