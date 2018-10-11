import numpy as np
import math


def minimize(func_to_minimize, derivative, starting_point, eta=1e-5):
	counts = 0
	current_point = starting_point
	current_derivative = derivative(current_point)

	while not np.isclose(current_derivative, 0, atol=1e-9):
		counts += 1
		current_point = current_point - (eta * current_derivative)
		current_derivative = derivative(current_point)

	print(current_point)
	print(counts)
	print("Found minimum value of {} at x = {} after {} iterations.".format(func_to_minimize(current_point), current_point, counts))

def main():
	minimize(lambda x : np.power(x - 4, 2) + 2*np.exp(x), lambda x : 2 * (-4 + x + np.exp(x)), 2)

if __name__ == '__main__':
	main()