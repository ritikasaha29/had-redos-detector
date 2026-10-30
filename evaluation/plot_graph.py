import matplotlib.pyplot as plt
import numpy as np

n = np.array([10, 50, 100, 200, 400])

linear = n
quadratic = n**2
cubic = n**3

plt.plot(n, linear, label="O(n)")
plt.plot(n, quadratic, label="O(n²)")
plt.plot(n, cubic, label="O(n³)")

plt.xlabel("Input Size (n)")
plt.yscale("log")
plt.ylabel("Time (log scale)")
plt.title("Complexity Growth")

plt.legend()
plt.grid()

plt.savefig("evaluation/graphs/complexity.png")
plt.show()
