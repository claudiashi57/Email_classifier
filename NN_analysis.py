import numpy as np
import matplotlib.pyplot as plt

some_results = np.load('results.npy')
more_results = np.load('1NNresults.npy')


averages = some_results.mean(axis=2)
splits = [0.9, 0.75, 0.5, 0.25]

plt.plot(splits, more_results.T)
plt.legend(["k=1", "k=3", "k=5"])
plt.title("kNN with 1 norm")
plt.xlabel("Percent of Data Used for Training")
plt.ylabel("Testing Accuracy (1 - error)")
plt.savefig("1norm.png")


plt.figure()
plt.plot(splits, averages[0:3, :].T)
plt.legend(["k=1", "k=3", "k=5"])
plt.title("kNN with 2 norm")
plt.xlabel("Percent of Data Used for Training")
plt.ylabel("Testing Accuracy (1 - error)")
plt.savefig("2norm.png")

plt.figure()
plt.plot(splits, averages[3:6, :].T)
plt.legend(["k=1", "k=3", "k=5"])
plt.title("kNN with INF norm")
plt.xlabel("Percent of Data Used for Training")
plt.ylabel("Testing Accuracy (1 - error)")
plt.savefig("infnorm.png")

plt.figure()
plt.plot(splits, more_results[0, :].T, splits, averages[0, :].T, splits, averages[4, :].T)
plt.legend(["1NN (1 norm)", "1NN (2 norm)", "3NN (INF norm)"])
plt.title("Comparison of best k's for each choice of norm")
plt.xlabel("Percent of Data Used for Training")
plt.ylabel("Testing Accuracy (1 - error)")
plt.savefig("bestks.png")


# 3nn INF
# 1nn 1, 2