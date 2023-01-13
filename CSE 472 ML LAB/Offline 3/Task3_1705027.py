from Task1_1705027 import *

DATASET = "data6D.txt"

if __name__ == "__main__":
    data = load_data(DATASET)
    gmm_runner = GMM_Runner(data, K_RANGE)
    gmm_runner.run_gmm_greater_than_2D(k_star=5, mean_selection="random", max_iter=50, verbose=False)

# 3D --> 4
# 6D --> 5