from Task1_1705027 import *

if __name__ == "__main__":
    data = load_data(DATASET)
    gmm_runner = GMM_Runner(data, K_RANGE)
    gmm_runner.run_gmm_for_k(k_star=6, mean_selection="random", max_iter=100, verbose=False)

#data2D --> 3
#data3D --> 6