from Task1_1705027 import *

data = load_data(DATASET)
gmm_runner = GMM_Runner(data, K_RANGE)
gmm_runner.run_gmm_for_k(3)
