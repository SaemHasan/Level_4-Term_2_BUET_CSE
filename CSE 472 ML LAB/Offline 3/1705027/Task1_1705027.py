import numpy as np
from scipy.stats import multivariate_normal
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import pandas as pd

DATASET = "data6D.txt"
# DATASET = "3D_spatial_network.txt"
K_RANGE = range(2, 11)
EPSILON = 1e-6
NORMALIZE = False


def load_data(file_name):
    """Load the data from a file and return it as a matrix."""
    data = []
    SEPERATOR = " "
    with open(file_name, 'r') as f:
        for line in f:
            data.append(list(map(float, line.strip().split(SEPERATOR))))
    data = np.array(data)
    if NORMALIZE:
        data = normalize(data)
    return data

def load_dataset_from_csv(file_name):
    """
    function for reading data from csv
    and processing to return a 2D feature matrix and a vector of class
    :return:
    """
    SEPERATOR = ";"
    data = []
    first = True
    with open(file_name, 'r') as f:
        for line in f:
            if first:
                first = False
                continue
            data.append(list(map(float, line.strip().split(SEPERATOR))))
    data = np.array(data)
    # print(data)
    if NORMALIZE:
        data = normalize(data)
    # print(data)
    return data

def normalize(data):
    # data of shape (n, m)
    # n = number of data points
    # m = number of features
    m = data.shape[1]
    for i in range(m):
        data[:, i] = (data[:, i] - np.mean(data[:, i])) / np.std(data[:, i])
    return data


def plot_2D_data(x, y, title, x_label, y_label):
    plt.figure()
    plt.plot(x, y)
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.show()

def run_pca(X):
    pca = PCA(n_components=2)
    principalComponents = pca.fit_transform(X)
    
    # print(pca.explained_variance_ratio_)
    # print(principalComponents)
    return np.array(principalComponents)

# Gaussian Mixture Model
class GMM:
    def __init__(self, data, k, mean_selection='data_points'):
        self.n = data.shape[0] # number of data points
        self.m = data.shape[1] # number of features
        self.data = data # data matrix
        self.k = k # number of clusters
        
        if mean_selection == 'random':
            self.means = np.random.rand(k, self.m)
        elif mean_selection == 'data_points':
            self.means = data[np.random.choice(self.n, k, replace=False), :]
        else:
            self.means = np.random.rand(k, self.m)    
        
        self.covariances = np.array([np.eye(data.shape[1])] * k) + np.array([np.eye(data.shape[1])] * k) * EPSILON
        self.weights = np.ones(k) / k # weights of clusters k * 1 (pi_k)
        self.gamma_nk = np.zeros((self.n, k)) # gamma_nk = p(z_n = k | x_n, theta)
        self.totals = np.zeros((self.n, 1)) # totals = sum_k gamma_nk

    def print_values(self):
        print("Initial values:")
        print("Means: \n", self.means)
        print("Covariances: \n", self.covariances)
        print("Weights: \n", self.weights)
    
    '''def multivariate_normal(self, x, mean, covariance):
        """pdf of the multivariate normal distribution."""
        x_m = x - mean
        return np.exp(-0.5 * np.sum(np.dot(x_m, np.linalg.inv(covariance)) * x_m, axis=1)) / np.sqrt((2 * np.pi) ** self.m * np.linalg.det(covariance))
    '''
    def expectation_step(self):
        "E step for EM Algorithm"

        for k in range(self.k):
            mul_normal = multivariate_normal.pdf(self.data, self.means[k], self.covariances[k], allow_singular=True)
            # mul_normal_self = self.multivariate_normal(self.data, self.means[k], self.covariances[k])
            self.gamma_nk[:, k] = self.weights[k] * mul_normal
        
        self.totals = np.sum(self.gamma_nk, axis=1)
        self.gamma_nk /= np.expand_dims(self.totals, axis=1)
    
    def maximization_step(self):
        "M step for EM Algorithm"
        for k in range(self.k):
            self.means[k] = np.sum(self.gamma_nk[:, k] * self.data.T, axis=1) / np.sum(self.gamma_nk[:, k])
            self.covariances[k] = np.dot((self.gamma_nk[:, k] * (self.data - self.means[k]).T), (self.data - self.means[k])) / np.sum(self.gamma_nk[:, k])
            self.weights[k] = np.sum(self.gamma_nk[:, k]) / self.n
    
    def get_likelihood(self):
        sample_likelihood = np.log(self.totals)
        return np.sum(sample_likelihood)
    
    def get_likelihood_each_sample(self, data):
        likelihood = np.zeros(data.shape[0])
        for k in range(self.k):
            mul_normal = multivariate_normal.pdf(data, self.means[k], self.covariances[k], allow_singular=True)
            likelihood += self.weights[k] * mul_normal
        # print(likelihood.shape)
        return np.log(likelihood)
    
    def fit(self, max_iter=100, verbose=False):
        for i in range(max_iter+1):
            self.expectation_step()
            self.maximization_step()
            if i % 10 == 0 and verbose:
                print("k: ",self.k,"\tIteration: ", i, "\tLikelihood: ", self.get_likelihood())
            
        return self.get_likelihood()
    
    def fit_show(self, max_iter=100, verbose=False):
        # Turn on interactive mode
        plt.ion()
        for i in range(max_iter+1):
            self.expectation_step()
            self.maximization_step()
            
            if self.m == 2:
                self.display_contour(self.data, self.means, i)
            
            if i % 10 == 0 and verbose:
                print("Iteration: ", i, "\tLikelihood: ", self.get_likelihood())
        # Turn off interactive mode
        plt.ioff()
        plt.show()
    
    def display_contour(self, data, mean, iter):
        plt.clf()
        plt.scatter(data[:, 0], data[:, 1], .8)
        plt.scatter(mean[:, 0], mean[:, 1], 1000, marker="+")

        x_max = np.max(data[:, 0])
        x_min = np.min(data[:, 0])
        y_max = np.max(data[:, 1])
        y_min = np.min(data[:, 1])
        x = np.linspace(x_min-5, x_max+5, 100)
        y = np.linspace(y_min-5, y_max+5, 100)
        X, Y = np.meshgrid(x, y)
        XX = np.array([X.ravel(), Y.ravel()]).T
        Z = -self.get_likelihood_each_sample(XX)
        Z = Z.reshape(X.shape)
        
        X= np.array(X)
        Y= np.array(Y)
        Z= np.array(Z)
        # print(X.shape)
        # print(Y.shape)
        # print(Z.shape)
        # print(np.unique(Z))

        CS = plt.contour(X, Y, Z, norm=LogNorm(vmin=1.0, vmax=1000.0), levels=np.logspace(0, 3, 10))
        # CS = plt.contour(X, Y, Z)
        CB = plt.colorbar(CS, shrink=0.8, extend='both')

        plt.title('Iteration : ' + str(iter))
        plt.axis('tight')
        plt.draw()
        plt.pause(0.1)
    


    
# run GMM for range of k
class GMM_Runner:
    def __init__(self, data, k_range):
        self.data = data
        self.k_range = k_range
        self.likelihoods = np.zeros(len(k_range))
    
    def run_gmm(self, mean_selection= "data_points" ,max_iter=100, verbose=False):
        for i, k in enumerate(self.k_range):
            gmm = GMM(self.data, k, mean_selection=mean_selection)
            # gmm.print_values()
            self.likelihoods[i] = gmm.fit(max_iter=max_iter, verbose=verbose)
        plot_2D_data(self.k_range, self.likelihoods, "Log-Likelihood Plot vs k", "k", "Log-Likelihood")
        
    def run_gmm_for_k(self, k_star, mean_selection="data_points", max_iter=100, verbose=False):
        gmm = GMM(self.data, k_star, mean_selection=mean_selection)
        gmm.fit_show(max_iter=max_iter, verbose=verbose)
        # gmm.display_contour(self.data)
        print("Final values:")
        print("Means: \n", gmm.means)
    
    def run_gmm_greater_than_2D(self, k_star, mean_selection="data_points", max_iter=100, verbose=False):
        gmm = GMM(self.data, k_star, mean_selection=mean_selection)
        gmm.fit(max_iter=max_iter, verbose=verbose)

        print("Final values:")
        print("Means: \n", gmm.means)

        data_2D = run_pca(self.data)
        gmm_2d = GMM(data_2D, k_star, mean_selection=mean_selection)
        gmm_2d.fit_show(max_iter=max_iter, verbose=False)



if __name__ == "__main__":
    data = load_data(DATASET)
    # data = load_dataset_from_csv(DATASET)
    print(data.shape)
    gmm_runner = GMM_Runner(data, K_RANGE)
    gmm_runner.run_gmm(mean_selection="data_points", max_iter=300, verbose=True)