import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
import seaborn as sns
from sklearn.model_selection import train_test_split
from scipy import stats
from sklearn.utils.extmath import weighted_mode 
from sklearn.metrics import accuracy_score


# Read in palmer penguisn da
penguins = pd.read_csv('https://raw.githubusercontent.com/allisonhorst/palmerpenguins/1a19e36ba583887a4630b1f821e3a53d5a4ffb76/data-raw/penguins_raw.csv')
cols = ['Species', 'Culmen Length (mm)', 'Culmen Depth (mm)', 'Flipper Length (mm)', 'Body Mass (g)']
df = penguins.loc[:, cols]

# Drop nulls and factorize
df = df.dropna()
df['Species'] = pd.factorize(df['Species'])[0]

X = df.drop('Species', axis = 1).to_numpy()
y = df['Species']


df.describe()


# Plot variation across each axis
g = sns.PairGrid(df, hue = 'Species')
g.map(plt.scatter);


class KNN:
    def __init__(self, k = 3):
        self.k = k
    
    def predict(self, X_train, X_test, y_train):
        # Compute pairwise distances between each train and test point:
        distances = self.__compute_distances(X_train, X_test)
        
        # For each test point, get indices of k closest train points
        k_closest_idx = np.argsort(distances)[:, :self.k]
        
        # Take mode of labels of those k points:
        labels = np.array(y_train)[k_closest_idx]
        k_closest_labels = stats.mode(labels, axis = 1)[0].reshape(-1)
        
        return k_closest_labels        
        
    def __compute_distances(self, a, b):
        # Compute pairwise Euclidian distances using numpy broadcasting
        distances = np.sum(((a.T[:, None] - b.T[:, :, None])**2), axis = 0)
        return distances


# Generate data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.33, random_state=42)

# Run regression
clf = KNN(k = 2)
y_hat = clf.predict(X_train, X_test, y_train)

# Assess performance
print(f"My KNN accuracy is {accuracy_score(y_test, y_hat) * 100:.2f}get_ipython().run_line_magic("")", "")

# Compare to SKlearn
from sklearn.neighbors import KNeighborsClassifier
neigh = KNeighborsClassifier(n_neighbors = 5)
neigh.fit(X_train, y_train)
sk_y_hat = neigh.predict(X_test)
print(f"SKLearn KNN accuracy is {accuracy_score(y_test, sk_y_hat) * 100:.2f}get_ipython().run_line_magic("")", "")


get_ipython().run_line_magic("timeit", " clf.predict(X_train, X_test, y_train)")


get_ipython().run_cell_magic("timeit", " ", """neigh.fit(X_train, y_train)
neigh.predict(X_test)""")


from sklearn.neighbors._base import _get_weights

class KNN:
    def __init__(self, k = 3, weights = 'uniform'):
        self.k = k
        self.weights = weights
    
    def predict(self, X_train, X_test, y_train):
        # Compute pairwise distances between each train and test point:
        distances = self.__compute_distances(X_train, X_test)
        
        # For each test point, get indices of k closest train points
        k_closest_idx = np.argsort(distances)[:, :self.k]
                
        # Take mode of labels of those k points:
        labels = np.array(y_train)[k_closest_idx]
        
        if self.weights == 'distance':
            # Get distances of k closest points per:
            # https://stackoverflow.com/questions/20103779/index-2d-numpy-array-by-a-2d-array-of-indices-without-loops
            k_closest_distances = distances[np.arange(k_closest_idx.shape[0])[:, None], k_closest_idx]
            weights = 1/k_closest_distances
            k_closest_labels = weighted_mode(labels, weights, axis = 1)[0].reshape(-1)
        else:
            k_closest_labels = stats.mode(labels, axis = 1)[0].reshape(-1)
        
        return k_closest_labels        
        
    def __compute_distances(self, a, b):
        # Compute pairwise Euclidian distances using numpy broadcasting
        distances = np.sqrt(np.sum(((a.T[:, None] - b.T[:, :, None])**2), axis = 0))
        return distances


# Run regression
clf = KNN(k = 10, weights = 'distance')
y_hat = clf.predict(X_train, X_test, y_train)

# Assess performance
print(f"My KNN accuracy is {accuracy_score(y_test, y_hat) * 100:.2f}get_ipython().run_line_magic("")", "")

# Compare to SKlearn
from sklearn.neighbors import KNeighborsClassifier
neigh = KNeighborsClassifier(n_neighbors = 10, weights = 'distance')
neigh.fit(X_train, y_train)
sk_y_hat = neigh.predict(X_test)
print(f"SKLearn KNN accuracy is {accuracy_score(y_test, sk_y_hat) * 100:.2f}get_ipython().run_line_magic("")", "")



