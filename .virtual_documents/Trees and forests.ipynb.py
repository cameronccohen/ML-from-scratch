import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.tree import DecisionTreeRegressor as SKDecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor as SKRandomForestRegressor


# Read in tips data -- Run for regression

df = sns.load_dataset('tips')
X = df.drop('tip', axis = 1).to_numpy()
y = df['tip']

# Generate data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.33, random_state=42)


df.describe()


## Helper functions:

def rss(left, right):
    rss_left = np.sum((left - np.mean(left))**2)
    rss_right = np.sum((right - np.mean(right))**2)
    
    return rss_left + rss_right



class DecisionTreeRegressor:
    
    # Node object that constitutes the tree
    class node:
        def __init__(self, feat = None, threshold = None, y_hat = None, left_child = None, right_child = None):
            self.feat = feat
            self.threshold = threshold
            self.y_hat = y_hat
            self.left_child = left_child
            self.right_child = right_child
            
    # Basic fitting function
    def fit(self, X, y, max_depth = 50, min_size = 2, max_features = None):
        self.X = X
        self.y = y
        self.max_depth = max_depth
        self.min_size = min_size
        self.max_features = max_features
        
        dtypes = [np.array(list(self.X[:,d])).dtype for d in range(self.X.shape[1])]
        self.dtypes = ['quant' if (dtype == float or dtype == int) else 'cat' for dtype in dtypes]
        
        self.root = self.split(X, y, 0, max_depth, min_size, max_features)
        
    # Find the best splitting rule among all possible splits
    def find_splitting_rule(self, X, y, max_features):
        
        def order_x_by_y(X, y, feat):
            thresholds = np.unique(X[:, feat])
            category_means = [y[X[:, feat] == threshold].mean() for threshold in thresholds]
            ordered_cats = thresholds[np.argsort(category_means)]
            return ordered_cats
                
        # Later change this to not require all feats (e.g. for RF)
        num_total_feats = X.shape[1]
        if max_features is not None:
            feats = np.random.choice(np.arange(num_total_feats), max_features, replace = False)
        else:
            feats = np.arange(num_total_feats)
        
        min_rss, best_feat, best_thresh = np.inf, None, None
        for feat in feats:
            if self.dtypes[feat] == 'quant':
                thresholds = np.unique(X[:, feat])
                for threshold in thresholds:
                    idxs = X[:, feat] < threshold
                    y_left, y_right = y[idxs], y[~idxs]
            else:
                categories = order_x_by_y(X, y, feat)
                for i in range(len(categories)):
                    category_subset = categories[:i]
                    idxs = np.isin(X[:, feat], category_subset)
                    y_left, y_right = y[idxs], y[~idxs]
                     
            # Only split if some observations in each node; otherwise pick another feature
            if len(y_left) == 0 or len(y_right) == 0:
                continue

            # If its an allowed split, check if best possible split
            split_rss = rss(y_left, y_right)
            if split_rss < min_rss:
                min_rss = split_rss
                best_thresh = threshold
                best_feat = feat
            
        return best_feat, best_thresh
                
    # Recursively split the tree
    def split(self, X, y, current_depth, max_depth, min_size, max_features):
        
        # If we hit stopping rule, make prediction and return leaf node 
        if current_depth == max_depth or X.shape[0] <= min_size:
            return self.node(y_hat = np.mean(y))
        
        ## Otherwise, split and recurse:
        # Find new splitting rule
        feature, threshold = self.find_splitting_rule(X, y, max_features)
        
        # If no split found, stop searching and return leaf node
        if feature is None:
            return self.node(y_hat = np.mean(y))
        
        # Get the observations on each side 
        if self.dtypes[feature] == 'quant':
            idxs = X[:, feature] < threshold
        else:
            idxs = np.isin(X[:, feature], threshold)
        # Assign to new nodes   
        new_node = self.node(feature, threshold)
        new_node.left_child = self.split(X[idxs], y[idxs], current_depth + 1, max_depth, min_size, max_features)
        new_node.right_child = self.split(X[~idxs], y[~idxs], current_depth + 1, max_depth, min_size, max_features)
            
        return new_node
    
    def predict_one(self, x_test):
        # Traverse the tree
        y_hat = None
        current_node = self.root
        while y_hat is None:
            feature, threshold = current_node.feat, current_node.threshold
            if self.dtypes[feature] == 'quant':
                if x_test[feature] < threshold:
                    current_node = current_node.left_child
                else:
                    current_node = current_node.right_child
            else:
                if np.isin(x_test[feature], threshold):
                    current_node = current_node.left_child
                else:
                    current_node = current_node.right_child
            y_hat = current_node.y_hat
            
        return y_hat

    def predict(self, X_test):
        y_hats = np.apply_along_axis(self.predict_one, 1, X_test)
        return y_hats


tree = DecisionTreeRegressor()
tree.fit(X_train, y_train, max_features = 4, max_depth = 6, min_size = 5)
y_test_hat = tree.predict(X_test)


y_test_hat


## Visualize predictions
fig, ax = plt.subplots(figsize = (7, 5))
sns.scatterplot(y_test, tree.predict(X_test))
ax.set(xlabel = r'$y$', ylabel = r'$\hat{y}$', title = r'Test Sample $y$ vs. $\hat{y}$')
sns.despine()

print(f"MSE from our model is {mean_squared_error(y_test, y_test_hat):.2f}")



class RandomForest:
    def fit(self, X, y, max_features, num_trees = 10, max_depth = 50, min_size = 2):
        self.trees = []
        self.num_trees = num_trees
        
        for B in range(self.num_trees):
            sample = np.random.choice(np.arange(X.shape[0]), size = X.shape[0], replace = True)
            X_b = X[sample]
            y_b = y.to_numpy()[sample]

            tree = DecisionTreeRegressor()
            tree.fit(X_b, y_b, max_depth, min_size, max_features)
            self.trees.append(tree)
            
            
    def predict(self, X_test):
        y_hats = np.empty((len(self.trees), X_test.shape[0]))
        for i, tree in enumerate(self.trees):
            y_hats[i] = tree.predict(X_test)
            #print(tree.predict(X_test))
        return np.mean(y_hats, axis = 0)


rf = RandomForest()
rf.fit(X_train, y_train, max_features = 4, num_trees = 100, max_depth = 6, min_size = 5)
y_test_hat = rf.predict(X_test)
#y_test_hat.shape


## Visualize predictions
fig, ax = plt.subplots(figsize = (7, 5))
sns.scatterplot(y_test, rf.predict(X_test))
ax.set(xlabel = r'$y$', ylabel = r'$\hat{y}$', title = r'Test Sample $y$ vs. $\hat{y}$')
sns.despine()

print(f"MSE from our model is {mean_squared_error(y_test, y_test_hat):.2f}")


# SKLearn's model doesn't support factors, so have to one-hot encode

df = sns.load_dataset('tips')
X = df.drop('tip', axis = 1)
X = pd.get_dummies(X, drop_first = True)
X = X.to_numpy()
y = df['tip']

# Generate data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.33, random_state=42)


rf = RandomForest()
rf.fit(X_train, y_train, max_features = 4, num_trees = 100, max_depth = 6, min_size = 5)
y_test_hat = rf.predict(X_test)
print(f"MSE from our RF model is {mean_squared_error(y_test, y_test_hat):.2f}")

rf = SKRandomForestRegressor(max_depth = 6, min_samples_split = 5, n_estimators = 100, max_features = 4)
rf.fit(X_train, y_train)
y_test_hat = rf.predict(X_test)
print(f"SKLearn RF MSE is is {mean_squared_error(y_test, y_test_hat):.2f}")
