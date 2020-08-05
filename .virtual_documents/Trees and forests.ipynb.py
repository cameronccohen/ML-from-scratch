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

# Generate data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.33, random_state=42)


df.describe()



