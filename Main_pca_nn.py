import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt
from Neural_Netwoks import NeuralNetwork
from PCA import PCA

# Data generation for PCA
iris = load_iris()
x= iris['data']
y = iris['target']

n_samples, n_features = x.shape
df = pd.DataFrame(
    iris.data,
    columns=iris.feature_names
    )
df["label"] = iris.target
column_names = iris.feature_names

# Compute the correlation matrix
corr_matrix = df.corr()
# Plot the correlation matrix using a heatmap




#Data generation for Neural Networks
var = 0.2
n = 800
class_0_a = var * np.random.randn(n//4,2)
class_0_b =var * np.random.randn(n//4,2) + (2,2)

class_1_a = var* np.random.randn(n//4,2) + (0,2)
class_1_b = var * np.random.randn(n//4,2) +  (2,0)

X = np.concatenate([class_0_a, class_0_b,class_1_a,class_1_b], axis =0)
Y = np.concatenate([np.zeros((n//2,1)), np.ones((n//2,1))])
X.shape, Y.shape

# Shuffle the data
rand_perm = np.random.permutation(n)

X = X[rand_perm, :]
Y = Y[rand_perm, :]

X = X.T
Y = Y.T

# train test split
ratio = 0.8
X_train = X [:, :int (n*ratio)]
Y_train = Y [:, :int (n*ratio)]

X_test = X [:, int (n*ratio):]
Y_test = Y [:, int (n*ratio):]




model_NN = NeuralNetwork(2,10,1,0.001,10000)
model_pca = PCA(n_component=2)


def main():
     
     print("Choose model type you want to use")
     user = input("1 for Neural network and 2 for PCA ")
     if user=="1":
         plt.scatter(X_train[0,:], X_train[1,:], c=Y_train[0,:])
         plt.show()
         model_NN.fit(X_train,Y_train,X_test, Y_test)
     else:
         print(df)

         plt.figure(figsize=(16,4))
         plt.subplot(1, 3, 1)
         plt.title(f"{column_names[0]} vs {column_names[1]}")
         plt.scatter(x[:, 0], x[:, 1], c=y)
         plt.subplot(1, 3, 2)
         plt.title(f"{column_names[1]} vs {column_names[2]}")
         plt.scatter(x[:, 1], x[:, 3], c=y)
         plt.subplot(1, 3, 3)
         plt.title(f"{column_names[2]} vs {column_names[3]}")
         plt.scatter(x[:, 2], x[:, 3], c=y)
         plt.show()

         sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
         plt.show()

         model_pca.fit(x)
         new_X = model_pca.transform(x,y)

         
if __name__ == "__main__":
  main()
