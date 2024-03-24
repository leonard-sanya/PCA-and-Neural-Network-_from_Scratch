# Principal Component Analysis

- PCA is a simple dimensionality reduction technique that can capture linear correlations between the features. For a given (standardized) data, PCA can be calculated by eigenvalue decomposition of covariance (or correlation) matrix of the data, or Singular Value Decomposition (SVD) of the data matrix. The data standardization includes mean removal and variance normalization.
- We used iris dataset which consists of 50 samples from each of three species of Iris. The rows being the samples and the columns being: Sepal Length, Sepal Width, Petal Length and Petal Width.

Steps in performing the PCA
1. Data Normalization/Standardization
$$\text{Mean} = \bar{X} = \frac{1}{n}\sum_{i=1}^{n} X_i $$
$$\text{Variance} = \sigma^2 = \frac{1}{n-1}\sum_{i=1}^{n}(X_i - \bar{X})^2 $$
Where $X_{i}$ represent the $i^{th}$ variable in the data set.
 To standardize a variable X using its mean ($\bar{X}$) and standard deviation ($\sigma$):

$$\text{Standardized }X = \frac{X - \bar{X}}{\sigma} $$

In this code, $\bar{X}$ represents the sample mean of the variable X, $\sigma$ represents the sample standard deviation of X, and X represents the original unstandardized variable. The resulting standardized variable has a mean of 0 and a standard deviation of 1.

2. Compute the covariance matrix

Determine the covariance matrix of the data set

$$\text{Cov}(X_i,X_j) = \frac{1}{n-1}\sum_{k=1}^{n}(X_{i}^{k}-\bar{X}i)(X_{j}^{k}-\bar{X}_j)$$

$$\mathbf{S} = \frac{1}{n-1}\mathbf{X}^\top\mathbf{X},$$


where $\mathbf{X}$ is the $n \times p$ matrix of standardized data, and $\mathbf{S}$ is the $p \times p$ sample covariance matrix. The $(i,j)^{th}$ entry of $\mathbf{S}$ is given by

$$s_{i,j} = \frac{1}{n-1}\sum_{k=1}^{n} x_{k,i}x_{k,j},$$


where $x_{k,i}$ and $x_{k,j}$ are the $i$th and $j^{th}$ standardized variables, respectively, for the $k^{th}$ observation.
It is important to note that the covariance matrix is a square, postive definate ,symmetric matric of dimension p by p where p is the number of variables

3. Compute the eigenvalue and eigenvector of our covariance matrix
   
Compute eigen values and standardised eigen vectors of the covariance matrix
Let $A$ be the covariance matrix of a dataset $X$, then the eigenvalue equation is given by:


$$A\mathbf{v} = \lambda \mathbf{v}$$

where $\mathbf{v}$ is the eigenvector of $A$ and $\lambda$ is the corresponding eigenvalue.

To find the eigenvalues and eigenvectors, we can solve this equation using the characteristic polynomial of $A$:
$$\det(A - \lambda I) = 0$$

where $I$ is the identity matrix of the same size as $A$. Solving for $\lambda$ gives the eigenvalues, and substituting each eigenvalue back into the equation $A\mathbf{v} = \lambda \mathbf{v}$ gives the corresponding eigenvectors.


4. Choose the number component and project your data on the new space

*   To be able to visualize our data on the new subspace we will choose 2 PCs 
*   Retain at least 95% percent from the cumulayive explained variance

# Neural Networks 

Neural networks are a class of machine learning algorithms inspired by the structure and function of the human brain. They consist of interconnected nodes (neurons) arranged in layers. Each neuron receives input signals, processes them, and generates an output signal.

This neural network implementation includes the following components:

1. Weight initialization
   - The neural network is initialized with random weights and biases.
2. Forward Pass
   - Input data is fed forward through the network, computing the output of each layer using activation functions.
     $$\hat{y} = \sigma\big(\sum_{i=1}^k W_i X_i +b \big)$$

     where $σ(z)= \frac{1}{1+ e^{-z}} \hspace{0.2cm}\text{and}\hspace{0.2cm} z= XW + b $.
    - Derivative of Logistic/sigmoid function with respective to $z$ as 
$$\sigma^{'}(z)= \sigma(z)(1-\sigma(z))$$
3. Compute the loss between the model prediction and the true value using the Negative log likelihood or Cross-entropy loss given as

$$\mathcal{L}= -\frac{1}{N}\sum_{i= 1}^{N} \big(y_{true} * log (y_{pred} )+ (1-y_{true})*\log (1-y_{pred}) \big)$$

4. Backpropagation and Updating Weights
- The errors are propagated backward by computing the gradients of the loss through the network to adjust the weights and biases using gradient descent.

$$\theta^{(k)} \leftarrow \theta^{(k)} - \eta \frac{\partial \mathcal L}{\partial \theta^{(k)}} $$

$${b}^{(k)}  \leftarrow {b}^{(k)} - \eta \frac{\partial \mathcal L}{\partial {b}^{(k)}}$$
where $\eta$ is the learning rate.
