# PCA-and-Neural-Network
##PCA
Principal Component Analysis (PCA) is a simple dimensionality reduction technique that can capture linear correlations between the features. For a given (standardized) data, PCA can be calculated by eigenvalue decomposition of covariance (or correlation) matrix of the data, or Singular Value Decomposition (SVD) of the data matrix. The data standardization includes mean removal and variance normalization.


We useD iris dataset which consists of 50 samples from each of three species of Iris. The rows being the samples and the columns being: Sepal Length, Sepal Width, Petal Length and Petal Width.

Compute the mean as follows for each variables as
\begin{align*}
\text{Mean} &= \bar{X} = \frac{1}{n}\sum_{i=1}^{n} X_i \
\end{align*}
\begin{align*}
\text{Variance} &= \sigma^2 = \frac{1}{n-1}\sum_{i=1}^{n}(X_i - \bar{X})^2
\end{align*}
Where $X_{i}$ represent the $i^{th}$ variable in the data set.
 To standardize a variable X using its mean ($\bar{X}$) and standard deviation ($\sigma$):

\begin{align*}
\text{Standardized }X &= \frac{X - \bar{X}}{\sigma}
\end{align*}

In this code, $\bar{X}$ represents the sample mean of the variable X, $\sigma$ represents the sample standard deviation of X, and X represents the original unstandardized variable. The resulting standardized variable has a mean of 0 and a standard deviation of 1.
