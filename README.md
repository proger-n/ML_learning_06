# Unsupervised Learning <br/> Dimensionality Reduction

Summary: This project will cover some unsupervised learning tasks focusing on dimensionality reduction.

💡 [Tap here](https://new.oprosso.net/p/4cb31ec3f47a4596bc758ea1861fb624) **to leave your feedback on the project**. It's anonymous and will help our team make your educational experience better. We recommend completing the survey immediately after the project.

## Contents

1. [Chapter I. Preamble](#chapter-i-preamble)
2. [Chapter II. Introduction](#chapter-ii-introduction) \
    2.1. [Что такое уменьшение размерности](#what-is-dimensional-reduction) \
    2.2. [Matrix Factorization methods](#matrix-factorization-methods) \
    2.3. [Manifold learning](#manifold-learning) 
3. [Chapter III. Goal](#chapter-iii-goal) 
4. [Chapter IV. Instructions](#chapter-iv-instructions)
5. [Chapter V. Task](#chapter-v-task) \
   5.1. [Dataset](#dataset) \
         5.1.1. [Book Recommendation Dataset](#book-recommendation-dataset) \
         5.1.2. [Mnist](#mnist) \
         5.1.3. [Background models challenge](#background-models-challenge) \
   5.2. [Task](#task) \
   5.3. [Submission](#submission)
6. [Chapter VI. Bonus part](#chapter-vi-bonus-part)

## Chapter I. Preamble

In previous projects, we focused on supervised learning - when each data sample we describe with features, 
having a target variable that we want to learn how to predict. In this project, we will get acquainted with the 
problems of unsupervised learning - the case when we do not have the target variable, but we still want to find patterns 
in the data and use them for good.

We can begin with the dimensionality reduction problem, one of the unsupervised learning subtasks. 
To explain the use cases of these models, let’s go back to supervised learning. 
Imagine a situation where we have already trained some baseline model and realized that its quality is not very good. 
Then we decided to correct this situation by adding new features. But we don’t really want to create them manually, 
so we decided to encode each feature into several other by using the formula *sin (ax + b) * c*. 
Where a, b and c are the parameters which we iterate through along the given grid. How dangerous is this situation? 
It is likely that most of these features will be useless. 
While some of them can even be randomly transformed in such a way that model will find a pattern between the 
target variable and this feature, which will lead to model overfitting.

This problem is closely related to Curse of Dimensionality, which was first introduced by  Richard E. Bellman. 
If we will add and add new features (in other words, increase the dimension), 
then in order to provide the same density of observations, we will need n times more observations each time 
(see the figure below). Thus in order to obtain an estimate of a function unknown to us as accurately as before, 
the number of necessary data must grow exponentially with increasing dimension.

![](misc/images/preamble.png)

Source: [What do you mean by Curse of Dimensionality? What are the different ways to deal with it?](https://www.i2tutorials.com/what-do-you-mean-by-curse-of-dimensionality-what-are-the-different-ways-to-deal-with-it/)

For the task of supervised learning, this problem can result in:
1. that we increase the distance between samples
2. that, as mentioned earlier, we can add noise, from which the model will extract patterns and thereby overfit
3. that models fitting time and memory usage will increase

Fortunately, not all models are equally affected by this problem. 
Think for which of the models that we already studied this problem is relevant. Explain why.

To deal with this problem, you can perform feature selection before training the model. 
For example, train a lasso model and look at the coefficients of the model. 
Or use other approaches that you use in the 3rd project.

While the purpose of the current project is to introduce you to a number of approaches that are used for 
dimensionality reduction. Besides this, some of these approaches can be applied to other tasks of unsupervised learning. 
For example, for compressing media content, for highlighting the background of a video, for building recommender systems, 
for interpreting results, and much more. In the project, we will apply some of them.

## Chapter II. Introduction

### What is Dimensional Reduction

In the case of dimensionality reduction, our main goal is to transform data from high-dimensional feature space to a 
low-dimensional feature space. And do it in such a way that information about the important properties of the samples 
should not be lost: similar samples should remain similar, and different samples, on the contrary, should be different. 
Mostly, however, this approach is used to visualize datasets in two-dimensional or three-dimensional space 
(see the figure below) or to speed up the training of other machine learning models

![Dimensional Reduction](misc/images/dimensionality_reduction.png)

Images dimensionality reduction of hand-written numbers from MNIST Dataset. 
Source: https://neptune.ai/blog/dimensionality-reduction

Within the problem of dimensionality reduction, there are many approaches, 
and to make it easier to navigate through them, you can divide them into groups.

### Matrix Factorization methods

Above, we mentioned that we do not have a target variable, but features that we collect in the form of a matrix. 
From linear algebra we know that matrices can be decomposed in different ways. 
We can represent the matrix as a product of lower triangular and upper triangular (LU-decomposition), 
or as a product of an orthogonal matrix and upper triangular (QR-decomposition), 
or find eigenvectors and use them as a basis for matrix decomposition, etc. 
The most important of all these decompositions is the Singular Value Decomposition (SVD) decomposition.

The SVD is a factorization, when input matrix $$ M_{n \times m} $$ is decomposed into:

$$ M_{n \times m}=U_{n \times r}\Sigma_{r \times r}V_{m \times r}^T, \text{ where } r =\min (n, m)$$

where
* *U* and *V* are orthogonal matrices, 
* $`\Sigma`$ is diagonal matrix that contains singular values of matrix A

Recall that orthogonal matrices are just rotations and reflections, that leave the sphere the same. 
From a geometric point of view this means that an arbitrary matrix can be decomposed as a rotation, 
followed by a stretch, followed by another rotation.

![](misc/images/rotation.png)

Source: [wikipedia](https://en.wikipedia.org/wiki/Singular_value_decomposition)

The crucial property of this decomposition is that using SVD we can get the k-rank best approximation of the original 
matrix. Let me explain what this means:
* Matrices *U* and *V* contains left and right eigenvectors, matrix $`\Sigma`$ contains singular values
* If we crop *U*, *V*, $`\Sigma`$ and take first k eigenvectors with corresponding singular values and multiply cropped matrices
$$ U_{n \times k}\Sigma_{k \times k}V_{m \times k}^T = M_{n \times m}^k $$ the resulted matrix $$ M_{n \times m}^k $$ will have the rank k and the same shape as matrix *M*. 
* Thus we can calculate how similar the elements of both matrices are. To calculate their similarity we will use the Frobenius norm. And this property said that out of all possible approximation with rank k, $$ M_{n \times m}^k $$ will be the best (in terms of Frobenius norm)

$$M_{n \times m}^k = \arg \min_{\bar{M}: \text{rank}(\bar{M})=k} \|M - \bar{M}\|_F = \arg \min_{\bar{M}: \text{rank}(\bar{M})=k} \sqrt{\sum_{i,j}\left(M_{ij}-\bar{M_{ij}}\right)^2}$$

Ok, we got some properties, but how can we use SVD for dimensionality reduction? 
Basically, we can just use $`U_k`$ if we wish to compress a high dimensional matrix, and that is it. 
$`\Sigma_k V_k^T`$ will consist of base vectors that we use for factorization. 
And the property above will guarantee us that this will be the best approximation.

Another important property of SVD is that the first right eigenvector (from matrix V) will cover the direction where 
the variance of our sample will have the maximal value (see gif below, you can also visit source link for better example). 
What does this mean? If we take the left eigenvector as a new axis, and then project all samples on it, we get the 
sequence of values. If we then calculate the variance of this sequence it will be the maximal from all possible variants. 
In other words, this direction explains our data in the most effective way. And what about the second eigenvector? 
It will represent the direction with the best correction to our first eigenvector. And so on.

![](misc/images/stats.gif)

Source: [stats.stackexchange.com](https://stats.stackexchange.com/questions/2691/making-sense-of-principal-component-analysis-eigenvectors-eigenvalues/140579) 

To use SVD in sklearn you should use PCA class inside the sklearn.decomposition module. 
Please provide the answer in your notebook to the question: “What is the difference between PCA and SVD?”

SVD is not the only way we can decompose the matrix. 
Therefore not the only way we can decrease dimensionality. 
Another way is to use non-negative matrix factorization (NMF). To understand what is the difference 
between NMF and SVD is also part of this project.

All the decompositions we have mentioned have transformed our features linearly. 
But what if there are non-linear dependencies in the data? You can use kernel functions, just like for SVM. 
In simple words, the use of the kernel function will be similar to the non-linear transformation of input features 
and the further use of linear methods. For example, if we use a polynomial kernel with degree = 2 
and we will have 2 features [x1; x2], then this is equivalent to if we will use features [1; x1; x2; x12; x1x2; x22]. 
However, there is a separate class of methods that allow learning non-linear dependencies. Let's move to them.


### Manifold learning

Manifold learning is an approach to non-linear dimensionality reduction.

To begin we will dive into t-Distributed Stochastic Neighbor Embedding or t-SNE. 
This method is perhaps the most popular for visualizing features in 2D or 3D space. 
I will try to explain how it works on the fingers:
1. First, in compressed space, we create random vectors for each observation
2. Secondly, in high-dimensional space we count the distance between all possible pairs of our observations. 
And then we convert this distance into a probability, so that the sum of the probabilities to all other points is equal to 1. 
(Use softmax for this)
3. At the same time we repeat the distance measuring in low-dimensional space..
4. Let's take 1 observation. If the probability distributions to every other point in both spaces are similar, 
then we have built an excellent compressed representation. 
To measure this distance, there is a special loss that measures the similarity of distributions. 
This loss is called the Kullback-Leiber divergence: <br/>
$$ D_{KL}(P\|Q)=\sum_{i=1}^n p_i \log \frac{p_i}{q_i}$$
5. Now, since we have a loss, we can calculate the gradient and update the sample vector in the compressed representation. 
By repeating 2-5 steps many times the approach will converge to a solution that will tend to minimize the distance 
between distributions 

It is worth mentioning that for better convergence, instead of using random vectors in the 1st step, 
it is better to use some pre-trained compression. Often PCA is applied to generate starting points for algorithms. 
Together with the fact that t-SNE requires a lot of computing resources, this allows saving on calculations.

However, not so long ago, an alternative method appeared that added several optimizations to the t-SNE approach - 
called Uniform Manifold Approximation and Projection (UMAP). Instead of building a probability distribution on 
all samples, the authors proposed to extract k-nearest neighbors and use only them during optimization. 
This approach is available in a separate UMAP library.

Also from the proven approaches there is Locally Linear Embedding. Understanding and describing how it works is part 
of your task.

## Chapter III. Goal

The goals of this project are:
* to get a deep understanding of how to build dimensionality reduction models and for which task they can be applied
* try out methods from dimensionality reduction field in other unsupervised learning tasks

## Chapter IV. Instructions

* This project will only be evaluated by humans. You are free to organize and name your files as you desire.
* Here and further we use Python 3 as the only correct version of Python.
* For training deep learning algorithms you can try [Google Colab](https://colab.research.google.com). It offers kernels (Runtime) with GPU for free that is faster than CPU for such tasks. The norm is not applied to this project. Nevertheless, you are asked to be clear and structured in the conception of your source code.
* Store the datasets in the subfolder data

## Chapter V. Task

### Dataset

#### Book Recommendation Dataset

Source: https://www.kaggle.com/datasets/arashnic/book-recommendation-dataset

We need to download 2 files: Ratings.csv and Users.csv. Ratings are required to create sparse features for every user. 
i-th feature is whether a user put the rating for book i or not. Users are required to get age as a target variable.

#### Mnist

Source: http://yann.lecun.com/exdb/mnist/ 

“Hello world” dataset for hand-written digit recognition. Many libraries have built-in tools to load this dataset.

![](misc/images/mnist.png)

#### Background models challenge

Source: http://backgroundmodelschallenge.eu/

Consists of several videos from street cameras. We need only one of them inside bmc_real.zip

### Task

1. Answer questions from Preamble and Introduction
   1. Для каких из уже изученных моделей проклятье размерности актуально и объясните почему
   2. What is the difference between PCA and SVD?
   3. What is the difference  between NMF and SVD?
   4. Описать устройство алгоритма уменьшения размерности Locally Linear Embedding
2. Classification with sparse features
   1. Load “Book Recommendation Dataset” dataset ([source](https://www.kaggle.com/datasets/arashnic/book-recommendation-dataset))
   2. Create a matrix of user-book interactions. Use sparse matrices
   3. Split dataset into train, valid and test part
   4. Build linear model and random forest model to predict age of user based on user-book interactions. Do not forget to choose optimal parameter
   5. Use PCA and UMAP to reduce dimensionality of features
   6. Build linear model and random forest model on the compressed features
   7. Compare fitting time and quality for models from d and f
3. Visualizations
   1. Load MNIST dataset ([source](http://yann.lecun.com/exdb/mnist/))
   2. Represent every digit picture as a vector
   3. Transform vectors up to 2 dimensions using PCA, SVD, Randomized-SVD, TSNE, UMAP and LLE
   4. Suggest metric and compare models based on how well they separate digit classes
4. Image compression using SVD
   1. Choose 3 any gray-scale pictures and load it with python into matrix
   2. Decompose matrix using SVD
   3. Vary rank, calculate low-rank approximation of initial matrix and plot restored image. Use up to 10 different rank values. What does the singular value spectrum look like and how many modes are necessary for good image reconstructions?
   4. Plot the distribution of explained variance over rank. Explain what is “explained variance”
5. Background detection using SVD
   1. Load “Video_008.avi” ([source](http://backgroundmodelschallenge.eu/) - bmc_real.zip)
   2. Represent the video as a matrix and decompose it using SVD
   3. Restore the first frame using low-rank approximation. What rank should you use to get the background?
   
### Submission

Save one of the models from “Visualizations” part of the task. 
Your peer will load it and use it to compress one of the hand-written digits. 
Compressed vector should have the same values.

Your repository should contain one or several notebooks with your solutions.

## Chapter VI. Bonus part

* Using [face expression recognition dataset](https://www.kaggle.com/datasets/jonathanoheix/face-expression-recognition-dataset) fit the dimensionality reduction model to compress faces. Using representations try to make a sad face smiley and vice versa.


>Please leave feedback on the project in the [feedback form.](https://forms.yandex.ru/cloud/646b476102848f2ee1031b24/) 
