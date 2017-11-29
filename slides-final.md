---
title: "An introduction to Machine Learning..."
author: Nico Scherf
date: Nov, 2017
theme: black 
---

# What is Machine Learning?

---

<!-- .slide: data-background="https://cdn.redshift.autodesk.com/2016/05/Machine-Learning-hero.jpg" -->  

Note: An integral part of Artificial Intelligence


----

### TOC

- Part I
	- Historical note
	- Definition(s) of Machine Learning
		- A taxonomy of different aspects of Machine Learning
- Part II
	- A few useful things to know about Machine Learning

----

### The idea of AI has a long history...
  
<img src="https://upload.wikimedia.org/wikipedia/commons/0/0b/Medeia_and_Talus.png" height=500>

----

### The idea of AI has a long history...

- *The Analytical Engine has no pretensions whatever to originate anything.	It can do whatever we know how to order it to perform...* --- Ada Lovelace 1843

- The Imitation Game - Alan Turing "Computing Machinery and Intelligence". Mind. 1950

----


AI solved problems that are intellectually difficult for humans

<img src="https://images.chesscomfiles.com/uploads/v1/blog/299228.1a04250c.5000x5000o.9607e919a884.jpeg" height = 500 >

----

The true challenge: tasks that are easy for people to do but hard to describe formally 
<img src="https://www.peonderey.com/wp-content/uploads/2016/05/13242063_10154042837576675_1639904989_o.jpg" height = 500 >


----

- Solving a visual pattern recognition problem is hard by using prescribed rules: Think about how to detect a handwritten number...
- This is where Machine Learning came into play...

----

- 1763 | Bayes 
- 1913 | Markov chains
- 1950 | Turing “Computing Machinery and Intelligence.” Mind.
- 1957 | Rosenblatt - Perceptron 
- 1980 | Fukushima - Neocognitron 
- 1982 | Hopfield - Recurrent Networks 
- 1986 | Rumelhart, Hinton, Williams: Backpropagation 
- 1995 | Ho - Random Forest 
- 1995 | Cortes and Vapnik - Support Vector Machines
- 1997 | Hochreiter and Schmidhuber - LSTM networks

----

### Knowledge from data

<img src="https://upload.wikimedia.org/wikipedia/commons/d/da/Brahe_notebook.jpg" height=400 >

Tycho Brahe's observations were turned into laws of planetary motion by Kepler.

Note: Tycho Brahe's careful observations were used by Kepler to derive his three laws of planetary motion

----

## Machine Learning:

 ...a set of methods that can **automatically detect patterns** in data, and then use the uncovered patterns to **predict future data**, or to perform other kinds of decision making under uncertainty. -- Murphy 2012, Machine Learning

----

## Prerequisites:

- There is a pattern in the data
- There is (enough) data
- We do not have a mathematical model

----

## But then you can do a lot...

----

### Colorization

<img src=https://3qeqpr26caki16dnhd19sv6by6v-wpengine.netdna-ssl.com/wp-content/uploads/2016/07/Colorization-of-Black-and-White-Photographs.png height = 500>

----

### Detection 

<img src=http://cs.stanford.edu/people/karpathy/deepimagesent/dogball.png height = 500>

----

### Image captioning

<img src=https://3qeqpr26caki16dnhd19sv6by6v-wpengine.netdna-ssl.com/wp-content/uploads/2016/07/Automatic-Image-Caption-Generation.png height = 500>

----

### Image translation

<img src=https://3qeqpr26caki16dnhd19sv6by6v-wpengine.netdna-ssl.com/wp-content/uploads/2016/07/Instant-Visual-Translation.png>

----

### Face generation 

<img src=https://camo.githubusercontent.com/d371bad7ae1f9f5bf9e0f1906e46726e503ab99d/687474703a2f2f692e696d6775722e636f6d2f45334d677a6e422e6a7067>

----

### Automated stylised drawing

<img src=https://3qeqpr26caki16dnhd19sv6by6v-wpengine.netdna-ssl.com/wp-content/uploads/2016/07/Automatically-Create-Styled-Image-From-Sketch.png height>

----

- Machine Learning is not magic; it can't get something from nothing.
- It is like farming: 
	- Farmers combine seeds with nutrients to grow crops.
	- Learners combine knowledge with data to grow programs.

---

## A Taxonomy

 A computer program is said to *learn* from experience E with respect to some class of tasks T and performance measure P if its performance at tasks in T, as measured by P, improves with experience E. -- Mitchell (1997). Machine Learning. 

---

## The Task, T 
- Classification vs. Regression

----

## Classification

- learn mapping from $\vec{x}$ to $y \in \\{ 1, ..., C \\} $
- or probability distribution over classes

Note: - response variable y categorical
				- C = 2 binary
				- C > 2 multiclass 

----

<img src=https://upload.wikimedia.org/wikipedia/commons/thumb/f/fe/Kernel_Machine.svg/1200px-Kernel_Machine.svg.png style="background-color:white;">

----

## Classification examples

- email spam filtering,
- object recognition / image classification:
	- handwriting recognition,
	- face detection. 

----

MNIST: The drosophila of Machine Learning:
<img src="https://raw.github.com/nscherf/01-ML-introduction/gh-pages/img/mnist.png" height=500 >

----

## Regression

- predict numerical value
- approximate $ y = f(\vec{x}) $ by $ \hat{y} = \hat{f}(\vec{x}) $

Note: - response variable real-valued 

----


<img src="https://upload.wikimedia.org/wikipedia/commons/7/7b/Gaussian_Kernel_Regression.png" height = 300 >


----

## Regression examples

- price of item (e.g. used car) based on covariates
- predict stock market tomorrow
- level of biomarker given clinical measurements

----


## Other tasks

- Structured output (a vector with relationships between elements) 
	- translation (e.g. Klingon to French)
- Anomaly detection (e.g. credit card fraud)
- Density estimation: structure $p(x)$ from data

----

## Other tasks 

- Imputaton of missing values: $ p(x\_i | x\_{-i})$
- Denoising: clean $x$ from corrupted $ \tilde{x} $: $ p(x| \widetilde{x}) $

<img src = https://ars.els-cdn.com/content/image/1-s2.0-S1047320317301803-gr1.jpg>

---

## The Performance Measure, P
- Objective/Loss functions

----

## The Performance Measure, P

- Describes what is a good/bad solution:
	- Accuracy / Error rate
	- Precision and Recall
	- Squared Error
	- Likelihood
	- Information Gain
	- KL Divergence
	- Depends on application:
		- what are problematic mistakes?


Note: error rate = Expected 0-1 loss
	- e.g. regression: regular medium mistakes or rare large mistakes

---

## The Experience, E

- *supervised* vs *unsupervised* methods

----

## Supervised Learning (predictive learning)

- Learn from data containing features and associated label(s) (given by an "instructor").
- Learn mapping from features to response variable.
- Conditional density: learn $p(\vec{y}| \vec{x})$ given $\vec{x}$ and $\vec{y}$.

----

## Typical examples supervised learning

- Regression,
- Classification,
- Denoising.

----

## Unsupervised Learning (descriptive learning) 

- Learn from unlabelled dataset:
	- structural properties (knowledge discovery),
	- patterns (density estimation).
- Unconditional density: learn $p(\vec{x})$
	

Note: Unsupervised learning might be much closer to how animals learn: "When we’re learning to see, nobody’s telling us what the right answers are — we just look. Every so often, your mother says “that’s a dog”, but that’s very little information. You’d be lucky if you got a few bits of information — even one bit per second — that way. The brain’s visual system has 10^14 neural connections. And you only live for 10^9 seconds. So it’s no use learning one bit per second. You need more like 10^5 bits per second. And there’s only one place you can get that much information: from the input itself." — Geoffrey Hinton, 1996 (quoted in (Gorder 2006)).

----


## Typical examples unsupervised Learning

- Clustering,
- Density estimation,
- Imputation,
- Latent factor discovery,
- Dimensionality reduction.

----

Clustering

<img src="https://upload.wikimedia.org/wikipedia/commons/7/76/Sidney_Hall_-_Urania%27s_Mirror_-_Lacerta%2C_Cygnus%2C_Lyra%2C_Vulpecula_and_Anser.jpg" height = 500>

----

Density Estimation
  
<img src="https://c2.staticflickr.com/4/3258/5851679238_23b1b2bafe_b.jpg" height = 500>

----

Dimensionality Reduction

<img src="https://blog.sourced.tech/post/lapjv/mnist_after.png" height = "500">


----


## Beyond (un)supervised 

- Semi-supervised learning:
	- only few labelled examples.
- Multi-instance learning
	- Collection contains example of a class or not.

---

## Reinforcement Learning

- Learning successful behavioural strategies from occasional rewards.

----

<iframe width="840" height="472" src="https://www.youtube.com/embed/TmPfTpjtdgg" frameborder="0" allowfullscreen></iframe>

---

### Many different representations...

- Instances
	- kNN
	- SVM
- Hyperplanes
	- Naive Bayes
	- Logistic Regression
- Decision Trees
- Set of rules
- Neural Networks
- Graphical Models

---

## Some useful things to know about Machine Learning*

*From the paper of the same name by Pedro Domingos

---

### Generalisation counts

- Fundamental goal is to generalise to **unseen** data, not a perfect fit to the training data. 
- There are (loose) theoretical bounds (e.g. VC dimension).
- The empirical way:
	- Split data into training and test set
	- Never touch the test set !

---

### Data alone is not enough: The problem with induction

>  "Even after the observation of the frequent conjunction of objects, we have no reason to draw any inference concerning any object beyond those of which we have had experience." --- Hume, A treatise of Human Nature, Book I, part 3, sec 12

- The "No free lunch" theorem by Wolpert 

----

- Boolean function of 100 variables and a million examples:
- $2^{100} - 10^6$ unknown examples remain,
- without prior knowledge, do coin flipping.

----

- The functions we are interested in are not drawn randomly from the space of all possible functions...
- We use some general assumptions:
	- Smoothness,
	- Similar examples have similar classes,
	- Limited complexity or dependence.
- We can always infer something in a **probabilistic sense**.

---

### Overfitting has many faces

- If we model every minute variation in the training data, we are likely to fit the noise as well.
- perfect fit of training data = bad generalisation to unseen data

----

- Error decomposes into bias and variance
	- Bias: the tendency to consistently learn the wrong thing.
	- Variance: the tendency to learn random things irrespective of real signal

----


<img src=https://i.stack.imgur.com/8RlJk.png>

- linear model: high bias, low variance
- complex polynomial: low bias, high variance
- find the optimal trade-off
	- cross-validation (training, validation set)
	- use regularisation

----

### keep in mind: 

- We split the data into: 
	- Training set
	- (Validation set)
	- **Test set (never touch!)**
		- typical split: 80/20

---


### Intuition fails in high dimensions

- *The curse of dimensionality* (Bellman 1961)
- Sampling becomes exponentially harder:
	- In $D=100$, $10^{12}$ samples cover only a fraction of $10^{-18}$ of the input space

----

- Similarity-based reasoning breaks down in high dimensional space.
	- (Euclidean) distances become meaningless
- Most of the volume of a high-dimensional orange is in the skin, not the pulp.

<img src=https://c.pxhere.com/photos/05/79/agriculture_background_citrus_close_up_color_dessert_diet_drop-1112919.jpg!d height = 200>

Note: -Number of nearest neighbours increases until choice of NN is effectively random
- Most of the mass of a multivariate Gaussian is not near the mean but in a distant shell

----

- **BUT**: The data is usually not uniformly distributed in the embedding space. It forms a structured subset (a manifold)!
	- The "blessing of non-uniformity".

---

### More data beats clever algorithm

- The quickest path to success: just get more data.
- A dumb algorithm with lots of data is better than a clever one with modest amounts of it.

----

- Rule of thumb: try the simpler methods first!
	- Naive Bayes before Logistic Regression
	- kNN before SVM
	- Everything else before Deep Learning

----


- Labeled data is often scarce and costly to obtain:
	- That is why unsupervised learning is so interesting.

---

### Parametric vs. Non-parametric

- Learners can be divided into those where:
	- Representation size is fixed (parametric)
	- Representation grows with data (non-parametric)

----


### K-nearest neighbours

<img src=http://cs231n.github.io/assets/knn.jpeg >

- Guaranteed nearly optimal error rate (for $\infty $ data)
- Computation time scales with the size data

----


### Parametric models

- Reduce complexity by making assumptions about the data distribution (inductive bias).
- Parametric models:
	- fast to compute,
	- strong assumptions.

----

### Parametric models: regression

- linear regression: 
	- $y(x,w) = w_0 + w_1 x_1 + ... + w_D x_D$

<img src=https://cdn-images-1.medium.com/max/600/1*iuqVEjdtEMY8oIu3cGwC1g.png height = 400 style="background-color:white;">

----

### Parametric models: classification

- logistic regression:
	- linear regression + sigmoid nonlinearity
	- $ y(\Phi) = \sigma (\vec{w}^T \Phi)$

<img src=http://www.saedsayad.com/images/LogReg_1.png>

---


### Feature engineering is key

- The design matrix:
	- N examples, described by D features
		- NxD design matrix
	- predict the value of N target values

<img src=https://i.stack.imgur.com/VZtEr.jpg height = 200>

----

- Raw data often not useful to learn a task
- Ideal features are  independent and correlate well with the target.
	- How to find them is a 'black art'.
- Deep Learning methods extract the features themselves from the raw data.

---

### Learn many models, not just one

- Train and combine variations of the learner.
- Ensemble methods:
	- Bagging: resampling the training data.
	- Boosting: weighting examples.
	- Stacking: learn to combine output of lower level.
- Netflix prize: stacked ensembles of over 100 methods. 

Note: by how often the previous learner got them wrong

---


### Representable $\neq$ learnable

- "Multilayer Perceptron is a universal function approximator".
- Just because a function can be represented it doesn't mean it can be learned in practice.
- Some models are exponentially more compact/ efficient:
	- Deep Learning,
	- Capsule Networks.

---

### Correlation does not imply causation

- Machine Learning is done on *observational* data.
- Predictive variables can only be controlled in *experimental* data.

----


# Summary

----

- Basic concepts:
	- Supervised, Unsupervised
	- Classification, Regression
	- Parametric vs. Non-parametric 
	- Features
	- Generalisation, Overfitting, Underfitting
	- Training set, Validation set, Test set

---

# books 

----

- Bishop - Pattern Recognition and Machine Learning

<img src=https://www.microsoft.com/en-us/research/wp-content/uploads/2016/06/Springer-Cover-Image.jpg height = 500>

----

- Murphy - Machine Learning: A Probabilistic Perspective

<img src=https://mitpress.mit.edu/sites/default/files/9780262018029.jpg height = 500>

----

- Goodfellow, Bengio, Courville - Deep Learning

<img src=https://images.gr-assets.com/books/1478212695l/30422361.jpg height = 400>

----

- Geron - Hands-on Machine Learning

<img src=https://covers.oreillystatic.com/images/0636920052289/lrg.jpg height = 500>

----

- Abu-Mostafa, Magdon-Ismail, Lin - Learning From Data

<img src=http://amlbook.com/images/front.jpg height = 500>

----

- Hastie, Tibshirani, Friedman - The Elements of Statistical Learning

<img src=https://web.stanford.edu/~hastie/ElemStatLearn/CoverII_small.jpg height = 500>

---- 

The end