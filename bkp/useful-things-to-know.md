---
title: "A few useful things to know about Machine Learning..."
author: Nico Scherf
date: Oct, 2017
theme: black 
---

# A few useful things to know about Machine Learning

From the paper of the same name by Pedro Domingos

---

## Learning = Representation + Evaluation + Optimisation

- Representation: choice of classifier representation defines hypothesis space
- Evaluation: define objective or loss function
- Optimisation: a method to find highest-scoring classifier

----

### Representation

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

----

### Evaluation

- Accuracy / Error rate
- Precision and Recall
- Squared Error
- Likelihood
- Posterior probability
- Information Gain
- KL Divergence

----

### Optimization

- Combinatorial: 
	- Greedy
	- Beam
	- Branch-and-bound
- Continuous unconstrained:
	- Gradient descent
	- Quasi-Newton
- Constrained constrained:
	- Linear Programming

---

## Generalisation counts

- fundamental goal is to generalize to **unseen** data
- split data into training and test set
- never touch the test set !

---

## Data alone is not enough

-  "Even after the observation of the frequent conjunction of objects, we have no reason to draw any inference concerning any object beyond those of which we have had experience." --- Hume, A treatise of Human Nature, Book I, part 3, sec 12
- The "No free lunch" theorem by Wolpert 

----

- Boolean function of 100 variables and a million examples
- There are $2^100 - 10^6$ examples whose classes you don't know
- if we have no further information, nothing is better than coin flipping

----

- The functions we are interested in are not drawn randomly from the space of all possible functions!
- We can do very well if we use some very general assumptions:
	- smoothness
	- similar examples have similar classes
	- limited complexity
	- limited dependence

----

- Machine Learning is not magic; it can't get something from nothing.
- It is like farming: 
	- Farmers combine seeds with nutrients to grow crops.
	- Learners combine knowledge with data to grow programs.

---

## Overfitting has many faces

- Generalisation can be decomposed into bias and variance
	- Bias: the tendency to consistently learn the wrong thing.
	- Variance: the tendency to learn random things irrespective of real signal

----

- TBD: image
- linear regression: high bias, low variance
- degree 100 polynomial: low bias, high variance
- find the optimal trade-off
	- cross-validation (training, validation set)
	- use regularisation

---

## Intuition fails in high dimensions

- *The curse of dimensionality* (Bellman 1961)
- Sampling becomes exponentially harder:
	- In $D=100$, $10^{12}$ samples cover only a fraction of $10^{-18}$ of the input space

----

- Similarity-based reasoning breaks down in high dimensional space.
	- (Euclidean) distances become meaningless
	- Number of nearest neighbours increases until choice of NN is effectively random
- most of the mass of a multivariate Gaussian is not near the mean but in a distant shell
- most of the volume of a high-dimensional orange is in the skin, not the pulp.

----

- BUT: The data is usually not uniformly distributed in the embedding space. It forms a structured subset (a manifold)!
	- The "blessing of non-uniformity".

---
  
## More data beats a cleverer algorithm

- The quickest path to success: just get more data.
- A dumb algorithm with lots of data is better than a clever one with modest amounts of it.
	- To a first approximation they all do the same: grouping nearby examples to the same class.

----

- Rule of thumb: it pays to try the simplest learners first!
	- Naive Bayes before Logistic Regression
	- kNN before SVM

----


- Labeled data is often scarce and costly to obtain:
	- That is why unsupervised learning is so interesting.

----

### parametric vs. Non-parametric

- learners can be divided into those where:
	- representation size is fixed (parametric)
	- representation grows with data (non-parametric)

----

TBD: kNN example and why parametric helps here...

---

## Theoretical guarantees are not what they seem

- Bounds like Hoeffding's inequality or VC dimension are rather loose.
- We don't live in "asymptopia"...
- Mostly use empirical estimates like error on test set.

---

## Feature engineering is the key

- TBD: Design Matrix
- Raw data often not useful to learn a task
- Features are needed: those that are independent and correlate well with the class.
	- More features are not necessarily better (the curse of dimensionality).
	- Feature engineering is a 'black art'.
- Deep Learning methods extract the features themselves from the raw data.

---

## Learn many models, not just one

- Train a number of variations of the learner.
	- Instead of using the best, combine all of them.
- Ensemble methods:
	- bagging: resampling the training data,
	- boosting: weighting examples by how often the previous learner got them wrong.
	- stacking: high level methods learn to combine output of lower level learners.
- Netflix prize: stacked ensembles of over 100 methods. 

---
  
## Simplicity does not imply accuracy

- Occam's razor: entities should not be multiplied beyond necessity.
- Not that simple in Machine Learning (cf. NFL-theorem).
	- Complexity is connected with size of hypothesis space.
- Simplicity is a virtue in its own right.

---

## Representable does not imply learnable

- Multilayer Perceptron is a universal function approximatior.
- But just because a function can be represented it doesn't mean it can be learned.
	- at least not efficiently...
- Some representation are exponentially more compact.
	- Distributed representations used in Deep Learning.

---

## Correlation does not imply causation

- Machine Learning is done on *observational* data.
- Predictive variables can only be controlled in *experimental* data.
