---
title: "An introduction to Machine Learning..."
author: Nico Scherf
date: Oct, 2017
theme: black 
---

# What is Machine Learning?

---

<!-- .slide: data-background="https://cdn.redshift.autodesk.com/2016/05/Machine-Learning-hero.jpg" -->  

Note: An integral part of Artificial Intelligence


----

## AI: a long history
  
<img src="https://upload.wikimedia.org/wikipedia/commons/0/0b/Medeia_and_Talus.png" height=500>

----

## AI: a long history

- *The Analytical Engine has no pretensions whatever to originate anything.	It can do whatever we know how to order it to perform...* --- Ada Lovelace 1843

- The Imitation Game - Alan Turing "Computing Machinery and Intelligence". Mind. 1950

----


AI solved problems that are intellectually difficult for humans

<img src="https://images.chesscomfiles.com/uploads/v1/blog/299228.1a04250c.5000x5000o.9607e919a884.jpeg" height = 500 >

----

The true challenge: tasks that are easy for people to do but hard to describe formally 
<img src="https://www.peonderey.com/wp-content/uploads/2016/05/13242063_10154042837576675_1639904989_o.jpg" height = 500 >

----

## Machine Learning and AI

- evolved in the 50s and 60s from AI, pattern recognition and computational learning theory	
- learn from data (and predict)


Note: - grew out of quest for AI
	- however, AI was dominated by logical, knowledge-based approaches 
	- flourished in the 1990s
		- from AI to tackle practical problems using data 
	- used where explicit solutions are hard to find:
		- OCR, SPAM, Network intrusion, Computer Vision

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

## Knowledge from data

<img src="https://upload.wikimedia.org/wikipedia/commons/d/da/Brahe_notebook.jpg" height=500 >

Note: Tycho Brahe's careful observations were used by Kepler to derive his three laws of planetary motion

----

## Knowledge from data 

 
> We are drowning in information and starving for knowledge. — John Naisbitt.

Note: Today we need computerised methods for most problems, <!-- .slide: data-background="https://upload.wikimedia.org/wikipedia/commons/6/69/NASA-HS201427a-HubbleUltraDeepField2014-20140603.jpg"-->

----

## Machine Learning:

 ...a set of methods that can **automatically detect patterns** in data, and then use the uncovered patterns to **predict future data**, or to perform other kinds of decision making under uncertainty. -- Murphy 2012, Machine Learning

----

## Prerequisites:

- There is a pattern in the data
- We cannot write down a mathematical model of it
- There is (enough) data

----

## But then you can do a lot...

----

### colorization

<img src=https://3qeqpr26caki16dnhd19sv6by6v-wpengine.netdna-ssl.com/wp-content/uploads/2016/07/Colorization-of-Black-and-White-Photographs.png height = 500>

----

### detection 

<img src=http://cs.stanford.edu/people/karpathy/deepimagesent/dogball.png height = 500>

----

### image captioning

<img src=https://3qeqpr26caki16dnhd19sv6by6v-wpengine.netdna-ssl.com/wp-content/uploads/2016/07/Automatic-Image-Caption-Generation.png height = 500>

----

### image translation

<img src=https://3qeqpr26caki16dnhd19sv6by6v-wpengine.netdna-ssl.com/wp-content/uploads/2016/07/Instant-Visual-Translation.png>

----

### face generation 

<img src=https://camo.githubusercontent.com/d371bad7ae1f9f5bf9e0f1906e46726e503ab99d/687474703a2f2f692e696d6775722e636f6d2f45334d677a6e422e6a7067>

----

### automated stylised drawing

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

 A computer program is said to learn from experience E with respect to some class of **tasks T** and performance measure P if its performance at tasks in T, as measured by P, improves with experience E. -- Mitchell (1997). Machine Learning.  

----

## The Task, T

- ML is interesting for tasks that are too difficult to solve with fixed algorithms 

----

## Classification

- learn mapping from $\vec{x}$ to $y \in \\{ 1, ..., C \\} $
- or probability distribution over classes

Note: - response variable y categorical
				- C = 2 binary
				- C > 2 multiclass 

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

## Regression examples

- price of item (e.g. used car) based on covariates
- predict stock market tomorrow
- level of biomarker given clinical measurements

----

## Regression examples

<img src="https://upload.wikimedia.org/wikipedia/commons/7/7b/Gaussian_Kernel_Regression.png" height = 300 >

----

## Other tasks

- Structured output (a vector with relationships between elements) 
	- transcription (e.g. image to text)
	- translation (e.g. Klingon to French)
- Anomaly detection (e.g. credit card fraud)

----

## Other tasks 

- Synthesis and sampling (e.g. textures in video games)
- Imputaton of missing values: $ p(x\_i | x\_{-i})$
- Denoising: clean $x$ from corrupted $ \tilde{x} $: $ p(x| \widetilde{x}) $
- Density estimation: structure $p(x)$ from data p(x)

---

## The Performance Measure, P

 A computer program is said to learn from experience E with respect to some class of tasks T and **performance measure P** if its performance at tasks in T, as measured by P, improves with experience E. -- Mitchell (1997). Machine Learning.  

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

 A computer program is said to learn from **experience E** with respect to some class of tasks T and performance measure P if its performance at tasks in T, as measured by P, improves with experience E. -- Mitchell (1997). Machine Learning.

----

## The Experience, E

- *supervised* vs *unsupervised* methods

----

## Supervised Learning (predictive learning)

- Learn from data containing features and associated label(s) (given by an "instructor").
- Learn mapping from features to response variable.
- Conditional density: learn $p(\vec{y}| \vec{x})$ given $\vec{x}$ and $\vec{y}$.

----

## Unsupervised Learning (descriptive learning) 

- From a dataset containing many features:
	- Learn structural properties (knowledge discovery),
	- Find interesting patterns (density estimation).
- Unconditional density: learn $p(\vec{x})$ given $\vec{x}$
	- Requires no training labels.

Note: Unsupervised learning might be much closer to how animals learn: "When we’re learning to see, nobody’s telling us what the right answers are — we just look. Every so often, your mother says “that’s a dog”, but that’s very little information. You’d be lucky if you got a few bits of information — even one bit per second — that way. The brain’s visual system has 10^14 neural connections. And you only live for 10^9 seconds. So it’s no use learning one bit per second. You need more like 10^5 bits per second. And there’s only one place you can get that much information: from the input itself." — Geoffrey Hinton, 1996 (quoted in (Gorder 2006)).

----

## Supervised vs Unsupervised


- Unsupervised problem could be split into a sequence of supervised problems: $ p(\vec{x}) = p(x_1) p(x_2 | x_1) p(x_3 | x_1, x_2) ... $
- and vice versa: $ p(y|x) = \frac{p(x,y)}{(\sum_{y'} p(x,y'))}$.

Note: - Unsupervised and supervised learning is mathematically/formally the same 
----

## Typical examples supervised learning

- Regression,
- Classification,
- Denoising.

----

## Typical examples unsupervised Learning

- Clustering,
- Density estimation,
- Imputation,
- Latent factor discovery,
- Dimensionality reduction,
- Graph structure estimation.

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

Dimensionality Reduction

<img src="https://c2.staticflickr.com/2/1501/26194566691_ef39f2c77b_b.jpg" height = "500">

----

## Beyond (un)supervised 

- Semi-supervised learning:
	- sparse labelling of dataset.
- Multi-instance learning
	- only known if a collection contains example of class or not.

---

## Reinforcement Learning

- Learning successful behavioural strategies from occasional rewards.

----

<iframe width="840" height="472" src="https://www.youtube.com/embed/L4KBBAwF_bE" frameborder="0" allowfullscreen></iframe>

---

### Representation of the learning model

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

----

<img src="https://raw.github.com/nscherf/01-ML-introduction/gh-pages/img/learning-algorithm-scheme.png" height = 500 >

----

### The design matrix

- N examples, described by D features
	- NxD design matrix
- predict the value of N target values

<img src=https://i.stack.imgur.com/VZtEr.jpg height = 200>

----

### Parametric vs. Non-parametric models

- parametric: closed form, i.e. #parameters is fixed	
- non-parametric: #parameters grows with amount of (training) data

----

### Parametric vs. Non-parametric models - K-nearest neighbours

<img src=http://cs231n.github.io/assets/knn.jpeg >

- guaranteed nearly optimal error rate (for $\infty $ data)
- computation time scales with the size data

----

### The curse of dimensionality

- Sampling effort grows exponentially with dimensionality,
- (Euclidean) distances become meaningless. 


Note: 
	- Bellman
	- ration between min and max distance approaches 1 with D -> Inf
	- collect data in hypercube: to sample 10% of data in 10 dimensions, we need a subvolume with 80% of max side-length.


----

### Parametric models

- Escape the curse of dimensionality by making assumptions about the data distribution (inductive bias).
- Parametric models with fixed number of parameters:
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

----

### Generalisation 

- The central question: How is the performance on **unseen data**?

----

### Overfitting

- If we model every minute variation in the training data, we are likely to fit the noise as well:
	- less accurate prediction for future data
- perfect fit of training data = bad generalisation to unseen data

----

### Model selection
- how to select a good model (e.g. the k in kNN)?
	- misclassification rate on training set
	- but: we care about generalisation error: 
		- the out-of sample error.
- Solution: using a **test set** separate from training set.

Note: link to u-curve.. <img src=https://pbs.twimg.com/media/CnRKSa8UsAAR2JC.jpg>

----

### the test set

- Split training data into training set and test set:
	- e.g. 80/20.
- And don't look at this test set!
	
Note:
- cross validation (k folds)
<img src= https://cdn-images-1.medium.com/max/1600/1*J2B_bcbd1-s1kpWOu_FZrg.png height = 500>

----

## model selection

- But how to optimise over different models?
	- e.g. tuning the hyperparameters
- Don't use the test set!
- Use an validation set from the training data!
	- e.g. use cross validation...

----

## model selection 

So we split the data into: 
- Training set
- (Validation set)
- Test set (don't touch!)

----

> All models are wrong, but some models are useful. — Box and Draper 1987.

----

### The *no free lunch* theorem 
	
- Wolpert 1996
- No universally best model across all problems!
- Assumptions that works well in one domain often fail in another.

----

#### The unreasonable effectiveness of data

- Simple models with lots of data will beat sophisticated models with few data!
- Labeled data is often scarce and expensive:
	- Try to use unsupervised learning if you can.

---

# Summary

----

- Types of ML methods:
	- Supervised, Unsupervised
	- Classification, Regression
	- Parametric vs. Non-parametric 
- Basic concepts
	- Design matrix, Features...
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