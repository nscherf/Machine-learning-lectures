---
title: "An introduction to Machine Learning..."
author: Nico Scherf
date: Oct, 2017
theme: black 
---

# What is Machine Learning?

---

<!-- .slide: data-background="img/robots.png" -->  
## An integral part of Artificial Intelligence


----

## AI: a long history
  
<img src="https://upload.wikimedia.org/wikipedia/commons/0/0b/Medeia_and_Talus.png" height=500>

----

## AI: a long history

> The Analytical Engine has no pretensions whatever to originate anything.	It can do whatever we know how to order it to perform... --- Ada Lovelace 1843

- The Imitation Game - Alan Turing "Computing Machinery and Intelligence", 1950

----


AI solved problems that are intellectually difficult for humans

<img src="http://www.publicdomainpictures.net/pictures/200000/velka/chess-board-1473668683CKw.jpg" height = 500 >

----

The true challenge: tasks that are easy for people to do but hard to describe formally 
<img src="img/faces.jpg" height = 500 >

----

## Machine Learning and AI

- evolved in the 50s and 60s from AI, pattern recognition and computational learning theory	
- learn from data (and predict)
- similar to computational statistics and scientific method


Note: - grew out of quest for AI
	- however, AI was dominated by logical, knowledge-based approaches 
	- flourished in the 1990s
		- from AI to tackle practical problems using data 
	- used where explicit solutions are hard to find:
		- OCR, SPAM, Network intrusion, Computer Vision

----

## A time line

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
<!-- .slide: data-background="https://upload.wikimedia.org/wikipedia/commons/6/69/NASA-HS201427a-HubbleUltraDeepField2014-20140603.jpg"-->
 
> We are drowning in information and starving for knowledge. — John Naisbitt.

Note: Today we need computerised methods for most problems

----

## Machine Learning:

> ...a set of methods that can automatically detect patterns in data, and then *use the uncovered patterns to predict future data*, or to perform other kinds of decision making under uncertainty. -- Murphy 2012, Machine Learning

----

<img src="img/learning-algorithm-scheme.png" height = 500 >

----

- Rules, Data  -> [Classical Programming] -> Answers <!-- .element: class="fragment" -->
- Data, Answers -> [ML] -> Rules (programs, computational models) <!-- .element: class="fragment" -->

----

## Prerequisites

- There is a pattern in the data
- We cannot write down a mathematical model of it
- There is (enough) data

----

## Awesome examples go here...

----

## Machine Learning: A Taxonomy

> A computer program is said to *learn* from experience E with respect to some class of tasks T and performance measure P if its performance at tasks in T, as measured by P, improves with experience E. -- from Mitchell, T. (1997). Machine Learning. McGraw Hill. 

---

## The Task, T
> A computer program is said to *learn* from experience E with respect to some class of **tasks T** and performance measure P if its performance at tasks in T, as measured by P, improves with experience E. -- from Mitchell, T. (1997). Machine Learning. McGraw Hill. 

----

## The Task, T

- ML is interesting for tasks that are too difficult to solve with fixed programs 

----

## Classification

- learn mapping from $\vec{x}$ to $y \in \\{ 1, ..., C \\} $
- or probability distribution over classes
- possibly with missing inputs (e.g. medical records)

Note: - response variable y categorical
				- C = 2 binary
				- C > 2 multiclass 

----

## Classification examples

- email spam filtering
- object recognition / image classification
	- handwriting recognition
	- face detection 

----

The drosophila of Machine Learning:
<img src="	../img/mnist.png" height=500 >

 

----

## Regression

- predict numerical value given input
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

> A computer program is said to *learn* from experience E with respect to some class of tasks T and **performance measure P** if its performance at tasks in T, as measured by P, improves with experience E. -- from Mitchell, T. (1997). Machine Learning. McGraw Hill. 

----

## The Performance Measure, P

- classification: accuracy, error rate (expected 0-1 loss)
- density estimation: average (log) probability of examples
- depends on application
	- what are problematic mistakes?
	- e.g. regression: regular medium mistakes or rare large mistakes


---

## The Experience, E

> A computer program is said to *learn* from **experience E** with respect to some class of tasks T and performance measure P if its performance at tasks in T, as measured by P, improves with experience E. -- from Mitchell, T. (1997). Machine Learning. McGraw Hill. 

----

## The Experience, E

- *supervised* vs *unsupervised* methods

----

## Supervised Learning (predictive learning)

- learn from data containing features and associated label(s) (given by an "instructor")
- learn mapping from features to response variable
- conditional density estimation: learn $p(\vec{y}| \vec{x})$ given $\vec{x}$ and $\vec{y}$

----

## Unsupervised Learning (descriptive learning) 

- dataset containing many features
	- learn useful properties of structure 
	- find interesting patterns in the data
- knowledge discovery, density estimation 
- unconditional density estimation: learn $p(\vec{x})$ given $\vec{x}$
- requires no training labels

Note: Unsupervised learning might be much closer to how animals learn: "When we’re learning to see, nobody’s telling us what the right answers are — we just look. Every so often, your mother says “that’s a dog”, but that’s very little information. You’d be lucky if you got a few bits of information — even one bit per second — that way. The brain’s visual system has 10^14 neural connections. And you only live for 10^9 seconds. So it’s no use learning one bit per second. You need more like 10^5 bits per second. And there’s only one place you can get that much information: from the input itself." — Geoffrey Hinton, 1996 (quoted in (Gorder 2006)).

----

## Supervised vs Unsupervised

- Not a formal distinction 
- unsupervised problem could be split into a sequence of supervised problems: $ p(\vec{x}) = p(x_1) p(x_2 | x_1) p(x_3 | x_1, x_2) ... $
- also works the other way around $ p(y|x) = \frac{p(x,y)}{(\sum_{y'} p(x,y'))}$
- i.e. most methods could do both supervised and unsupervised 

----

## Typical examples supervised learning

- regression
- classification,
- structured output,
- denoising

----

## Typical examples unsupervised Learning

- density estimation 
- clustering
- imputation 
- latent factor discovery / dimensionality reduction
- (graph) structure (i.e. influence/correlations between variables)

----

Clustering

<img src="https://upload.wikimedia.org/wikipedia/commons/7/76/Sidney_Hall_-_Urania%27s_Mirror_-_Lacerta%2C_Cygnus%2C_Lyra%2C_Vulpecula_and_Anser.jpg" height = 500>

----

Unsupervised Learning: Density Estimation
  
<img src="https://c2.staticflickr.com/4/3258/5851679238_23b1b2bafe_b.jpg" height = 500>

----

Dimensionality Reduction

<img src="https://c2.staticflickr.com/2/1501/26194566691_ef39f2c77b_b.jpg" height = "500">

----

## Beyond (un)supervised 

- semi-supervised learning
	- sparse labelling of dataset 
- multi-instance learning
	- only known if a collection contains example of class or not
	- individual members not labeled 

---

## Reinforcement Learning

- learning successful behavioural strategies from occasional rewards

----

<iframe width="840" height="472" src="https://www.youtube.com/embed/L4KBBAwF_bE" frameborder="0" allowfullscreen></iframe>

---

## Machine Learning: The hype

<img src="https://upload.wikimedia.org/wikipedia/commons/9/94/Gartner_Hype_Cycle.svg" height = "400">

----

## Machine Learning vs. Optimisation

Fitting data (optimisation) is different from finding patterns that generalise to unseen examples (machine learning)

----


Machine Learning | Statistics
---|---
Networks, graphs | model
Weights | parameters
Learning | fitting
Supervised learning | regression/classification
Unsupervised learning | density estimation, clustering
Large grant = $1,000,000 | large grant = $50,000

<small> *from http://statweb.stanford.edu/~tibs/stat315a/glossary.pdf* </small>

----

## ML vs statistics 

- Breiman, Leo. 2001. “Statistical Modeling: The Two Cultures.” Statistical Science: A Review Journal of the Institute of Mathematical Statistics 16 (3) 199–231.

---

## Some basic concepts


----


## The design matrix
- features
 - feature vectors
 - feature extraction
- classes/categories/groups

----

## parametric vs. Non-parametric models

- parametric: closed form, #parameters is fixed
	- fast to compute
	- strong assumptions
- non-parametric, #parameters grows with amount of (training) data
	- flexible
	- often computationally intractable

----

## parametric vs. Non-parametric models - K-nearest neighbours

----

## The curse of dimensionality
- Bellman 
- in high dimensions, (Euclidean) distances become meaningless 
- growing hypercube to collect data: e(f) = f^(1/D)
	- if we want f = 0.1 (10% of data) in D=10 dimensions, side length e = 0.8 (80% of max side-length) - not very local
- ration between min and max distance approaches 1 with D -> Inf

----

## parametric models

- to escape the curse of dimensionality we can make assumptions about the data distribution (inductive bias)
- parametric models with fixed number of parameters

----

## parametric models: regression

- linear regression

----

## parametric models: classification

- logistic regression

----

## generalisation 

- how is performance on **unseen data** ?
- using a **test set** separate from training set

----

## Overfitting

- if we model every minute variation in the training data, we are likely to fit the noise as well
- less accurate prediction for future data
- perfect fit of training data = bad generalisation to unseen data

----

## model selection and the feasibility of learning

- how to select a good model (e.g. the k in kNN)?
	- misclassification rate on training set
	- but: we care about generalisation error: misclassification rate on test set
	- out-of sample error 
- U-shaped curve

----

## the test set

- during training we don't have access to test set:
	- split training data into training set and validation set
		- e.g. 80/20
	

----

## cross validation

- cross validation (k folds) 

----

## model selection

- optimise over models (or hyperparameters)
- use a validation set

----

## model selection 

- training set
- validation set
- test set (don't touch!)

----

## No free lunch theorem 

> All models are wrong, but some models are useful. — George Box (Box and Draper 1987, p424).
- no free lunch theorem (Wolpert 1996)
- no universally best model across all problems
- assumptions that works well in one domain often fail in another

----

## The unreasonable effectiveness of data

---

# Summary

----

- types of ML methods
	- supervised, unsupervised
	- classification, regression
	- parametric vs. non-parametric 
- basic concepts
	- curse of dimensionality 
	- model selection
		- generalisation and overfitting
		- training set, test set, validation set 
	- There is no free lunch!

