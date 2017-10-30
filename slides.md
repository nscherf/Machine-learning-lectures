---
title: "An introduction to Machine Learning..."
author: N Scherf
date: Oct, 2017
output:
  revealjs::revealjs_presentation:
    theme: black
    center: true
---

# What is Machine Learning?

--- 

## A part of Artificial Intelligence {data-background="img/robots.png"}
  
--- 
  
## Thinking machines and AI: A long history

- Galatea, Talos, Pandora
- The Golem
- Frankenstein

## Thinking machines and AI: A long history

> The Analytical Engine has no pretensions whatever to originate anything.	It can do whatever we know how to order it to perform... --- Ada Lovelace 1843

> Can machines do what we (as thinking entities) can do? --- Alan Turing "Computing Machinery and Intelligence"

## Thinking machines and AI: A long history

- AI solved many problems that are intellectually difficult for humans 
- true challenge: tasks that are easy for people to do but hard to describe formally 
	- recognizing faces in images 

## Machine Learning

- evolved in the 50s and 60s from AI, Pattern Recognition and computational learning theory
	- grew out of quest for AI
	- however, AI was dominated by logical, knowledge-based approaches 
	- flourished in the 1990s
		- from AI to tackle practical problems using data
- construct algorithms to learn from data (and predict)
- very similar to computational statistics (and the scientific method) 
- used where explicit solutions are hard to find:
	- OCR, SPAM, Network intrusion, Computer Vision

## Machine Learning

- performance depends heavily on representation of data
	- example
- representations can be engineered or
	- learned from the data itself (representation learning)
- AI > Machine Learning > Representation Learning > Deep Learning 


## A time line

- 1763 | Bayes (actually by Laplace in 1812)
- 1913 | Markov chains (work on poems)
- 1950 | Turing, A. M. 1950. “Computing Machinery and Intelligence.” Mind 59 (236) 433–60.
- 1957 | Perceptron Rosenblatt
- 1980 | Neocognitron Fukushima
- 1982 | Recurrent Networks Hopfield
- 1986 | Backprop Rumelhart, Hinton, Williams
- 1995 | Random Forests, Tin Kam Ho
- 1995 | SVM, Cortes and Vapnik
- 1997 | LSTM networks, Hochreiter and Schmidhuber

## Knowledge from data {data-background="https://upload.wikimedia.org/wikipedia/commons/6/69/NASA-HS201427a-HubbleUltraDeepField2014-20140603.jpg"}

> We are drowning in information and starving for knowledge. — John Naisbitt.
 
 - the core example: Tycho Brahe and Johannes Kepler
 - now we need computerised methods

## ML: Definition(s)
> ...a set of methods that can automatically detect patterns in data, and then *use the uncovered patterns to predict future data*, or to perform other kinds of decision making under uncertainty. -- Murphy 2012, Machine Learning

## ML: Definition(s)

Fitting data (optimisation) is different from finding patterns that generalise to unseen examples (machine learning)

## ML: Definition(s)

Rules, Data  -> [Classical Programming] -> Answers
Data, Answers -> [ML] -> Rules (programs, computational models)

## ML: Definition(s)

> A computer program is said to *learn* from experience E with respect to some class of tasks T and performance measure P if its performance at tasks in T, as measured by P, improves with experience E. -- from Mitchell, T. (1997). Machine Learning. McGraw Hill. 

## ML vs statistics

Machine Learning | Statistics
---|---
Networks, graphs | model
Weights | parameters
Learning | fitting
Supervised learning | regression/classification
Unsupervised learning | density estimation, clustering
Large grant = $1,000,000 | large grant = $50,000

*from http://statweb.stanford.edu/~tibs/stat315a/glossary.pdf*

## ML vs statistics 

- Breiman, Leo. 2001. “Statistical Modeling: The Two Cultures.” Statistical Science: A Review Journal of the Institute of Mathematical Statistics 16 (3) 199–231.

# ML: A Taxonomy

## ML: Definition(s)

> A computer program is said to *learn* from experience E with respect to some class of tasks T and performance measure P if its performance at tasks in T, as measured by P, improves with experience E. -- from Mitchell, T. (1997). Machine Learning. McGraw Hill. 

# The task T

## The task T

- ML allows us to tackle tasks that are too difficult to solve with fixed programs 

## Classification

- (y) response variable categorical 
- from x to y \el {1, ..., C}
	- C = 2 binary
	- C > 2 multiclass 
- or probability distribution over classes
- possibly with missing inputs (e.g. medical records)

## Classification examples
- email spam filtering
- object recoginition / image classification
	- handwriting recognition
	- face detection 

## Regression

- response variable real-valued 
- predict numerical value given input
- function approximation y = f(x) by y_hat = f_hat(x)

## Regression examples

- predict price of item (e.g. house, used car) based on known covariates
- predict stock market tomorrow
- predict PSA given clinical measurements


## other tasks

- Structured output (output is a vector with relationships between different elements) 
	- transcription (e.g. image to text)
	- translation (e.g. Klingon to French)
- Anomaly detection (e.g. credit card fraud)

## other tasks 

- Synthesis and sampling (e.g. textures in video games)
- Imputaton of missing values (p(x_i | x_-i)
- Denoising (clean x from corrupted x~ or p(x|x~))
- Density estimation (lear structure from data p(x)) 


# The performance measure P

## The performance measure P

- for classification: accuracy or error rate (expected 0-1 loss)
- for density estimation: e.g. average log prob assigned to examples
- depends on application
	- what mistakes are more problematic ?
	- e.g. regression: regular medium mistakes or rare large mistakes

## generalisation 

- how is performance on unseen data ?
- using a **test set** separate from training set

# The Experience, E

## The Experience, E

- *supervised* vs *unsupervised* methods

## Unsupervised Learning (descriptive learning) 

- experience a dataset containing many features
	- learn useful properties of structure 
	- find interesting patterns in the data
- knowledge discovery, density estimation 
- statistically speaking given:
	- supervised learning is conditional density estimation
		- learn p(**y**|**x**) given **x** and **y**
	- unsupervised learning is unconditional density estimation.
		- learn p(**x**) given **x**
- requires no training labels

## Supervised Learning (predictive learning)

- learn from data containing features and associated label(s)
	- given by an "instructor" (i.e. supervised)
- from features, covariates, attributes to response variable

## (Un)supervised 

- no formal definition and borders are blurred 
- p(**x**) = p(x_1) p(x_2 | x_1) p(x_3 | x_1, x_2) ... 
	- unsupervised problem could be split into a sequence of supervised problems
	- also works the other way around p(y|x) = p(x,y) / (sum_y' p(x,y'))
	- i.e. most methods could do both supervised and unsupervised 

## Typical examples supervised learning

- regression
- classification,
- structured output,
- denoising

## Typical examples unsupervised Learning

- density estimation 
- clustering
- imputation 
- latent factor discovery / dimensionality reduction
- (graph) structure (i.e. influence/correlations between variables)

## Unsupervised Learning: Clustering

## Unsupervised Learning: Density Estimation

## Unsupervised Learning: Dimensionality Reduction {data-background="https://c2.staticflickr.com/2/1501/26194566691_ef39f2c77b_b.jpg"}

## Unsupervised Learning (descriptive learning) 
> When we’re learning to see, nobody’s telling us what the right answers are — we just look. Every so often, your mother says “that’s a dog”, but that’s very little information. You’d be lucky if you got a few bits of information — even one bit per second — that way. The brain’s visual system has 10^14 neural connections. And you only live for 10^9 seconds. So it’s no use learning one bit per second. You need more like 10^5 bits per second. And there’s only one place you can get that much information: from the input itself. — Geoffrey Hinton, 1996 (quoted in (Gorder 2006)).

## Beyond (un) vs. supervised 

- semi-supervised learning
	- sparse labelling of dataset 
- multi-instance learning
	- only known if a collection contains example of class or not
	- individual members not labeled 

## The Design Matrix 


# Reinforcement Learning

- learning successful behavioural strategies from occasional rewards

# Where are we now?
<img src="https://upload.wikimedia.org/wikipedia/commons/9/94/Gartner_Hype_Cycle.svg" height = "600">


# Some basic Machine Learning concepts and principles

## parametric vs. Non-parametric models

- parametric: closed form, #parameters is fixed
	- fast to compute
	- strong assumptions
- non-parametric, #parameters grows with amount of (training) data
	- flexible
	- often computationally intractable

## parametric vs. Non-parametric models - K-nearest neighbours

## The curse of dimensionality
- Bellman 
- in high dimensions, (Euclidean) distances become meaningless 
- growing hypercube to collect data: e(f) = f^(1/D)
	- if we want f = 0.1 (10% of data) in D=10 dimensions, side length e = 0.8 (80% of max side-length) - not very local
- ration between min and max distance approaches 1 with D -> Inf

## parametric models

- to escape the curse of dimensionality we can make assumptions about the data distribution (inductive bias)
- parametric models with fixed number of parameters

## parametric models: regression

- linear regression

## parametric models: classification

- logistic regression

## Overfitting

- if we model every minute variation in the training data, we are likely to fit the noise as well
- less accurate prediction for future data
- perfect fit of training data = bad generalisation to unseen data

## model selection

- how to select a good model (e.g. the k in kNN)?
	- misclassification rate on training set
	- but: we care about generalisation error: misclassification rate on test set 
- U-shaped curve

## model selection

- during training we don't have access to test set:
	- split training data into training set and validation set
		- e.g. 80/20
	- cross validation (k folds) 
- some interesting concepts
	- Occam's razor
	- Rashomon and the multiplicity of good models

## No free lunch theorem 

> All models are wrong, but some models are useful. — George Box (Box and Draper 1987, p424).
- no free lunch theorem (Wolpert 1996)
- no universally best model across all problems
- assumptions that works well in one domain often fail in another

# Summary

## what have we learned... 
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



## 

<iframe src="http://docs.google.com/gview?url=http://statweb.stanford.edu/~tibs/stat315a/glossary.pdf&embedded=true" 
style="width:600px; height:500px;" frameborder="0"></iframe>