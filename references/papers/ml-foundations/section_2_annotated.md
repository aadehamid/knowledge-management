# Backpropagation 

TA: Yi Wen 

April 17, 2020 CS231n Discussion Section 

Slides credits: Barak Oshri, Vincent Chen, Nish Khandwala, Yi Wen 

## Agenda 

● Motivation 

● Backprop Tips & Tricks 

● Matrix calculus primer 

## Agenda 

● **Motivation** 

● Backprop Tips & Tricks 

● Matrix calculus primer 

## Motivation 

Recall: Optimization objective is minimize loss 


![](references/papers/ml-foundations/section_2_annotated_images/section_2_annotated.pdf-0004-02.png)


## Motivation 

Recall: Optimization objective is minimize loss 


![](references/papers/ml-foundations/section_2_annotated_images/section_2_annotated.pdf-0005-02.png)


**Goal: how should we tweak the parameters to decrease the loss?** 


![](references/papers/ml-foundations/section_2_annotated_images/section_2_annotated.pdf-0005-04.png)


## Agenda 

● Motivation 

● **Backprop Tips & Tricks** 

● Matrix calculus primer 

## A Simple Example 

Loss 


![](references/papers/ml-foundations/section_2_annotated_images/section_2_annotated.pdf-0007-02.png)


Goal: Tweak the parameters to minimize loss 

=> **minimize a multivariable function** in parameter space 

## A Simple Example 

=> **minimize a multivariable function** 


![](references/papers/ml-foundations/section_2_annotated_images/section_2_annotated.pdf-0008-02.png)



![](references/papers/ml-foundations/section_2_annotated_images/section_2_annotated.pdf-0008-03.png)



![](references/papers/ml-foundations/section_2_annotated_images/section_2_annotated.pdf-0008-04.png)


Plotted on WolframAlpha 

## Approach #1: Random Search 

**Intuition:** the _step_ we take in the domain of function 


![](references/papers/ml-foundations/section_2_annotated_images/section_2_annotated.pdf-0009-02.png)



![](references/papers/ml-foundations/section_2_annotated_images/section_2_annotated.pdf-0009-03.png)


Approach #2: Numerical Gradient **Intuition:** rate of change of a function with respect to a variable surrounding a small region 


![](references/papers/ml-foundations/section_2_annotated_images/section_2_annotated.pdf-0010-01.png)



![](references/papers/ml-foundations/section_2_annotated_images/section_2_annotated.pdf-0010-02.png)


Approach #2: Numerical Gradient **Intuition:** rate of change of a function with respect to a variable surrounding a small region 


![](references/papers/ml-foundations/section_2_annotated_images/section_2_annotated.pdf-0011-01.png)


**Finite Differences:** 


![](references/papers/ml-foundations/section_2_annotated_images/section_2_annotated.pdf-0011-03.png)



![](references/papers/ml-foundations/section_2_annotated_images/section_2_annotated.pdf-0011-04.png)


Approach #3: Analytical Gradient **Recall** : partial derivative by limit definition 


![](references/papers/ml-foundations/section_2_annotated_images/section_2_annotated.pdf-0012-01.png)



![](references/papers/ml-foundations/section_2_annotated_images/section_2_annotated.pdf-0012-02.png)



![](references/papers/ml-foundations/section_2_annotated_images/section_2_annotated.pdf-0012-03.png)



![](references/papers/ml-foundations/section_2_annotated_images/section_2_annotated.pdf-0012-04.png)



![](references/papers/ml-foundations/section_2_annotated_images/section_2_annotated.pdf-0012-05.png)


Approach #3: Analytical Gradient 

**Recall** : chain rule 


![](references/papers/ml-foundations/section_2_annotated_images/section_2_annotated.pdf-0013-02.png)


Approach #3: Analytical Gradient **Recall** : chain rule 

E.g. 


![](references/papers/ml-foundations/section_2_annotated_images/section_2_annotated.pdf-0014-02.png)



![](references/papers/ml-foundations/section_2_annotated_images/section_2_annotated.pdf-0014-03.png)


Approach #3: Analytical Gradient **Recall** : chain rule 

E.g. 


![](references/papers/ml-foundations/section_2_annotated_images/section_2_annotated.pdf-0015-02.png)



![](references/papers/ml-foundations/section_2_annotated_images/section_2_annotated.pdf-0015-03.png)


Approach #3: Analytical Gradient 

**Recall** : chain rule 


![](references/papers/ml-foundations/section_2_annotated_images/section_2_annotated.pdf-0016-02.png)


**Intuition:** upstream gradient values propagate backwards -- we can reuse them! 

## Gradient 


![](references/papers/ml-foundations/section_2_annotated_images/section_2_annotated.pdf-0017-01.png)



![](references/papers/ml-foundations/section_2_annotated_images/section_2_annotated.pdf-0017-02.png)


**“** direction and rate of fastest increase” 

Numerical Gradient vs Analytical Gradient 

## What about Autograd? 

- Deep learning frameworks can automatically perform backprop! 

● Problems might surface related to underlying gradients when debugging your models 

**“Yes You Should Understand Backprop”** 

**<u>https://medium.com/@karpathy/yes-you-should-understand-backprop-e2f06eab496b</u>** 

Problem Statement: Backpropagation 


![](references/papers/ml-foundations/section_2_annotated_images/section_2_annotated.pdf-0019-01.png)



![](references/papers/ml-foundations/section_2_annotated_images/section_2_annotated.pdf-0019-02.png)


Given a function **_f_** with respect to inputs **_x_** , labels **_y_** , and parameters 𝜃 compute the gradient of **_Loss_** with respect to 𝜃 

## Problem Statement: Backpropagation 

An algorithm for computing the gradient of a **compound** function as a series of **local, intermediate gradients** : 

1. Identify intermediate functions (forward prop) 

2. Compute local gradients (chain rule) 

3. Combine with upstream error signal to get full gradient 


![](references/papers/ml-foundations/section_2_annotated_images/section_2_annotated.pdf-0020-05.png)


<!-- Start of picture text -->
local(x,W,b) => y<br>Input x W,b y output<br>dx,dW,db <= grad_local(dy,x,W,b)<br>dx dy<br>dW,db<br><!-- End of picture text -->

## Modularity: Previous Example 

Compound function 


![](references/papers/ml-foundations/section_2_annotated_images/section_2_annotated.pdf-0021-02.png)


Intermediate Variables (forward propagation) 


![](references/papers/ml-foundations/section_2_annotated_images/section_2_annotated.pdf-0021-04.png)



![](references/papers/ml-foundations/section_2_annotated_images/section_2_annotated.pdf-0021-05.png)



![](references/papers/ml-foundations/section_2_annotated_images/section_2_annotated.pdf-0021-06.png)


## Modularity: 2-Layer Neural Network 

### Compound function 


![](references/papers/ml-foundations/section_2_annotated_images/section_2_annotated.pdf-0022-02.png)


### Intermediate Variables 


![](references/papers/ml-foundations/section_2_annotated_images/section_2_annotated.pdf-0022-04.png)


(forward propagation) 


![](references/papers/ml-foundations/section_2_annotated_images/section_2_annotated.pdf-0022-06.png)



![](references/papers/ml-foundations/section_2_annotated_images/section_2_annotated.pdf-0022-07.png)



![](references/papers/ml-foundations/section_2_annotated_images/section_2_annotated.pdf-0022-08.png)



![](references/papers/ml-foundations/section_2_annotated_images/section_2_annotated.pdf-0022-09.png)



![](references/papers/ml-foundations/section_2_annotated_images/section_2_annotated.pdf-0022-10.png)



![](references/papers/ml-foundations/section_2_annotated_images/section_2_annotated.pdf-0022-11.png)


=> Squared Euclidean Distance between      and 

Intermediate **Variables** (forward propagation) 


![](references/papers/ml-foundations/section_2_annotated_images/section_2_annotated.pdf-0023-01.png)



![](references/papers/ml-foundations/section_2_annotated_images/section_2_annotated.pdf-0023-02.png)



![](references/papers/ml-foundations/section_2_annotated_images/section_2_annotated.pdf-0023-03.png)


? f(x;W,b) = Wx + b ? 

(↑lecture note) Input one feature **vector** 

(←here) Input a batch of data ( **matrix** ) 


![](references/papers/ml-foundations/section_2_annotated_images/section_2_annotated.pdf-0023-07.png)


Intermediate **Variables** (forward propagation) 

1. intermediate functions 2. local gradients 3. full gradients 

Intermediate **Gradients** (backward propagation) 


![](references/papers/ml-foundations/section_2_annotated_images/section_2_annotated.pdf-0024-03.png)



![](references/papers/ml-foundations/section_2_annotated_images/section_2_annotated.pdf-0024-04.png)


？？？ ？？？ ？？？ 


![](references/papers/ml-foundations/section_2_annotated_images/section_2_annotated.pdf-0024-06.png)



![](references/papers/ml-foundations/section_2_annotated_images/section_2_annotated.pdf-0024-07.png)



![](references/papers/ml-foundations/section_2_annotated_images/section_2_annotated.pdf-0024-08.png)



![](references/papers/ml-foundations/section_2_annotated_images/section_2_annotated.pdf-0024-09.png)


## Agenda 

● Motivation 

- Backprop Tips & Tricks 

● **Matrix calculus primer** 


![](references/papers/ml-foundations/section_2_annotated_images/section_2_annotated.pdf-0026-00.png)


## Derivative w.r.t. Vector 

### Scalar-by-Vector 

Vector-by-Vector 


![](references/papers/ml-foundations/section_2_annotated_images/section_2_annotated.pdf-0026-04.png)



![](references/papers/ml-foundations/section_2_annotated_images/section_2_annotated.pdf-0026-05.png)


1. intermediate functions 

## Derivative w.r.t. Vector: Chain Rule 

2. local gradients 

3. full gradients 


![](references/papers/ml-foundations/section_2_annotated_images/section_2_annotated.pdf-0027-04.png)



![](references/papers/ml-foundations/section_2_annotated_images/section_2_annotated.pdf-0027-05.png)



![](references/papers/ml-foundations/section_2_annotated_images/section_2_annotated.pdf-0027-06.png)


? 


![](references/papers/ml-foundations/section_2_annotated_images/section_2_annotated.pdf-0027-08.png)



![](references/papers/ml-foundations/section_2_annotated_images/section_2_annotated.pdf-0027-09.png)



![](references/papers/ml-foundations/section_2_annotated_images/section_2_annotated.pdf-0028-00.png)



![](references/papers/ml-foundations/section_2_annotated_images/section_2_annotated.pdf-0028-01.png)



![](references/papers/ml-foundations/section_2_annotated_images/section_2_annotated.pdf-0028-02.png)



![](references/papers/ml-foundations/section_2_annotated_images/section_2_annotated.pdf-0028-03.png)



![](references/papers/ml-foundations/section_2_annotated_images/section_2_annotated.pdf-0028-04.png)


Derivative w.r.t. Vector: Takeaway 

## Derivative w.r.t. Matrix 

Scalar-by-Matrix 

Vector-by-Matrix 


![](references/papers/ml-foundations/section_2_annotated_images/section_2_annotated.pdf-0029-03.png)


? 

## Derivative w.r.t. Matrix: Dimension Balancing 

When you take **scalar-by-matrix** gradients 

The gradient has **shape of denominator** 

● Dimension balancing is the “cheap” but **efficient** approach to gradient calculations in most practical settings 


![](references/papers/ml-foundations/section_2_annotated_images/section_2_annotated.pdf-0031-00.png)



![](references/papers/ml-foundations/section_2_annotated_images/section_2_annotated.pdf-0031-01.png)



![](references/papers/ml-foundations/section_2_annotated_images/section_2_annotated.pdf-0031-02.png)



![](references/papers/ml-foundations/section_2_annotated_images/section_2_annotated.pdf-0031-03.png)



![](references/papers/ml-foundations/section_2_annotated_images/section_2_annotated.pdf-0031-04.png)



![](references/papers/ml-foundations/section_2_annotated_images/section_2_annotated.pdf-0031-05.png)



![](references/papers/ml-foundations/section_2_annotated_images/section_2_annotated.pdf-0031-06.png)



![](references/papers/ml-foundations/section_2_annotated_images/section_2_annotated.pdf-0031-07.png)


Derivative w.r.t. Matrix: Takeaway 

1. intermediate functions 

Intermediate **Variables** (forward propagation) 

Intermediate **Gradients** (backward propagation) 

2. local gradients 

3. full gradients 


![](references/papers/ml-foundations/section_2_annotated_images/section_2_annotated.pdf-0032-05.png)



![](references/papers/ml-foundations/section_2_annotated_images/section_2_annotated.pdf-0032-06.png)



![](references/papers/ml-foundations/section_2_annotated_images/section_2_annotated.pdf-0032-07.png)



![](references/papers/ml-foundations/section_2_annotated_images/section_2_annotated.pdf-0032-08.png)



![](references/papers/ml-foundations/section_2_annotated_images/section_2_annotated.pdf-0032-09.png)



![](references/papers/ml-foundations/section_2_annotated_images/section_2_annotated.pdf-0032-10.png)



![](references/papers/ml-foundations/section_2_annotated_images/section_2_annotated.pdf-0032-11.png)



![](references/papers/ml-foundations/section_2_annotated_images/section_2_annotated.pdf-0032-12.png)



![](references/papers/ml-foundations/section_2_annotated_images/section_2_annotated.pdf-0032-13.png)



![](references/papers/ml-foundations/section_2_annotated_images/section_2_annotated.pdf-0032-14.png)



![](references/papers/ml-foundations/section_2_annotated_images/section_2_annotated.pdf-0032-15.png)



![](references/papers/ml-foundations/section_2_annotated_images/section_2_annotated.pdf-0032-16.png)


## Backprop Menu for Success 

1. Write down variable graph 


![](references/papers/ml-foundations/section_2_annotated_images/section_2_annotated.pdf-0033-02.png)


2. Keep track of error signals 

3. Compute derivative of loss function 

4. Enforce shape rule on error signals, especially when deriving over a linear transformation 

## Vector-by-vector 


![](references/papers/ml-foundations/section_2_annotated_images/section_2_annotated.pdf-0034-01.png)



![](references/papers/ml-foundations/section_2_annotated_images/section_2_annotated.pdf-0034-02.png)



![](references/papers/ml-foundations/section_2_annotated_images/section_2_annotated.pdf-0034-03.png)



![](references/papers/ml-foundations/section_2_annotated_images/section_2_annotated.pdf-0034-04.png)


<!-- Start of picture text -->
?<br><!-- End of picture text -->

## Vector-by-vector 


![](references/papers/ml-foundations/section_2_annotated_images/section_2_annotated.pdf-0035-01.png)



![](references/papers/ml-foundations/section_2_annotated_images/section_2_annotated.pdf-0035-02.png)



![](references/papers/ml-foundations/section_2_annotated_images/section_2_annotated.pdf-0035-03.png)



![](references/papers/ml-foundations/section_2_annotated_images/section_2_annotated.pdf-0035-04.png)



![](references/papers/ml-foundations/section_2_annotated_images/section_2_annotated.pdf-0035-05.png)



![](references/papers/ml-foundations/section_2_annotated_images/section_2_annotated.pdf-0035-06.png)



![](references/papers/ml-foundations/section_2_annotated_images/section_2_annotated.pdf-0035-07.png)


<!-- Start of picture text -->
?<br><!-- End of picture text -->

## Vector-by-vector 


![](references/papers/ml-foundations/section_2_annotated_images/section_2_annotated.pdf-0036-01.png)



![](references/papers/ml-foundations/section_2_annotated_images/section_2_annotated.pdf-0036-02.png)



![](references/papers/ml-foundations/section_2_annotated_images/section_2_annotated.pdf-0036-03.png)



![](references/papers/ml-foundations/section_2_annotated_images/section_2_annotated.pdf-0036-04.png)


<!-- Start of picture text -->
?<br><!-- End of picture text -->

## Vector-by-vector 


![](references/papers/ml-foundations/section_2_annotated_images/section_2_annotated.pdf-0037-01.png)



![](references/papers/ml-foundations/section_2_annotated_images/section_2_annotated.pdf-0037-02.png)



![](references/papers/ml-foundations/section_2_annotated_images/section_2_annotated.pdf-0037-03.png)



![](references/papers/ml-foundations/section_2_annotated_images/section_2_annotated.pdf-0037-04.png)



![](references/papers/ml-foundations/section_2_annotated_images/section_2_annotated.pdf-0037-05.png)



![](references/papers/ml-foundations/section_2_annotated_images/section_2_annotated.pdf-0037-06.png)


<!-- Start of picture text -->
?<br><!-- End of picture text -->


![](references/papers/ml-foundations/section_2_annotated_images/section_2_annotated.pdf-0038-00.png)


## Matrix multiplication [Backprop] 


![](references/papers/ml-foundations/section_2_annotated_images/section_2_annotated.pdf-0038-02.png)



![](references/papers/ml-foundations/section_2_annotated_images/section_2_annotated.pdf-0038-03.png)



![](references/papers/ml-foundations/section_2_annotated_images/section_2_annotated.pdf-0038-04.png)



![](references/papers/ml-foundations/section_2_annotated_images/section_2_annotated.pdf-0038-05.png)



![](references/papers/ml-foundations/section_2_annotated_images/section_2_annotated.pdf-0038-06.png)



![](references/papers/ml-foundations/section_2_annotated_images/section_2_annotated.pdf-0038-07.png)



![](references/papers/ml-foundations/section_2_annotated_images/section_2_annotated.pdf-0038-08.png)


<!-- Start of picture text -->
?<br><!-- End of picture text -->


![](references/papers/ml-foundations/section_2_annotated_images/section_2_annotated.pdf-0038-09.png)



![](references/papers/ml-foundations/section_2_annotated_images/section_2_annotated.pdf-0038-10.png)



![](references/papers/ml-foundations/section_2_annotated_images/section_2_annotated.pdf-0038-11.png)


<!-- Start of picture text -->
?<br><!-- End of picture text -->


![](references/papers/ml-foundations/section_2_annotated_images/section_2_annotated.pdf-0038-12.png)


## Elementwise function [Backprop] 


![](references/papers/ml-foundations/section_2_annotated_images/section_2_annotated.pdf-0039-01.png)



![](references/papers/ml-foundations/section_2_annotated_images/section_2_annotated.pdf-0039-02.png)



![](references/papers/ml-foundations/section_2_annotated_images/section_2_annotated.pdf-0039-03.png)



![](references/papers/ml-foundations/section_2_annotated_images/section_2_annotated.pdf-0039-04.png)



![](references/papers/ml-foundations/section_2_annotated_images/section_2_annotated.pdf-0039-05.png)



![](references/papers/ml-foundations/section_2_annotated_images/section_2_annotated.pdf-0039-06.png)



![](references/papers/ml-foundations/section_2_annotated_images/section_2_annotated.pdf-0039-07.png)


<!-- Start of picture text -->
?<br><!-- End of picture text -->


![](references/papers/ml-foundations/section_2_annotated_images/section_2_annotated.pdf-0039-08.png)


