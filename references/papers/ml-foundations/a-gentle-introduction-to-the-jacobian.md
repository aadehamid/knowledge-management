title: A Gentle Introduction to the Jacobian - MachineLearningMastery.com A Gentle Introduction to the Jacobian - MachineLearningMastery.com
description: In the literature, the term Jacobian is often interchangeably used to refer to both the Jacobian matrix or its determinant. Both the matrix and the determinant have useful and important applications: in machine learning, the Jacobian matrix aggregates the partial derivatives that are necessary for backpropagation; the determinant is useful in the process of changing \[…\] In the literature, the term Jacobian is often interchangeably used to refer to both the Jacobian matrix or its determinant. Both the matrix and the determinant have useful and important applications: in machine learning, the Jacobian matrix aggregates the partial derivatives that are necessary for backpropagation; the determinant is useful in the process of changing between variables. In this tutorial,…
author: Stefania Cristina

### [Navigation](#navigation)

# A Gentle Introduction to the Jacobian

By [Stefania Cristina](https://machinelearningmastery.com/author/scristina/) on June 4, 2022 in [Calculus](https://machinelearningmastery.com/category/calculus/ "View all items in Calculus") [ 25](https://machinelearningmastery.com/a-gentle-introduction-to-the-jacobian/#comments)

In the literature, the term *Jacobian* is often interchangeably used to refer to both the Jacobian matrix or its determinant. 

Both the matrix and the determinant have useful and important applications: in machine learning, the Jacobian matrix aggregates the partial derivatives that are necessary for backpropagation; the determinant is useful in the process of changing between variables.

In this tutorial, you will review a gentle introduction to the Jacobian. 

After completing this tutorial, you will know:

- The Jacobian matrix collects all first-order partial derivatives of a multivariate function that can be used for backpropagation.
- The Jacobian determinant is useful in changing between variables, where it acts as a scaling factor between one coordinate space and another. 

Let’s get started. 

[![](https://machinelearningmastery.com/wp-content/uploads/2021/07/jacobian_cover-1024x658.jpg){width=1024 height=658}](https://machinelearningmastery.com/wp-content/uploads/2021/07/jacobian_cover-scaled.jpg)

A Gentle Introduction to the Jacobian\
Photo by [Simon Berger](https://unsplash.com/@8moments), some rights reserved.

## **Tutorial Overview**

This tutorial is divided into three parts; they are:

- Partial Derivatives in Machine Learning
- The Jacobian Matrix
- Other Uses of the Jacobian

## **Partial Derivatives in Machine Learning**

We have thus far mentioned [gradients and partial derivatives](https://machinelearningmastery.com/a-gentle-introduction-to-partial-derivatives-and-gradient-vectors) as being important for an optimization algorithm to update, say, the model weights of a neural network to reach an optimal set of weights. The use of partial derivatives permits each weight to be updated independently of the others, by calculating the gradient of the error curve with respect to each weight in turn.

Many of the functions that we usually work with in machine learning are [multivariate](https://machinelearningmastery.com/?p=12606&preview=true), [vector-valued functions](https://machinelearningmastery.com/a-gentle-introduction-to-vector-valued-functions), which means that they map multiple real inputs, *n*, to multiple real outputs, *m*:

For example, consider a neural network that classifies grayscale images into several classes. The function being implemented by such a classifier would map the *n* pixel values of each single-channel input image, to *m* output probabilities of belonging to each of the different classes. 

In training a neural network, the backpropagation algorithm is responsible for sharing back the error calculated at the output layer, among the neurons comprising the different hidden layers of the neural network, until it reaches the input. 

> *The fundamental principle of the backpropagation algorithm in adjusting the weights in a network is that each weight in a network should be updated in proportion to the sensitivity of the overall error of the network to changes in that weight.* 
> – Page 222, [Deep Learning](https://www.amazon.com/Deep-Learning-Press-Essential-Knowledge/dp/0262537559/ref=sr_1_4?dchild=1&keywords=deep+learning&qid=1622968138&sr=8-4), 2019.

This sensitivity of the overall error of the network to changes in any one particular weight is measured in terms of the rate of change, which, in turn, is calculated by taking the partial derivative of the error with respect to the same weight. 

For simplicity, suppose that one of the hidden layers of some particular network consists of just a single neuron, *k*. We can represent this in terms of a simple computational graph:

[![](https://machinelearningmastery.com/wp-content/uploads/2021/07/jacobian_1-1024x289.png){width=510 height=144}](https://machinelearningmastery.com/wp-content/uploads/2021/07/jacobian_1.png)

A Neuron with a Single Input and a Single Output

Again, for simplicity, let’s suppose that a weight, *w* *~k~*, is applied to an input of this neuron to produce an output, *z* *~k~*, according to the function that this neuron implements (including the nonlinearity). Then, the weight of this neuron can be connected to the error at the output of the network as follows (the following formula is formally known as the *chain rule of calculus*, but more on this later in a separate tutorial):

[![](https://machinelearningmastery.com/wp-content/uploads/2021/07/jacobian_2.png){width=187 height=60}](https://machinelearningmastery.com/wp-content/uploads/2021/07/jacobian_2.png)

Here, the derivative, *dz* *~k~* / *dw* *~k~*, first connects the weight, *w* *~k~*, to the output, *z* *~k~*, while the derivative, *d*error / *dz* *~k~*, subsequently connects the output, *z* *~k~*, to the network error. 

It is more often the case that we’d have many connected neurons populating the network, each attributed a different weight. Since we are more interested in such a scenario, then we can generalise beyond the scalar case to consider multiple inputs and multiple outputs:

[![](https://machinelearningmastery.com/wp-content/uploads/2021/07/jacobian_3-1024x130.png){width=473 height=60}](https://machinelearningmastery.com/wp-content/uploads/2021/07/jacobian_3.png)

This sum of terms can be represented more compactly as follows:

[![](https://machinelearningmastery.com/wp-content/uploads/2021/07/jacobian_4-1.png){width=207 height=65}](https://machinelearningmastery.com/wp-content/uploads/2021/07/jacobian_4-1.png)

Or, equivalently, in [vector notation](https://machinelearningmastery.com/a-gentle-introduction-to-partial-derivatives-and-gradient-vectors) using the del operator, ∇, to represent the gradient of the error with respect to either the weights, **w** *~k~*, or the outputs, **z** *~k~*:

[![](https://machinelearningmastery.com/wp-content/uploads/2021/07/jacobian_5.png){width=219 height=68}](https://machinelearningmastery.com/wp-content/uploads/2021/07/jacobian_5.png)

> *The back-propagation algorithm consists of performing such a Jacobian-gradient product for each operation in the graph.*
> – Page 207, [Deep Learning](https://www.amazon.com/Deep-Learning-Adaptive-Computation-Machine/dp/0262035618/ref=sr_1_1?dchild=1&keywords=deep+learning&qid=1622968138&sr=8-1), 2017.

This means that the backpropagation algorithm can relate the sensitivity of the network error to changes in the weights, through a multiplication by the *Jacobian matrix*, (∂**z** *~k~* / ∂**w** *~k~*)^T^.

Hence, what does this Jacobian matrix contain?

## **The Jacobian Matrix**

The Jacobian matrix collects all first-order partial derivatives of a multivariate function.

Specifically, consider first a function that maps *u* real inputs, to a single real output:

Then, for an input vector, **x**, of length, *u*, the Jacobian vector of size, 1 × *u*, can be defined as follows:

[![](https://machinelearningmastery.com/wp-content/uploads/2021/07/jacobian_7-1024x289.png){width=231 height=65}](https://machinelearningmastery.com/wp-content/uploads/2021/07/jacobian_7.png)

Now, consider another function that maps *u* real inputs, to *v* real outputs:

Then, for the same input vector, **x**, of length, *u*, the Jacobian is now a *v* × *u* matrix, **J** ∈ ℝ*^v×^* *^u^*, that is defined as follows:

[![](https://machinelearningmastery.com/wp-content/uploads/2021/07/jacobian_9-1024x305.png){width=395 height=118}](https://machinelearningmastery.com/wp-content/uploads/2021/07/jacobian_9.png)

Reframing the Jacobian matrix into the machine learning problem considered earlier, while retaining the same number of *u* real inputs and *v* real outputs, we find that this matrix would contain the following partial derivatives:

[![](https://machinelearningmastery.com/wp-content/uploads/2021/08/jacobian_10.png){width=188 height=147}](https://machinelearningmastery.com/wp-content/uploads/2021/08/jacobian_10.png) \

### Want to Get Started With Calculus for Machine Learning?

Take my free 7-day email crash course now (with sample code).

Click to sign-up and also get a free PDF Ebook version of the course.

## **Other Uses of the Jacobian**

An important technique when working with [integrals](https://machinelearningmastery.com/?p=12637&preview=true) involves the *change of variables* (also referred to as, *integration by substitution* or *u-substitution*), where an integral is simplified into another integral that is easier to compute. 

In the single variable case, substituting some variable, *x*, with another variable, *u*, can transform the original function into a simpler one for which it is easier to find an antiderivative. In the two variable case, an additional reason might be that we would also wish to transform the region of terms over which we are integrating, into a different shape. 

> *In the single variable case, there’s typically just one reason to want to change the variable: to make the function “nicer” so that we can find an antiderivative. In the two variable case, there is a second potential reason: the two-dimensional region over which we need to integrate is somehow unpleasant, and we want the region in terms of u and v to be nicer—to be a rectangle, for example.* 
> – Page 412, [Single and Multivariable Calculus](https://www.whitman.edu/mathematics/multivariable/multivariable.pdf), 2020.

When performing a substitution between two (or possibly more) variables, the process starts with a definition of the variables between which the substitution is to occur. For example, *x* \= *f*(*u*, *v*) and *y* \= *g*(*u*, *v*). This is then followed by a conversion of the integral limits depending on how the functions, *f* and *g*, will transform the *u*–*v* plane into the *x*–*y* plane. Finally, the absolute value of the *Jacobian determinant* is computed and included, to act as a scaling factor between one coordinate space and another. 

## **Further Reading**

This section provides more resources on the topic if you are looking to go deeper.

### **Books**

- [Deep Learning](https://www.amazon.com/Deep-Learning-Adaptive-Computation-Machine/dp/0262035618/ref=sr_1_1?dchild=1&keywords=deep+learning&qid=1622968138&sr=8-1), 2017.
- [Mathematics for Machine Learning](https://www.amazon.com/Mathematics-Machine-Learning-Peter-Deisenroth/dp/110845514X/ref=as_li_ss_tl?dchild=1&keywords=calculus+machine+learning&qid=1606171788&s=books&sr=1-3&linkCode=sl1&tag=inspiredalgor-20&linkId=209ba69202a6cc0a9f2b07439b4376ca&language=en_US), 2020.
- [Single and Multivariable Calculus](https://www.whitman.edu/mathematics/multivariable/multivariable.pdf), 2020.
- [Deep Learning](https://www.amazon.com/Deep-Learning-Press-Essential-Knowledge/dp/0262537559/ref=sr_1_4?dchild=1&keywords=deep+learning&qid=1622968138&sr=8-4), 2019.

### **Articles**

- [Jacobian matrix and determinant, Wikipedia](https://en.wikipedia.org/wiki/Jacobian_matrix_and_determinant).
- [Integration by substitution, Wikipedia](https://en.wikipedia.org/wiki/Integration_by_substitution).

## **Summary**

In this tutorial, you discovered a gentle introduction to the Jacobian.

Specifically, you learned:

- The Jacobian matrix collects all first-order partial derivatives of a multivariate function that can be used for backpropagation.
- The Jacobian determinant is useful in changing between variables, where it acts as a scaling factor between one coordinate space and another.

Do you have any questions?\
 Ask your questions in the comments below and I will do my best to answer.

## Get a Handle on Calculus for Machine Learning!

[![Calculus For Machine Learning](https://machinelearningmastery.com/wp-content/uploads/2022/11/C4ML-220.png)](https://machinelearningmastery.com/calculus-for-machine-learning/)

#### Feel Smarter with Calculus Concepts

...by getting a better sense on the calculus symbols and terms

Discover how in my new Ebook:\
 [Calculus for Machine Learning](https://machinelearningmastery.com/calculus-for-machine-learning/)

It provides **self-study tutorials** with **full working code** on:\
 *differntiation*, *gradient*, *Lagrangian mutiplier approach*, *Jacobian matrix*, and much more...

#### Bring Just Enough Calculus Knowledge to \
Your Machine Learning Projects

\

[See What's Inside](https://machinelearningmastery.com/calculus-for-machine-learning/)

### More On This Topic

- [![nick-night-8LTlfHL47Ac-unsplash](https://machinelearningmastery.com/wp-content/uploads/2026/01/nick-night-8LTlfHL47Ac-unsplash-200x200.jpg "A Gentle Introduction to Language Model Fine-tuning"){width=200 height=200} A Gentle Introduction to Language Model Fine-tuning](https://machinelearningmastery.com/a-gentle-introduction-to-language-model-fine-tuning/)
- [![mlm-ipc-gentle-introduction-batch-normalization](https://machinelearningmastery.com/wp-content/uploads/2025/08/mlm-ipc-gentle-introduction-batch-normalization-200x200.png "A Gentle Introduction to Batch Normalization"){width=200 height=200} A Gentle Introduction to Batch Normalization](https://machinelearningmastery.com/a-gentle-introduction-to-batch-normalization/)
- [![mlm-ipc-gentle-introduction-bayesian-regression](https://machinelearningmastery.com/wp-content/uploads/2025/07/mlm-ipc-gentle-introduction-bayesian-regression-200x200.png "A Gentle Introduction to Bayesian Regression"){width=200 height=200} A Gentle Introduction to Bayesian Regression](https://machinelearningmastery.com/a-gentle-introduction-to-bayesian-regression/)
- [![mlm-ipc-gentle-introduction-q-learning](https://machinelearningmastery.com/wp-content/uploads/2025/07/mlm-ipc-gentle-introduction-q-learning-200x200.png "A Gentle Introduction to Q-Learning"){width=200 height=200} A Gentle Introduction to Q-Learning](https://machinelearningmastery.com/a-gentle-introduction-to-q-learning/)
- [![caleb-jack-jUxMsNZZCJ8-unsplash](https://machinelearningmastery.com/wp-content/uploads/2025/06/caleb-jack-jUxMsNZZCJ8-unsplash-200x200.jpg "A Gentle Introduction to Attention Masking in Transformer Models"){width=200 height=200} A Gentle Introduction to Attention Masking in…](https://machinelearningmastery.com/a-gentle-introduction-to-attention-masking-in-transformer-models/)
- [![victoriano-izquierdo-29Rh5DOS5Qs-unsplash](https://machinelearningmastery.com/wp-content/uploads/2025/06/victoriano-izquierdo-29Rh5DOS5Qs-unsplash-200x200.jpg "A Gentle Introduction to Multi-Head Latent Attention (MLA)"){width=200 height=200} A Gentle Introduction to Multi-Head Latent Attention (MLA)](https://machinelearningmastery.com/a-gentle-introduction-to-multi-head-latent-attention-mla/)

 [backpropagation](https://machinelearningmastery.com/tag/backpropagation/), [first-order](https://machinelearningmastery.com/tag/first-order/), [integral calculus](https://machinelearningmastery.com/tag/integral-calculus/), [jacobian](https://machinelearningmastery.com/tag/jacobian/), [jacobian determinant](https://machinelearningmastery.com/tag/jacobian-determinant/), [jacobian matrix](https://machinelearningmastery.com/tag/jacobian-matrix/), [multivariate](https://machinelearningmastery.com/tag/multivariate/), [partial derivatives](https://machinelearningmastery.com/tag/partial-derivatives/)

[ Higher-Order Derivatives](https://machinelearningmastery.com/higher-order-derivatives/) [A Gentle Introduction To Hessian Matrices ](https://machinelearningmastery.com/a-gentle-introduction-to-hessian-matrices/)

### 25 Responses to *A Gentle Introduction to the Jacobian* {#comments-title}

1. 
[Asmare Belay](http://no) August 3, 2021 at 11:38 am [#](https://machinelearningmastery.com/a-gentle-introduction-to-the-jacobian/#comment-619628 "Direct link to this comment")very interesting    - 
Stefania Cristina August 4, 2021 at 2:07 am [#](https://machinelearningmastery.com/a-gentle-introduction-to-the-jacobian/#comment-619691 "Direct link to this comment")Thank you!    - 
Italo August 16, 2021 at 12:59 am [#](https://machinelearningmastery.com/a-gentle-introduction-to-the-jacobian/#comment-621964 "Direct link to this comment")So great, Stefania. Thanks!!!        - 
Stefania Cristina August 17, 2021 at 4:48 pm [#](https://machinelearningmastery.com/a-gentle-introduction-to-the-jacobian/#comment-622282 "Direct link to this comment")You’re welcome!    - 
Wuraola December 16, 2021 at 6:51 pm [#](https://machinelearningmastery.com/a-gentle-introduction-to-the-jacobian/#comment-644956 "Direct link to this comment")Very interesting, l am working on the application of Jacobian matrix to algebra of order 42. 
Ananthapadmanabhan G August 5, 2021 at 11:14 pm [#](https://machinelearningmastery.com/a-gentle-introduction-to-the-jacobian/#comment-620169 "Direct link to this comment")What are the uses of these in simultaneous equations in applied side.    - 
Stefania Cristina August 10, 2021 at 5:50 pm [#](https://machinelearningmastery.com/a-gentle-introduction-to-the-jacobian/#comment-621132 "Direct link to this comment")If you would like to solve a system of nonlinear equations by Newton’s method, [this is how](https://en.wikipedia.org/wiki/Newton%27s_method#Nonlinear_systems_of_equations) the Jacobian would be used.3. 
Tarique Ahmad August 7, 2021 at 3:54 am [#](https://machinelearningmastery.com/a-gentle-introduction-to-the-jacobian/#comment-620482 "Direct link to this comment")Well explained    - 
Stefania Cristina August 10, 2021 at 5:44 pm [#](https://machinelearningmastery.com/a-gentle-introduction-to-the-jacobian/#comment-621128 "Direct link to this comment")Thank you!4. 
[JG](http://www.acehl.org) August 7, 2021 at 5:01 am [#](https://machinelearningmastery.com/a-gentle-introduction-to-the-jacobian/#comment-620512 "Direct link to this comment")Great tutorial about math compact notations! Thank you!I did not know that Jacobian matrix was behind backpropagation algorithm! so it is one of the main pillars of machine learning …because of first order derivatives are behind the way to distribute the cost function (error) into each of the neuron layers’s weights ..as some kind of error sensitivity to all weighs !Anyway, many times math scares people because of symbols or compact notation …where so many things are expressed in a “simple” expression …they call it an elegant way …\
 So I like the way where such compact functions are explained in detail, or break-down into an operative way …an later on rebuilt the expression with your own invented symbols :-))    - 
Stefania Cristina August 10, 2021 at 5:44 pm [#](https://machinelearningmastery.com/a-gentle-introduction-to-the-jacobian/#comment-621127 "Direct link to this comment")Thank you for the insight!5. 
Rushikesh August 7, 2021 at 10:35 am [#](https://machinelearningmastery.com/a-gentle-introduction-to-the-jacobian/#comment-620616 "Direct link to this comment")Nicely explain    - 
Stefania Cristina August 10, 2021 at 5:43 pm [#](https://machinelearningmastery.com/a-gentle-introduction-to-the-jacobian/#comment-621126 "Direct link to this comment")Thank you.6. 
[Keang](http://www.keang.be/) August 7, 2021 at 4:11 pm [#](https://machinelearningmastery.com/a-gentle-introduction-to-the-jacobian/#comment-620711 "Direct link to this comment")Is the lower left term in this formula a typo?\
 Should it be ∂zk(v) / ∂wk(1) instead?[https://machinelearningmastery.com/wp-content/uploads/2021/07/jacobian\_10.png](https://machinelearningmastery.com/wp-content/uploads/2021/07/jacobian_10.png)    - 
Stefania Cristina August 10, 2021 at 6:07 pm [#](https://machinelearningmastery.com/a-gentle-introduction-to-the-jacobian/#comment-621134 "Direct link to this comment")Indeed, thank you for that!7. 
Bhavin August 8, 2021 at 5:55 am [#](https://machinelearningmastery.com/a-gentle-introduction-to-the-jacobian/#comment-620847 "Direct link to this comment")Thanks Stephanie for a nice article.\
 Just one small comment that while you defined Jacobian matrix, I guess you want to say $J \in \R\^{v\times u}$    - 
Stefania Cristina August 10, 2021 at 5:43 pm [#](https://machinelearningmastery.com/a-gentle-introduction-to-the-jacobian/#comment-621125 "Direct link to this comment")Indeed, thank you for this!8. 
Atul January 2, 2022 at 4:28 am [#](https://machinelearningmastery.com/a-gentle-introduction-to-the-jacobian/#comment-649096 "Direct link to this comment")Great article, Stefania! Shouldn’t the Jacobian for the first function be of size 1 x u and not u x 1?    - 
James Carmichael February 27, 2022 at 12:48 pm [#](https://machinelearningmastery.com/a-gentle-introduction-to-the-jacobian/#comment-657072 "Direct link to this comment")Hi Atul…Please clarify which function you are referring to so that I may better assist you.9. 
Risto Lankinen June 3, 2022 at 11:12 pm [#](https://machinelearningmastery.com/a-gentle-introduction-to-the-jacobian/#comment-671340 "Direct link to this comment")There is an error in the article. It states that f:R\^u -> R\^v produces v\*u matrix, hence f:R\^u -> R\^1 should produce 1\*u matrix. Previous paragraph however states that f:R\^u -> R produces a u\*1 matrix. This is an error because 1\*u matrix and u\*1 matrix are mathematically different objects.    - 
James Carmichael June 4, 2022 at 10:18 am [#](https://machinelearningmastery.com/a-gentle-introduction-to-the-jacobian/#comment-671369 "Direct link to this comment")Thank you for the feedback Risto! We will review the items you noted.10. 
sarah February 8, 2024 at 1:11 am [#](https://machinelearningmastery.com/a-gentle-introduction-to-the-jacobian/#comment-708900 "Direct link to this comment")Thank you for the article, very clear !\
 I want to know how can we compute uncertainty of the output of the input from the jacobian? Thank you in advance.    - 
James Carmichael February 8, 2024 at 10:03 am [#](https://machinelearningmastery.com/a-gentle-introduction-to-the-jacobian/#comment-708918 "Direct link to this comment")Hi Sarah…The following resource may be of interest to you:[https://www.cambridge.org/core/journals/design-science/article/uncertainty-quantification-and-reduction-using-jacobian-and-hessian-information/957A5E1284BB22E1DC734187E9625396](https://www.cambridge.org/core/journals/design-science/article/uncertainty-quantification-and-reduction-using-jacobian-and-hessian-information/957A5E1284BB22E1DC734187E9625396)11. 
Oguz November 29, 2024 at 1:32 am [#](https://machinelearningmastery.com/a-gentle-introduction-to-the-jacobian/#comment-727462 "Direct link to this comment")I don’t understand these parts: “Or, equivalently, in vector notation using the del operator, ∇, to represent the gradient of the error with respect to either the weights, wk, or the outputs, zk” “This means that the backpropagation algorithm can relate the sensitivity of the network error to changes in the weights, through a multiplication by the Jacobian matrix, (∂zk / ∂wk)T.” “Reframing the Jacobian matrix into the machine learning problem considered earlier, while retaining the same number of u real inputs and v real outputs, we find that this matrix would contain the following partial derivatives:”Loss function is a scalar function, also the other ones (activation function, sum of weights vice versa) are SCALAR functions, not vector-valued. So we need to compute the gradient, not the Jacobian.\
 Why do we need Jacobian, where all the functions are scalar??    - 
James Carmichael November 30, 2024 at 3:53 am [#](https://machinelearningmastery.com/a-gentle-introduction-to-the-jacobian/#comment-727500 "Direct link to this comment")Hi Oguz…Your confusion arises from the context in which the \*\*Jacobian\*\* is mentioned and why it might be relevant despite the functions being scalar-valued. Let me clarify this step by step:Backpropagation computes the gradient of the \*\*scalar loss\*\* \\( \mathcal{L} \\) with respect to all the weights. To do so, it needs to account for how changes in weights affect the \*\*layer outputs\*\* (\\( z\_k \\)), and how those outputs propagate forward to the final loss. This is where the \*\*Jacobian\*\* comes in.While the \*\*loss function\*\* \\( \mathcal{L} \\) is scalar, intermediate functions (e.g., the activations \\( z\_k \\)) are vector-valued. To compute the gradient of \\( \mathcal{L} \\) with respect to the weights, backpropagation relies on chain rule expressions like:\
 \\\[\
 \frac{\partial \mathcal{L}}{\partial w\_k} \= \frac{\partial \mathcal{L}}{\partial z\_k} \cdot \frac{\partial z\_k}{\partial w\_k}.\
 \\\]Here:\
 – \\( \frac{\partial \mathcal{L}}{\partial z\_k} \\) is a gradient (row vector).\
 – \\( \frac{\partial z\_k}{\partial w\_k} \\) is the \*\*Jacobian matrix\*\*.### 4. \*\*Simplified view for scalar-valued functions\*\*\
 For scalar-valued functions (like the loss), the full Jacobian isn’t always needed explicitly:\
 – In many cases, the Jacobian reduces to a simpler derivative because of the structure of scalar functions.\
 – However, when working with intermediate vector-valued quantities, the Jacobian implicitly represents the relationships needed for gradient computation.### 5. \*\*The key insight\*\*\
 The \*\*Jacobian matrix\*\* appears because the network layers are vector-valued mappings. While the \*\*loss function\*\* is scalar, backpropagation computes how the \*\*vector-valued activations\*\* depend on the \*\*weights\*\* and uses the Jacobian to relate those dependencies to the scalar loss.If you’re only computing a scalar loss with respect to scalar parameters (e.g., in a single-variable function), the Jacobian is unnecessary. However, in neural networks, where layers and weights involve vectors, the Jacobian naturally arises in the computation process.
### Leave a Reply  {#reply-title}

*[June 4, 2022]: 2022-06-04T01:01:00+1000
