# Lecture 4: Neural Networks and Backpropagation 

<mark>Fei-Fei Li, Ranjay Krishna, Danfei Xu</mark> 

<mark>April 16, 2020</mark> 

<mark>1</mark> 

<mark>Lecture 4 -</mark> 

#### Administrative: Assignment 1 

**Assignment 1** due **Wednesday April 22** , 11:59pm If using Google Cloud, **you don’t need GPUs** for this assignment! 

<mark>Fei-Fei Li, Ranjay Krishna, Danfei Xu</mark> 

<mark>April 16, 2020</mark> 

<mark>Lecture 4 - 2</mark> 

#### Administrative: Project Proposal 

###### Project proposal due 4/27 

<mark>Fei-Fei Li, Ranjay Krishna, Danfei Xu</mark> 

<mark>April 16, 2020</mark> 

<mark>3</mark> 

<mark>Lecture 4 -</mark> 

#### Administrative: Discussion Section 

Discussion section tomorrow: Backpropagation 

<mark>Fei-Fei Li, Ranjay Krishna, Danfei Xu</mark> 

<mark>April 16, 2020</mark> 

<mark>Lecture 4 -</mark> 

<mark>4</mark> 

#### Administrative: Midterm Updates 

University has updated guidance on administering exams in spring quarter. In order to comply with the current policies, we have changed the exam format as the following to be consistent with exams in previous offerings of cs 231n: **Date** : released on Tuesday 5/12 (open for 24 hours to choose 1hr 40 mins time frame) **Format** : Timestamped with Gradescope 

<mark>Fei-Fei Li, Ranjay Krishna, Danfei Xu</mark> 

<mark>April 16, 2020</mark> 

<mark>Lecture 4 -</mark> 

<mark>5</mark> 

#### Where we are... 

Linear score function 

SVM loss (or softmax) 

data loss + regularization 

<mark>Fei-Fei Li, Ranjay Krishna, Danfei Xu</mark> 

<mark>April 16, 2020</mark> 

<mark>Lecture 4 - 6</mark> 

#### Finding the best W: Optimize with Gradient Descent 


![](references/papers/ml-foundations/lecture_4_images/lecture_4.pdf-0007-01.png)



![](references/papers/ml-foundations/lecture_4_images/lecture_4.pdf-0007-02.png)


<u>Landscape image</u> is CC0 1.0 public domain <u>Walking man image is CC0 1.0</u> public domain 


![](references/papers/ml-foundations/lecture_4_images/lecture_4.pdf-0007-04.png)


<mark>Fei-Fei Li, Ranjay Krishna, Danfei Xu</mark> 

<mark>April 16, 2020</mark> 

<mark>Lecture 4 -</mark> 

<mark>7</mark> 

#### Gradient descent 


![](references/papers/ml-foundations/lecture_4_images/lecture_4.pdf-0008-01.png)


**Numerical gradient** : slow :(, approximate :(, easy to write :) **Analytic gradient** : fast :), exact :), error-prone :( In practice: Derive analytic gradient, check your implementation with numerical gradient 

<mark>Fei-Fei Li, Ranjay Krishna, Danfei Xu</mark> 

<mark>April 16, 2020</mark> 

<mark>8</mark> 

<mark>Lecture 4 -</mark> 

#### Where we are... 

Linear score function SVM loss (or softmax) data loss + regularization How to find the best W? 

<mark>Fei-Fei Li, Ranjay Krishna, Danfei Xu</mark> 

<mark>April 16, 2020</mark> 

<mark>9</mark> 

<mark>Lecture 4 -</mark> 

#### Problem: Linear Classifiers are not very powerful 

###### Visual Viewpoint 

Geometric Viewpoint 


![](references/papers/ml-foundations/lecture_4_images/lecture_4.pdf-0010-03.png)



![](references/papers/ml-foundations/lecture_4_images/lecture_4.pdf-0010-04.png)



![](references/papers/ml-foundations/lecture_4_images/lecture_4.pdf-0010-05.png)


Linear classifiers learn one template per class 

Linear classifiers can only draw linear decision boundaries 

<mark>Fei-Fei Li, Ranjay Krishna, Danfei Xu</mark> 

<mark>April 16, 2020</mark> 

<mark>Lecture 4 - 10</mark> 

#### Pixel Features 


![](references/papers/ml-foundations/lecture_4_images/lecture_4.pdf-0011-01.png)


f(x) = Wx 


![](references/papers/ml-foundations/lecture_4_images/lecture_4.pdf-0011-03.png)


Class scores 


![](references/papers/ml-foundations/lecture_4_images/lecture_4.pdf-0011-05.png)


<mark>Fei-Fei Li, Ranjay Krishna, Danfei Xu</mark> 

<mark>11</mark> 

<mark>April 16, 2020</mark> 

<mark>Lecture 4 -</mark> 

#### Image Features 


![](references/papers/ml-foundations/lecture_4_images/lecture_4.pdf-0012-01.png)


f(x) = Wx Class scores Feature Representation 

<mark>Fei-Fei Li, Ranjay Krishna, Danfei Xu</mark> 

<mark>12</mark> 

<mark>April 16, 2020</mark> 

<mark>Lecture 4 -</mark> 

#### Image Features: Motivation 


![](references/papers/ml-foundations/lecture_4_images/lecture_4.pdf-0013-01.png)


<!-- Start of picture text -->
y<br>x<br><!-- End of picture text -->

Cannot separate red and blue points with linear classifier 

<mark>Fei-Fei Li, Ranjay Krishna, Danfei Xu</mark> 

<mark>13</mark> 

<mark>April 16, 2020</mark> 

<mark>Lecture 4 -</mark> 

#### Image Features: Motivation 


![](references/papers/ml-foundations/lecture_4_images/lecture_4.pdf-0014-01.png)


<!-- Start of picture text -->
y θ<br>f(x, y) = (r(x, y), θ(x, y))<br>x<br>r<br><!-- End of picture text -->

Cannot separate red and blue points with linear classifier 

After applying feature transform, points can be separated by linear classifier 

<mark>Fei-Fei Li, Ranjay Krishna, Danfei Xu</mark> 

<mark>14</mark> 

<mark>April 16, 2020</mark> 

<mark>Lecture 4 -</mark> 

#### Example: Color Histogram 


![](references/papers/ml-foundations/lecture_4_images/lecture_4.pdf-0015-01.png)



![](references/papers/ml-foundations/lecture_4_images/lecture_4.pdf-0015-02.png)



![](references/papers/ml-foundations/lecture_4_images/lecture_4.pdf-0015-03.png)



![](references/papers/ml-foundations/lecture_4_images/lecture_4.pdf-0015-04.png)


<!-- Start of picture text -->
+1<br><!-- End of picture text -->


![](references/papers/ml-foundations/lecture_4_images/lecture_4.pdf-0015-05.png)


<mark>Fei-Fei Li, Ranjay Krishna, Danfei Xu</mark> 

<mark>15</mark> 

<mark>April 16, 2020</mark> 

<mark>Lecture 4 -</mark> 

#### Example: Histogram of Oriented Gradients (HoG) 


![](references/papers/ml-foundations/lecture_4_images/lecture_4.pdf-0016-01.png)



![](references/papers/ml-foundations/lecture_4_images/lecture_4.pdf-0016-02.png)


Divide image into 8x8 pixel regions Within each region quantize edge direction into 9 bins 

Example: 320x240 image gets divided into 40x30 bins; in each bin there are 9 numbers so feature vector has 30*40*9 = 10,800 numbers 

Lowe, “Object recognition from local scale-invariant features”, ICCV 1999 Dalal and Triggs, "Histograms of oriented gradients for human detection," CVPR 2005 

<mark>Fei-Fei Li, Ranjay Krishna, Danfei Xu</mark> 

<mark>16</mark> 

<mark>April 16, 2020</mark> 

<mark>Lecture 4 -</mark> 

#### Example: Bag of Words 

###### **Step 1: Build codebook** 


![](references/papers/ml-foundations/lecture_4_images/lecture_4.pdf-0017-02.png)



![](references/papers/ml-foundations/lecture_4_images/lecture_4.pdf-0017-03.png)


<!-- Start of picture text -->
Extract random<br>patches<br><!-- End of picture text -->


![](references/papers/ml-foundations/lecture_4_images/lecture_4.pdf-0017-04.png)


###### **Step 2: Encode images** 


![](references/papers/ml-foundations/lecture_4_images/lecture_4.pdf-0017-06.png)



![](references/papers/ml-foundations/lecture_4_images/lecture_4.pdf-0017-07.png)



![](references/papers/ml-foundations/lecture_4_images/lecture_4.pdf-0017-08.png)



![](references/papers/ml-foundations/lecture_4_images/lecture_4.pdf-0017-09.png)



![](references/papers/ml-foundations/lecture_4_images/lecture_4.pdf-0017-10.png)



![](references/papers/ml-foundations/lecture_4_images/lecture_4.pdf-0017-11.png)



![](references/papers/ml-foundations/lecture_4_images/lecture_4.pdf-0017-12.png)



![](references/papers/ml-foundations/lecture_4_images/lecture_4.pdf-0017-13.png)



![](references/papers/ml-foundations/lecture_4_images/lecture_4.pdf-0017-14.png)


<!-- Start of picture text -->
Cluster patches to<br>form “codebook”<br>of “visual words”<br><!-- End of picture text -->


![](references/papers/ml-foundations/lecture_4_images/lecture_4.pdf-0017-15.png)



![](references/papers/ml-foundations/lecture_4_images/lecture_4.pdf-0017-16.png)



![](references/papers/ml-foundations/lecture_4_images/lecture_4.pdf-0017-17.png)



![](references/papers/ml-foundations/lecture_4_images/lecture_4.pdf-0017-18.png)



![](references/papers/ml-foundations/lecture_4_images/lecture_4.pdf-0017-19.png)



![](references/papers/ml-foundations/lecture_4_images/lecture_4.pdf-0017-20.png)



![](references/papers/ml-foundations/lecture_4_images/lecture_4.pdf-0017-21.png)



![](references/papers/ml-foundations/lecture_4_images/lecture_4.pdf-0017-22.png)



![](references/papers/ml-foundations/lecture_4_images/lecture_4.pdf-0017-23.png)



![](references/papers/ml-foundations/lecture_4_images/lecture_4.pdf-0017-24.png)


Fei-Fei and Perona, “A bayesian hierarchical model for learning natural scene categories”, CVPR 2005 

<mark>Fei-Fei Li, Ranjay Krishna, Danfei Xu</mark> 

<mark>17</mark> 

<mark>April 16, 2020</mark> 

<mark>Lecture 4 -</mark> 

#### Image Features 


![](references/papers/ml-foundations/lecture_4_images/lecture_4.pdf-0018-01.png)



![](references/papers/ml-foundations/lecture_4_images/lecture_4.pdf-0018-02.png)



![](references/papers/ml-foundations/lecture_4_images/lecture_4.pdf-0018-03.png)


<mark>Fei-Fei Li, Ranjay Krishna, Danfei Xu</mark> 

<mark>Lecture 4 - 18</mark> 

<mark>April 16, 2020</mark> 

#### Image features vs ConvNets 


![](references/papers/ml-foundations/lecture_4_images/lecture_4.pdf-0019-01.png)



![](references/papers/ml-foundations/lecture_4_images/lecture_4.pdf-0019-02.png)



![](references/papers/ml-foundations/lecture_4_images/lecture_4.pdf-0019-03.png)


<!-- Start of picture text -->
f<br>Feature Extraction<br>10  numbers giving<br>scores for classes<br>training<br>Krizhevsky, Sutskever, and Hinton, “Imagenet classification<br>with deep convolutional neural networks”, NIPS 2012.<br>Figure copyright Krizhevsky, Sutskever, and Hinton, 2012.<br>Reproduced with permission.<br>10  numbers giving<br>scores for classes<br>training<br><!-- End of picture text -->

<mark>Fei-Fei Li, Ranjay Krishna, Danfei Xu</mark> 

<mark>19</mark> 

<mark>April 16, 2020</mark> 

<mark>Lecture 4 -</mark> 

#### One Solution: Feature Transformation 


![](references/papers/ml-foundations/lecture_4_images/lecture_4.pdf-0020-01.png)


<!-- Start of picture text -->
f(x, y) = (r(x, y), θ(x, y))<br>Transform data with a cleverly<br>chosen  feature transform  f,<br>then apply linear classifier<br>Color Histogram Histogram of Oriented Gradients (HoG)<br><!-- End of picture text -->

<mark>Fei-Fei Li, Ranjay Krishna, Danfei Xu</mark> 

<mark>April 16, 2020</mark> 

<mark>Lecture 4 - 20</mark> 

Today: Neural Networks 

<mark>Fei-Fei Li, Ranjay Krishna, Danfei Xu</mark> 

<mark>Lecture 4 - 21 April 16, 2020</mark> 

#### Neural networks: without the brain stuff 

( **Before** ) Linear score function: 


![](references/papers/ml-foundations/lecture_4_images/lecture_4.pdf-0022-02.png)



![](references/papers/ml-foundations/lecture_4_images/lecture_4.pdf-0022-03.png)


<mark>Fei-Fei Li, Ranjay Krishna, Danfei Xu</mark> 

<mark>April 16, 2020</mark> 

<mark>Lecture 4 - 22</mark> 

#### Neural networks: without the brain stuff 

( **Before** ) Linear score function: <mark>(</mark> **<mark>Now</mark>** <mark>) 2-layer Neural Network</mark> 


![](references/papers/ml-foundations/lecture_4_images/lecture_4.pdf-0023-02.png)



![](references/papers/ml-foundations/lecture_4_images/lecture_4.pdf-0023-03.png)



![](references/papers/ml-foundations/lecture_4_images/lecture_4.pdf-0023-04.png)


(In practice we will usually add a learnable bias at each layer as well) 

<mark>Fei-Fei Li, Ranjay Krishna, Danfei Xu</mark> 

<mark>Lecture 4 - 23 April 16, 2020</mark> 

#### Neural networks: without the brain stuff 

( **Before** ) Linear score function: <mark>(</mark> **<mark>Now</mark>** <mark>) 2-layer Neural Network</mark> 


![](references/papers/ml-foundations/lecture_4_images/lecture_4.pdf-0024-02.png)



![](references/papers/ml-foundations/lecture_4_images/lecture_4.pdf-0024-03.png)



![](references/papers/ml-foundations/lecture_4_images/lecture_4.pdf-0024-04.png)


“Neural Network” is a very broad term; these are more accurately called “fully-connected networks” or sometimes “multi-layer perceptrons” (MLP) (In practice we will usually add a learnable bias at each layer as well) 

<mark>Fei-Fei Li, Ranjay Krishna, Danfei Xu</mark> 

<mark>Lecture 4 - 24 April 16, 2020</mark> 

#### Neural networks: without the brain stuff 

( **Before** ) Linear score function: <mark>(</mark> **<mark>Now</mark>** <mark>) 2-layer Neural Network</mark> or 3-layer Neural Network 


![](references/papers/ml-foundations/lecture_4_images/lecture_4.pdf-0025-02.png)



![](references/papers/ml-foundations/lecture_4_images/lecture_4.pdf-0025-03.png)



![](references/papers/ml-foundations/lecture_4_images/lecture_4.pdf-0025-04.png)



![](references/papers/ml-foundations/lecture_4_images/lecture_4.pdf-0025-05.png)


(In practice we will usually add a learnable bias at each layer as well) 

<mark>Fei-Fei Li, Ranjay Krishna, Danfei Xu</mark> 

<mark>Lecture 4 - 25 April 16, 2020</mark> 

#### Neural networks: without the brain stuff 

<mark>(</mark> **<mark>Before</mark>** <mark>) Linear score function: (</mark> **<mark>Now</mark>** <mark>) 2-layer Neural Network</mark> 


![](references/papers/ml-foundations/lecture_4_images/lecture_4.pdf-0026-02.png)



![](references/papers/ml-foundations/lecture_4_images/lecture_4.pdf-0026-03.png)



![](references/papers/ml-foundations/lecture_4_images/lecture_4.pdf-0026-04.png)


<!-- Start of picture text -->
x W1 h W2 s<br>10<br>3072 100<br><!-- End of picture text -->


![](references/papers/ml-foundations/lecture_4_images/lecture_4.pdf-0026-05.png)


<mark>Fei-Fei Li, Ranjay Krishna, Danfei Xu</mark> 

<mark>April 16, 2020</mark> 

<mark>Lecture 4 - 26</mark> 

#### Neural networks: without the brain stuff 


![](references/papers/ml-foundations/lecture_4_images/lecture_4.pdf-0027-01.png)


<!-- Start of picture text -->
( Before ) Linear score function:<br>( Now ) 2-layer Neural Network<br>x W1 h W2 s<br>10<br>3072 100<br><!-- End of picture text -->

Learn 100 templates instead of 10.                               Share templates between classes 

<mark>Fei-Fei Li, Ranjay Krishna, Danfei Xu</mark> 

<mark>April 16, 2020</mark> 

<mark>Lecture 4 - 27</mark> 

#### Neural networks: without the brain stuff 

<mark>(</mark> **<mark>Before</mark>** <mark>) Linear score function: (</mark> **<mark>Now</mark>** <mark>) 2-layer Neural Network</mark> 


![](references/papers/ml-foundations/lecture_4_images/lecture_4.pdf-0028-02.png)



![](references/papers/ml-foundations/lecture_4_images/lecture_4.pdf-0028-03.png)


The function                   is called the **activation function. Q:** What if we try to build a neural network without one? 


![](references/papers/ml-foundations/lecture_4_images/lecture_4.pdf-0028-05.png)


<mark>Fei-Fei Li, Ranjay Krishna, Danfei Xu</mark> 

<mark>April 16, 2020</mark> 

<mark>28</mark> 

<mark>Lecture 4 -</mark> 

#### Neural networks: without the brain stuff 

<mark>(</mark> **<mark>Before</mark>** <mark>) Linear score function: (</mark> **<mark>Now</mark>** <mark>) 2-layer Neural Network</mark> 


![](references/papers/ml-foundations/lecture_4_images/lecture_4.pdf-0029-02.png)



![](references/papers/ml-foundations/lecture_4_images/lecture_4.pdf-0029-03.png)


The function                   is called the **activation function. Q:** What if we try to build a neural network without one? **A** : We end up with a linear classifier again! 

<mark>Fei-Fei Li, Ranjay Krishna, Danfei Xu</mark> 

<mark>April 16, 2020</mark> 

<mark>29</mark> 

<mark>Lecture 4 -</mark> 

#### Activation functions 

**<mark>Sigmoid</mark>** 


![](references/papers/ml-foundations/lecture_4_images/lecture_4.pdf-0030-02.png)


**<mark>tanh ReLU</mark>** 


![](references/papers/ml-foundations/lecture_4_images/lecture_4.pdf-0030-04.png)



![](references/papers/ml-foundations/lecture_4_images/lecture_4.pdf-0030-05.png)



![](references/papers/ml-foundations/lecture_4_images/lecture_4.pdf-0030-06.png)



![](references/papers/ml-foundations/lecture_4_images/lecture_4.pdf-0030-07.png)



![](references/papers/ml-foundations/lecture_4_images/lecture_4.pdf-0030-08.png)


<!-- Start of picture text -->
Leaky ReLU<br><!-- End of picture text -->


![](references/papers/ml-foundations/lecture_4_images/lecture_4.pdf-0030-09.png)



![](references/papers/ml-foundations/lecture_4_images/lecture_4.pdf-0030-10.png)



![](references/papers/ml-foundations/lecture_4_images/lecture_4.pdf-0030-11.png)


<!-- Start of picture text -->
Maxout<br>ELU<br><!-- End of picture text -->

<mark>Fei-Fei Li, Ranjay Krishna, Danfei Xu</mark> 

<mark>Lecture 4 - 30 April 16, 2020</mark> 

#### Activation functions 

ReLU is a good default choice for most problems 


![](references/papers/ml-foundations/lecture_4_images/lecture_4.pdf-0031-02.png)


<!-- Start of picture text -->
Leaky ReLU<br>Sigmoid<br>tanh<br>Maxout<br>ELU<br>ReLU<br><!-- End of picture text -->

<mark>Fei-Fei Li, Ranjay Krishna, Danfei Xu</mark> 

<mark>Lecture 4 - 31 April 16, 2020</mark> 

#### Neural networks: Architectures 


![](references/papers/ml-foundations/lecture_4_images/lecture_4.pdf-0032-01.png)



![](references/papers/ml-foundations/lecture_4_images/lecture_4.pdf-0032-02.png)


“3-layer Neural Net”, or “2-layer Neural Net”, or “2-hidden-layer Neural Net” “1-hidden-layer Neural Net” **“Fully-connected” layers** 

<mark>Fei-Fei Li, Ranjay Krishna, Danfei Xu</mark> 

<mark>April 16, 2020</mark> 

<mark>Lecture 4 - 32</mark> 

##### Example feed-forward computation of a neural network 


![](references/papers/ml-foundations/lecture_4_images/lecture_4.pdf-0033-01.png)



![](references/papers/ml-foundations/lecture_4_images/lecture_4.pdf-0033-02.png)


<mark>Fei-Fei Li, Ranjay Krishna, Danfei Xu</mark> 

<mark>April 16, 2020</mark> 

<mark>Lecture 4 - 33</mark> 

###### <mark>Full implementation of training a 2-layer Neural Network needs ~20 lines:</mark> 


![](references/papers/ml-foundations/lecture_4_images/lecture_4.pdf-0034-01.png)


<mark>Fei-Fei Li, Ranjay Krishna, Danfei Xu</mark> 

<mark>Lecture 4 - 34 April 16, 2020</mark> 

###### <mark>Full implementation of training a 2-layer Neural Network needs ~20 lines:</mark> 


![](references/papers/ml-foundations/lecture_4_images/lecture_4.pdf-0035-01.png)


Define the network 

<mark>Fei-Fei Li, Ranjay Krishna, Danfei Xu</mark> 

<mark>Lecture 4 - 35 April 16, 2020</mark> 

###### <mark>Full implementation of training a 2-layer Neural Network needs ~20 lines:</mark> 


![](references/papers/ml-foundations/lecture_4_images/lecture_4.pdf-0036-01.png)


Define the network 

###### Forward pass 

<mark>Fei-Fei Li, Ranjay Krishna, Danfei Xu</mark> 

<mark>Lecture 4 - 36 April 16, 2020</mark> 

<mark>Full implementation of training a 2-layer Neural Network needs ~20 lines:</mark> 


![](references/papers/ml-foundations/lecture_4_images/lecture_4.pdf-0037-01.png)


Define the network 

Forward pass 

Calculate the analytical gradients 

<mark>Fei-Fei Li, Ranjay Krishna, Danfei Xu</mark> 

<mark>April 16, 2020</mark> 

<mark>Lecture 4 - 37</mark> 

###### <mark>Full implementation of training a 2-layer Neural Network needs ~20 lines:</mark> 


![](references/papers/ml-foundations/lecture_4_images/lecture_4.pdf-0038-01.png)


Define the network 

Forward pass 

Calculate the analytical gradients 

Gradient descent 

<mark>Fei-Fei Li, Ranjay Krishna, Danfei Xu</mark> 

<mark>Lecture 4 - 38 April 16, 2020</mark> 

### Setting the number of layers and their sizes 


![](references/papers/ml-foundations/lecture_4_images/lecture_4.pdf-0039-01.png)


more neurons = more capacity 

**Fei-Fei Li & Andrej Karpathy & Justin Johnson** 

**Lecture 4 -** 39 **13 Jan 2016** 

###### Do not use size of neural network as a regularizer. Use stronger regularization instead: 


![](references/papers/ml-foundations/lecture_4_images/lecture_4.pdf-0040-01.png)



![](references/papers/ml-foundations/lecture_4_images/lecture_4.pdf-0040-02.png)


(Web demo with ConvNetJS: <u><mark>http://cs.stanford.edu/people/karpathy/convnetjs/demo /classify2d.html)</mark></u> 

**Fei-Fei Li & Andrej Karpathy & Justin Johnson Lecture 4 -** 40 **13 Jan 2016** 


![](references/papers/ml-foundations/lecture_4_images/lecture_4.pdf-0041-00.png)


<u>This image</u> by Fotis Bobolas is licensed under <u><mark>CC-BY 2.0</mark></u> 

<mark>Fei-Fei Li, Ranjay Krishna, Danfei Xu</mark> 

<mark>Lecture 4 - 41 April 16, 2020</mark> 

###### Impulses carried toward cell body 


![](references/papers/ml-foundations/lecture_4_images/lecture_4.pdf-0042-01.png)


<!-- Start of picture text -->
dendrite<br>presynaptic<br>  terminal<br>axon<br>cell body<br>Impulses carried away<br>from cell body<br><!-- End of picture text -->

<u>This image</u> by Felipe Perucho is licensed under <u><mark>CC-BY 3.0</mark></u> 

<mark>Fei-Fei Li, Ranjay Krishna, Danfei Xu</mark> 

<mark>April 16, 2020</mark> 

<mark>42</mark> 

<mark>Lecture 4 -</mark> 

###### Impulses carried toward cell body 


![](references/papers/ml-foundations/lecture_4_images/lecture_4.pdf-0043-01.png)


<!-- Start of picture text -->
dendrite<br>presynaptic<br>  terminal<br>axon<br>cell body<br>Impulses carried away<br>from cell body<br>This image by Felipe Perucho<br>is licensed under  CC-BY 3.0<br><!-- End of picture text -->

<mark>Fei-Fei Li, Ranjay Krishna, Danfei Xu</mark> 

<mark>April 16, 2020</mark> 

<mark>Lecture 4 - 43</mark> 

###### Impulses carried toward cell body 


![](references/papers/ml-foundations/lecture_4_images/lecture_4.pdf-0044-01.png)


<!-- Start of picture text -->
dendrite<br>presynaptic<br>  terminal<br>axon<br>cell body<br>Impulses carried away<br>from cell body<br>This image by Felipe Perucho<br>is licensed under  CC-BY 3.0<br>sigmoid activation function<br><!-- End of picture text -->

<mark>Fei-Fei Li, Ranjay Krishna, Danfei Xu Lecture 4 - 44</mark> 

<mark>April 16, 2020</mark> 

###### Impulses carried toward cell body 


![](references/papers/ml-foundations/lecture_4_images/lecture_4.pdf-0045-01.png)


<!-- Start of picture text -->
dendrite<br>presynaptic<br>  terminal<br>axon<br>cell body<br>Impulses carried away<br>from cell body<br>This image by Felipe Perucho<br>is licensed under  CC-BY 3.0<br><!-- End of picture text -->

<mark>Fei-Fei Li, Ranjay Krishna, Danfei Xu</mark> 

<mark>April 16, 2020</mark> 

<mark>Lecture 4 -</mark> 

**<mark>45</mark>** 

Biological Neurons: Complex connectivity patterns 


![](references/papers/ml-foundations/lecture_4_images/lecture_4.pdf-0046-01.png)


Neurons in a neural network: Organized into regular layers for computational efficiency 


![](references/papers/ml-foundations/lecture_4_images/lecture_4.pdf-0046-03.png)


<u>This image</u> is CC0 Public Domain 

<mark>Fei-Fei Li, Ranjay Krishna, Danfei Xu</mark> 

<mark>April 16, 2020</mark> 

<mark>46</mark> 

<mark>Lecture 4 -</mark> 

Biological Neurons: Complex connectivity patterns 


![](references/papers/ml-foundations/lecture_4_images/lecture_4.pdf-0047-01.png)


<u>This image</u> is CC0 Public Domain 

But neural networks with random connections can work too! 


![](references/papers/ml-foundations/lecture_4_images/lecture_4.pdf-0047-04.png)


Xie et al, “Exploring Randomly Wired Neural Networks for Image Recognition”, arXiv 2019 

<mark>Fei-Fei Li, Ranjay Krishna, Danfei Xu</mark> 

<mark>April 16, 2020</mark> 

<mark>Lecture 4 -</mark> 

<mark>47</mark> 

##### Be very careful with your brain analogies! 

###### **Biological Neurons:** 

- Many different types 

- Dendrites can perform complex non-linear computations 

- ● Synapses are not a single weight but a complex non-linear dynamical system 

- [Dendritic Computation. London and Hausser] 

<mark>Fei-Fei Li, Ranjay Krishna, Danfei Xu</mark> 

<mark>April 16, 2020</mark> 

<mark>48</mark> 

<mark>Lecture 4 -</mark> 

Problem: How to compute gradients? 

Nonlinear score function SVM Loss on predictions Regularization Total loss: data loss + regularization If we can compute                     then we can learn W1 and W2 

<mark>Fei-Fei Li, Ranjay Krishna, Danfei Xu Lecture 4 -</mark> 

<mark>April 16, 2020</mark> 

<mark>49</mark> 

#### (Bad) Idea: Derive on paper 

**Problem** : Very tedious: Lots of matrix calculus, need lots of paper **Problem** : What if we want to change loss? E.g. use softmax instead of SVM? Need to re-derive from scratch =( **Problem** : Not feasible for very complex models! 

<mark>Fei-Fei Li, Ranjay Krishna, Danfei Xu Lecture 4 - 50 April 16, 2020</mark> 

#### Better Idea: Computational graphs + Backpropagation 


![](references/papers/ml-foundations/lecture_4_images/lecture_4.pdf-0051-01.png)


<!-- Start of picture text -->
x<br>s  (scores)<br>* hinge  +<br>loss L<br>W<br>R<br><!-- End of picture text -->

<mark>Fei-Fei Li, Ranjay Krishna, Danfei Xu</mark> 

<mark>April 16, 2020</mark> 

<mark>51</mark> 

<mark>Lecture 4 -</mark> 


![](references/papers/ml-foundations/lecture_4_images/lecture_4.pdf-0052-00.png)


<!-- Start of picture text -->
Convolutional network<br>(AlexNet)<br>input image<br>weights<br>loss<br>Figure copyright Alex Krizhevsky, Ilya Sutskever, and<br><!-- End of picture text -->

Figure copyright Alex Krizhevsky, Ilya Sutskever, and Geoffrey Hinton, 2012. Reproduced with permission. 

<mark>Fei-Fei Li, Ranjay Krishna, Danfei Xu</mark> 

<mark>April 16, 2020</mark> 

<mark>Lecture 4 - 52</mark> 


![](references/papers/ml-foundations/lecture_4_images/lecture_4.pdf-0053-00.png)


<!-- Start of picture text -->
Neural Turing Machine<br>input image<br>loss<br>Figure reproduced with permission from a  Twitter post by Andrej Karpathy.<br><!-- End of picture text -->

<mark>Fei-Fei Li, Ranjay Krishna, Danfei Xu</mark> 

<mark>Lecture 4 - 53 April 16, 2020</mark> 

#### Neural Turing Machine 


![](references/papers/ml-foundations/lecture_4_images/lecture_4.pdf-0054-01.png)



![](references/papers/ml-foundations/lecture_4_images/lecture_4.pdf-0054-02.png)


<!-- Start of picture text -->
Figure reproduced with permission from a  Twitter post by Andrej Karpathy.<br><!-- End of picture text -->

<mark>Fei-Fei Li, Ranjay Krishna, Danfei Xu</mark> 

<mark>April 16, 2020</mark> 

<mark>Lecture 4 -</mark> 

## Solution: Backpropagation 

<mark>Fei-Fei Li, Ranjay Krishna, Danfei Xu</mark> 

<mark>April 16, 2020</mark> 

<mark>Lecture 4 -</mark> 

<mark>55</mark> 

###### Backpropagation: a simple example 


![](references/papers/ml-foundations/lecture_4_images/lecture_4.pdf-0056-01.png)


<mark>Fei-Fei Li, Ranjay Krishna, Danfei Xu</mark> 

<mark>April 16, 2020</mark> 

<mark>56</mark> 

<mark>Lecture 4 -</mark> 

###### Backpropagation: a simple example 


![](references/papers/ml-foundations/lecture_4_images/lecture_4.pdf-0057-01.png)



![](references/papers/ml-foundations/lecture_4_images/lecture_4.pdf-0057-02.png)


<mark>Fei-Fei Li, Ranjay Krishna, Danfei Xu</mark> 

<mark>April 16, 2020</mark> 

<mark>Lecture 4 -</mark> 

<mark>57</mark> 

###### Backpropagation: a simple example 


![](references/papers/ml-foundations/lecture_4_images/lecture_4.pdf-0058-01.png)


e.g. x = -2, y = 5, z = -4 


![](references/papers/ml-foundations/lecture_4_images/lecture_4.pdf-0058-03.png)


Fei-Fei Li & Justin Johnson & Serena Yeung 

Lecture 4 - 58 April 13, 2017 

Backpropagation: a simple example 


![](references/papers/ml-foundations/lecture_4_images/lecture_4.pdf-0059-01.png)


e.g. x = -2, y = 5, z = -4 


![](references/papers/ml-foundations/lecture_4_images/lecture_4.pdf-0059-03.png)



![](references/papers/ml-foundations/lecture_4_images/lecture_4.pdf-0059-04.png)


Fei-Fei Li & Justin Johnson & Serena Yeung Lecture 4 - 59 

April 13, 2017 

Backpropagation: a simple example 


![](references/papers/ml-foundations/lecture_4_images/lecture_4.pdf-0060-01.png)


e.g. x = -2, y = 5, z = -4 


![](references/papers/ml-foundations/lecture_4_images/lecture_4.pdf-0060-03.png)



![](references/papers/ml-foundations/lecture_4_images/lecture_4.pdf-0060-04.png)


Fei-Fei Li & Justin Johnson & Serena Yeung 

Lecture 4 - 60 

April 13, 2017 

Backpropagation: a simple example 


![](references/papers/ml-foundations/lecture_4_images/lecture_4.pdf-0061-01.png)


e.g. x = -2, y = 5, z = -4 


![](references/papers/ml-foundations/lecture_4_images/lecture_4.pdf-0061-03.png)



![](references/papers/ml-foundations/lecture_4_images/lecture_4.pdf-0061-04.png)


Want: 


![](references/papers/ml-foundations/lecture_4_images/lecture_4.pdf-0061-06.png)


Fei-Fei Li & Justin Johnson & Serena Yeung 

Lecture 4 - 61 

April 13, 2017 

Backpropagation: a simple example 


![](references/papers/ml-foundations/lecture_4_images/lecture_4.pdf-0062-01.png)


e.g. x = -2, y = 5, z = -4 


![](references/papers/ml-foundations/lecture_4_images/lecture_4.pdf-0062-03.png)



![](references/papers/ml-foundations/lecture_4_images/lecture_4.pdf-0062-04.png)


Want: 


![](references/papers/ml-foundations/lecture_4_images/lecture_4.pdf-0062-06.png)



![](references/papers/ml-foundations/lecture_4_images/lecture_4.pdf-0062-07.png)


Fei-Fei Li & Justin Johnson & Serena Yeung 

Lecture 4 - 62 

April 13, 2017 

Backpropagation: a simple example 


![](references/papers/ml-foundations/lecture_4_images/lecture_4.pdf-0063-01.png)


e.g. x = -2, y = 5, z = -4 


![](references/papers/ml-foundations/lecture_4_images/lecture_4.pdf-0063-03.png)



![](references/papers/ml-foundations/lecture_4_images/lecture_4.pdf-0063-04.png)


Want: 


![](references/papers/ml-foundations/lecture_4_images/lecture_4.pdf-0063-06.png)



![](references/papers/ml-foundations/lecture_4_images/lecture_4.pdf-0063-07.png)


Fei-Fei Li & Justin Johnson & Serena Yeung 

Lecture 4 - 63 

April 13, 2017 

Backpropagation: a simple example 


![](references/papers/ml-foundations/lecture_4_images/lecture_4.pdf-0064-01.png)


e.g. x = -2, y = 5, z = -4 


![](references/papers/ml-foundations/lecture_4_images/lecture_4.pdf-0064-03.png)



![](references/papers/ml-foundations/lecture_4_images/lecture_4.pdf-0064-04.png)


Want: 


![](references/papers/ml-foundations/lecture_4_images/lecture_4.pdf-0064-06.png)



![](references/papers/ml-foundations/lecture_4_images/lecture_4.pdf-0064-07.png)


Fei-Fei Li & Justin Johnson & Serena Yeung 

Lecture 4 - 64 

April 13, 2017 

Backpropagation: a simple example 


![](references/papers/ml-foundations/lecture_4_images/lecture_4.pdf-0065-01.png)


e.g. x = -2, y = 5, z = -4 


![](references/papers/ml-foundations/lecture_4_images/lecture_4.pdf-0065-03.png)



![](references/papers/ml-foundations/lecture_4_images/lecture_4.pdf-0065-04.png)


Want: 


![](references/papers/ml-foundations/lecture_4_images/lecture_4.pdf-0065-06.png)



![](references/papers/ml-foundations/lecture_4_images/lecture_4.pdf-0065-07.png)


Fei-Fei Li & Justin Johnson & Serena Yeung 

Lecture 4 - 65 

April 13, 2017 

Backpropagation: a simple example 


![](references/papers/ml-foundations/lecture_4_images/lecture_4.pdf-0066-01.png)


e.g. x = -2, y = 5, z = -4 


![](references/papers/ml-foundations/lecture_4_images/lecture_4.pdf-0066-03.png)



![](references/papers/ml-foundations/lecture_4_images/lecture_4.pdf-0066-04.png)


Want: 


![](references/papers/ml-foundations/lecture_4_images/lecture_4.pdf-0066-06.png)



![](references/papers/ml-foundations/lecture_4_images/lecture_4.pdf-0066-07.png)


Fei-Fei Li & Justin Johnson & Serena Yeung 

Lecture 4 - 66 

April 13, 2017 

Backpropagation: a simple example 


![](references/papers/ml-foundations/lecture_4_images/lecture_4.pdf-0067-01.png)


e.g. x = -2, y = 5, z = -4 


![](references/papers/ml-foundations/lecture_4_images/lecture_4.pdf-0067-03.png)



![](references/papers/ml-foundations/lecture_4_images/lecture_4.pdf-0067-04.png)


Want: 


![](references/papers/ml-foundations/lecture_4_images/lecture_4.pdf-0067-06.png)



![](references/papers/ml-foundations/lecture_4_images/lecture_4.pdf-0067-07.png)


Fei-Fei Li & Justin Johnson & Serena Yeung 

Lecture 4 - 67 

April 13, 2017 

Backpropagation: a simple example 


![](references/papers/ml-foundations/lecture_4_images/lecture_4.pdf-0068-01.png)


e.g. x = -2, y = 5, z = -4 


![](references/papers/ml-foundations/lecture_4_images/lecture_4.pdf-0068-03.png)



![](references/papers/ml-foundations/lecture_4_images/lecture_4.pdf-0068-04.png)


Want: 


![](references/papers/ml-foundations/lecture_4_images/lecture_4.pdf-0068-06.png)



![](references/papers/ml-foundations/lecture_4_images/lecture_4.pdf-0068-07.png)


<mark>Chain rule:</mark> Upstream Local gradient gradient 

Fei-Fei Li & Justin Johnson & Serena Yeung 

Lecture 4 - 68 April 13, 2017 

Backpropagation: a simple example 


![](references/papers/ml-foundations/lecture_4_images/lecture_4.pdf-0069-01.png)


e.g. x = -2, y = 5, z = -4 


![](references/papers/ml-foundations/lecture_4_images/lecture_4.pdf-0069-03.png)



![](references/papers/ml-foundations/lecture_4_images/lecture_4.pdf-0069-04.png)


Want: 


![](references/papers/ml-foundations/lecture_4_images/lecture_4.pdf-0069-06.png)



![](references/papers/ml-foundations/lecture_4_images/lecture_4.pdf-0069-07.png)


<mark>Chain rule:</mark> Upstream Local gradient gradient 

Fei-Fei Li & Justin Johnson & Serena Yeung 

Lecture 4 - 69 April 13, 2017 

Backpropagation: a simple example 


![](references/papers/ml-foundations/lecture_4_images/lecture_4.pdf-0070-01.png)


e.g. x = -2, y = 5, z = -4 


![](references/papers/ml-foundations/lecture_4_images/lecture_4.pdf-0070-03.png)



![](references/papers/ml-foundations/lecture_4_images/lecture_4.pdf-0070-04.png)


Want: 


![](references/papers/ml-foundations/lecture_4_images/lecture_4.pdf-0070-06.png)



![](references/papers/ml-foundations/lecture_4_images/lecture_4.pdf-0070-07.png)


<mark>Chain rule:</mark> Upstream Local gradient gradient 

Lecture 4 - 70 April 13, 2017 

Fei-Fei Li & Justin Johnson & Serena Yeung 

Backpropagation: a simple example 


![](references/papers/ml-foundations/lecture_4_images/lecture_4.pdf-0071-01.png)


e.g. x = -2, y = 5, z = -4 


![](references/papers/ml-foundations/lecture_4_images/lecture_4.pdf-0071-03.png)



![](references/papers/ml-foundations/lecture_4_images/lecture_4.pdf-0071-04.png)


Want: 


![](references/papers/ml-foundations/lecture_4_images/lecture_4.pdf-0071-06.png)



![](references/papers/ml-foundations/lecture_4_images/lecture_4.pdf-0071-07.png)


<mark>Chain rule:</mark> Upstream Local gradient gradient 

Lecture 4 - 71 April 13, 2017 

Fei-Fei Li & Justin Johnson & Serena Yeung 


![](references/papers/ml-foundations/lecture_4_images/lecture_4.pdf-0072-00.png)


<!-- Start of picture text -->
f<br><!-- End of picture text -->


![](references/papers/ml-foundations/lecture_4_images/lecture_4.pdf-0072-01.png)


<mark>Fei-Fei Li, Ranjay Krishna, Danfei Xu</mark> 

<mark>April 16, 2020</mark> 

<mark>Lecture 4 - 72</mark> 


![](references/papers/ml-foundations/lecture_4_images/lecture_4.pdf-0073-00.png)


<!-- Start of picture text -->
“local gradient”<br>f<br><!-- End of picture text -->


![](references/papers/ml-foundations/lecture_4_images/lecture_4.pdf-0073-01.png)


<mark>Fei-Fei Li, Ranjay Krishna, Danfei Xu</mark> 

<mark>April 16, 2020</mark> 

<mark>Lecture 4 - 73</mark> 


![](references/papers/ml-foundations/lecture_4_images/lecture_4.pdf-0074-00.png)


<!-- Start of picture text -->
“local gradient”<br>f<br>“Upstream<br>gradient”<br><!-- End of picture text -->

<mark>Fei-Fei Li, Ranjay Krishna, Danfei Xu</mark> 

<mark>April 16, 2020</mark> 

<mark>Lecture 4 - 74</mark> 


![](references/papers/ml-foundations/lecture_4_images/lecture_4.pdf-0075-00.png)


<!-- Start of picture text -->
“local gradient”<br>“Downstream<br>f<br>gradients”<br>“Upstream<br>gradient”<br><!-- End of picture text -->

<mark>Fei-Fei Li, Ranjay Krishna, Danfei Xu</mark> 

<mark>Lecture 4 - 75 April 16, 2020</mark> 


![](references/papers/ml-foundations/lecture_4_images/lecture_4.pdf-0076-00.png)


<!-- Start of picture text -->
“local gradient”<br>“Downstream<br>f<br>gradients”<br>“Upstream<br>gradient”<br><!-- End of picture text -->

<mark>Fei-Fei Li, Ranjay Krishna, Danfei Xu</mark> 

<mark>Lecture 4 - 76 April 16, 2020</mark> 


![](references/papers/ml-foundations/lecture_4_images/lecture_4.pdf-0077-00.png)


<!-- Start of picture text -->
“local gradient”<br>“Downstream<br>f<br>gradients”<br>“Upstream<br>gradient”<br><!-- End of picture text -->

<mark>Fei-Fei Li, Ranjay Krishna, Danfei Xu</mark> 

<mark>April 16, 2020</mark> 

<mark>Lecture 4 - 77</mark> 

###### <mark>Another example:</mark> 


![](references/papers/ml-foundations/lecture_4_images/lecture_4.pdf-0078-01.png)



![](references/papers/ml-foundations/lecture_4_images/lecture_4.pdf-0078-02.png)


<mark>Fei-Fei Li, Ranjay Krishna, Danfei Xu</mark> 

<mark>April 16, 2020</mark> 

<mark>78</mark> 

<mark>Lecture 4 -</mark> 

###### <mark>Another example:</mark> 


![](references/papers/ml-foundations/lecture_4_images/lecture_4.pdf-0079-01.png)



![](references/papers/ml-foundations/lecture_4_images/lecture_4.pdf-0079-02.png)


<mark>Fei-Fei Li, Ranjay Krishna, Danfei Xu</mark> 

<mark>April 16, 2020</mark> 

<mark>Lecture 4 - 79</mark> 

###### <mark>Another example:</mark> 


![](references/papers/ml-foundations/lecture_4_images/lecture_4.pdf-0080-01.png)



![](references/papers/ml-foundations/lecture_4_images/lecture_4.pdf-0080-02.png)


<mark>Fei-Fei Li, Ranjay Krishna, Danfei Xu</mark> 

<mark>Lecture 4 - 80 April 16, 2020</mark> 

###### <mark>Another example:</mark> 


![](references/papers/ml-foundations/lecture_4_images/lecture_4.pdf-0081-01.png)



![](references/papers/ml-foundations/lecture_4_images/lecture_4.pdf-0081-02.png)



![](references/papers/ml-foundations/lecture_4_images/lecture_4.pdf-0081-03.png)



![](references/papers/ml-foundations/lecture_4_images/lecture_4.pdf-0081-04.png)


<mark>Fei-Fei Li, Ranjay Krishna, Danfei Xu</mark> 

<mark>Lecture 4 - 81 April 16, 2020</mark> 

###### <mark>Another example:</mark> 


![](references/papers/ml-foundations/lecture_4_images/lecture_4.pdf-0082-01.png)



![](references/papers/ml-foundations/lecture_4_images/lecture_4.pdf-0082-02.png)



![](references/papers/ml-foundations/lecture_4_images/lecture_4.pdf-0082-03.png)



![](references/papers/ml-foundations/lecture_4_images/lecture_4.pdf-0082-04.png)


<mark>Fei-Fei Li, Ranjay Krishna, Danfei Xu</mark> 

<mark>Lecture 4 - 82 April 16, 2020</mark> 

###### <mark>Another example:</mark> 


![](references/papers/ml-foundations/lecture_4_images/lecture_4.pdf-0083-01.png)


Upstream Local gradient gradient 


![](references/papers/ml-foundations/lecture_4_images/lecture_4.pdf-0083-03.png)



![](references/papers/ml-foundations/lecture_4_images/lecture_4.pdf-0083-04.png)



![](references/papers/ml-foundations/lecture_4_images/lecture_4.pdf-0083-05.png)


<mark>Fei-Fei Li, Ranjay Krishna, Danfei Xu</mark> 

<mark>Lecture 4 - 83 April 16, 2020</mark> 

###### <mark>Another example:</mark> 


![](references/papers/ml-foundations/lecture_4_images/lecture_4.pdf-0084-01.png)



![](references/papers/ml-foundations/lecture_4_images/lecture_4.pdf-0084-02.png)



![](references/papers/ml-foundations/lecture_4_images/lecture_4.pdf-0084-03.png)



![](references/papers/ml-foundations/lecture_4_images/lecture_4.pdf-0084-04.png)


<mark>Fei-Fei Li, Ranjay Krishna, Danfei Xu</mark> 

<mark>Lecture 4 - 84 April 16, 2020</mark> 


![](references/papers/ml-foundations/lecture_4_images/lecture_4.pdf-0085-00.png)


<!-- Start of picture text -->
Another example:<br>Upstream  Local<br>gradient gradient<br><!-- End of picture text -->

###### <mark>Another example:</mark> 

<mark>Fei-Fei Li, Ranjay Krishna, Danfei Xu</mark> 

<mark>Lecture 4 - 85 April 16, 2020</mark> 

###### <mark>Another example:</mark> 


![](references/papers/ml-foundations/lecture_4_images/lecture_4.pdf-0086-01.png)



![](references/papers/ml-foundations/lecture_4_images/lecture_4.pdf-0086-02.png)



![](references/papers/ml-foundations/lecture_4_images/lecture_4.pdf-0086-03.png)



![](references/papers/ml-foundations/lecture_4_images/lecture_4.pdf-0086-04.png)


<mark>Fei-Fei Li, Ranjay Krishna, Danfei Xu</mark> 

<mark>Lecture 4 - 86 April 16, 2020</mark> 


![](references/papers/ml-foundations/lecture_4_images/lecture_4.pdf-0087-00.png)


<!-- Start of picture text -->
Another example:<br>Upstream  Local<br>gradient gradient<br><!-- End of picture text -->

###### <mark>Another example:</mark> 

<mark>Fei-Fei Li, Ranjay Krishna, Danfei Xu</mark> 

<mark>Lecture 4 - 87 April 16, 2020</mark> 

###### <mark>Another example:</mark> 


![](references/papers/ml-foundations/lecture_4_images/lecture_4.pdf-0088-01.png)



![](references/papers/ml-foundations/lecture_4_images/lecture_4.pdf-0088-02.png)



![](references/papers/ml-foundations/lecture_4_images/lecture_4.pdf-0088-03.png)



![](references/papers/ml-foundations/lecture_4_images/lecture_4.pdf-0088-04.png)


<mark>Fei-Fei Li, Ranjay Krishna, Danfei Xu</mark> 

<mark>Lecture 4 - 88 April 16, 2020</mark> 

###### <mark>Another example:</mark> 


![](references/papers/ml-foundations/lecture_4_images/lecture_4.pdf-0089-01.png)



![](references/papers/ml-foundations/lecture_4_images/lecture_4.pdf-0089-02.png)


<!-- Start of picture text -->
Upstream  Local<br>gradient gradient<br><!-- End of picture text -->


![](references/papers/ml-foundations/lecture_4_images/lecture_4.pdf-0089-03.png)



![](references/papers/ml-foundations/lecture_4_images/lecture_4.pdf-0089-04.png)


<mark>Fei-Fei Li, Ranjay Krishna, Danfei Xu</mark> 

<mark>Lecture 4 - 89 April 16, 2020</mark> 

###### <mark>Another example:</mark> 


![](references/papers/ml-foundations/lecture_4_images/lecture_4.pdf-0090-01.png)



![](references/papers/ml-foundations/lecture_4_images/lecture_4.pdf-0090-02.png)



![](references/papers/ml-foundations/lecture_4_images/lecture_4.pdf-0090-03.png)



![](references/papers/ml-foundations/lecture_4_images/lecture_4.pdf-0090-04.png)


<mark>Fei-Fei Li, Ranjay Krishna, Danfei Xu</mark> 

<mark>Lecture 4 - 90 April 16, 2020</mark> 

###### <mark>Another example:</mark> 


![](references/papers/ml-foundations/lecture_4_images/lecture_4.pdf-0091-01.png)


[upstream gradient] x [local gradient] [0.2] x [1] = 0.2 [0.2] x [1] = 0.2  (both inputs!) 


![](references/papers/ml-foundations/lecture_4_images/lecture_4.pdf-0091-03.png)



![](references/papers/ml-foundations/lecture_4_images/lecture_4.pdf-0091-04.png)


<mark>Fei-Fei Li, Ranjay Krishna, Danfei Xu</mark> 

<mark>Lecture 4 - 91 April 16, 2020</mark> 

###### <mark>Another example:</mark> 


![](references/papers/ml-foundations/lecture_4_images/lecture_4.pdf-0092-01.png)



![](references/papers/ml-foundations/lecture_4_images/lecture_4.pdf-0092-02.png)



![](references/papers/ml-foundations/lecture_4_images/lecture_4.pdf-0092-03.png)



![](references/papers/ml-foundations/lecture_4_images/lecture_4.pdf-0092-04.png)


<mark>Fei-Fei Li, Ranjay Krishna, Danfei Xu</mark> 

<mark>Lecture 4 - 92 April 16, 2020</mark> 

###### <mark>Another example:</mark> 


![](references/papers/ml-foundations/lecture_4_images/lecture_4.pdf-0093-01.png)


[upstream gradient] x [local gradient] w0: [0.2] x [-1] = -0.2 x0: [0.2] x [2] = 0.4 


![](references/papers/ml-foundations/lecture_4_images/lecture_4.pdf-0093-03.png)



![](references/papers/ml-foundations/lecture_4_images/lecture_4.pdf-0093-04.png)


<mark>Fei-Fei Li, Ranjay Krishna, Danfei Xu</mark> 

<mark>Lecture 4 - 93 April 16, 2020</mark> 


![](references/papers/ml-foundations/lecture_4_images/lecture_4.pdf-0094-00.png)


<!-- Start of picture text -->
Another example: Computational graph<br>representation may not<br>be unique. Choose one<br>Sigmoid  where local gradients at<br>function each node can be easily<br>expressed!<br>Sigmoid<br><!-- End of picture text -->

###### <mark>Another example:</mark> 

<mark>Fei-Fei Li, Ranjay Krishna, Danfei Xu</mark> 

<mark>Lecture 4 - 94 April 16, 2020</mark> 


![](references/papers/ml-foundations/lecture_4_images/lecture_4.pdf-0095-00.png)


<!-- Start of picture text -->
Another example: Computational graph<br>representation may not<br>be unique. Choose one<br>Sigmoid  where local gradients at<br>function each node can be easily<br>expressed!<br>Sigmoid<br>Sigmoid local<br>gradient:<br><!-- End of picture text -->

<mark>Fei-Fei Li, Ranjay Krishna, Danfei Xu</mark> 

<mark>Lecture 4 - 95 April 16, 2020</mark> 


![](references/papers/ml-foundations/lecture_4_images/lecture_4.pdf-0096-00.png)


<!-- Start of picture text -->
Another example: Computational graph<br>representation may not<br>be unique. Choose one<br>Sigmoid  where local gradients at<br>function each node can be easily<br>expressed!<br>Sigmoid<br>[upstream gradient] x [local gradient]<br>[1.00] x [(1 - 1/(1+e 1 )) (1/(1+e 1 ))] = 0.2<br>Sigmoid local<br>gradient:<br><!-- End of picture text -->

<mark>Fei-Fei Li, Ranjay Krishna, Danfei Xu</mark> 

<mark>Lecture 4 - 96 April 16, 2020</mark> 


![](references/papers/ml-foundations/lecture_4_images/lecture_4.pdf-0097-00.png)


<!-- Start of picture text -->
Another example: Computational graph<br>representation may not<br>be unique. Choose one<br>Sigmoid  where local gradients at<br>function each node can be easily<br>expressed!<br>Sigmoid<br>[upstream gradient] x [local gradient]<br>[1.00] x [(1 - 0.73) (0.73)] = 0.2<br>Sigmoid local<br>gradient:<br><!-- End of picture text -->

<mark>Fei-Fei Li, Ranjay Krishna, Danfei Xu</mark> 

<mark>April 16, 2020</mark> 

<mark>Lecture 4 - 97</mark> 

#### Patterns in gradient flow 

###### **add** gate: gradient distributor 


![](references/papers/ml-foundations/lecture_4_images/lecture_4.pdf-0098-02.png)


<!-- Start of picture text -->
3<br>2<br>7<br>+<br>2<br>4<br>2<br><!-- End of picture text -->

<mark>Fei-Fei Li, Ranjay Krishna, Danfei Xu</mark> 

<mark>April 16, 2020</mark> 

<mark>Lecture 4 - 98</mark> 

#### Patterns in gradient flow 

###### **add** gate: gradient distributor 


![](references/papers/ml-foundations/lecture_4_images/lecture_4.pdf-0099-02.png)


<!-- Start of picture text -->
3<br>2<br>7<br>+<br>2<br>4<br>2<br><!-- End of picture text -->

###### **mul** gate: “swap multiplier” 


![](references/papers/ml-foundations/lecture_4_images/lecture_4.pdf-0099-04.png)


<!-- Start of picture text -->
2<br>5*3=15<br>6<br>×<br>3 5<br>2*5=10<br><!-- End of picture text -->

<mark>Fei-Fei Li, Ranjay Krishna, Danfei Xu</mark> 

<mark>April 16, 2020</mark> 

<mark>99</mark> 

<mark>Lecture 4 -</mark> 

#### Patterns in gradient flow 

###### **add** gate: gradient distributor 

###### **mul** gate: “swap multiplier” 


![](references/papers/ml-foundations/lecture_4_images/lecture_4.pdf-0100-03.png)


<!-- Start of picture text -->
3 2<br>2 5*3=15<br>7 6<br>+ ×<br>4 2 3 5<br>2 2*5=10<br><!-- End of picture text -->

###### **copy** gate: gradient adder 


![](references/papers/ml-foundations/lecture_4_images/lecture_4.pdf-0100-05.png)


<!-- Start of picture text -->
7<br>4<br>7<br>4+2=6 7<br>2<br><!-- End of picture text -->

<mark>Fei-Fei Li, Ranjay Krishna, Danfei Xu</mark> 

<mark>April 16, 2020</mark> 

<mark>100</mark> 

<mark>Lecture 4 -</mark> 

#### Patterns in gradient flow 

###### **add** gate: gradient distributor 


![](references/papers/ml-foundations/lecture_4_images/lecture_4.pdf-0101-02.png)


<!-- Start of picture text -->
3<br>2<br>7<br>+<br>2<br>4<br>2<br><!-- End of picture text -->


![](references/papers/ml-foundations/lecture_4_images/lecture_4.pdf-0101-03.png)


<!-- Start of picture text -->
copy  gate: gradient adder<br>7<br>4<br>7<br>4+2=6 7<br>2<br><!-- End of picture text -->

###### **mul** gate: “swap multiplier” 


![](references/papers/ml-foundations/lecture_4_images/lecture_4.pdf-0101-05.png)


<!-- Start of picture text -->
2<br>5*3=15<br>6<br>×<br>3 5<br>2*5=10<br>max  gate: gradient router<br>4<br>0<br>5<br>max<br>9<br>5<br>9<br><!-- End of picture text -->

###### **max** gate: gradient router 

<mark>Fei-Fei Li, Ranjay Krishna, Danfei Xu</mark> 

<mark>April 16, 2020</mark> 

<mark>101</mark> 

<mark>Lecture 4 -</mark> 

Backprop Implementation: “Flat” code 


![](references/papers/ml-foundations/lecture_4_images/lecture_4.pdf-0102-01.png)



![](references/papers/ml-foundations/lecture_4_images/lecture_4.pdf-0102-02.png)



![](references/papers/ml-foundations/lecture_4_images/lecture_4.pdf-0102-03.png)


Forward pass: Compute output 

Backward pass: Compute grads 


![](references/papers/ml-foundations/lecture_4_images/lecture_4.pdf-0102-06.png)


<mark>Fei-Fei Li, Ranjay Krishna, Danfei Xu</mark> 

<mark>Lecture 4 - 102 April 16, 2020</mark> 

###### Backprop Implementation: “Flat” code 


![](references/papers/ml-foundations/lecture_4_images/lecture_4.pdf-0103-01.png)



![](references/papers/ml-foundations/lecture_4_images/lecture_4.pdf-0103-02.png)



![](references/papers/ml-foundations/lecture_4_images/lecture_4.pdf-0103-03.png)


Forward pass: Compute output 

Base case 


![](references/papers/ml-foundations/lecture_4_images/lecture_4.pdf-0103-06.png)


<mark>Fei-Fei Li, Ranjay Krishna, Danfei Xu</mark> 

<mark>April 16, 2020</mark> 

<mark>Lecture 4 - 103</mark> 

###### Backprop Implementation: “Flat” code 


![](references/papers/ml-foundations/lecture_4_images/lecture_4.pdf-0104-01.png)



![](references/papers/ml-foundations/lecture_4_images/lecture_4.pdf-0104-02.png)


Forward pass: Compute output Sigmoid 


![](references/papers/ml-foundations/lecture_4_images/lecture_4.pdf-0104-04.png)


<mark>Fei-Fei Li, Ranjay Krishna, Danfei Xu</mark> 

<mark>April 16, 2020</mark> 

<mark>Lecture 4 - 104</mark> 

###### Backprop Implementation: “Flat” code 


![](references/papers/ml-foundations/lecture_4_images/lecture_4.pdf-0105-01.png)



![](references/papers/ml-foundations/lecture_4_images/lecture_4.pdf-0105-02.png)



![](references/papers/ml-foundations/lecture_4_images/lecture_4.pdf-0105-03.png)


Forward pass: Compute output 

Add gate 


![](references/papers/ml-foundations/lecture_4_images/lecture_4.pdf-0105-06.png)


<mark>Fei-Fei Li, Ranjay Krishna, Danfei Xu</mark> 

<mark>April 16, 2020</mark> 

<mark>Lecture 4 - 105</mark> 

###### Backprop Implementation: “Flat” code 


![](references/papers/ml-foundations/lecture_4_images/lecture_4.pdf-0106-01.png)



![](references/papers/ml-foundations/lecture_4_images/lecture_4.pdf-0106-02.png)



![](references/papers/ml-foundations/lecture_4_images/lecture_4.pdf-0106-03.png)


Forward pass: Compute output 

Add gate 


![](references/papers/ml-foundations/lecture_4_images/lecture_4.pdf-0106-06.png)


<mark>Fei-Fei Li, Ranjay Krishna, Danfei Xu</mark> 

<mark>April 16, 2020</mark> 

<mark>Lecture 4 - 106</mark> 

###### Backprop Implementation: “Flat” code 


![](references/papers/ml-foundations/lecture_4_images/lecture_4.pdf-0107-01.png)



![](references/papers/ml-foundations/lecture_4_images/lecture_4.pdf-0107-02.png)



![](references/papers/ml-foundations/lecture_4_images/lecture_4.pdf-0107-03.png)


Forward pass: Compute output 

Multiply gate 


![](references/papers/ml-foundations/lecture_4_images/lecture_4.pdf-0107-06.png)


<mark>Fei-Fei Li, Ranjay Krishna, Danfei Xu</mark> 

<mark>April 16, 2020</mark> 

<mark>Lecture 4 - 107</mark> 

###### Backprop Implementation: “Flat” code 


![](references/papers/ml-foundations/lecture_4_images/lecture_4.pdf-0108-01.png)



![](references/papers/ml-foundations/lecture_4_images/lecture_4.pdf-0108-02.png)



![](references/papers/ml-foundations/lecture_4_images/lecture_4.pdf-0108-03.png)


Forward pass: Compute output 

Multiply gate 


![](references/papers/ml-foundations/lecture_4_images/lecture_4.pdf-0108-06.png)


<mark>Fei-Fei Li, Ranjay Krishna, Danfei Xu</mark> 

<mark>April 16, 2020</mark> 

<mark>Lecture 4 - 108</mark> 

#### “Flat” Backprop: Do this for assignment 1! 

Stage your forward/backward computation! 

E.g. for the SVM: 

margins 


![](references/papers/ml-foundations/lecture_4_images/lecture_4.pdf-0109-04.png)


<mark>Fei-Fei Li, Ranjay Krishna, Danfei Xu</mark> 

<mark>April 16, 2020</mark> 

<mark>Lecture 4 - 109</mark> 

#### “Flat” Backprop: Do this for assignment 1! 

E.g. for two-layer neural net: 


![](references/papers/ml-foundations/lecture_4_images/lecture_4.pdf-0110-02.png)


<mark>Fei-Fei Li, Ranjay Krishna, Danfei Xu</mark> 

<mark>April 16, 2020</mark> 

<mark>Lecture 4 - 110</mark> 

##### Backprop Implementation: Modularized API 

Graph (or Net) object _(rough pseudo code)_ 


![](references/papers/ml-foundations/lecture_4_images/lecture_4.pdf-0111-02.png)



![](references/papers/ml-foundations/lecture_4_images/lecture_4.pdf-0111-03.png)



![](references/papers/ml-foundations/lecture_4_images/lecture_4.pdf-0111-04.png)



![](references/papers/ml-foundations/lecture_4_images/lecture_4.pdf-0111-05.png)


<mark>Fei-Fei Li, Ranjay Krishna, Danfei Xu</mark> 

<mark>Lecture 4 - 111 April 16, 2020</mark> 

##### Modularized implementation: forward / backward API 

Gate / Node / Function object: Actual PyTorch code 


![](references/papers/ml-foundations/lecture_4_images/lecture_4.pdf-0112-02.png)


<!-- Start of picture text -->
x<br>z<br>*<br>y<br>(x,y,z are scalars)<br><!-- End of picture text -->


![](references/papers/ml-foundations/lecture_4_images/lecture_4.pdf-0112-03.png)


|Need to stash<br>some values for<br>use in backward|
|---|
|Upstream<br>gradient|
|Multiply upstream|
|and local gradients|



<mark>Fei-Fei Li, Ranjay Krishna, Danfei Xu</mark> 

<mark>April 16, 2020</mark> 

<mark>Lecture 4 - 112</mark> 

##### Example: PyTorch operators 


![](references/papers/ml-foundations/lecture_4_images/lecture_4.pdf-0113-01.png)



![](references/papers/ml-foundations/lecture_4_images/lecture_4.pdf-0113-02.png)


<mark>Fei-Fei Li, Ranjay Krishna, Danfei Xu</mark> 

<mark>Lecture 4 - 113 April 16, 2020</mark> 

##### PyTorch sigmoid layer 

Forward 


![](references/papers/ml-foundations/lecture_4_images/lecture_4.pdf-0114-02.png)


<u>Source</u> 

<mark>Fei-Fei Li, Ranjay Krishna, Danfei Xu</mark> 

<mark>Lecture 4 - 114 April 16, 2020</mark> 

##### PyTorch sigmoid layer 

Forward Forward actually defined <u><mark>elsewhere.</mark></u> .. 


![](references/papers/ml-foundations/lecture_4_images/lecture_4.pdf-0115-02.png)


<u>Source</u> 

<mark>Fei-Fei Li, Ranjay Krishna, Danfei Xu</mark> 

<mark>Lecture 4 - 115 April 16, 2020</mark> 


![](references/papers/ml-foundations/lecture_4_images/lecture_4.pdf-0116-00.png)


<!-- Start of picture text -->
PyTorch sigmoid layer<br>Forward<br>Forward actually<br>defined  elsewhere. ..<br>Backward<br>Source<br><!-- End of picture text -->

<mark>Fei-Fei Li, Ranjay Krishna, Danfei Xu</mark> 

<mark>April 16, 2020</mark> 

<mark>Lecture 4 - 116</mark> 

So far: backprop with scalars What about vector-valued functions? 

<mark>Fei-Fei Li, Ranjay Krishna, Danfei Xu</mark> 

<mark>April 16, 2020</mark> 

<mark>Lecture 4 - 117</mark> 

### Recap: Vector derivatives 

###### Scalar to Scalar 


![](references/papers/ml-foundations/lecture_4_images/lecture_4.pdf-0118-02.png)


###### Regular derivative: 


![](references/papers/ml-foundations/lecture_4_images/lecture_4.pdf-0118-04.png)


If x changes by a small amount, how much will y change? 

<mark>Fei-Fei Li, Ranjay Krishna, Danfei Xu</mark> 

<mark>April 16, 2020</mark> 

<mark>Lecture 4 - 118</mark> 

### Recap: Vector derivatives 

Scalar to Scalar Vector to Scalar 


![](references/papers/ml-foundations/lecture_4_images/lecture_4.pdf-0119-02.png)



![](references/papers/ml-foundations/lecture_4_images/lecture_4.pdf-0119-03.png)


Regular derivative: Derivative is **Gradient** : If x changes by a For each element of x, small amount, how if it changes by a small much will y change? amount then how much will y change? 

<mark>Fei-Fei Li, Ranjay Krishna, Danfei Xu</mark> 

<mark>April 16, 2020</mark> 

<mark>Lecture 4 - 119</mark> 

### Recap: Vector derivatives 

Scalar to Scalar Vector to Scalar 


![](references/papers/ml-foundations/lecture_4_images/lecture_4.pdf-0120-02.png)



![](references/papers/ml-foundations/lecture_4_images/lecture_4.pdf-0120-03.png)


Regular derivative: 

###### Derivative is **Gradient** : 


![](references/papers/ml-foundations/lecture_4_images/lecture_4.pdf-0120-06.png)



![](references/papers/ml-foundations/lecture_4_images/lecture_4.pdf-0120-07.png)



![](references/papers/ml-foundations/lecture_4_images/lecture_4.pdf-0120-08.png)


If x changes by a small amount, how much will y change? 

For each element of x, if it changes by a small amount then how much will y change? 

###### Vector to Vector 


![](references/papers/ml-foundations/lecture_4_images/lecture_4.pdf-0120-12.png)


###### Derivative is **Jacobian** : 


![](references/papers/ml-foundations/lecture_4_images/lecture_4.pdf-0120-14.png)



![](references/papers/ml-foundations/lecture_4_images/lecture_4.pdf-0120-15.png)


For each element of x, if it changes by a small amount then how much will each element of y change? 

<mark>Fei-Fei Li, Ranjay Krishna, Danfei Xu</mark> 

<mark>April 16, 2020</mark> 

<mark>120</mark> 

<mark>Lecture 4 -</mark> 

###### Backprop with Vectors 


![](references/papers/ml-foundations/lecture_4_images/lecture_4.pdf-0121-01.png)



![](references/papers/ml-foundations/lecture_4_images/lecture_4.pdf-0121-02.png)



![](references/papers/ml-foundations/lecture_4_images/lecture_4.pdf-0121-03.png)


<!-- Start of picture text -->
Loss L still a scalar!<br>f<br><!-- End of picture text -->

<mark>Fei-Fei Li, Ranjay Krishna, Danfei Xu</mark> 

<mark>Lecture 4 - 121 April 16, 2020</mark> 

###### Backprop with Vectors 


![](references/papers/ml-foundations/lecture_4_images/lecture_4.pdf-0122-01.png)


<!-- Start of picture text -->
Loss L still a scalar!<br>D x<br>Dz<br>f<br>Dy<br><!-- End of picture text -->

<mark>Fei-Fei Li, Ranjay Krishna, Danfei Xu</mark> 

<mark>Lecture 4 - 122 April 16, 2020</mark> 

###### Backprop with Vectors 


![](references/papers/ml-foundations/lecture_4_images/lecture_4.pdf-0123-01.png)


<!-- Start of picture text -->
Loss L still a scalar!<br>D x<br>Dz<br>f<br>Dy<br>“Upstream gradient”<br><!-- End of picture text -->

<mark>Fei-Fei Li, Ranjay Krishna, Danfei Xu</mark> 

<mark>Lecture 4 - 123 April 16, 2020</mark> 

###### Backprop with Vectors 


![](references/papers/ml-foundations/lecture_4_images/lecture_4.pdf-0124-01.png)


<!-- Start of picture text -->
Loss L still a scalar!<br>D x<br>Dz<br>f<br>D z<br>Dy<br>“Upstream gradient”<br>For each element of z, how<br>much does it influence L?<br><!-- End of picture text -->

<mark>Fei-Fei Li, Ranjay Krishna, Danfei Xu</mark> 

<mark>Lecture 4 - 124 April 16, 2020</mark> 

###### Backprop with Vectors 


![](references/papers/ml-foundations/lecture_4_images/lecture_4.pdf-0125-01.png)


<!-- Start of picture text -->
Loss L still a scalar!<br>“local<br>D x<br>gradients”<br>Dz<br>“Downstream<br>f<br>gradients”<br>D z<br>Dy<br>“Upstream gradient”<br>For each element of z, how<br>much does it influence L?<br><!-- End of picture text -->

<mark>Fei-Fei Li, Ranjay Krishna, Danfei Xu</mark> 

<mark>Lecture 4 - 125 April 16, 2020</mark> 

###### Backprop with Vectors 


![](references/papers/ml-foundations/lecture_4_images/lecture_4.pdf-0126-01.png)


<!-- Start of picture text -->
Loss L still a scalar!<br>“local<br>D x<br>gradients”<br>[Dx x Dz]<br>Dz<br>“Downstream<br>f<br>gradients”<br>[Dy x Dz]<br>D z<br>Jacobian<br>Dy<br>matrices<br>“Upstream gradient”<br>For each element of z, how<br>much does it influence L?<br>Fei-Fei Li, Ranjay Krishna, Danfei Xu Lecture 4 - 126 April 16, 2020<br><!-- End of picture text -->

###### Backprop with Vectors 


![](references/papers/ml-foundations/lecture_4_images/lecture_4.pdf-0127-01.png)


<!-- Start of picture text -->
Loss L still a scalar!<br>“local<br>D x<br>gradients”<br>D x [Dx x Dz]<br>Dz<br>“Downstream<br>Matrix-vector f<br>gradients”<br>multiply<br>[Dy x Dz]<br>D z<br>Jacobian<br>Dy<br>matrices<br>“Upstream gradient”<br>D y For each element of z, how much does it influence L?<br><!-- End of picture text -->

<mark>Fei-Fei Li, Ranjay Krishna, Danfei Xu</mark> 

<mark>127 April 16, 2020</mark> 

<mark>Lecture 4 -</mark> 

Gradients of variables wrt loss have same dims as the original variable 


![](references/papers/ml-foundations/lecture_4_images/lecture_4.pdf-0128-01.png)


<!-- Start of picture text -->
Loss L still a scalar!<br>D x<br>D x<br>Dz<br>f<br>D z<br>Dy<br>“Upstream gradient”<br>D y For each element of z, how much does it influence L?<br><!-- End of picture text -->

<mark>Fei-Fei Li, Ranjay Krishna, Danfei Xu</mark> 

<mark>Lecture 4 - 128 April 16, 2020</mark> 

#### Backprop with Vectors 


![](references/papers/ml-foundations/lecture_4_images/lecture_4.pdf-0129-01.png)


<!-- Start of picture text -->
4D input x: 4D output z:<br>[  1  ] [  1  ]<br>[ -2  ] [  0  ]<br>f(x) = max(0,x)<br>[  3  ] [  3  ]<br>(elementwise)<br>[  -1 ] [  0  ]<br><!-- End of picture text -->

<mark>Fei-Fei Li, Ranjay Krishna, Danfei Xu</mark> 

<mark>April 16, 2020</mark> 

<mark>Lecture 4 - 129</mark> 

#### Backprop with Vectors 


![](references/papers/ml-foundations/lecture_4_images/lecture_4.pdf-0130-01.png)


<!-- Start of picture text -->
4D input x: 4D output z:<br>[  1  ] [  1  ]<br>[ -2  ] [  0  ]<br>f(x) = max(0,x)<br>[  3  ] [  3  ]<br>(elementwise)<br>[  -1 ] [  0  ]<br>4D dL/dz:<br>[  4  ]<br>[  -1 ] Upstream<br>[  5  ] gradient<br>[  9  ]<br><!-- End of picture text -->

<mark>Fei-Fei Li, Ranjay Krishna, Danfei Xu</mark> 

<mark>Lecture 4 - 130 April 16, 2020</mark> 

#### Backprop with Vectors 


![](references/papers/ml-foundations/lecture_4_images/lecture_4.pdf-0131-01.png)


<!-- Start of picture text -->
4D input x: 4D output z:<br>[  1  ] [  1  ]<br>[ -2  ] [  0  ]<br>f(x) = max(0,x)<br>[  3  ] [  3  ]<br>(elementwise)<br>[  -1 ] [  0  ]<br>Jacobian dz/dx 4D dL/dz:<br>[ 1 0 0 0 ]  [  4  ]<br>[ 0 0 0 0 ]  [  -1 ] Upstream<br>[ 0 0 1 0 ]  [  5  ] gradient<br>[ 0 0 0 0 ]  [  9  ]<br><!-- End of picture text -->

<mark>Fei-Fei Li, Ranjay Krishna, Danfei Xu</mark> 

<mark>April 16, 2020</mark> 

<mark>131</mark> 

<mark>Lecture 4 -</mark> 

#### Backprop with Vectors 


![](references/papers/ml-foundations/lecture_4_images/lecture_4.pdf-0132-01.png)


<!-- Start of picture text -->
4D input x: 4D output z:<br>[  1  ] [  1  ]<br>[ -2  ] [  0  ]<br>f(x) = max(0,x)<br>[  3  ] [  3  ]<br>(elementwise)<br>[  -1 ] [  0  ]<br>[dz/dx] [dL/dz] 4D dL/dz:<br>[ 1 0 0 0 ] [ 4  ] [  4  ]<br>[ 0 0 0 0 ] [ -1 ] [  -1 ] Upstream<br>[ 0 0 1 0 ] [ 5  ] [  5  ] gradient<br>[ 0 0 0 0 ] [ 9  ] [  9  ]<br><!-- End of picture text -->

<mark>Fei-Fei Li, Ranjay Krishna, Danfei Xu</mark> 

<mark>April 16, 2020</mark> 

<mark>132</mark> 

<mark>Lecture 4 -</mark> 

#### Backprop with Vectors 


![](references/papers/ml-foundations/lecture_4_images/lecture_4.pdf-0133-01.png)


<!-- Start of picture text -->
4D input x: 4D output z:<br>[  1  ] [  1  ]<br>[ -2  ] [  0  ]<br>f(x) = max(0,x)<br>[  3  ] [  3  ]<br>(elementwise)<br>[  -1 ] [  0  ]<br>4D dL/dx:  [dz/dx] [dL/dz] 4D dL/dz:<br>[ 4 ] [ 1 0 0 0 ] [ 4  ] [  4  ]<br>[ 0 ] [ 0 0 0 0 ] [ -1 ] [  -1 ] Upstream<br>[ 5 ] [ 0 0 1 0 ] [ 5  ] [  5  ] gradient<br>[ 0 ] [ 0 0 0 0 ] [ 9  ] [  9  ]<br><!-- End of picture text -->

<mark>Fei-Fei Li, Ranjay Krishna, Danfei Xu</mark> 

<mark>Lecture 4 - 133 April 16, 2020</mark> 

#### Backprop with Vectors 


![](references/papers/ml-foundations/lecture_4_images/lecture_4.pdf-0134-01.png)


<!-- Start of picture text -->
4D input x: 4D output z:<br>[  1  ] [  1  ]<br>[ -2  ] [  0  ]<br>f(x) = max(0,x)<br>Jacobian is  sparse :<br>[  3  ] [  3  ]<br>off-diagonal entries<br>(elementwise)<br>always zero! Never  [  -1 ] [  0  ]<br>explicitly  form<br>Jacobian -- instead<br>use  implicit 4D dL/dx:  [dz/dx] [dL/dz] 4D dL/dz:<br>multiplication [ 4 ] [ 1 0 0 0 ] [ 4  ] [  4  ]<br>[ 0 ] [ 0 0 0 0 ] [ -1 ] [  -1 ] Upstream<br>[ 5 ] [ 0 0 1 0 ] [ 5  ] [  5  ] gradient<br>[ 0 ] [ 0 0 0 0 ] [ 9  ] [  9  ]<br><!-- End of picture text -->

<mark>Fei-Fei Li, Ranjay Krishna, Danfei Xu</mark> 

<mark>April 16, 2020</mark> 

<mark>134</mark> 

<mark>Lecture 4 -</mark> 

#### Backprop with Vectors 


![](references/papers/ml-foundations/lecture_4_images/lecture_4.pdf-0135-01.png)


<!-- Start of picture text -->
4D input x: 4D output z:<br>[  1  ] [  1  ]<br>[ -2  ] [  0  ]<br>f(x) = max(0,x)<br>Jacobian is  sparse :<br>[  3  ] [  3  ]<br>off-diagonal entries<br>(elementwise)<br>always zero! Never  [  -1 ] [  0  ]<br>explicitly  form<br>Jacobian -- instead<br>use  implicit 4D dL/dx:  [dz/dx] [dL/dz] 4D dL/dz:<br>multiplication [ 4 ] [  4  ]<br>[ 0 ] z [  -1 ] Upstream<br>[ 5 ] [  5  ] gradient<br>[ 0 ] [  9  ]<br><!-- End of picture text -->

<mark>Fei-Fei Li, Ranjay Krishna, Danfei Xu</mark> 

<mark>April 16, 2020</mark> 

<mark>135</mark> 

<mark>Lecture 4 -</mark> 


![](references/papers/ml-foundations/lecture_4_images/lecture_4.pdf-0136-00.png)


<!-- Start of picture text -->
Loss L still a scalar!<br>Backprop with Matrices (or Tensors)<br>dL/dx always has the<br>[Dx×Mx]<br>same shape as x!<br>[Dz×Mz]<br>Matrix-vector f<br>multiply<br>[Dy×My]<br>Jacobian<br>matrices<br><!-- End of picture text -->

<mark>Fei-Fei Li, Ranjay Krishna, Danfei Xu</mark> 

<mark>Lecture 4 - 136 April 16, 2020</mark> 


![](references/papers/ml-foundations/lecture_4_images/lecture_4.pdf-0137-00.png)


<!-- Start of picture text -->
Loss L still a scalar!<br>Backprop with Matrices (or Tensors)<br>dL/dx always has the<br>[Dx×Mx]<br>same shape as x!<br>[Dx×Mx]<br>[Dz×Mz]<br>“Downstream<br>Matrix-vector f<br>gradients”<br>multiply<br>[Dz×Mz]<br>[Dy×My]<br>Jacobian<br>matrices<br>“Upstream gradient”<br>[Dy×My] For each element of z, how<br>much does it influence L?<br><!-- End of picture text -->

<mark>Fei-Fei Li, Ranjay Krishna, Danfei Xu</mark> 

<mark>137 April 16, 2020</mark> 

<mark>Lecture 4 -</mark> 


![](references/papers/ml-foundations/lecture_4_images/lecture_4.pdf-0138-00.png)


<!-- Start of picture text -->
Loss L still a scalar!<br>Backprop with Matrices (or Tensors)<br>dL/dx always has the<br>“local<br>[Dx×Mx]<br>same shape as x!<br>gradients”<br>[Dx×Mx]<br>[Dz×Mz]<br>“Downstream<br>Matrix-vector<br>gradients”<br>multiply<br>[Dz×Mz]<br>[Dy×My]<br>Jacobian<br>matrices<br>“Upstream gradient”<br>[Dy×My] For each element of z, how<br>For each element of y, how much<br>much does it influence L?<br>does it influence each element of z?<br><!-- End of picture text -->

<mark>Fei-Fei Li, Ranjay Krishna, Danfei Xu</mark> 

<mark>Lecture 4 - 138 April 16, 2020</mark> 


![](references/papers/ml-foundations/lecture_4_images/lecture_4.pdf-0139-00.png)


<!-- Start of picture text -->
Loss L still a scalar!<br>Backprop with Matrices (or Tensors)<br>dL/dx always has the<br>“local<br>[Dx×Mx]<br>same shape as x!<br>gradients”<br>[Dx×Mx]<br>[(Dx×Mx)×(Dz×Mz)]<br>[Dz×Mz]<br>“Downstream<br>Matrix-vector<br>gradients”<br>multiply [(Dy×My)×(Dz×Mz)]<br>[Dz×Mz]<br>[Dy×My]<br>Jacobian<br>matrices<br>“Upstream gradient”<br>[Dy×My] For each element of z, how<br>For each element of y, how much<br>much does it influence L?<br>does it influence each element of z?<br><!-- End of picture text -->

<mark>Fei-Fei Li, Ranjay Krishna, Danfei Xu</mark> 

<mark>April 16, 2020</mark> 

<mark>139</mark> 

<mark>Lecture 4 -</mark> 


![](references/papers/ml-foundations/lecture_4_images/lecture_4.pdf-0140-00.png)


<!-- Start of picture text -->
Backprop with Matrices y: [N×M]<br>[ 13  9  -2  -6  ]<br>x: [N×D]<br>Matrix Multiply [  5  2  17  1 ]<br>[  2    1   -3 ]<br>[ -3   4   2 ]<br>dL/dy: [N×M]<br>w: [D×M] [  2  3 -3  9 ]<br>[  3  2  1 -1] [ -8  1  4  6 ]<br>[  2  1  3  2]<br>[  3  2  1 -2]<br>Also see derivation in the course notes:<br>http://cs231n.stanford.edu/handouts/linear-backprop.pdf<br><!-- End of picture text -->

<mark>Fei-Fei Li, Ranjay Krishna, Danfei Xu</mark> 

<mark>140 April 16, 2020</mark> 

<mark>Lecture 4 -</mark> 


![](references/papers/ml-foundations/lecture_4_images/lecture_4.pdf-0141-00.png)


<!-- Start of picture text -->
Backprop with Matrices y: [N×M]<br>[ 13  9  -2  -6  ]<br>x: [N×D]<br>Matrix Multiply [  5  2  17  1 ]<br>[  2    1   -3 ]<br>[ -3   4   2 ]<br>dL/dy: [N×M]<br>w: [D×M] [  2  3 -3  9 ]<br>[  3  2  1 -1] Jacobians : [ -8  1  4  6 ]<br>[  2  1  3  2]<br>dy/dx: [(N×D)×(N×M)]<br>[  3  2  1 -2]<br>dy/dw: [(D×M)×(N×M)]<br>For a neural net we may have<br>N=64, D=M=4096<br>Each Jacobian takes 256 GB of memory!<br>Must work with them implicitly!<br><!-- End of picture text -->

<mark>Fei-Fei Li, Ranjay Krishna, Danfei Xu</mark> 

<mark>April 16, 2020</mark> 

<mark>141</mark> 

<mark>Lecture 4 -</mark> 


![](references/papers/ml-foundations/lecture_4_images/lecture_4.pdf-0142-00.png)


<!-- Start of picture text -->
Backprop with Matrices y: [N×M]<br>[ 13  9  -2  -6  ]<br>x: [N×D]<br>Matrix Multiply [  5  2  17  1 ]<br>[  2    1   -3 ]<br>[ -3   4   2 ]<br>dL/dy: [N×M]<br>w: [D×M] [  2  3 -3  9 ]<br>[  3  2  1 -1] Q : What parts of y  [ -8  1  4  6 ]<br>[  2  1  3  2] are affected by one<br>[  3  2  1 -2] element of x?<br><!-- End of picture text -->

<mark>Fei-Fei Li, Ranjay Krishna, Danfei Xu</mark> 

<mark>April 16, 2020</mark> 

<mark>142</mark> 

<mark>Lecture 4 -</mark> 


![](references/papers/ml-foundations/lecture_4_images/lecture_4.pdf-0143-00.png)


<!-- Start of picture text -->
Backprop with Matrices y: [N×M]<br>[ 13  9  -2  -6  ]<br>x: [N×D]<br>Matrix Multiply [  5  2  17  1 ]<br>[  2    1   -3 ]<br>[ -3   4   2 ]<br>dL/dy: [N×M]<br>w: [D×M] [  2  3 -3  9 ]<br>[  3  2  1 -1] Q : What parts of y  [ -8  1  4  6 ]<br>[  2  1  3  2] are affected by one<br>[  3  2  1 -2] element of x?<br>A : affects the<br>whole row<br><!-- End of picture text -->

<mark>Fei-Fei Li, Ranjay Krishna, Danfei Xu</mark> 

<mark>Lecture 4 - 143 April 16, 2020</mark> 


![](references/papers/ml-foundations/lecture_4_images/lecture_4.pdf-0144-00.png)


<!-- Start of picture text -->
Backprop with Matrices y: [N×M]<br>[ 13  9  -2  -6  ]<br>x: [N×D]<br>Matrix Multiply [  5  2  17  1 ]<br>[  2    1   -3 ]<br>[ -3   4   2 ]<br>dL/dy: [N×M]<br>w: [D×M] [  2  3 -3  9 ]<br>[  3  2  1 -1] Q : What parts of y  Q : How much  [ -8  1  4  6 ]<br>[  2  1  3  2] are affected by one  does<br>[  3  2  1 -2] element of x? affect ?<br>A : affects the<br>whole row<br><!-- End of picture text -->

<mark>Fei-Fei Li, Ranjay Krishna, Danfei Xu</mark> 

<mark>Lecture 4 - 144 April 16, 2020</mark> 


![](references/papers/ml-foundations/lecture_4_images/lecture_4.pdf-0145-00.png)


<!-- Start of picture text -->
Backprop with Matrices y: [N×M]<br>[ 13  9  -2  -6  ]<br>x: [N×D]<br>Matrix Multiply [  5  2  17  1 ]<br>[  2    1   -3 ]<br>[ -3   4   2 ]<br>dL/dy: [N×M]<br>w: [D×M] [  2  3 -3  9 ]<br>[  3  2  1 -1] Q : What parts of y  Q : How much  [ -8  1  4  6 ]<br>[  2  1  3  2] are affected by one  does<br>[  3  2  1 -2] element of x? affect ?<br>A : affects the  A:<br>whole row<br><!-- End of picture text -->

<mark>Fei-Fei Li, Ranjay Krishna, Danfei Xu</mark> 

<mark>April 16, 2020</mark> 

<mark>145</mark> 

<mark>Lecture 4 -</mark> 


![](references/papers/ml-foundations/lecture_4_images/lecture_4.pdf-0146-00.png)


<!-- Start of picture text -->
Backprop with Matrices y: [N×M]<br>[ 13  9  -2  -6  ]<br>x: [N×D]<br>Matrix Multiply [  5  2  17  1 ]<br>[  2    1   -3 ]<br>[ -3   4   2 ]<br>dL/dy: [N×M]<br>w: [D×M] [  2  3 -3  9 ]<br>[  3  2  1 -1] Q : What parts of y  Q : How much  [ -8  1  4  6 ]<br>[  2  1  3  2] are affected by one  does<br>[  3  2  1 -2] element of x? affect ?<br>A : affects the  A:<br>[N×D]  [N×M] [M×D]<br>whole row<br><!-- End of picture text -->

<mark>Fei-Fei Li, Ranjay Krishna, Danfei Xu</mark> 

<mark>April 16, 2020</mark> 

<mark>146</mark> 

<mark>Lecture 4 -</mark> 

###### Backprop with Matrices 

x: [N×D] Matrix Multiply [  2 **1** -3 ] [ -3   4   2 ] w: [D×M] [  3  2  1 -1] [  2  1  3  2] By similar logic: [  3  2  1 -2] [N×D]  [N×M] [M×D] [D×M]  [D×N] [N×M] 

[D×M]  [D×N] [N×M] 


![](references/papers/ml-foundations/lecture_4_images/lecture_4.pdf-0147-03.png)



![](references/papers/ml-foundations/lecture_4_images/lecture_4.pdf-0147-04.png)


y: [N×M] [ **13  9  -2  -6** ] [  5  2  17  1 ] dL/dy: [N×M] [  2  3 -3  9 ] [ -8  1  4  6 ] 

These formulas are easy to remember: they are the only way to make shapes match up! 

<mark>Fei-Fei Li, Ranjay Krishna, Danfei Xu</mark> 

<mark>April 16, 2020</mark> 

<mark>147</mark> 

<mark>Lecture 4 -</mark> 

#### Summary for today: 

- **(Fully-connected) Neural Networks** are stacks of linear functions and nonlinear activation functions; they have much more representational power than linear classifiers 

- **backpropagation** = recursive application of the chain rule along a computational graph to compute the gradients of all inputs/parameters/intermediates 

- implementations maintain a graph structure, where the nodes implement the **forward** () / **backward** () API 

- **forward** : compute result of an operation and save any intermediates needed for gradient computation in memory 

- **backward** : apply the chain rule to compute the gradient of the loss function with respect to the inputs 

<mark>Fei-Fei Li, Ranjay Krishna, Danfei Xu Lecture 4 -</mark> 

<mark>148 April 16, 2020</mark> 

#### Next Time: Convolutional Networks! 


![](references/papers/ml-foundations/lecture_4_images/lecture_4.pdf-0149-01.png)


<mark>Fei-Fei Li, Ranjay Krishna, Danfei Xu</mark> 

<mark>April 16, 2020</mark> 

<mark>Lecture 4 - 149</mark> 

<mark>A vectorized example:</mark> 


![](references/papers/ml-foundations/lecture_4_images/lecture_4.pdf-0150-01.png)


<mark>Fei-Fei Li, Ranjay Krishna, Danfei Xu</mark> 

<mark>April 16, 2020</mark> 

<mark>Lecture 4 - 150</mark> 

###### <mark>A vectorized example:</mark> 


![](references/papers/ml-foundations/lecture_4_images/lecture_4.pdf-0151-01.png)



![](references/papers/ml-foundations/lecture_4_images/lecture_4.pdf-0151-02.png)



![](references/papers/ml-foundations/lecture_4_images/lecture_4.pdf-0151-03.png)



![](references/papers/ml-foundations/lecture_4_images/lecture_4.pdf-0151-04.png)


<mark>Fei-Fei Li, Ranjay Krishna, Danfei Xu</mark> 

<mark>Lecture 4 - 151 April 16, 2020</mark> 

<mark>A vectorized example:</mark> 


![](references/papers/ml-foundations/lecture_4_images/lecture_4.pdf-0152-01.png)



![](references/papers/ml-foundations/lecture_4_images/lecture_4.pdf-0152-02.png)


<mark>Fei-Fei Li, Ranjay Krishna, Danfei Xu</mark> 

<mark>April 16, 2020</mark> 

<mark>Lecture 4 - 152</mark> 


![](references/papers/ml-foundations/lecture_4_images/lecture_4.pdf-0153-00.png)


###### <mark>A vectorized example:</mark> 


![](references/papers/ml-foundations/lecture_4_images/lecture_4.pdf-0153-02.png)



![](references/papers/ml-foundations/lecture_4_images/lecture_4.pdf-0153-03.png)



![](references/papers/ml-foundations/lecture_4_images/lecture_4.pdf-0153-04.png)



![](references/papers/ml-foundations/lecture_4_images/lecture_4.pdf-0153-05.png)


<mark>Fei-Fei Li, Ranjay Krishna, Danfei Xu</mark> 

<mark>Lecture 4 - 153 April 16, 2020</mark> 


![](references/papers/ml-foundations/lecture_4_images/lecture_4.pdf-0154-00.png)


###### <mark>A vectorized example:</mark> 


![](references/papers/ml-foundations/lecture_4_images/lecture_4.pdf-0154-02.png)



![](references/papers/ml-foundations/lecture_4_images/lecture_4.pdf-0154-03.png)



![](references/papers/ml-foundations/lecture_4_images/lecture_4.pdf-0154-04.png)



![](references/papers/ml-foundations/lecture_4_images/lecture_4.pdf-0154-05.png)


<mark>Fei-Fei Li, Ranjay Krishna, Danfei Xu</mark> 

<mark>Lecture 4 - 154 April 16, 2020</mark> 


![](references/papers/ml-foundations/lecture_4_images/lecture_4.pdf-0155-00.png)


###### <mark>A vectorized example:</mark> 


![](references/papers/ml-foundations/lecture_4_images/lecture_4.pdf-0155-02.png)



![](references/papers/ml-foundations/lecture_4_images/lecture_4.pdf-0155-03.png)



![](references/papers/ml-foundations/lecture_4_images/lecture_4.pdf-0155-04.png)



![](references/papers/ml-foundations/lecture_4_images/lecture_4.pdf-0155-05.png)


<mark>Fei-Fei Li, Ranjay Krishna, Danfei Xu</mark> 

<mark>Lecture 4 - 155 April 16, 2020</mark> 


![](references/papers/ml-foundations/lecture_4_images/lecture_4.pdf-0156-00.png)


###### <mark>A vectorized example:</mark> 


![](references/papers/ml-foundations/lecture_4_images/lecture_4.pdf-0156-02.png)



![](references/papers/ml-foundations/lecture_4_images/lecture_4.pdf-0156-03.png)



![](references/papers/ml-foundations/lecture_4_images/lecture_4.pdf-0156-04.png)



![](references/papers/ml-foundations/lecture_4_images/lecture_4.pdf-0156-05.png)



![](references/papers/ml-foundations/lecture_4_images/lecture_4.pdf-0156-06.png)



![](references/papers/ml-foundations/lecture_4_images/lecture_4.pdf-0156-07.png)


<mark>Fei-Fei Li, Ranjay Krishna, Danfei Xu</mark> 

<mark>April 16, 2020</mark> 

<mark>156</mark> 

<mark>Lecture 4 -</mark> 


![](references/papers/ml-foundations/lecture_4_images/lecture_4.pdf-0157-00.png)


###### <mark>A vectorized example:</mark> 


![](references/papers/ml-foundations/lecture_4_images/lecture_4.pdf-0157-02.png)



![](references/papers/ml-foundations/lecture_4_images/lecture_4.pdf-0157-03.png)



![](references/papers/ml-foundations/lecture_4_images/lecture_4.pdf-0157-04.png)



![](references/papers/ml-foundations/lecture_4_images/lecture_4.pdf-0157-05.png)



![](references/papers/ml-foundations/lecture_4_images/lecture_4.pdf-0157-06.png)



![](references/papers/ml-foundations/lecture_4_images/lecture_4.pdf-0157-07.png)


<mark>Fei-Fei Li, Ranjay Krishna, Danfei Xu</mark> 

<mark>April 16, 2020</mark> 

<mark>157</mark> 

<mark>Lecture 4 -</mark> 


![](references/papers/ml-foundations/lecture_4_images/lecture_4.pdf-0158-00.png)


###### <mark>A vectorized example:</mark> 


![](references/papers/ml-foundations/lecture_4_images/lecture_4.pdf-0158-02.png)



![](references/papers/ml-foundations/lecture_4_images/lecture_4.pdf-0158-03.png)



![](references/papers/ml-foundations/lecture_4_images/lecture_4.pdf-0158-04.png)



![](references/papers/ml-foundations/lecture_4_images/lecture_4.pdf-0158-05.png)


<mark>Fei-Fei Li, Ranjay Krishna, Danfei Xu</mark> 

<mark>Lecture 4 - 158 April 16, 2020</mark> 


![](references/papers/ml-foundations/lecture_4_images/lecture_4.pdf-0159-00.png)


###### <mark>A vectorized example:</mark> 


![](references/papers/ml-foundations/lecture_4_images/lecture_4.pdf-0159-02.png)



![](references/papers/ml-foundations/lecture_4_images/lecture_4.pdf-0159-03.png)



![](references/papers/ml-foundations/lecture_4_images/lecture_4.pdf-0159-04.png)



![](references/papers/ml-foundations/lecture_4_images/lecture_4.pdf-0159-05.png)



![](references/papers/ml-foundations/lecture_4_images/lecture_4.pdf-0159-06.png)



![](references/papers/ml-foundations/lecture_4_images/lecture_4.pdf-0159-07.png)



![](references/papers/ml-foundations/lecture_4_images/lecture_4.pdf-0159-08.png)


<mark>Fei-Fei Li, Ranjay Krishna, Danfei Xu</mark> 

<mark>April 16, 2020</mark> 

<mark>Lecture 4 - 159</mark> 


![](references/papers/ml-foundations/lecture_4_images/lecture_4.pdf-0160-00.png)


###### <mark>A vectorized example:</mark> 


![](references/papers/ml-foundations/lecture_4_images/lecture_4.pdf-0160-02.png)



![](references/papers/ml-foundations/lecture_4_images/lecture_4.pdf-0160-03.png)



![](references/papers/ml-foundations/lecture_4_images/lecture_4.pdf-0160-04.png)



![](references/papers/ml-foundations/lecture_4_images/lecture_4.pdf-0160-05.png)



![](references/papers/ml-foundations/lecture_4_images/lecture_4.pdf-0160-06.png)



![](references/papers/ml-foundations/lecture_4_images/lecture_4.pdf-0160-07.png)



![](references/papers/ml-foundations/lecture_4_images/lecture_4.pdf-0160-08.png)


<mark>Fei-Fei Li, Ranjay Krishna, Danfei Xu</mark> 

<mark>April 16, 2020</mark> 

<mark>Lecture 4 - 160</mark> 

###### <mark>A vectorized example:</mark> 


![](references/papers/ml-foundations/lecture_4_images/lecture_4.pdf-0161-01.png)



![](references/papers/ml-foundations/lecture_4_images/lecture_4.pdf-0161-02.png)



![](references/papers/ml-foundations/lecture_4_images/lecture_4.pdf-0161-03.png)



![](references/papers/ml-foundations/lecture_4_images/lecture_4.pdf-0161-04.png)



![](references/papers/ml-foundations/lecture_4_images/lecture_4.pdf-0161-05.png)



![](references/papers/ml-foundations/lecture_4_images/lecture_4.pdf-0161-06.png)


<mark>Fei-Fei Li, Ranjay Krishna, Danfei Xu</mark> 

<mark>April 16, 2020</mark> 

<mark>Lecture 4 - 161</mark> 


![](references/papers/ml-foundations/lecture_4_images/lecture_4.pdf-0162-00.png)


###### <mark>A vectorized example:</mark> 


![](references/papers/ml-foundations/lecture_4_images/lecture_4.pdf-0162-02.png)



![](references/papers/ml-foundations/lecture_4_images/lecture_4.pdf-0162-03.png)


Always check: The gradient with respect to a variable should have the same shape as the variable 


![](references/papers/ml-foundations/lecture_4_images/lecture_4.pdf-0162-05.png)



![](references/papers/ml-foundations/lecture_4_images/lecture_4.pdf-0162-06.png)



![](references/papers/ml-foundations/lecture_4_images/lecture_4.pdf-0162-07.png)



![](references/papers/ml-foundations/lecture_4_images/lecture_4.pdf-0162-08.png)



![](references/papers/ml-foundations/lecture_4_images/lecture_4.pdf-0162-09.png)


<mark>Fei-Fei Li, Ranjay Krishna, Danfei Xu</mark> 

<mark>April 16, 2020</mark> 

<mark>Lecture 4 - 162</mark> 


![](references/papers/ml-foundations/lecture_4_images/lecture_4.pdf-0163-00.png)


###### <mark>A vectorized example:</mark> 


![](references/papers/ml-foundations/lecture_4_images/lecture_4.pdf-0163-02.png)



![](references/papers/ml-foundations/lecture_4_images/lecture_4.pdf-0163-03.png)



![](references/papers/ml-foundations/lecture_4_images/lecture_4.pdf-0163-04.png)



![](references/papers/ml-foundations/lecture_4_images/lecture_4.pdf-0163-05.png)


<mark>Fei-Fei Li, Ranjay Krishna, Danfei Xu</mark> 

<mark>Lecture 4 - 163 April 16, 2020</mark> 


![](references/papers/ml-foundations/lecture_4_images/lecture_4.pdf-0164-00.png)


###### <mark>A vectorized example:</mark> 


![](references/papers/ml-foundations/lecture_4_images/lecture_4.pdf-0164-02.png)



![](references/papers/ml-foundations/lecture_4_images/lecture_4.pdf-0164-03.png)



![](references/papers/ml-foundations/lecture_4_images/lecture_4.pdf-0164-04.png)



![](references/papers/ml-foundations/lecture_4_images/lecture_4.pdf-0164-05.png)



![](references/papers/ml-foundations/lecture_4_images/lecture_4.pdf-0164-06.png)



![](references/papers/ml-foundations/lecture_4_images/lecture_4.pdf-0164-07.png)


<mark>Fei-Fei Li, Ranjay Krishna, Danfei Xu</mark> 

<mark>April 16, 2020</mark> 

<mark>Lecture 4 - 164</mark> 


![](references/papers/ml-foundations/lecture_4_images/lecture_4.pdf-0165-00.png)


###### <mark>A vectorized example:</mark> 


![](references/papers/ml-foundations/lecture_4_images/lecture_4.pdf-0165-02.png)



![](references/papers/ml-foundations/lecture_4_images/lecture_4.pdf-0165-03.png)



![](references/papers/ml-foundations/lecture_4_images/lecture_4.pdf-0165-04.png)



![](references/papers/ml-foundations/lecture_4_images/lecture_4.pdf-0165-05.png)



![](references/papers/ml-foundations/lecture_4_images/lecture_4.pdf-0165-06.png)



![](references/papers/ml-foundations/lecture_4_images/lecture_4.pdf-0165-07.png)



![](references/papers/ml-foundations/lecture_4_images/lecture_4.pdf-0165-08.png)


<mark>Fei-Fei Li, Ranjay Krishna, Danfei Xu</mark> 

<mark>Lecture 4 - 165 April 16, 2020</mark> 

###### <mark>In discussion section: A matrix example...</mark> 


![](references/papers/ml-foundations/lecture_4_images/lecture_4.pdf-0166-01.png)


**? ?** 


![](references/papers/ml-foundations/lecture_4_images/lecture_4.pdf-0166-03.png)



![](references/papers/ml-foundations/lecture_4_images/lecture_4.pdf-0166-04.png)


16 Lecture 4 - April 13, 2017 6 

Fei-Fei Li & Justin Johnson & Serena Yeung 

