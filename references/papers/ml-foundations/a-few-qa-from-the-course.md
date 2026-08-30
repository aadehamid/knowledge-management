title: A few QA’s from the course F’20 \<Deep Learning\> – Kyunghyun Cho

# A few QA’s from the course F’20 \<Deep Learning> – Kyunghyun Cho

i’ve just finished teaching \<Deep Learning> this semester together with [Yann](http://yann.lecun.com/) and [Alfredo](https://www.linkedin.com/in/alfredocanziani/). the course was in a “blended mode”, implying that lectures were given in person and live-streamed with a limited subset of students allowed to join each week and all the other students joining remotely via Zoom. this has resulted in more active online discussion among students, instructors and assistants over the course, and indeed there were quite a few interesting questions posted on the course page which was run on [campuswire](https://campuswire.com/). 

i enjoyed answering those questions, because they made me think quite a bit about them myself. of course, as usual i ended up leaving only a short answer to each, but i thought i’d share them here in the case any students in the future run into the same questions. although my questions are all quite speculative and based on experience rather than rigorously justified, what’s fun in rigorously proven and well-known answers?

of course, there were so much more questions asked and answered during live lectures and at the chatrooms, but i just cannot recall all of them easily nor am i energetic enough after this unprecedented semester to go through the whole chat log to dig out interesting questions. i just ask you to trust me that the list of questions below is a tiny subset of interesting questions.

i will paraphrase/shorten the answers below and remove any identifying information (if any):

1. [Why was backprop controversial? Yann mentioned that one of the big things that made the use of ConvNets in various applications controversial was the use of backpropagation. backprop is just an application of the chain rule, so why would anyone be suspicious of using it?](#q1)
2. [Professor LeCun said that mini-batch has no advantage over single-batch SGD besides being easier to parallelize, and online SGD is actually superior. Is there any other theoretical reason why single-batch is preferable?](#q2)
3. [Why we would do batch normalization instead of normalizing the whole dataset all at once at first? Is it for when normalizing the whole dataset is too computationally expensive? I understood that normalization makes the optimization process easier through making the eigenvalues equal. However, if you’re only normalizing over the batch, your normalization for each batch is subject to noise and might still lead to bad learning rates for each dimension.](#q3)
4. [Batch normalization in VAE: While implementing the convolutional VAE model, I noticed that removing these BatchNorm layers enabled the model to train as expected. I was wondering why does BatchNorm cause this issue in the VAE model?](#q4)
5. [In semi-supervised VAE, how do we decide the embedding dimensions for the class? Also, BERT used position embedding to represent the position, so how do we determine the position embedding dimensions in BERT?](#q5)
6. [Why do we divide the input to the softmax in dot product attention by the square root of the dimensionality?](#q6)
7. [DL appears to add double descent as a caveat in addition to bias-variance tradeoff learned earlier. Do you have any insights on how we should think about double-descent?](#q7)
8. [In your opinion, will we achieve AGI?](#q8)

**1. Why was backprop controversial? Yann mentioned that one of the big things that made the use of ConvNets in various applications controversial was the use of backpropagation. backprop is just an application of the chain rule, so why would anyone be suspect of using it?**

when yann said it was controversial to use backprop earlier, i believe he meant it in two different ways: (1) backprop itself to compute the gradient of the loss function w.r.t. the parameters and (2) backprop to refer to gradient-based optimization. i’ll explain a bit of each below, but neither of them is considered a serious argument against using backprop anymore.

(1) backprop was controversial and is under great scrutiny when artificial neural nets (what we learn) are compared against biological neural nets (what we have). it’s quite clear due to biological constraints that backprop is not implemented in brains, as it is in our deep learning toolkits (see e.g., [https://openreview.net/forum?id\=HJgPEXtIUS](https://openreview.net/forum?id=HJgPEXtIUS) for some of interesting biological constraints/properties that should be satisfied by any biologically plausible learning algorithms.) to some people, this is a make-or-break kind of issue, because there seems to exist a learning algorithm that results in a superior neural net (human brains!) of course, this could just mean that a biological brain is approximating the gradient computation as well as it could under the constraints, but it’s not easy to verify this (see, e.g., [https://www.youtube.com/watch?v\=VIRCybGgHts](https://www.youtube.com/watch?v=VIRCybGgHts) for how a brain might implement backprop.)

another criticism or objection along this line is that biological brains seem to have either zero or multiple objectives that are being optimized simultaneously. this is unlike our usual practice in deep learning where we start by defining one clear objective function to minimize.

(2) gradient-based optimization often refers to a set of techniques developed for (constrained/unconstrained) convex optimization. when such a technique is used for a non-convex problem, we are often working with the local quadratic approximation, that is, given any point in the space, the underlying non-convex objective function can be approximated by a convex quadratic function ($\theta\^\top H \theta \+ g\^\top \theta \+ c$.) under this assumption, gradient-based optimization would be attracted toward the minimum of this local quadratic approximation, regardless of whether there exists a better minimum far away from the current point in the space. this is often used as a reason for criticizing the use of gradient-based optimization with a non-convex objective function, thereby for criticizing the use of backprop. see e.g. [http://leon.bottou.org/publications/pdf/online-1998.pdf](http://leon.bottou.org/publications/pdf/online-1998.pdf) for extensive study on the convergence properties of SGD.

this criticism however requires one big assumption that there is a big gap of quality between one of the nearby local minimum (we’ll talk about it in a few weeks at the course) and the global minimum. if there is a big gap, this would indeed be a trouble, but what if there isn’t?

it turned out that we’ve known for already a few decades that most of local minima are of reasonable quality (in terms of both training and test accuracies) as long as we make neural nets larger than necessary. let me quote [Rumelhart, Hinton & Williams (1986)](https://www.nature.com/articles/323533a0):

> “*The most obvious drawback of the learning procedure is that the error-surface may contain local minima so that gradient descent is not guaranteed to find a global minimum. However, experience with many tasks shows that the network very rarely gets stuck in poor local minima that are significantly worse than the global minimum. We have only encountered this undesirable behaviour in networks that have just enough connections to perform the task. Adding a few more connections creates extra dimensions in weight-space and these dimensions provide paths around the barriers that create poor local minima in the lower dimensional subspaces.*“
> [\<Learning representations by back-propagating errors>](https://www.nature.com/articles/323533a0) by [Rumelhart, Hinton & Williams (1986)](https://www.nature.com/articles/323533a0)

this phenomenon has been and is being studied quite extensively from various angles. if you’re interested in this topic, see e.g. [http://papers.nips.cc/paper/5486-identifying-and-attacking-the-saddle-point-problem-in-high-dimensional-non-convex-optimization](http://papers.nips.cc/paper/5486-identifying-and-attacking-the-saddle-point-problem-in-high-dimensional-non-convex-optimization) and [https://arxiv.org/abs/1803.03635](https://arxiv.org/abs/1803.03635) for some recent directions. or, if you feel lazy, you can see my slides at [https://drive.google.com/file/d/1YxHbQ0NeSaAANaFEmlo9H5fUsZRsiGJK/view](https://drive.google.com/file/d/1YxHbQ0NeSaAANaFEmlo9H5fUsZRsiGJK/view) which i prepared recently.

**2. Professor LeCun said that mini-batch has no advantage over single-batch SGD besides being easier to parallelize, and SGD is actually superior. Is there any other theoretical reason why single-batch is preferable?**

this is an interesting & important question, and the answer to this varies from one expert to another, including Yann and myself as well, based on what are being implicitly assumed and what are being used as criteria to tell which is preferred (computational efficiency, generalization accuracy, etc.)

Yann’s view is that noise in SGD greatly helps generalization because it prevents learning from being stuck at a sharp local minimum and drives learning to find a flatter local minimum which would imply that the final neural net is more robust to perturbation to the parameters, which naturally translates to the robust to the perturbation to the input, implying that it would generalize better. under this perspective, you want to maximize the level of noise, as long as they roughly cancel out on average across all the stochastic gradients computed from the training examples. that would correspond to using just one training example for computing each stochastic gradient.

of course, the amount of noise, which is proportional to the variance of the stochastic gradient, does impact the speed at which learning happens. in recent years, we (as the community of deep learning researchers) have found that certain network architectures require stochastic gradients computed using large minibatches (though, it’s unclear what large means, as it’s quite relative to the size of the training set) to be trained at all. in these cases, it looks like high level of noise sometimes prevents any progress in learning especially in the early stage.

so, in short, it’s still an open question. yann’s perspective may turn out to be the correct one (and that wouldn’t be the first time this happend,) or we may find a completely different explanation in the future.

**3. Why we would do batch normalization instead of normalizing the whole dataset all at once at first? Is it for when normalizing the whole dataset is too computationally expensive?** **I understood that normalization makes the optimization process easier through making the eigenvalues equal. However, if you’re only normalizing over the batch, your normalization for each batch is subject to noise and might still lead to bad learning rates for each dimension.**

there are three questions/points here. let me address each separately below:

“*normalization makes the optimization process easier through making the eigenvalues equal*“

we need to specify what kind of normalization you refer to, but in general, it’s not possible to make the hessian to be identity by simply normalizing the input. this is only possible when we are considering a linear network with a specific loss function (e.g., l2 loss for regression and cross-entropy for classification.) however, it is empirically known and for some cases rigorously as well that normalizing the input variables to be zero-mean and unit-variance makes the conditioning number (the ratio between the largest and smallest real eigenvalues of the hessian matrix) close to 1 (which is good.)

“*why we would do batch normalization instead of normalizing the whole dataset all at once at first?*“

now, in the case of a network with multiple layers, it turned out that we can maximize the benefit of normalization by normalizing the input to each layer to be zero-mean and unit-variance. unfortunately, this is not trivial, because the input to each layer changes as the lower layers’ weights and biases evolve. in other words, if we wanted to normalize the input to each layer, we would need to sweep through the entire dataset every time we update the weight matrices and bias vectors, which would make it intolerable. furthermore, renormalizing the input at a lower layer changes the input to the upper layers, ultimately resulting in the loss function to change dramatically each time we renormalize all the layers, likely making learning impossible. though, this is up to a certain degree addressible (see [http://www.jmlr.org/proceedings/papers/v22/raiko12/raiko12.pdf](http://www.jmlr.org/proceedings/papers/v22/raiko12/raiko12.pdf) by Tapani Raiko, my phd advisor, and Yann LeCun.)

“*your normalization for each batch is subject to noise*“

this is indeed true, and that’s precisely why it’s a customary practice to keep the running averages of the mean and variance of each dimension in batch normalization. assuming that the parameters of the network evolve slowly, such practice ultimately converges to the population mean and variance.

**4. Batch normalization in VAE: While implementing the convolutional VAE model, I noticed that removing these BatchNorm layers enabled the model to train as expected. I was wondering why does BatchNorm cause this issue in the VAE model?**

i don’t have a clear answer unfortunately, but can speculate a bit on why this is the case. my answer will depend on where batchnorm was used. of course, before reading the answer below, make sure your implementation of batchnorm doesn’t have a bug.

if batchnorm was used in the approximate posterior (encoder), it shouldn’t really matter, since the approximate posterior can be anything by definition. it can depend not only on the current observation $x$\
, but can be anything else that helps minimizing the KL divergence from this approximate posterior to the true posterior. so, i wouldn’t be surprised if it’s totally fine leaving batchnorm in the encoder.

if batchnorm was used in the decoder, it may matter, as the likelihood distribution (generative distribution) is over the observation space $\mathcal{X}$ conditioned on the latent variable configuration $z$. with batchnorm, instead, the decoder is conditioned on the entire minibatch of latent variable configurations, that is, the latent variable configurations of the other examples. this may hinder optimization in the early stage of learning (in the later stage of learning, it shouldn’t really matter much, though.)

in general, batchnorm is a tricky technique and makes it difficult to analyze SGD, because it introduces correlation across per-example stochastic gradients within each minibatch.

\5. **In semi-supervised VAE, how do we decide the embedding dimensions for the class**? **Also, BERT used position embedding to represent the position, so how do we determine the position embedding dimensions in BERT?**

this question can be answered from two angles.

a. network size

the embedding dimensionality is a part of a neural net, and it can be thought of as a part of determining the size of your neural network. it’s a good rule of thumb to use as large as neural net as you can within your computational and financial budget to maximize your gain in terms of generalization. this might sound counter-intuitive, if you have learned from earlier courses that we want to choose the most succinct model (according to the principle of occam’s razor,) but in neural nets, it’s not simply the size of the model, but the choice of optimization and regularization that matters perhaps even more. in particular, as we will learn next week, SGD is inherently working in a low-dimensional subspace of the parameter space and cannot explore the whole space of the parameters, a larger network does not imply that it’s more prone to overfitting.

b. why more than one dimension?

let’s think of the class embedding (though, the same argument applies to positional embedding.) take as an example handwritten digit classification, where our classes consists of 0, 1, 2, .., 9. it seems quite natural that there’s a clear one-dimensional structure behind these classes, and we would only need a one-dimensional embedding. why we do need then multi-dimensional class embedding?

it turned out that there are multiple degrees of similarity among these classes, and that the similarity among these classes is context-dependent. that is, depending on what we see as an input, the class similarity changes. for instance, when the input is a slanted 3 (3 significantly rotated clock-wise), it looks like either 3 or 2 but not 8 nor 0. when the input is a straight-standing 3, it looks like either 3 or 8 but not 2. in other words, the classes 3 and 2 are similar to each other when the input was a slanted 3, while the classes 3 and 8 are similar to each other when the input was a upright 3.

having multiple dimensions to represent each class allows us to capture these different degrees of similarity among classes. a few dimensions in the class embeddings of 3 and 2 will point toward a similar direction, while a few other dimensions in the class embeddings of 3 and 8 will point toward another similar direction. when the input is a slanted 3, the feature extractor (a convolutional net) will output a vector that will emphasize the first few dimensions and suppress the other dimensions to exploit the similarity between 3 and 2. a similar mechanism would lead to a feature vector of an upright 3 that would suppress the first few dimensions and emphasize the latter few to exploit the similarity between 3 and 8.

it’s impossible to tell in advance how many such degrees of similarity exist and how to encode them. that’s why we need to use as high dimensional embedding as possible for encoding any discrete, one-hot input.

**6. Why do we divide the input to the softmax in dot product attention by the square root of the dimensionality?** 

This question was asked at one of the office hours, and [Richard Pang](https://yzpang.github.io/) (one of the TA’s) and i attempted at reverse-engineering the motivations behind the scaled dot-product attention from the transformers.

assume each key vector $k \in \mathbb{R}\^d$ is a sample drawn from a multivariate, standard Normal distribution, i.e., $k\_i \sim \mathcal{N}(0, 1\^2).$ given a query vector $q \in \mathbb{R}\^d$, we can now compute the variance of the dot product between the query and key vectors as $\mathbb{V}\[q\^\top k\] \= \mathbb{V}\[\sum\_{i\=1}\^d q\_i k\_i\] \= \sum\_{i\=1}\^d q\_i\^2 \mathbb{V}\[k\_i\] \= \sum\_{i\=1}\^d q\_i\^2$. in other words, the variance of each logit is the squared norm of the query vector.

assume the query vector $q$ is also a sample drawn from a multivariate, standard Normal distribution, i.e., $q\_i \sim \mathcal{N}(0, 1\^2)$. in other words, $\mathbb{E}\[q\_i\]\=0$ and $\mathbb{V}\[q\_i\]\=\mathbb{E}{q\_i} \left\[(q\_i – \mathbb{E}\[q\_i\])\^2\right\] \= \mathbb{E}{q\_i} \left\[ q\_i\^2 \right\] \= 1$. then, the expected variance of the logit ends up being $\mathbb{E}{q} \left\[ \mathbb{V}\[q\^\top k\] \right\] \= \mathbb{E}{q} \sum\_{i\=1}\^d q\_i\^2 \= \sum\_{i\=1}\^d \mathbb{E}{q\_i} q\_i\^2 \= \sum{i\=1}\^d 1 \= d.$

we can now standardize the logit to be $0$-mean and unit-variance (or more precisely, we make the logit’s scale to be invariant to the dimensionality of the key and query vectors) by dividing it with the standard deviation $\sqrt{\mathbb{E}\_q \mathbb{V}\[q\^\top k\]}\=\sqrt{d}.$

these assumptions of Normality do not hold in reality, but as we talked about it earlier, Normality is one of the safest things to assume when we don’t know much about the underlying process.

As [Ilya Kulikov](https://iliakulikov.ru/) kindly pointed out, this explanation doesn’t answer “why” and instead answers “what” scaling does. “why” is a bit more difficult to answer (perhaps unsurprisingly,) but one answer is that softmax saturates as the logits (the input to softmax) grow in their magnitudes, which may slow down learning due to the vanishing gradient. though, it’s unclear what’s the right way to quantify it.

**7. DL appears to add double descent as a caveat in addition to bias-variance tradeoff learned early on. Do you have any insights about how we should think about double-descent?** 

The so-called double descent phenomenon is a relatively recently popularized concept that’s still being studied heavily (though, it was observed and reported by Yann already in the early 90s. see, e.g., [https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.66.2396](https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.66.2396) and also [https://iopscience.iop.org/article/10.1088/0305-4470/25/5/020](https://iopscience.iop.org/article/10.1088/0305-4470/25/5/020) by Krogh and Hartz) The issue I have with double descent in deep neural nets is that it’s unclear how we define a model capacity. the # of parameters is certainly not the best proxy, because the parameters are all heavily correlated and redundant. perhaps it should be the number of SGD steps, because we learned that the size of the hypothesis space is in fact the function of the number of SGD steps. 

One particular proxy I find interesting and convincing is the fraction of positive eigenvalues of the Hessian at a solution. With this proxy, it looks like the apparent double descent phenomenon often lessens. see e.g. [https://arxiv.org/abs/2003.02139](https://arxiv.org/abs/2003.02139). 

So, in short, the model capacity is a key to understanding the bias-variance trade-off or more generally generalization in machine learning, but is not a simple concept to grasp with deep neural networks.

**8. In your opinion, will we achieve AGI?**

Of course, I’m far from being qualified to answer this question well. Instead, let me quote Yann:

> *Yann LeCun, a professor at the Courant Institute of Mathematical Sciences at New York University (NYU), is much more direct: “It’s hard to explain to non-specialists that AGI is not a ‘thing’, and that most venues that have AGI in their name deal in highly speculative and theoretical issues…*
> [\<An executive primer on artificial general intelligence>](https://www.mckinsey.com/business-functions/operations/our-insights/an-executive-primer-on-artificial-general-intelligence?fbclid=IwAR0Sq4kztjaspZcwGJgTsOYXKtXBdnztrEXbVgKHRu07cwpVm0JoQHM-bxA) by Federico Berruti, Pieter Nel, and Rob Whiteman
