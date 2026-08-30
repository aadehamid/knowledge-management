[![Theory Dish](https://theorydish.blog/wp-content/uploads/2017/03/cropped-nightdish11.jpg)

# Theory Dish

## Stanford's CS Theory Research Blog](https://theorydish.blog/ "Theory Dish")

* [About](https://theorydish.blog/about/)

# Backpropagation ≠ Chain Rule

Posted on December 16, 2021 by [Lunjia Hu](https://theorydish.blog/author/lunjiah/) in [Uncategorized](https://theorydish.blog/category/uncategorized/) // [4 Comments](https://theorydish.blog/2021/12/16/backpropagation-%E2%89%A0-chain-rule/#comments)

[![](https://theorydish.blog/wp-content/uploads/2021/12/picture2.png?w=608&h=264&crop=1 "Picture2")](https://theorydish.blog/wp-content/uploads/2021/12/picture2.png)

The chain rule is a fundamental result in calculus. Roughly speaking, it states that if a variable ![c](https://s0.wp.com/latex.php?latex=c&bg=ffffff&fg=000000&s=0&c=20201002) is a differentiable function of intermediate variables ![b_1,\ldots,b_n](https://s0.wp.com/latex.php?latex=b_1%2C%5Cldots%2Cb_n&bg=ffffff&fg=000000&s=0&c=20201002), and each intermediate variable ![b_i](https://s0.wp.com/latex.php?latex=b_i&bg=ffffff&fg=000000&s=0&c=20201002) is itself a differentiable function of ![a](https://s0.wp.com/latex.php?latex=a&bg=ffffff&fg=000000&s=0&c=20201002), then we can compute the derivative ![\frac{{\mathrm d} c}{{\mathrm d} a}](https://s0.wp.com/latex.php?latex=%5Cfrac%7B%7B%5Cmathrm+d%7D+c%7D%7B%7B%5Cmathrm+d%7D+a%7D&bg=ffffff&fg=000000&s=0&c=20201002) as follows:

![\begin{aligned}\frac{{\mathrm d} c}{{\mathrm d} a} = \frac{{\mathrm d}}{{\mathrm d} a}c(b_1(a),\ldots,b_n(a)) = \sum_{i=1}^n \frac{\partial c}{\partial b_i}\frac{{\mathrm d} b_i}{{\mathrm d} a}. && (1)\end{aligned}](https://s0.wp.com/latex.php?latex=%5Cbegin%7Baligned%7D%5Cfrac%7B%7B%5Cmathrm+d%7D+c%7D%7B%7B%5Cmathrm+d%7D+a%7D+%3D+%5Cfrac%7B%7B%5Cmathrm+d%7D%7D%7B%7B%5Cmathrm+d%7D+a%7Dc%28b_1%28a%29%2C%5Cldots%2Cb_n%28a%29%29+%3D+%5Csum_%7Bi%3D1%7D%5En+%5Cfrac%7B%5Cpartial+c%7D%7B%5Cpartial+b_i%7D%5Cfrac%7B%7B%5Cmathrm+d%7D+b_i%7D%7B%7B%5Cmathrm+d%7D+a%7D.+%26%26+%281%29%5Cend%7Baligned%7D&bg=ffffff&fg=000000&s=0&c=20201002)

Besides being a handy tool for computing derivatives in calculus homework, the chain rule is closely related to the backpropagation algorithm that is widely-used for computing derivatives (gradients) in neural network training. This [blog post](https://windowsontheory.org/2020/11/03/yet-another-backpropagation-tutorial/) by Boaz Barak is a beautiful tutorial on the chain rule and the backpropagation algorithm.

As in Barak’s post, the backpropagation algorithm is usually taught as an application of the chain rule in machine learning classes. This leads to a common belief that “backpropagation is just applying the chain rule repeatedly”. While this is in a sense true, we wish to point out in this blog post that this belief is over-simplifying and can lead to incorrect implementations of the backpropagation algorithm. It has been discussed in many [blog](https://timvieira.github.io/blog/post/2017/08/18/backprop-is-not-just-the-chain-rule/) [posts](https://wordpress.cs.vt.edu/optml/2018/04/28/backpropagation-is-not-just-the-chain-rule/) that backpropagation is not just the chain rule, but we want to focus on a simple and basic difference here.

![](https://theorydish.blog/wp-content/uploads/2021/12/picture1-3.png)

Consider the “neural network” above: it consists of an input variable ![a](https://s0.wp.com/latex.php?latex=a&bg=ffffff&fg=000000&s=0&c=20201002), an output variable ![c](https://s0.wp.com/latex.php?latex=c&bg=ffffff&fg=000000&s=0&c=20201002), and two intermediate variables ![b_1](https://s0.wp.com/latex.php?latex=b_1&bg=ffffff&fg=000000&s=0&c=20201002) and ![b_2](https://s0.wp.com/latex.php?latex=b_2&bg=ffffff&fg=000000&s=0&c=20201002). The neural network describes ![c](https://s0.wp.com/latex.php?latex=c&bg=ffffff&fg=000000&s=0&c=20201002) as a function of ![a](https://s0.wp.com/latex.php?latex=a&bg=ffffff&fg=000000&s=0&c=20201002) in the following way:

![\begin{aligned}b_1 = 2a,\ b_2 = 3a + b_1,\ c = 4b_1 + 5b_2.\end{aligned}](https://s0.wp.com/latex.php?latex=%5Cbegin%7Baligned%7Db_1+%3D+2a%2C%5C+b_2+%3D+3a+%2B+b_1%2C%5C+c+%3D+4b_1+%2B+5b_2.%5Cend%7Baligned%7D&bg=ffffff&fg=000000&s=0&c=20201002)

Using the chain rule [(1)](#eq-1) twice, we can compute the derivative ![\frac{{\mathrm d} c}{{\mathrm d} a}](https://s0.wp.com/latex.php?latex=%5Cfrac%7B%7B%5Cmathrm+d%7D+c%7D%7B%7B%5Cmathrm+d%7D+a%7D&bg=ffffff&fg=000000&s=0&c=20201002) as follows:

![\begin{aligned} \frac{{\mathrm d} c}{{\mathrm d} a} & = \frac{\partial c}{\partial b_1}\frac{{\mathrm d} b_1}{{\mathrm d} a} + \frac{\partial c}{\partial b_2}\frac{{\mathrm d} b_2}{{\mathrm d} a} \\ & = \frac{\partial c}{\partial b_1}\frac{{\mathrm d} b_1}{{\mathrm d} a} + \frac{\partial c}{\partial b_2}\left(\frac{\partial b_2}{\partial a} \frac{{\mathrm d} a}{{\mathrm d} a} + \frac{\partial b_2}{\partial b_1} \frac{{\mathrm d} b_1}{{\mathrm d} a}\right) \\ & = 4 \times 2 + 5 \times (3\times 1 + 1\times 2) \\ & = 33. & (2) \end{aligned}](https://s0.wp.com/latex.php?latex=%5Cbegin%7Baligned%7D+%5Cfrac%7B%7B%5Cmathrm+d%7D+c%7D%7B%7B%5Cmathrm+d%7D+a%7D+%26+%3D+%5Cfrac%7B%5Cpartial+c%7D%7B%5Cpartial+b_1%7D%5Cfrac%7B%7B%5Cmathrm+d%7D+b_1%7D%7B%7B%5Cmathrm+d%7D+a%7D+%2B+%5Cfrac%7B%5Cpartial+c%7D%7B%5Cpartial+b_2%7D%5Cfrac%7B%7B%5Cmathrm+d%7D+b_2%7D%7B%7B%5Cmathrm+d%7D+a%7D+%5C%5C+%26+%3D+%5Cfrac%7B%5Cpartial+c%7D%7B%5Cpartial+b_1%7D%5Cfrac%7B%7B%5Cmathrm+d%7D+b_1%7D%7B%7B%5Cmathrm+d%7D+a%7D+%2B+%5Cfrac%7B%5Cpartial+c%7D%7B%5Cpartial+b_2%7D%5Cleft%28%5Cfrac%7B%5Cpartial+b_2%7D%7B%5Cpartial+a%7D+%5Cfrac%7B%7B%5Cmathrm+d%7D+a%7D%7B%7B%5Cmathrm+d%7D+a%7D+%2B+%5Cfrac%7B%5Cpartial+b_2%7D%7B%5Cpartial+b_1%7D+%5Cfrac%7B%7B%5Cmathrm+d%7D+b_1%7D%7B%7B%5Cmathrm+d%7D+a%7D%5Cright%29+%5C%5C+%26+%3D+4+%5Ctimes+2+%2B+5+%5Ctimes+%283%5Ctimes+1+%2B+1%5Ctimes+2%29+%5C%5C+%26+%3D+33.+%26+%282%29+%5Cend%7Baligned%7D&bg=ffffff&fg=000000&s=0&c=20201002)

Backpropagation computes the derivative ![\frac{{\mathrm d} c}{{\mathrm d} a}](https://s0.wp.com/latex.php?latex=%5Cfrac%7B%7B%5Cmathrm+d%7D+c%7D%7B%7B%5Cmathrm+d%7D+a%7D&bg=ffffff&fg=000000&s=0&c=20201002) via a different route:

![\begin{aligned}\frac{{\mathrm d} c}{{\mathrm d} a} & = \frac{{\mathrm d} c}{{\mathrm d} b_1}\frac{\partial b_1}{\partial a} + \frac{{\mathrm d} c}{{\mathrm d} b_2}\frac{\partial b_2}{\partial a} & (3) \\ & = \left(\frac{{\mathrm d} c}{{\mathrm d} b_2}\frac{\partial b_2}{\partial b_1} + \frac{{\mathrm d} c}{{\mathrm d} c}\frac{\partial c}{\partial b_1}\right)\frac{\partial b_1}{\partial a} + \frac{{\mathrm d} c}{{\mathrm d} b_2}\frac{\partial b_2}{\partial a} \\ & = (5\times 1+ 1\times 4)\times 2 + 5\times 3 \\ & = 33. \end{aligned}](https://s0.wp.com/latex.php?latex=%5Cbegin%7Baligned%7D%5Cfrac%7B%7B%5Cmathrm+d%7D+c%7D%7B%7B%5Cmathrm+d%7D+a%7D+%26+%3D+%5Cfrac%7B%7B%5Cmathrm+d%7D+c%7D%7B%7B%5Cmathrm+d%7D+b_1%7D%5Cfrac%7B%5Cpartial+b_1%7D%7B%5Cpartial+a%7D+%2B+%5Cfrac%7B%7B%5Cmathrm+d%7D+c%7D%7B%7B%5Cmathrm+d%7D+b_2%7D%5Cfrac%7B%5Cpartial+b_2%7D%7B%5Cpartial+a%7D+%26+%283%29+%5C%5C+%26+%3D+%5Cleft%28%5Cfrac%7B%7B%5Cmathrm+d%7D+c%7D%7B%7B%5Cmathrm+d%7D+b_2%7D%5Cfrac%7B%5Cpartial+b_2%7D%7B%5Cpartial+b_1%7D+%2B+%5Cfrac%7B%7B%5Cmathrm+d%7D+c%7D%7B%7B%5Cmathrm+d%7D+c%7D%5Cfrac%7B%5Cpartial+c%7D%7B%5Cpartial+b_1%7D%5Cright%29%5Cfrac%7B%5Cpartial+b_1%7D%7B%5Cpartial+a%7D+%2B+%5Cfrac%7B%7B%5Cmathrm+d%7D+c%7D%7B%7B%5Cmathrm+d%7D+b_2%7D%5Cfrac%7B%5Cpartial+b_2%7D%7B%5Cpartial+a%7D+%5C%5C+%26+%3D+%285%5Ctimes+1%2B+1%5Ctimes+4%29%5Ctimes+2+%2B+5%5Ctimes+3+%5C%5C+%26+%3D+33.+%5Cend%7Baligned%7D&bg=ffffff&fg=000000&s=0&c=20201002)

In the calculations above, note the difference between partial and full derivatives. For example, the partial derivative ![\frac{\partial b_2}{\partial a} = 3](https://s0.wp.com/latex.php?latex=%5Cfrac%7B%5Cpartial+b_2%7D%7B%5Cpartial+a%7D+%3D+3&bg=ffffff&fg=000000&s=0&c=20201002), whereas the full derivative ![\frac{{\mathrm d} b_2}{{\mathrm d} a} = 3\times 1 + 1\times 2 = 5](https://s0.wp.com/latex.php?latex=%5Cfrac%7B%7B%5Cmathrm+d%7D+b_2%7D%7B%7B%5Cmathrm+d%7D+a%7D+%3D+3%5Ctimes+1+%2B+1%5Ctimes+2+%3D+5&bg=ffffff&fg=000000&s=0&c=20201002). Similarly, the partial derivative ![\frac{\partial c}{\partial b_1} = 4](https://s0.wp.com/latex.php?latex=%5Cfrac%7B%5Cpartial+c%7D%7B%5Cpartial+b_1%7D+%3D+4&bg=ffffff&fg=000000&s=0&c=20201002), whereas the full derivative ![\frac{{\mathrm d} c}{{\mathrm d} b_1} = 5\times 1 + 1\times 4 = 9](https://s0.wp.com/latex.php?latex=%5Cfrac%7B%7B%5Cmathrm+d%7D+c%7D%7B%7B%5Cmathrm+d%7D+b_1%7D+%3D+5%5Ctimes+1+%2B+1%5Ctimes+4+%3D+9&bg=ffffff&fg=000000&s=0&c=20201002). Intuitively, for every edge in the graph, we can compute the corresponding partial derivative locally (after the *forward pass*, to be precise), but a full derivative may require longer calculations.

Now it should be clear that equation [(3)](#eq-3) cannot be *directly* explained as the standard chain rule [(1)](#eq-1). The key difference is that in the chain rule, we need partial derivatives of a single variable w.r.t. multiple other variables, whereas in [(3)](#eq-3), we need partial derivatives of multiple variables w.r.t. the same variable.

Of course, one can prove the correctness of backpropagation using the chain rule in various ways, but the simple proof “backpropagation uses the standard chain rule at every step” is incomplete. Also, it is certainly possible to compute derivatives (gradients) on a neural network directly using the chain rule similarly to [(2)](#eq-2), but in neural network training one typically wants to calculate the derivatives of a single output variable w.r.t. a large number of input variables, in which case backpropagation allows a more efficient implementation than using the standard chain rule directly.

## The real chain rule in actual backpropagation

When implementing the backpropagation algorithm, it is more convenient and efficient to add in the two terms (or any number of terms in general) on the right-hand-side of [(3)](#eq-3) at *different* steps of the algorithm. This is described in Barak’s [tutorial](https://windowsontheory.org/2020/11/03/yet-another-backpropagation-tutorial/), which also has an actual Python implementation! In contrast to the failure of naively explaining equation [(3)](#eq-3) as the chain rule, there is a way to explain this implementation using the chain rule directly.

To describe the implementation, suppose ![b_1,\ldots,b_n](https://s0.wp.com/latex.php?latex=b_1%2C%5Cldots%2Cb_n&bg=ffffff&fg=000000&s=0&c=20201002) are all the variables, including input variables ![b_1,\ldots,b_m](https://s0.wp.com/latex.php?latex=b_1%2C%5Cldots%2Cb_m&bg=ffffff&fg=000000&s=0&c=20201002), intermediate variables ![b_{m+1},\ldots,b_{n-1}](https://s0.wp.com/latex.php?latex=b_%7Bm%2B1%7D%2C%5Cldots%2Cb_%7Bn-1%7D&bg=ffffff&fg=000000&s=0&c=20201002), and the only output variable ![c = b_n](https://s0.wp.com/latex.php?latex=c+%3D+b_n&bg=ffffff&fg=000000&s=0&c=20201002). Assume the variables are arranged in topological order: for every ![i = m+1,\ldots,n](https://s0.wp.com/latex.php?latex=i+%3D+m%2B1%2C%5Cldots%2Cn&bg=ffffff&fg=000000&s=0&c=20201002), variable ![b_i](https://s0.wp.com/latex.php?latex=b_i&bg=ffffff&fg=000000&s=0&c=20201002) is locally a function of variables ![b_j](https://s0.wp.com/latex.php?latex=b_j&bg=ffffff&fg=000000&s=0&c=20201002) with ![j < i](https://s0.wp.com/latex.php?latex=j+%3C+i&bg=ffffff&fg=000000&s=0&c=20201002). We can write this as ![b_i = b_i(b_1,\ldots,b_{i-1})](https://s0.wp.com/latex.php?latex=b_i+%3D+b_i%28b_1%2C%5Cldots%2Cb_%7Bi-1%7D%29&bg=ffffff&fg=000000&s=0&c=20201002). Note that some variables ![b_j](https://s0.wp.com/latex.php?latex=b_j&bg=ffffff&fg=000000&s=0&c=20201002) with ![j < i](https://s0.wp.com/latex.php?latex=j+%3C+i&bg=ffffff&fg=000000&s=0&c=20201002) may not have an edge to ![b_i](https://s0.wp.com/latex.php?latex=b_i&bg=ffffff&fg=000000&s=0&c=20201002), in which case ![\frac{\partial b_i}{\partial b_j} = 0](https://s0.wp.com/latex.php?latex=%5Cfrac%7B%5Cpartial+b_i%7D%7B%5Cpartial+b_j%7D+%3D+0&bg=ffffff&fg=000000&s=0&c=20201002).

For each variable ![b_i](https://s0.wp.com/latex.php?latex=b_i&bg=ffffff&fg=000000&s=0&c=20201002), backpropagation stores in ![{\mathsf {grad}}_i](https://s0.wp.com/latex.php?latex=%7B%5Cmathsf+%7Bgrad%7D%7D_i&bg=ffffff&fg=000000&s=0&c=20201002) a “temporary derivative/gradient” w.r.t. ![b_i](https://s0.wp.com/latex.php?latex=b_i&bg=ffffff&fg=000000&s=0&c=20201002). Initially, ![{\mathsf {grad}}_n = 1](https://s0.wp.com/latex.php?latex=%7B%5Cmathsf+%7Bgrad%7D%7D_n+%3D+1&bg=ffffff&fg=000000&s=0&c=20201002) and ![{\mathsf {grad}}_i = 0](https://s0.wp.com/latex.php?latex=%7B%5Cmathsf+%7Bgrad%7D%7D_i+%3D+0&bg=ffffff&fg=000000&s=0&c=20201002) for ![i < n](https://s0.wp.com/latex.php?latex=i+%3C+n&bg=ffffff&fg=000000&s=0&c=20201002). The backpropagation algorithm iterates over ![i = n,n-1,\ldots,m+1](https://s0.wp.com/latex.php?latex=i+%3D+n%2Cn-1%2C%5Cldots%2Cm%2B1&bg=ffffff&fg=000000&s=0&c=20201002) and performs the following updates in each iteration:

![\begin{aligned}{\mathsf {grad}}_j \gets {\mathsf {grad}}_j + {\mathsf {grad}}_i \cdot \frac{\partial b_i}{\partial b_j} ,\ \textnormal{for all}\ j = 1,\ldots,i-1. && (4) \end{aligned}](https://s0.wp.com/latex.php?latex=%5Cbegin%7Baligned%7D%7B%5Cmathsf+%7Bgrad%7D%7D_j+%5Cgets+%7B%5Cmathsf+%7Bgrad%7D%7D_j+%2B+%7B%5Cmathsf+%7Bgrad%7D%7D_i+%5Ccdot+%5Cfrac%7B%5Cpartial+b_i%7D%7B%5Cpartial+b_j%7D+%2C%5C+%5Ctextnormal%7Bfor+all%7D%5C+j+%3D+1%2C%5Cldots%2Ci-1.+%26%26+%284%29+%5Cend%7Baligned%7D&bg=ffffff&fg=000000&s=0&c=20201002)

Of course, it suffices to update ![{\mathsf {grad}}_j](https://s0.wp.com/latex.php?latex=%7B%5Cmathsf+%7Bgrad%7D%7D_j&bg=ffffff&fg=000000&s=0&c=20201002) only when there is an edge from ![b_j](https://s0.wp.com/latex.php?latex=b_j&bg=ffffff&fg=000000&s=0&c=20201002) to ![b_i](https://s0.wp.com/latex.php?latex=b_i&bg=ffffff&fg=000000&s=0&c=20201002), because otherwise ![\frac{\partial b_i}{\partial b_j} = 0](https://s0.wp.com/latex.php?latex=%5Cfrac%7B%5Cpartial+b_i%7D%7B%5Cpartial+b_j%7D+%3D+0&bg=ffffff&fg=000000&s=0&c=20201002) and the update does not change ![{\mathsf {grad}}_j](https://s0.wp.com/latex.php?latex=%7B%5Cmathsf+%7Bgrad%7D%7D_j&bg=ffffff&fg=000000&s=0&c=20201002).

The update rule [(4)](#eq-4) can be explained as a direct application of the chain rule as follows. If we know the values of ![b_1,\ldots,b_i](https://s0.wp.com/latex.php?latex=b_1%2C%5Cldots%2Cb_i&bg=ffffff&fg=000000&s=0&c=20201002), we can evaluate the remaining variables ![b_{i+1},\ldots,b_n](https://s0.wp.com/latex.php?latex=b_%7Bi%2B1%7D%2C%5Cldots%2Cb_n&bg=ffffff&fg=000000&s=0&c=20201002) one by one, so in this sense, the output variable ![c = b_n](https://s0.wp.com/latex.php?latex=c+%3D+b_n&bg=ffffff&fg=000000&s=0&c=20201002) is a function ![c_i](https://s0.wp.com/latex.php?latex=c_i&bg=ffffff&fg=000000&s=0&c=20201002) of ![b_1,\ldots,b_i](https://s0.wp.com/latex.php?latex=b_1%2C%5Cldots%2Cb_i&bg=ffffff&fg=000000&s=0&c=20201002). Using the function ![b_i = b_i(b_1,\ldots,b_{i-1})](https://s0.wp.com/latex.php?latex=b_i+%3D+b_i%28b_1%2C%5Cldots%2Cb_%7Bi-1%7D%29&bg=ffffff&fg=000000&s=0&c=20201002), we can relate functions ![c_i](https://s0.wp.com/latex.php?latex=c_i&bg=ffffff&fg=000000&s=0&c=20201002) and ![c_{i-1}](https://s0.wp.com/latex.php?latex=c_%7Bi-1%7D&bg=ffffff&fg=000000&s=0&c=20201002) as follows:

![\begin{aligned}c_{i-1}(b_1,\ldots,b_{i-1}) = c_i(b_1,\ldots,b_{i-1}, b_i(b_1,\ldots,b_{i-1})).\end{aligned}](https://s0.wp.com/latex.php?latex=%5Cbegin%7Baligned%7Dc_%7Bi-1%7D%28b_1%2C%5Cldots%2Cb_%7Bi-1%7D%29+%3D+c_i%28b_1%2C%5Cldots%2Cb_%7Bi-1%7D%2C+b_i%28b_1%2C%5Cldots%2Cb_%7Bi-1%7D%29%29.%5Cend%7Baligned%7D&bg=ffffff&fg=000000&s=0&c=20201002)

By the chain rule,

![\begin{aligned}\frac{\partial c_{i-1}}{\partial b_j} = \frac{\partial c_i}{\partial b_j} + \frac{\partial c_i}{\partial b_i} \cdot \frac{\partial b_i}{\partial b_j},\ \text{for all}\ j = 1,\ldots, i-1. && (5) \end{aligned}](https://s0.wp.com/latex.php?latex=%5Cbegin%7Baligned%7D%5Cfrac%7B%5Cpartial+c_%7Bi-1%7D%7D%7B%5Cpartial+b_j%7D+%3D+%5Cfrac%7B%5Cpartial+c_i%7D%7B%5Cpartial+b_j%7D+%2B+%5Cfrac%7B%5Cpartial+c_i%7D%7B%5Cpartial+b_i%7D+%5Ccdot+%5Cfrac%7B%5Cpartial+b_i%7D%7B%5Cpartial+b_j%7D%2C%5C+%5Ctext%7Bfor+all%7D%5C+j+%3D+1%2C%5Cldots%2C+i-1.+%26%26+%285%29+%5Cend%7Baligned%7D&bg=ffffff&fg=000000&s=0&c=20201002)

The correspondence between [(4)](#eq-4) and [(5)](#eq-5) completes the explanation: if ![{\mathsf {grad}}_j = \frac{\partial c_i}{\partial b_j}](https://s0.wp.com/latex.php?latex=%7B%5Cmathsf+%7Bgrad%7D%7D_j+%3D+%5Cfrac%7B%5Cpartial+c_i%7D%7B%5Cpartial+b_j%7D&bg=ffffff&fg=000000&s=0&c=20201002) for every ![j \le i](https://s0.wp.com/latex.php?latex=j+%5Cle+i&bg=ffffff&fg=000000&s=0&c=20201002) before the update, we have ![{\mathsf {grad}}_j = \frac{\partial c_{i-1}}{\partial b_j}](https://s0.wp.com/latex.php?latex=%7B%5Cmathsf+%7Bgrad%7D%7D_j+%3D+%5Cfrac%7B%5Cpartial+c_%7Bi-1%7D%7D%7B%5Cpartial+b_j%7D&bg=ffffff&fg=000000&s=0&c=20201002) for every ![j < i](https://s0.wp.com/latex.php?latex=j+%3C+i&bg=ffffff&fg=000000&s=0&c=20201002) after the update. By induction, after the final iteration with ![i = m+1](https://s0.wp.com/latex.php?latex=i+%3D+m%2B1&bg=ffffff&fg=000000&s=0&c=20201002), ![{\mathsf {grad}}_j](https://s0.wp.com/latex.php?latex=%7B%5Cmathsf+%7Bgrad%7D%7D_j&bg=ffffff&fg=000000&s=0&c=20201002) contains the desired value ![\frac{\partial c_m}{\partial b_j}](https://s0.wp.com/latex.php?latex=%5Cfrac%7B%5Cpartial+c_m%7D%7B%5Cpartial+b_j%7D&bg=ffffff&fg=000000&s=0&c=20201002) for every ![j = 1,\ldots,m](https://s0.wp.com/latex.php?latex=j+%3D+1%2C%5Cldots%2Cm&bg=ffffff&fg=000000&s=0&c=20201002), where ![c = c_{m}(b_1,\ldots,b_m)](https://s0.wp.com/latex.php?latex=c+%3D+c_%7Bm%7D%28b_1%2C%5Cldots%2Cb_m%29&bg=ffffff&fg=000000&s=0&c=20201002) is exactly the function whose derivatives we want to compute.

### Share this:

* [Share on X (Opens in new window)
  X](https://theorydish.blog/2021/12/16/backpropagation-%E2%89%A0-chain-rule/?share=twitter)
* [Share on Facebook (Opens in new window)
  Facebook](https://theorydish.blog/2021/12/16/backpropagation-%E2%89%A0-chain-rule/?share=facebook)

Like Loading...

### *Related*

#### 3 Comments on Backpropagation ≠ Chain Rule

1. ![Unknown's avatar](https://2.gravatar.com/avatar/e7fcf5506bd871a2894c76c02118fcc72ba9639936faf88ea4cbaab5af2c572f?s=30&d=identicon&r=G) [Boaz Barak](http://www.boazbarak.org) //
   [December 16, 2021 at 8:00 pm](https://theorydish.blog/2021/12/16/backpropagation-%E2%89%A0-chain-rule/#comment-204330) //
   [Reply](https://theorydish.blog/2021/12/16/backpropagation-%E2%89%A0-chain-rule/?replytocom=204330#respond)

   Thanks! I think in theory terms, the way to describe it is as follows: if you follow the chain rule in the standard “forward” direction as we learned in basic calculus, you will pay a price that scales in the \*formula size\* for the output.

   Backpropagation allows us to pay a price that only scales with the \*circuit size\*. This is what enables automatic differentiation since a computation graph is simply a circuit.

   I still maintain that it’s the (multivariate) chain rule, but it applied in a clever way.

   [Like](https://theorydish.blog/2021/12/16/backpropagation-%E2%89%A0-chain-rule/?like_comment=204330&_wpnonce=4423dcb31e)Liked by 1 person

   * ![Unknown's avatar](https://1.gravatar.com/avatar/a6a6519f182e3a2288e847feb14032cc3e26472563e418725a67727a24f433a0?s=30&d=identicon&r=G) Lunjia Hu //
     [December 16, 2021 at 9:53 pm](https://theorydish.blog/2021/12/16/backpropagation-%E2%89%A0-chain-rule/#comment-204334) //
     [Reply](https://theorydish.blog/2021/12/16/backpropagation-%E2%89%A0-chain-rule/?replytocom=204334#respond)

     Thanks for the reply! Just to make a small clarification: if a circuit has only one input variable, there is a way to implement the “forward chain rule” algorithm efficiently (with “time” complexity proportional to the circuit size) via dynamic programming, no matter how many output variables we have. When there are m input variables, doing this for each input variable achieves complexity roughly m \* circuit size, which can be much smaller than the formula size. By operating in the reversed direction, backpropagation replaces m by the number of \*output\* variables. This is a significant improvement because there is usually only one output variable but a huge number m of input variables. To achieve this, backpropagation applies a reversed version of the chain rule like in (3). The difference between the reversed version and the standard version is usually neglected, partly because they are the “same thing” if, for example, there is no edge from b\_1 to b\_2 in the example. Adding the edge highlights the difference.

     One purpose of this blog post is to show that backpropogation is an arguably non-trivial algorithm, despite its simplicity (which is a lesson that I failed to learn when I was an undergrad…) Thank you for your contribution in helping more people appreciate the ideas behind the backpropagation algorithm!

     [Like](https://theorydish.blog/2021/12/16/backpropagation-%E2%89%A0-chain-rule/?like_comment=204334&_wpnonce=e6030ce579)Like

     + ![Unknown's avatar](https://2.gravatar.com/avatar/e7fcf5506bd871a2894c76c02118fcc72ba9639936faf88ea4cbaab5af2c572f?s=30&d=identicon&r=G) [Boaz Barak](http://www.boazbarak.org) //
       [December 17, 2021 at 3:27 am](https://theorydish.blog/2021/12/16/backpropagation-%E2%89%A0-chain-rule/#comment-204344) //

       Thank you! That’s a good point. Note also that m \* circuit\_size complexity can be achieved via numerical differentiation

       [Like](https://theorydish.blog/2021/12/16/backpropagation-%E2%89%A0-chain-rule/?like_comment=204344&_wpnonce=a14f238b9b)Liked by 1 person

#### 1 Trackback / Pingback

1. [Yet another backpropagation tutorial – Windows On Theory](http://windowsontheory.org/2020/11/03/yet-another-backpropagation-tutorial/)

### Leave a comment [Cancel reply](/2021/12/16/backpropagation-%E2%89%A0-chain-rule/#respond)

Δ

[Follow Theory Dish on WordPress.com](https://theorydish.blog)

* [RSS - Posts](https://theorydish.blog/feed/ "Subscribe to Posts")
* [RSS - Comments](https://theorydish.blog/comments/feed/ "Subscribe to Comments")

#### Recent Comments

|  |  |
| --- | --- |
| [![Roy Andrews's avatar](https://1.gravatar.com/avatar/7f15de8340cc2a2b284e3d3ae7a9f2d53aa78fb34e495d8a40097778807e5b32?s=48&d=identicon&r=G)](https://www.royandrews.com/) | [Roy Andrews](https://www.royandrews.com/) on [Prediction with a short m…](https://theorydish.blog/2017/11/29/prediction-with-a-short-memory/comment-page-1/#comment-226366) |
|  | [Multiparty Computati…](https://digitalfinancenews.com/research-reports/multiparty-computation-foundations-applications-and-cryptographic-techniques/) on [A few lessons from the history…](https://theorydish.blog/2021/05/26/few-lessons-from-the-history-of-multiparty-computation/comment-page-1/#comment-226365) |
| ![Arick Grootveld's avatar](https://1.gravatar.com/avatar/a309471110dc88c6c3c8ba52a95225f9ed3f3d2677929cc57bc5ae46ccea1e0b?s=48&d=identicon&r=G) | Arick Grootveld on [Trace Reconstruction from Comp…](https://theorydish.blog/2021/06/29/trace-reconstruction/comment-page-1/#comment-226352) |
| [![Unknown's avatar](https://theorydish.blog/wp-content/uploads/2017/03/cropped-nightdish1.jpg?w=48)](https://theorydish.blog/2023/09/05/dnf-minimization-part-ii/) | [DNF Minimization, Pa…](https://theorydish.blog/2023/09/05/dnf-minimization-part-ii/) on [DNF Minimization, Part I](https://theorydish.blog/2023/08/28/dnf-minimization-part-i/comment-page-1/#comment-226329) |
| ![Hao Sun's avatar](https://graph.facebook.com/v6.0/10208712753015118/picture?type=large) | Hao Sun on [RANDOM & APPROX 2023](https://theorydish.blog/2023/03/25/random-approx-2023/comment-page-1/#comment-223048) |
|  | [Yet another backprop…](http://windowsontheory.org/2020/11/03/yet-another-backpropagation-tutorial/) on [Backpropagation ≠ Chain R…](https://theorydish.blog/2021/12/16/backpropagation-%E2%89%A0-chain-rule/comment-page-1/#comment-204349) |
| [![Boaz Barak's avatar](https://2.gravatar.com/avatar/e7fcf5506bd871a2894c76c02118fcc72ba9639936faf88ea4cbaab5af2c572f?s=48&d=identicon&r=G)](http://www.boazbarak.org) | [Boaz Barak](http://www.boazbarak.org) on [Backpropagation ≠ Chain R…](https://theorydish.blog/2021/12/16/backpropagation-%E2%89%A0-chain-rule/comment-page-1/#comment-204344) |
| ![Lunjia Hu's avatar](https://2.gravatar.com/avatar/56f2e0fad03202f62e33ccffef369544a96dcf39b92c297ee0954d53bb57ed74?s=48&d=identicon&r=G) | Lunjia Hu on [Backpropagation ≠ Chain R…](https://theorydish.blog/2021/12/16/backpropagation-%E2%89%A0-chain-rule/comment-page-1/#comment-204334) |
| [![Boaz Barak's avatar](https://2.gravatar.com/avatar/e7fcf5506bd871a2894c76c02118fcc72ba9639936faf88ea4cbaab5af2c572f?s=48&d=identicon&r=G)](http://www.boazbarak.org) | [Boaz Barak](http://www.boazbarak.org) on [Backpropagation ≠ Chain R…](https://theorydish.blog/2021/12/16/backpropagation-%E2%89%A0-chain-rule/comment-page-1/#comment-204330) |
| ![kodlu's avatar](https://2.gravatar.com/avatar/bd55a766f82092a26bc17ada5c639843fb8b6f894285ad48b33592c67a5482d0?s=48&d=identicon&r=G) | kodlu on [Average-Case Fine-Grained Hard…](https://theorydish.blog/2021/07/23/average-case-fine-grained-hardness-part-i/comment-page-1/#comment-189650) |

Copyright © 2026
[Create a website or blog at WordPress.com](https://wordpress.com/?ref=footer_custom_svg "Create a website or blog at WordPress.com")

* [Comment](https://theorydish.blog/2021/12/16/backpropagation-%E2%89%A0-chain-rule/#comments)
* Reblog
* Subscribe
  Subscribed

  + [![](https://theorydish.blog/wp-content/uploads/2017/03/cropped-nightdish1.jpg?w=50) Theory Dish](https://theorydish.blog)

  Join 80 other subscribers

  Sign me up

  + Already have a WordPress.com account? [Log in now.](https://wordpress.com/log-in?redirect_to=https%3A%2F%2Fr-login.wordpress.com%2Fremote-login.php%3Faction%3Dlink%26back%3Dhttps%253A%252F%252Ftheorydish.blog%252F2021%252F12%252F16%252Fbackpropagation-%2525e2%252589%2525a0-chain-rule%252F)
* + [![](https://theorydish.blog/wp-content/uploads/2017/03/cropped-nightdish1.jpg?w=50) Theory Dish](https://theorydish.blog)
  + Subscribe
    Subscribed
  + [Sign up](https://wordpress.com/start/)
  + [Log in](https://wordpress.com/log-in?redirect_to=https%3A%2F%2Fr-login.wordpress.com%2Fremote-login.php%3Faction%3Dlink%26back%3Dhttps%253A%252F%252Ftheorydish.blog%252F2021%252F12%252F16%252Fbackpropagation-%2525e2%252589%2525a0-chain-rule%252F)
  + [Copy shortlink](https://wp.me/p8xbPW-IO)
  + [Report this content](https://wordpress.com/abuse/?report_url=https://theorydish.blog/2021/12/16/backpropagation-%e2%89%a0-chain-rule/)
  + [View post in Reader](https://wordpress.com/reader/blogs/126121016/posts/2778)
  + [Manage subscriptions](https://subscribe.wordpress.com/)
  + Collapse this bar

Loading Comments...

Write a Comment...

Email (Required)

Name (Required)

Website

%d

![](https://pixel.wp.com/b.gif?v=noscript)