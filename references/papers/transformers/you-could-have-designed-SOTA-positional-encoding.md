[# fleetwood.dev](/)

[Essays](/)[Reads](/reads)

# You could have designed state of the art Positional Encoding

Rotary Position Encoding (RoPE) can seem intimidating at first. This post endeavours to rediscover RoPE by iteratively improving upon proposed positional encoding schemes.

Published:

17/11/2024

⏱ 21 Min Read

AIPedagogy

> **Gall's Law**
> A complex system that works is invariably found to have evolved from a simple
> system that worked
> — John Gall

This post walks you through the step-by-step discovery of state-of-the-art positional encoding in transformer models. We will achieve
this by iteratively improving our approach to encoding position, arriving at **Ro**tary **P**ostional **E**ncoding (RoPE) used in the latest [LLama 3.2](https://ai.meta.com/blog/llama-3-2-connect-2024-vision-edge-mobile-devices/) release and most modern transformers. This post intends to limit the mathematical knowledge required to follow along, but some basic linear algebra, trigonometry and understanding of self attention is expected.

## [Problem Statement](#problem-statement)

> You shall know a word by the company it keeps
> — John Rupert Firth

As with all problems, it is best to first start with understanding **exactly** what we are trying to achieve. The self attention mechanism in transformers is utilized to understand relationships
between tokens in a sequence. Self attention is a **set** operation, which
means it is **permutation equivariant**. If we do not
enrich self attention with positional information, many important relationships are
**incapable of being determined**.

This is best demonstrated by example.

## [Motivating Example](#motivating-example)

Consider this sentence with the same word in different positions:

The dog chased another dog\text{The dog chased another dog}The dog chased another dog

Intuitively, "dog" refers to two different entities. Let's see what happens if we first tokenize them, map to the real token embeddings of **Llama 3.2 1B** and pass them through [torch.nn.MultiheadAttention](https://pytorch.org/docs/stable/generated/torch.nn.MultiheadAttention.html).

```
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel

model_id = "meta-llama/Llama-3.2-1B"
tok = AutoTokenizer.from_pretrained(model_id)
model = AutoModel.from_pretrained(model_id)

text = "The dog chased another dog"
tokens = tok(text, return_tensors="pt")["input_ids"]
embeddings = model.embed_tokens(tokens)
hdim = embeddings.shape[-1]

W_q = nn.Linear(hdim, hdim, bias=False)
W_k = nn.Linear(hdim, hdim, bias=False)
W_v = nn.Linear(hdim, hdim, bias=False)
mha = nn.MultiheadAttention(embed_dim=hdim, num_heads=4, batch_first=True)

with torch.no_grad():
    for param in mha.parameters():
        nn.init.normal_(param, std=0.1) # Initialize weights to be non-negligible

output, _ = mha(W_q(embeddings), W_k(embeddings), W_v(embeddings))

dog1_out = output[0, 2]
dog2_out = output[0, 5]
print(f"Dog output identical?: {torch.allclose(dog1_out, dog2_out, atol=1e-6)}") #True
```

As we can see, without any positional information, the output of a (multi
headed) self attention operation is **identical for the same token in
different positions**, despite the tokens clearly representing distinct entities. Let's begin designing a method of enhancing self attention with positional information, such that it can determine relationships between words encoded by
their positions.

To understand and design an optimal encoding scheme, let's explore some desirable properties such a scheme should have.

## [Desirable Properties](#desirable-properties)

Let's try and define some desirable properties that will make the optimization
process as easy as possible.

#### [Property 1 - Unique encoding for each position (across sequences)](#property-1---unique-encoding-for-each-position-across-sequences)

Each position needs a unique encoding that remains consistent regardless of sequence length - a token at position 5 should have the same encoding whether the current sequence is of length 10 or 10,000.

#### [Property 2 - Linear relation between two encoded positions](#property-2---linear-relation-between-two-encoded-positions)

The relationship between positions should be mathematically simple. If we know the encoding for position ppp, it should be straightforward to compute the encoding for position p+kp+kp+k, making it easier for the model to learn positional patterns.

If you think about how we represent numbers on a number line, it's easy to understand that 5 is 2 steps away from 3, or that 10 is 5 steps from 15. The same intuitive relationship should exist in our encodings.

#### [Property 3 - Generalizes to longer sequences than those encountered in training](#property-3---generalizes-to-longer-sequences-than-those-encountered-in-training)

To increase our models' utility in the real world, they should generalize outside
their training distribution. Therefore, our encoding scheme needs to be
adaptable enough to handle unexpected input lengths, without
violating any of our other desirable properties.

#### [Property 4 - Generated by a deterministic process the model can learn](#property-4---generated-by-a-deterministic-process-the-model-can-learn)

It would be ideal if our positional encodings could be drawn from a
deterministic process. This should allow the model to learn the mechanism
behind our encoding scheme efficiently.

#### [Property 5 - Extensible to multiple dimensions](#property-5---extensible-to-multiple-dimensions)

With multimodal models becoming the norm, it is crucial that our positional
encoding scheme can naturally extend from 1D1D1D to nDnDnD. This will allow models to consume data like images or brain scans, which are 2D2D2D and 4D4D4D
respectively.

Now we know the ideal properties (henceforth referred to as PrnPr\_nPrn​), let's start designing and iterating on our encoding scheme.

## [Integer Position Encoding](#integer-position-encoding)

The first approach that may jump to mind is simply to add the integer value of the token position to each component of the token embedding, with values ranging from 0→L0 \rightarrow L0→L where LLL is the
length of our current sequence.

[Your browser does not support the video tag.](/positional-encoding/IntegerEncoding.mp4)

In the above animation, we create our positional encoding vector for the token chased\color{#699C52}\text{chased}chased from the index and add it to our token embedding. The embedding values here are a subset of the real values from **Llama 3.2 1B**. We can observe that they're clustered around 0. This
is desirable to avoid [vanishing or exploding gradients](https://www.cs.toronto.edu/~rgrosse/courses/csc321_2017/readings/L15%20Exploding%20and%20Vanishing%20Gradients.pdf) during training and therefore is something we'd like to maintain throughout the model.

It's clear that our current naïve approach is going to cause problems. The magnitude of the position value
vastly exceeds the actual values of our input. This means the signal-to-noise
ratio is very low, and it's hard for the model to separate the semantic
information from the positional information.

With this new knowledge, a natural follow on might be to normalize the position value by 1N\frac{1}{N}N1​. This constrains the values between 0 and 1, but introduces another problem. If we choose NNN to be the length of the current sequence, then the position values will be completely different for each sequence of differing lengths, violating Pr1Pr\_1Pr1​.

Is there a better way to ensure our numbers are between 0 and 1?
If we thought really hard about this for a while, we might come up with switching
from decimal to binary numbers.

## [Binary Position Encoding](#binary-position-encoding)

Instead of adding our (potentially normalized) integer position to each
component of the embedding, we could instead convert it into its binary
representation and *s t r e t c h* our value out to match our embedding dimension, as demonstrated below.

[Your browser does not support the video tag.](/positional-encoding/BinaryEncoding.mp4)

We've converted the position of interest (252) into its binary representation
(11111100) and added each bit to the corresponding component of the
token embedding. The least significant bit (LSB) will cycle between 0 and 1 for every
subsequent token, whilst the most significant bit (MSB) will cycle every
2n−12^{n-1}2n−1 tokens where nnn is the number of bits.
You can see the positional encoding vector for different indices in the animation below [1](#user-content-fn-1).

[Your browser does not support the video tag.](/positional-encoding/BinaryPositionalEncodingPlot.mp4)

We've solved the value range problem, and we now have unique encodings that are
consistent across different sequence lengths. What happens if we plot a low dimensional version of our token embedding and visualize the addition of our binary positional vector for different values.

[Your browser does not support the video tag.](/positional-encoding/BinaryVector3D.mp4)

We can see that the result is very "jumpy" (as we might expect from the
discrete nature of binary). The optimization process likes smooth, continuous and
predictable changes. Do we know any functions with similar value ranges that are smooth and continuous?

If we looked around a little, we might notice that both sin⁡\sinsin and cos⁡\coscos fit the bill!

## [Sinusoidal positional encoding](#sinusoidal-positional-encoding)

[Your browser does not support the video tag.](/positional-encoding/SteppedPositionalEncodingPlot.mp4)

The above animation visualizes our position embedding if each component is
alternatively drawn from sin⁡\sinsin and cos⁡\coscos with gradually increasing
wavelengths. If you compare it with the previous animation, you'll notice a striking similarity!

We've now arrived at Sinusoidal embeddings; originally defined in the [Attention is all you need](https://arxiv.org/abs/1706.03762) paper.
Let's look at the equations:

PE(pos,2i)=sin⁡(pos100002i/d)PE(pos,2i+1)=cos⁡(pos100002i/d)PE\_{(pos,2i)} = \color{#58C4DD}\sin\left(\color{black}\frac{pos}{10000^{2i/d}}\color{#58C4DD}\right)\color{black} \\
\quad \\
PE\_{(pos,2i+1)} = \color{#FC6255}\cos\left(\color{black}\frac{pos}{10000^{2i/d}}\color{#FC6255}\right)\color{black} \\ PE(pos,2i)​=sin(100002i/dpos​)PE(pos,2i+1)​=cos(100002i/dpos​)

where pospospos is the tokens position index, iii is the component index
in the positional encoding vector, and ddd is the model dimension. 10,00010,00010,000 is the **base wavelength** (henceforth referred to as
θ\thetaθ), which we stretch or compress as a function of the component index. I encourage you to plug in some realistic values to get a feel for this
geometric progression.

There's a few parts of this equation that are confusing at first glance. How did the
authors choose 10,00010,00010,000? Why are we using sin⁡\sinsin **and** cos⁡\coscos for even and odd positions respectively?

It seems that using 10,00010,00010,000 for the base wavelength was determined experimentally [2](#user-content-fn-2). Deciphering the usage of both sin⁡\sinsin and cos⁡\coscos is more involved, but crucial
for our iterative approach to understanding. The key here is our desire for a linear relation between two encoded positions (Pr2Pr\_2Pr2​). To understand how using sin⁡\sinsin and cos⁡\coscos in tandem produce this linear relation, we will have to dive into some trigonometry.

Consider a sequence of sine and cosine pairs, each associated with a frequency ωi\omega\_iωi​. Our goal is to find a linear transformation matrix M\mathbf{M}M that can shift these sinusoidal functions by a fixed offset kkk:

M⋅[sin⁡(ωip)cos⁡(ωip)]=[sin⁡(ωi(p+k))cos⁡(ωi(p+k))]\mathbf{M} \cdot \begin{bmatrix} \sin(\omega\_i p) \\ \cos(\omega\_i p) \end{bmatrix} = \begin{bmatrix} \sin(\omega\_i(p + k)) \\ \cos(\omega\_i(p + k)) \end{bmatrix}M⋅[sin(ωi​p)cos(ωi​p)​]=[sin(ωi​(p+k))cos(ωi​(p+k))​]

The frequencies ωi\omega\_iωi​ follow a geometric progression that decreases with dimension index iii, defined as:

ωi=1100002i/d\omega\_i = \frac{1}{10000^{2i/d}}ωi​=100002i/d1​

To find this transformation matrix, we can express it as a general 2×2 matrix with unknown coefficients u1u\_1u1​, v1v\_1v1​, u2u\_2u2​, and v2v\_2v2​:

[u1v1u2v2]⋅[sin⁡(ωip)cos⁡(ωip)]=[sin⁡(ωi(p+k))cos⁡(ωi(p+k))]\begin{bmatrix} u\_1 & v\_1 \\ u\_2 & v\_2 \end{bmatrix} \cdot \begin{bmatrix} \sin(\omega\_i p) \\ \cos(\omega\_i p) \end{bmatrix} = \begin{bmatrix} \sin(\omega\_i(p+k)) \\ \cos(\omega\_i(p+k)) \end{bmatrix}[u1​u2​​v1​v2​​]⋅[sin(ωi​p)cos(ωi​p)​]=[sin(ωi​(p+k))cos(ωi​(p+k))​]

By applying the trigonometric addition theorem to the right-hand side, we can expand this into:

[u1v1u2v2]⋅[sin⁡(ωip)cos⁡(ωip)]=[sin⁡(ωip)cos⁡(ωik)+cos⁡(ωip)sin⁡(ωik)cos⁡(ωip)cos⁡(ωik)−sin⁡(ωip)sin⁡(ωik)]\begin{bmatrix} u\_1 & v\_1 \\ u\_2 & v\_2 \end{bmatrix} \cdot \begin{bmatrix} \sin(\omega\_i p) \\ \cos(\omega\_i p) \end{bmatrix} = \begin{bmatrix} \sin(\omega\_i p)\cos(\omega\_i k) + \cos(\omega\_i p)\sin(\omega\_i k) \\ \cos(\omega\_i p)\cos(\omega\_i k) - \sin(\omega\_i p)\sin(\omega\_i k) \end{bmatrix}[u1​u2​​v1​v2​​]⋅[sin(ωi​p)cos(ωi​p)​]=[sin(ωi​p)cos(ωi​k)+cos(ωi​p)sin(ωi​k)cos(ωi​p)cos(ωi​k)−sin(ωi​p)sin(ωi​k)​]

This expansion gives us a system of two equations by matching coefficients:

u1sin⁡(ωip)+v1cos⁡(ωip)=cos⁡(ωik)sin⁡(ωip)+sin⁡(ωik)cos⁡(ωip)u2sin⁡(ωip)+v2cos⁡(ωip)=−sin⁡(ωik)sin⁡(ωip)+cos⁡(ωik)cos⁡(ωip)\begin{align}
u\_1\sin(\omega\_i p) + v\_1\cos(\omega\_i p) &= \cos(\omega\_i k)\sin(\omega\_i p) + \sin(\omega\_i k)\cos(\omega\_i p) \\
u\_2\sin(\omega\_i p) + v\_2\cos(\omega\_i p) &= -\sin(\omega\_i k)\sin(\omega\_i p) + \cos(\omega\_i k)\cos(\omega\_i p)
\end{align}u1​sin(ωi​p)+v1​cos(ωi​p)u2​sin(ωi​p)+v2​cos(ωi​p)​=cos(ωi​k)sin(ωi​p)+sin(ωi​k)cos(ωi​p)=−sin(ωi​k)sin(ωi​p)+cos(ωi​k)cos(ωi​p)​​

By comparing terms with sin⁡(ωip)\sin(\omega\_i p)sin(ωi​p) and cos⁡(ωip)\cos(\omega\_i p)cos(ωi​p) on both sides, we can solve for the unknown coefficients:

u1=cos⁡(ωik)v1=sin⁡(ωik)u2=−sin⁡(ωik)v2=cos⁡(ωik)\begin{align}
u\_1 &= \cos(\omega\_i k) & v\_1 &= \sin(\omega\_i k) \\
u\_2 &= -\sin(\omega\_i k) & v\_2 &= \cos(\omega\_i k)
\end{align}u1​u2​​=cos(ωi​k)=−sin(ωi​k)​v1​v2​​=sin(ωi​k)=cos(ωi​k)​​

These solutions give us our final transformation matrix Mk\mathbf{M\_k}Mk​:

Mk=[cos⁡(ωik)sin⁡(ωik)−sin⁡(ωik)cos⁡(ωik)]\mathbf{M\_k} = \begin{bmatrix} \cos(\omega\_i k) & \sin(\omega\_i k) \\ -\sin(\omega\_i k) & \cos(\omega\_i k) \end{bmatrix}Mk​=[cos(ωi​k)−sin(ωi​k)​sin(ωi​k)cos(ωi​k)​]

If you've done any game programming before, you might notice that the
result of our derivation is oddly familiar. That's right, it's the [Rotation Matrix!](https://en.wikipedia.org/wiki/Rotation_matrix) [3](#user-content-fn-3).

So the encoding scheme designed by [Noam Shazeer](https://en.wikipedia.org/wiki/Noam_Shazeer) in [Attention is all you need](https://arxiv.org/abs/1706.03762) was already encoding relative position as a rotation back in 2017! It took another **4 years** to go from Sinusoidal Encoding to RoPE, despite rotations already being on the table...

## [Absolute vs Relative Position Encoding](#absolute-vs-relative-position-encoding)

With the knowledge in hand that rotations are important here, let's
return to our motivating example and try to discover some intuitions for our next iteration.

01234The dog chased another dog-2-1012The dog chased another dog\begin{align\*}
&\hspace{0.7em}0 \hspace{1.4em} 1 \hspace{2em} 2 \hspace{2.6em} 3 \hspace{2.4em} 4\\
&\text{The dog chased another dog} \\
\\
&\hspace{0.3em}\text{-2} \hspace{1.4em} \text{-1} \hspace{1.7em} 0 \hspace{2.6em} 1 \hspace{2.4em} 2\\
&\text{The dog \color{#699C52}chased \color{black}another dog}
\end{align\*}​01234The dog chased another dog-2-1012The dog chased another dog​

Above we can see the absolute positions of our tokens, and the relative
positions from chased\color{#699C52}\text{chased}chased to every other token. With Sinusoidal Encoding, we
generated a separate vector which represents the absolute position,
and using some trigonometric trickery we were able to encode relative positions.

When we're trying to understand these sentences, does it matter that *this* word is the 2157th word in this blog post? Or do we care about its relationship to the words around it? The absolute position of a word rarely matters for meaning - what matters is how words relate to each other.

## [Positional encoding in context](#positional-encoding-in-context)

From this point on, it's key to consider positional encoding **in the context of**
self attention. To reiterate, the self-attention mechanism enables the model to weigh the importance of different elements in an input sequence and dynamically adjust their influence on the output.

Attn(Q,K,V)=softmax(QKTdk)V\text{Attn}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d\_k}}\right)VAttn(Q,K,V)=softmax(dk​​QKT​)V

In all our previous iterations, we've generated a separate positional encoding
vector and **added** it to our token embedding prior to our QQQ, KKK and VVV projections.
By adding the positional information directly to our token embedding, we are
**polluting** the semantic information with the positional information. We should
be attempting to encode the information without modifying the norm. Shifting to multiplicative is the
key.

Using the dictionary analogy, when looking up a word (query) in our dictionary (keys), nearby words should have more influence than distant ones. The influence of one token upon another is determined by the QKTQK^TQKT dot product - so that's exactly where we should focus our positional encoding!

a⃗⋅b⃗=∣a⃗∣∣b⃗∣cos⁡θ\vec{a} \cdot \vec{b} = |\vec{a}| |\vec{b}| \cos \thetaa⋅b=∣a∣∣b∣cosθ

The geometric interpretation of the dot product shown above gives us a magnificent insight.
We can modulate the result of our dot product of two vectors purely by
increasing or decreasing the angle between them. Furthermore, by rotating the
vector, we have absolutely zero impact on the norm of the vector, which encodes
the semantic information of our token.

So now we know where to focus our *attention*, and have seen from another *angle* why a
rotation might be a sensible "channel" in which to encode our positional
information, let's put it all together!

## [**Ro**tary **P**ositional **E**mbedding](#rotary-positional-embedding)

**Ro**tary **P**ostional **E**mbedding or RoPE was defined in the
[RoFormer paper](https://arxiv.org/pdf/2104.09864) ([Jianlin Su](https://x.com/bojone1993) designed it independently on his blog [here](https://kexue.fm/archives/8130) and [here](https://kexue.fm/archives/8265)).
While it may seem like voodoo if you skip to the end result, by thinking about Sinusoidal Encoding in the
context of self attention (and more specifically dot products), we can see how
it all comes together.

Much like in Sinusoidal Encoding, we decompose our vectors (q\mathbf{q}q or k\mathbf{k}k, instead of pre-projection x\mathbf{x}x) into 2D pairs/chunks. Rather than encoding *absolute* position directly by adding a vector we drew from sinusoidal functions of slowly decreasing frequencies, we cut to the chase and encode *relative* position by **multiplying each pair with the rotation matrix**.

Let q\mathbf{q}q or k\mathbf{k}k be our input vector at position ppp. We create a block diagonal matrix
where Mi\mathbf{M\_i}Mi​ is the corresponding rotation matrix for that component
pairs desired rotation:

R(q,p)=(M1M2⋱Md/2)(q1q2⋮qd)R(\mathbf{q}, p) = \begin{pmatrix} \mathbf{M\_1} & & & \\ & \mathbf{M\_2} & & \\ & & \ddots & \\ & & & \mathbf{M\_{d/2}} \end{pmatrix} \begin{pmatrix} q\_1 \\ q\_2 \\ \vdots \\ q\_d \end{pmatrix} R(q,p)=​M1​​M2​​⋱​Md/2​​​​q1​q2​⋮qd​​​

Much the same as Sinusoidal Encoding, Mi\mathbf{M\_i}Mi​ is simply:

Mi=[cos⁡(ωip)sin⁡(ωip)−sin⁡(ωip)cos⁡(ωip)]\mathbf{M\_i} = \begin{bmatrix} \cos(\omega\_i p) & \sin(\omega\_i p) \\ -\sin(\omega\_i p) & \cos(\omega\_i p) \end{bmatrix}Mi​=[cos(ωi​p)−sin(ωi​p)​sin(ωi​p)cos(ωi​p)​]

[Your browser does not support the video tag.](/positional-encoding/RopeEncoding.mp4)

In practice, we don't use a matrix multiplication to compute RoPE as it would be
computationally inefficient with such a sparse matrix. Instead, we can directly apply the rotations to pairs of elements independently, taking advantage of the regular pattern in the computation:

RΘ,pdq=(q1q2q3q4⋮qd−1qd)⊙(cos⁡pθ1cos⁡pθ1cos⁡pθ2cos⁡pθ2⋮cos⁡pθd/2cos⁡pθd/2)+(−q2q1−q4q3⋮−qdqd−1)⊙(sin⁡pθ1sin⁡pθ1sin⁡pθ2sin⁡pθ2⋮sin⁡pθd/2sin⁡pθd/2)R\_{\Theta,p}^d q = \begin{pmatrix}
q\_1 \\
q\_2 \\
q\_3 \\
q\_4 \\
\vdots \\
q\_{d-1} \\
q\_d
\end{pmatrix} \odot \begin{pmatrix}
\cos p\theta\_1 \\
\cos p\theta\_1 \\
\cos p\theta\_2 \\
\cos p\theta\_2 \\
\vdots \\
\cos p\theta\_{d/2} \\
\cos p\theta\_{d/2}
\end{pmatrix} + \begin{pmatrix}
-q\_2 \\
q\_1 \\
-q\_4 \\
q\_3 \\
\vdots \\
-q\_d \\
q\_{d-1}
\end{pmatrix} \odot \begin{pmatrix}
\sin p\theta\_1 \\
\sin p\theta\_1 \\
\sin p\theta\_2 \\
\sin p\theta\_2 \\
\vdots \\
\sin p\theta\_{d/2} \\
\sin p\theta\_{d/2}
\end{pmatrix}RΘ,pd​q=​q1​q2​q3​q4​⋮qd−1​qd​​​⊙​cospθ1​cospθ1​cospθ2​cospθ2​⋮cospθd/2​cospθd/2​​​+​−q2​q1​−q4​q3​⋮−qd​qd−1​​​⊙​sinpθ1​sinpθ1​sinpθ2​sinpθ2​⋮sinpθd/2​sinpθd/2​​​

That's all there is to it! By artfully applying our rotations to 2D chunks of q\mathbf{q}q and
k\mathbf{k}k prior to their dot product, and switching from additive to
multiplicative, we can gain a big performance boost in evaluations [4](#user-content-fn-4).

## [Extending RoPE to nnn-Dimensions](#extending-rope-to-nnn-dimensions)

We've explored the 1D1D1D case for RoPE and by this point I hope you've gained an
intuitive understanding of an admittedly unintuitive component of transformers.
Finally, let's explore extending it to higher dimensions, such as images.

A natural first intuition could be to directly use the [xy] \begin{bmatrix} x \\ y \end{bmatrix}[xy​] coordinate pairs from the image. This might seem intuitive, after all, we were almost arbitrarily pairing up our components previously. However, this would be a mistake!

In the 1D1D1D case, we encode the relative position m−nm - nm−n through a rotation of pairs
of values from our input vector. For 2D2D2D data, we need to encode both horizontal and vertical relative positions, say m−nm - nm−n and i−ji - ji−j independently. RoPE's brilliance lies in how it handles multiple dimensions. Instead of trying to encode all positional information in a single rotation, we pair components **within the same dimension** and rotate those, otherwise we would be intermixing the xxx and yyy offset information. By handling each dimension independently, we maintain the natural structure of the space. This can generalize to as many dimensions as required!

## [The future of positional encoding](#the-future-of-positional-encoding)

Is RoPE the final incarnation of positional encoding? This [recent paper](https://arxiv.org/pdf/2410.06205) from DeepMind deeply analyses RoPE and highlights some fundamental problems.

I anticipate some future breakthroughs, perhaps taking inspiration from
signal processing with ideas like wavelets or hierarchical implementations. As models
are increasingly quantized for deployment, I'd also expect to see some
innovation in encoding schemes that remain robust under low-precision arithmetic.

## [Conclusion](#conclusion)

Positional encoding has and continues to be treated as an after thought in
transformers. I believe we should view it differently - self attention has an
Achilles heel that has been repeatedly patched.

I hope this blog post showed you that you too could have discovered state of the
art positional encoding, despite it being unintuitive at first. In a follow up
post I'd love to explore practical implementation details for RoPE in order to
maximise performance.

Thanks to Madeline Ephgrave for proof reading this, and thanks to [James
McDermott](http://www.jmmcd.net/) for his insightful corrections.

## [References](#references)

* <https://kazemnejad.com/blog/transformer_architecture_positional_encoding/>
* <https://blog.eleuther.ai/rotary-embeddings/>
* <https://www.youtube.com/watch?v=T3OT8kqoqjc>
* <https://arxiv.org/pdf/1706.03762>
* <https://arxiv.org/pdf/2410.06205>
* <https://arxiv.org/pdf/2104.09864>

## [Footnotes](#footnote-label)

1. Binary and Sinusoidal animations are reproductions of animations contained
   in [this](https://www.youtube.com/watch?v=T3OT8kqoqjc0) video. [↩](#user-content-fnref-1)
2. Using θ=10000\theta = 10000θ=10000 gives us ⌊2π⋅10000⌋\lfloor 2 \pi \cdot 10000 \rfloor⌊2π⋅10000⌋ unique positions, or a
   theoretical upper bound on the context length at ~63,000. [↩](#user-content-fnref-2)
3. Pieces of this post are based on [this fantastic
   post](https://kazemnejad.com/blog/transformer_architecture_positional_encoding/)
   by [Amirhossein Kazemnejad](https://kazemnejad.com/). [↩](#user-content-fnref-3)
4. For empirical evidence, see [this](https://blog.eleuther.ai/rotary-embeddings/) great post by EleutherAI. [↩](#user-content-fnref-4)

## Cite This Post

```
@online{christopherfleetwood2024,
  author = {Christopher Fleetwood},
  title = {You could have designed state of the art Positional Encoding},
  date = {2024-11-17},
  url = {https://fleetwood.dev/posts/you-could-have-designed-SOTA-positional-encoding},
  urldate = {2026-02-27}
}
```

Christopher Fleetwood 2026

[CV](/resume)