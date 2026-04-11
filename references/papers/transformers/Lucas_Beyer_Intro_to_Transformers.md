**Transformers** Lucas Beyer 

� lucasb.eyer.be@gmail.com 𝕏 @giffmana http://lb.eyer.be/transformer 


![](references/papers/transformers/Lucas_Beyer_Intro_to_Transformers_images/Lucas_Beyer_Intro_to_Transformers.pdf-0001-02.png)



![](references/papers/transformers/Lucas_Beyer_Intro_to_Transformers_images/Lucas_Beyer_Intro_to_Transformers.pdf-0001-03.png)



![](references/papers/transformers/Lucas_Beyer_Intro_to_Transformers_images/Lucas_Beyer_Intro_to_Transformers.pdf-0001-04.png)



![](references/papers/transformers/Lucas_Beyer_Intro_to_Transformers_images/Lucas_Beyer_Intro_to_Transformers.pdf-0001-05.png)


**Google Research Brain Team, Zürich** 

Photo CC BY-SA 2.0, Gabriel Synnaeve 

# **The classic landscape: One architecture per "community"** 

## Computer Vision 

## Natural Lang. Proc. 

Convolutional NNs (+ResNets) 

Recurrent NNs (+LSTMs) 


![](references/papers/transformers/Lucas_Beyer_Intro_to_Transformers_images/Lucas_Beyer_Intro_to_Transformers.pdf-0003-04.png)


**----- Start of picture text -----**<br>
[1]<br>**----- End of picture text -----**<br>



![](references/papers/transformers/Lucas_Beyer_Intro_to_Transformers_images/Lucas_Beyer_Intro_to_Transformers.pdf-0003-05.png)


**----- Start of picture text -----**<br>
[2]<br>**----- End of picture text -----**<br>



![](references/papers/transformers/Lucas_Beyer_Intro_to_Transformers_images/Lucas_Beyer_Intro_to_Transformers.pdf-0003-06.png)



![](references/papers/transformers/Lucas_Beyer_Intro_to_Transformers_images/Lucas_Beyer_Intro_to_Transformers.pdf-0003-07.png)



![](references/papers/transformers/Lucas_Beyer_Intro_to_Transformers_images/Lucas_Beyer_Intro_to_Transformers.pdf-0003-08.png)


**----- Start of picture text -----**<br>
h         e         l                                    l          o<br>**----- End of picture text -----**<br>


## Speech 

## Translation 

RL 


![](references/papers/transformers/Lucas_Beyer_Intro_to_Transformers_images/Lucas_Beyer_Intro_to_Transformers.pdf-0003-12.png)


**----- Start of picture text -----**<br>
Deep Belief Nets (+non-DL) Seq2Seq<br>RBM DBN s   a   l    u   t  </s><br>RBM DBN<br>GRBM DBN<br>h   e   l    l   o  </s>  s   a   l    u   t<br>**----- End of picture text -----**<br>


DQN/BC/GAIL 


![](references/papers/transformers/Lucas_Beyer_Intro_to_Transformers_images/Lucas_Beyer_Intro_to_Transformers.pdf-0003-14.png)


[1] CNN image CC-BY-SA by Aphex34 for Wikipedia https://commons.wikimedia.org/wiki/File:Typical_cnn.png [2] RNN image CC-BY-SA by GChe for Wikipedia https://commons.wikimedia.org/wiki/File:The_LSTM_Cell.svg 

# **The Transformer's takeover: One community at a time** 

## Computer Vision 

## Natural Lang. Proc. 


![](references/papers/transformers/Lucas_Beyer_Intro_to_Transformers_images/Lucas_Beyer_Intro_to_Transformers.pdf-0005-02.png)



![](references/papers/transformers/Lucas_Beyer_Intro_to_Transformers_images/Lucas_Beyer_Intro_to_Transformers.pdf-0005-03.png)


Speech Translation 


![](references/papers/transformers/Lucas_Beyer_Intro_to_Transformers_images/Lucas_Beyer_Intro_to_Transformers.pdf-0005-05.png)



![](references/papers/transformers/Lucas_Beyer_Intro_to_Transformers_images/Lucas_Beyer_Intro_to_Transformers.pdf-0005-06.png)


## Reinf. Learning 


![](references/papers/transformers/Lucas_Beyer_Intro_to_Transformers_images/Lucas_Beyer_Intro_to_Transformers.pdf-0005-08.png)


Graphs/Science 


![](references/papers/transformers/Lucas_Beyer_Intro_to_Transformers_images/Lucas_Beyer_Intro_to_Transformers.pdf-0005-10.png)


Transformer image source: "Attention Is All You Need" paper 

# **The origins: Translation, learned alignment** 

Neural Machine Translation by Jointly Learning to Align and Translate **2014** , Dzmitry Bahdanau, KyungHyun Cho, Yoshua Bengio 


![](references/papers/transformers/Lucas_Beyer_Intro_to_Transformers_images/Lucas_Beyer_Intro_to_Transformers.pdf-0007-01.png)


**----- Start of picture text -----**<br>
Source (input)<br>Target (output)<br>**----- End of picture text -----**<br>


## Attention Is All You Need 

**2017** , Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Łukasz Kaiser, Illia Polosukhin 

Attention is a building block. 

kv Think of it as a "soft" dictionary lookup: 


![](references/papers/transformers/Lucas_Beyer_Intro_to_Transformers_images/Lucas_Beyer_Intro_to_Transformers.pdf-0008-04.png)


**----- Start of picture text -----**<br>
Attention weights a<br>... ...<br>Dictionary<br>W W<br>k v<br>...<br>Output z<br>Query q<br>Keys k Values v<br>Input x<br>**----- End of picture text -----**<br>


1. Attention weights a1:N are query-key similarities: âi = q · ki Normalized via softmax: ai = e[âi] / ∑je[âj] 

2. Output z is attention-weighted average of values v1:N: 


![](references/papers/transformers/Lucas_Beyer_Intro_to_Transformers_images/Lucas_Beyer_Intro_to_Transformers.pdf-0008-07.png)


3. Usually, k and v are derived from the same input x: 


![](references/papers/transformers/Lucas_Beyer_Intro_to_Transformers_images/Lucas_Beyer_Intro_to_Transformers.pdf-0008-09.png)


The query q can come from a separate input y: 

q = W · y q x Or from the same input ! Then we call it "self attention": 


![](references/papers/transformers/Lucas_Beyer_Intro_to_Transformers_images/Lucas_Beyer_Intro_to_Transformers.pdf-0008-12.png)


Historical side-note: "non-local NNs" in computer vision and "relational NNs" in RL appeared almost at the same time and contain the same core idea! 

## Attention Is All You Need 

**2017** , Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Łukasz Kaiser, Illia Polosukhin 


![](references/papers/transformers/Lucas_Beyer_Intro_to_Transformers_images/Lucas_Beyer_Intro_to_Transformers.pdf-0009-02.png)


**----- Start of picture text -----**<br>
But that's not actually it! There are a few more details:<br>1. We usually use  many queries q1:M, not just one.<br>Attention matrix A<br>Stacking them leads to the Attention matrix  A 1:N,1:M<br>and subsequently to many outputs:<br>z1:M = Attn(q1:M,x) = [Attn(q1,x) | Attn(q2,x) | ... | Attn(qM,x)]<br>2.  We usually use " multi-head " attention. This means the            Attn1(qi,x)  Attention matrix A<br>Attention matrix A<br>operation is repeated K times and the results are zi = ⎛Attn2(qi,x) ⎞ Attention matrix AAttention matrix A<br>                  ....<br>concatenated along the feature dimension.  W s differ. ⎝ AttnK(qi,x)⎠<br>3.  The most commonly seen formulation:<br>z = softmax(QK'/√d )V Attention matrix A<br>key Attention matrix AAttention matrix A<br>Note that the complexity is O(N²)<br>1:M<br>1:M<br>Outputs z<br>Queries q<br>1:M<br>1:M1:M1:M 1:M1:M1:M<br>1:M<br>Outputs z<br>Queries qQueries qQueries q Outputs zOutputs zOutputs z<br>Queries q<br>1:M<br>1:M<br>Outputs z<br>Queries q<br>**----- End of picture text -----**<br>


Attention Is All You Need - The Transformer architecture **2017** , Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Łukasz Kaiser, Illia Polosukhin 


![](references/papers/transformers/Lucas_Beyer_Intro_to_Transformers_images/Lucas_Beyer_Intro_to_Transformers.pdf-0010-01.png)


## **Encoder** 

Task: read and “understand” the user’s input. 

## **Decoder** 

Task: generate the output (eg answer the user’s query). 

Thanks to Basil Mustafa for slide inspiration Transformer image source: "Attention Is All You Need" paper 

## Attention Is All You Need - The Transformer architecture 

**2017** , Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Łukasz Kaiser, Illia Polosukhin 


![](references/papers/transformers/Lucas_Beyer_Intro_to_Transformers_images/Lucas_Beyer_Intro_to_Transformers.pdf-0011-02.png)


**----- Start of picture text -----**<br>
d<br>mode<br>l<br>0.123, -5.234, …<br>Vocab size (32k)<br>**----- End of picture text -----**<br>


**Input (Tokenization and) Embedding** Input text is first split into pieces. Can be characters, word, "tokens": "The detective investigated" -> [The_] [detective_] [invest] [igat] [ed_] Tokens are indices into the "vocabulary": [The_] [detective_] [invest] [igat] [ed_] -> [3 721 68 1337 42] Each vocab entry corresponds to a learned dmodel-dimensional vector. [3 721 68 1337 42] -> [ [0.123, -5.234, ...], [...], [...], [...], [...] ] **Positional Encoding** Remember attention is permutation invariant, but language is not! (“The mouse ate the cat” vs “The cat ate the mouse”) Need to encode position of each word; just add something. Think  [The_] + 10   [detective_] + 20  [invest] + 30   ... but smarter. 

Thanks to Basil Mustafa for slide inspiration Transformer image source: "Attention Is All You Need" paper 

## Attention Is All You Need - The Transformer architecture 

**2017** , Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Łukasz Kaiser, Illia Polosukhin 


![](references/papers/transformers/Lucas_Beyer_Intro_to_Transformers_images/Lucas_Beyer_Intro_to_Transformers.pdf-0012-02.png)


## **Multi-headed Self-Attention** 

Meaning the input sequence is used to create queries, keys, and values! Each token can "look around" the whole input, and decide how to update its representation based on what it sees. 


![](references/papers/transformers/Lucas_Beyer_Intro_to_Transformers_images/Lucas_Beyer_Intro_to_Transformers.pdf-0012-05.png)



![](references/papers/transformers/Lucas_Beyer_Intro_to_Transformers_images/Lucas_Beyer_Intro_to_Transformers.pdf-0012-06.png)


**----- Start of picture text -----**<br>
MHSA<br>**----- End of picture text -----**<br>



![](references/papers/transformers/Lucas_Beyer_Intro_to_Transformers_images/Lucas_Beyer_Intro_to_Transformers.pdf-0012-07.png)


[The_] [detective_] [invest] [igat] [ed_] 

Thanks to Basil Mustafa for slide inspiration Transformer image source: "Attention Is All You Need" paper 

## Attention Is All You Need - The Transformer architecture 

**2017** , Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Łukasz Kaiser, Illia Polosukhin 


![](references/papers/transformers/Lucas_Beyer_Intro_to_Transformers_images/Lucas_Beyer_Intro_to_Transformers.pdf-0013-02.png)



![](references/papers/transformers/Lucas_Beyer_Intro_to_Transformers_images/Lucas_Beyer_Intro_to_Transformers.pdf-0013-03.png)


**----- Start of picture text -----**<br>
GeLU GeLU GeLU GeLUGeLU<br>**----- End of picture text -----**<br>


## **Point-wise MLP** 

A simple MLP applied to each token individually: zi = W2 GeLU(W1x + b1) + b2 

Think of it as each token pondering for itself about what it has observed previously. There's some weak evidence this is where "world knowledge" is stored, too. 


![](references/papers/transformers/Lucas_Beyer_Intro_to_Transformers_images/Lucas_Beyer_Intro_to_Transformers.pdf-0013-07.png)


It contains the bulk of the parameters. When people make giant models and sparse/moe, this is what becomes giant. 

[The_] [detective_] [invest] [igat] [ed_] 

Some people like to call it 1x1 convolution. 

Thanks to Basil Mustafa for slide inspiration Transformer image source: "Attention Is All You Need" paper 

## Attention Is All You Need - The Transformer architecture 

**2017** , Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Łukasz Kaiser, Illia Polosukhin 


![](references/papers/transformers/Lucas_Beyer_Intro_to_Transformers_images/Lucas_Beyer_Intro_to_Transformers.pdf-0014-02.png)


Thanks to Basil Mustafa for slide inspiration Transformer image source: "Attention Is All You Need" paper 

"Skip connection" == "Residual block" **Residual/skip connections** 

Each module's output has the exact same shape as its input. Following ResNets, the module computes a "residual" instead of a new value: 

+ + 

z i = Module(xi) + xi This was shown to dramatically improve trainability. 

## **LayerNorm** 

Normalization also dramatically improves trainability. There's **post-norm** (original) and **pre-norm** (modern) z z i = LN(Module(xi) + xi) i = Module(LN(xi)) + xi 

## Attention Is All You Need - The Transformer architecture 

**2017** , Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Łukasz Kaiser, Illia Polosukhin 


![](references/papers/transformers/Lucas_Beyer_Intro_to_Transformers_images/Lucas_Beyer_Intro_to_Transformers.pdf-0015-02.png)


## **Encoding / Encoder** 

Since input and output shapes are identical, we can stack N such blocks. Typically, N=6 ("base"), N=12 ("large") or more. 

Encoder output is a "heavily processed" (think: "high level, contextualized") version of the input tokens, i.e. a sequence. This has nothing to do with the requested output yet (think: translation). That comes with the decoder. 

Thanks to Basil Mustafa for slide inspiration Transformer image source: "Attention Is All You Need" paper 

## Attention Is All You Need - The Transformer architecture 

**2017** , Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Łukasz Kaiser, Illia Polosukhin 

p(z3|z2,z1,x) **Decoding / the Decoder  (alternatively Generating / the Generator)** What we want to model: p(z|x) for example, in translation: p(z | "the detective investigated") ∀z Seems impossible at first, but we can _exactly_ decompose into tokens: p(z|x) = p(z1|x) p(z2|z1,x) p(z3|z2,z1,x)...1|x) p(z2|z1,x) p(z3|z2,z1,x)...|x) p(z2|z1,x) p(z3|z2,z1,x)...2|z1,x) p(z3|z2,z1,x)...|z1,x) p(z3|z2,z1,x)...1,x) p(z3|z2,z1,x)...,x) p(z3|z2,z1,x)...3|z2,z1,x)...|z2,z1,x)...2,z1,x)...,z1,x)...1,x)...,x)... Meaning, we can compute the likelihood of a given output z,z,, or generate/sample an answer z one token at a time.z one token at a time. one token at a time. Each p is a full pass through the model. For generating p(z3|z2,z1,x):p(z3|z2,z1,x):3|z2,z1,x):|z2,z1,x):2,z1,x):,z1,x):1,x):,x):: x comes from the encoder, comes from the encoder, z1, z2 is what we have predicted so far, goes into the decoder.1, z2 is what we have predicted so far, goes into the decoder., z2 is what we have predicted so far, goes into the decoder.2 is what we have predicted so far, goes into the decoder. is what we have predicted so far, goes into the decoder. z , z Once we have a p(zi|z:i,x) we still need to actually sample a sentence p(zi|z:i,x) we still need to actually sample a sentence i|z:i,x) we still need to actually sample a sentence |z:i,x) we still need to actually sample a sentence :i,x) we still need to actually sample a sentence ,x) we still need to actually sample a sentence  we still need to actually sample a sentence 1 2 

What we want to model: p(z|x) for example, in translation: p(z | "the detective investigated") ∀z Seems impossible at first, but we can _exactly_ decompose into tokens: p(z|x) = p(z1|x) p(z2|z1,x) p(z3|z2,z1,x)...1|x) p(z2|z1,x) p(z3|z2,z1,x)...|x) p(z2|z1,x) p(z3|z2,z1,x)...2|z1,x) p(z3|z2,z1,x)...|z1,x) p(z3|z2,z1,x)...1,x) p(z3|z2,z1,x)...,x) p(z3|z2,z1,x)...3|z2,z1,x)...|z2,z1,x)...2,z1,x)...,z1,x)...1,x)...,x)... Meaning, we can compute the likelihood of a given output z,z,, or generate/sample an answer z one token at a time.z one token at a time. one token at a time. Each p is a full pass through the model. For generating p(z3|z2,z1,x):p(z3|z2,z1,x):3|z2,z1,x):|z2,z1,x):2,z1,x):,z1,x):1,x):,x):: x comes from the encoder, comes from the encoder, z1, z2 is what we have predicted so far, goes into the decoder.1, z2 is what we have predicted so far, goes into the decoder., z2 is what we have predicted so far, goes into the decoder.2 is what we have predicted so far, goes into the decoder. is what we have predicted so far, goes into the decoder. Once we have a p(zi|z:i,x) we still need to actually sample a sentence p(zi|z:i,x) we still need to actually sample a sentence i|z:i,x) we still need to actually sample a sentence |z:i,x) we still need to actually sample a sentence :i,x) we still need to actually sample a sentence ,x) we still need to actually sample a sentence  we still need to actually sample a sentence such as "le détective a enquêté". Many strategies: greedy, beam, ... 


![](references/papers/transformers/Lucas_Beyer_Intro_to_Transformers_images/Lucas_Beyer_Intro_to_Transformers.pdf-0016-04.png)


**----- Start of picture text -----**<br>
x<br>**----- End of picture text -----**<br>


Thanks to Basil Mustafa for slide inspiration Transformer image source: "Attention Is All You Need" paper 

Attention Is All You Need - The Transformer architecture **2017** , Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Łukasz Kaiser, Illia Polosukhin 


![](references/papers/transformers/Lucas_Beyer_Intro_to_Transformers_images/Lucas_Beyer_Intro_to_Transformers.pdf-0017-01.png)


## **At training time: Masked self-attention** 

This is regular self-attention as in the encoder, to process what's been decoded so far, eg z2,z1 in p(z3|z2,z1,x), but with a trick... If we had to train on one single p(z3|z2,z1,x) at a time: SLOW! Instead, train on all p(zi|z1:i,x) for all i simultaneously. How? In the attention weights for zi, set all entries i:N to 0. This way, each token only sees the already generated ones. Let’s see what that means… 

Thanks to Basil Mustafa for slide inspiration Transformer image source: "Attention Is All You Need" paper 

## Attention Is All You Need - The Transformer architecture 

**2017** , Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Łukasz Kaiser, Illia Polosukhin 

**At training time: Masked self-attention** 


![](references/papers/transformers/Lucas_Beyer_Intro_to_Transformers_images/Lucas_Beyer_Intro_to_Transformers.pdf-0018-03.png)


**----- Start of picture text -----**<br>
At training time: Masked self-attention<br>[Le_] [détective_] [a_] [enquêt] [é_] <eos><br>Keys k<br><bos> 1          1          1          1          1          ✔ ✔ ✔ ✔ ✔ 1 ✔<br>[Le_] 0 ✗           1          1          1          1          ✔ ✔ ✔ ✔ 1 ✔<br>[détective_] 0  ✗ 0 ✗           1          1          1          ✔ ✔ ✔ 1 ✔<br>[a_] 0  ✗ 0 ✗ 0 ✗           1          1          ✔ ✔ 1 ✔<br>[enquêt] 0  ✗ 0 ✗ 0 ✗ 0          1          ✗ ✔ 1 ✔<br>[é_] 0 ✗ 0 ✗ 0 ✗ 0 ✗ 0 ✗ 1 ✔<br><bos> [Le_] [détective_] [a_] [enquêt] [é_]<br>Transformer image source: "Attention Is All You Need" paper<br>Key k Key k Key k Key k Key k Key k<br>Query q Query q Query q Query q Query q Query q<br>Input x Input x Input x Input x Input x Input x<br>**----- End of picture text -----**<br>


Transformer image source: "Attention Is All You Need" paper 

## Attention Is All You Need - The Transformer architecture 

**2017** , Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Łukasz Kaiser, Illia Polosukhin 


![](references/papers/transformers/Lucas_Beyer_Intro_to_Transformers_images/Lucas_Beyer_Intro_to_Transformers.pdf-0019-02.png)


Thanks to Basil Mustafa for slide inspiration Transformer image source: "Attention Is All You Need" paper 

## **At generation time** 

z There is no such trick. We need to generate one i at a time. This is why autoregressive decoding is extremely slow. 

But: we can cache k/v activations from previous tokens and re-use. 


![](references/papers/transformers/Lucas_Beyer_Intro_to_Transformers_images/Lucas_Beyer_Intro_to_Transformers.pdf-0019-07.png)


**----- Start of picture text -----**<br>
[enquêt]<br>KV-cache<br><bos><br>[Le_]<br>[détective_]<br>[a_]<br>Note cache size:<br>    (Seq x Heads x dmodel) x Layers x dtype<br>Example Gemma-9B:<br>    (4096 x 16 x 3584) x 42 x 4 = 36.75 GiB<br>⇒ Research: GQA, MQA, kv-compression, … [a_]<br>Key k<br>Query q<br>**----- End of picture text -----**<br>


## Attention Is All You Need - The Transformer architecture 

**2017** , Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Łukasz Kaiser, Illia Polosukhin 


![](references/papers/transformers/Lucas_Beyer_Intro_to_Transformers_images/Lucas_Beyer_Intro_to_Transformers.pdf-0020-02.png)


**----- Start of picture text -----**<br>
x<br>enc<br>x<br>dec<br>**----- End of picture text -----**<br>


## **"Cross" attention** 

Each decoded token can "look at" the encoder's output: Attn(q=Wqxdec, k=Wkxenc, v=Wvxenc) This is the same as in the 2014 paper. This is where |x in p(z3|z2,z1,x) comes from. 


![](references/papers/transformers/Lucas_Beyer_Intro_to_Transformers_images/Lucas_Beyer_Intro_to_Transformers.pdf-0020-05.png)


Because self-attention is so widely used, people have started just calling it "attention". Hence, we now often need to explicitly call this "cross attention". 

Thanks to Basil Mustafa for slide inspiration 

Transformer image source: "Attention Is All You Need" paper  ;  heatmap image source: "Neural Machine Translation by Jointly Learning to Align and Translate" paper 

## Attention Is All You Need - The Transformer architecture 

**2017** , Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Łukasz Kaiser, Illia Polosukhin 


![](references/papers/transformers/Lucas_Beyer_Intro_to_Transformers_images/Lucas_Beyer_Intro_to_Transformers.pdf-0021-02.png)


**Feedforward and stack layers.** 

Thanks to Basil Mustafa for slide inspiration Transformer image source: "Attention Is All You Need" paper 

## Attention Is All You Need - The Transformer architecture 

**2017** , Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Łukasz Kaiser, Illia Polosukhin 


![](references/papers/transformers/Lucas_Beyer_Intro_to_Transformers_images/Lucas_Beyer_Intro_to_Transformers.pdf-0022-02.png)


## **Output layer** 

Assume we have already generated K tokens, generate the next one. 

The decoder was used to gather all information necessary to predict a probability distribution for the next token (K), over the whole vocab. Simple: 

linear projection of token K SoftMax normalization 

Thanks to Basil Mustafa for slide inspiration Transformer image source: "Attention Is All You Need" paper 

## Attention Is All You Need - Summary and results 

**2017** , Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Łukasz Kaiser, Illia Polosukhin 


![](references/papers/transformers/Lucas_Beyer_Intro_to_Transformers_images/Lucas_Beyer_Intro_to_Transformers.pdf-0023-02.png)



![](references/papers/transformers/Lucas_Beyer_Intro_to_Transformers_images/Lucas_Beyer_Intro_to_Transformers.pdf-0023-03.png)


## **Good and simple heuristic:** 

ExaFLOPs = k * [tokens in B] * [params in B] with k in [5…15] depending on many details 

Animation source: https://ai.googleblog.com/2017/08/transformer-novel-neural-network.html Table source: "Attention Is All You Need" paper 

Thanks to Stella Biderman for the heuristic idea: https://x.com/BlancheMinerva/status/1970189748682797415 

## Transformer compute budget heuristics 


![](references/papers/transformers/Lucas_Beyer_Intro_to_Transformers_images/Lucas_Beyer_Intro_to_Transformers.pdf-0024-01.png)


## **Good and simple heuristic:** 

ExaFLOPs = k × [tokens in B] × [params in B] ; k in [5…15] Base **tokens** = (25k + 25k) × 100k = **5B** Big **tokens** = (25k + 25k) × 300k = **15B** Base **params** = 65×10[6] = **0.065B** Big **params** = 213×10[6] = **0.213B** Base **ExaFLOPs** = k × 5 × 0.065 =(k:10)= **3.25** Big **ExaFLOPs** = k × 15 × 0.213 =(k:10)= **31.95** 


![](references/papers/transformers/Lucas_Beyer_Intro_to_Transformers_images/Lucas_Beyer_Intro_to_Transformers.pdf-0024-04.png)



![](references/papers/transformers/Lucas_Beyer_Intro_to_Transformers_images/Lucas_Beyer_Intro_to_Transformers.pdf-0024-05.png)


## **What about the fiddle factor k?** 

- Modern AR decoder-only: just use k=6 (2 fwd + 4 bwd) 

- - Changes with attn pattern/impl (AR, prefix, flash, …), remat ("activation checkpointing"), fusion, … 

- Changes with architecture details (eg Enc+Dec seen here) 

- Very large seqlen (128k+) starts breaking this 

Thanks to Stella Biderman for the heuristic slide idea: https://x.com/BlancheMinerva/status/1970189748682797415 

## Transformer compute budget heuristics 

## **Good and simple heuristic:** 

## **From ExaFLOPs to GPU-hours (and price)** 

ExaFLOPs = k × [tokens in B] × [params in B] ; k in [5…15] Base **tokens** = (25k + 25k) × 100k = **5B** Big **tokens** = (25k + 25k) × 300k = **15B** Base **params** = 65×10[6] = **0.065B** Big **params** = 213×10[6] = **0.213B** 

- Base **ExaFLOPs** = k × 5 × 0.065 =(k:10)= **3.25** Big **ExaFLOPs** = k × 15 × 0.213 =(k:10)= **31.95** 


![](references/papers/transformers/Lucas_Beyer_Intro_to_Transformers_images/Lucas_Beyer_Intro_to_Transformers.pdf-0025-05.png)


## **What about the fiddle factor k?** 

- Modern AR decoder-only: just use k=6 (2 fwd + 4 bwd) 

- Changes with attn pattern/impl (AR, prefix, flash, …), 

- Changes with architecture details (eg Enc+Dec s… 

- Very large seqlen (128k+) starts breaking this 

1. Find the GPU's peak FLOP/s (varies per dtype!) H100: ~1.979 PFLOP/s BF16 ≈ **7 EFLOP/h** 

2. Real code never runs at peak FLOP/s: a. Find codebase's "actual FLOP/s" by profiling 

b. This is called the "utilization". Typically: GPUs: 10- **30%** (but can be as bad as 1%!!) TPUs: 50-70% 

3. Combine: 7×0.3 ≈ **2 EFLOP per H100-hour** Base: 3.3 / 2 = 1.57 H100-hours Big: 23 / 2 = 11.5 H100-hours 

Chinchilla: 6×70×1400 / 2 ≈ 300k H100-hours 

4. Check public cloud prices: 

a. NeoClouds: $2-3 per H100-hour b. BigTech clouds: $4-12 per H100-hour Note: FLOPs ≠ FLOPS = FLOP/s 

Thanks to Stella Biderman for the heuristic slide idea: https://x.com/BlancheMinerva/status/1970189748682797415 

## Modern extensions that stuck: 1. Mixture of Experts 

**2021** William Fedus, Barret Zoph, Noam Shazeer: Switch Transformers: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity 

Remember: "knowledge stored in MLP" So: let's make it as big as possible? 


![](references/papers/transformers/Lucas_Beyer_Intro_to_Transformers_images/Lucas_Beyer_Intro_to_Transformers.pdf-0026-03.png)


**----- Start of picture text -----**<br>
GeLUGeLU<br>**----- End of picture text -----**<br>


**But: Too expensive!** 

## Modern extensions that stuck: 1. Mixture of Experts 

**2021** William Fedus, Barret Zoph, Noam Shazeer: Switch Transformers: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity 

Instead: have many of them, but only use one, "most fitting" => Same cost, more params! How to determine which is most fitting? 


![](references/papers/transformers/Lucas_Beyer_Intro_to_Transformers_images/Lucas_Beyer_Intro_to_Transformers.pdf-0027-03.png)


**----- Start of picture text -----**<br>
GeLU GeLU GeLU GeLU GeLU<br>**----- End of picture text -----**<br>


## Modern extensions that stuck: 1. Mixture of Experts 

**2021** William Fedus, Barret Zoph, Noam Shazeer: Switch Transformers: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity 

How to determine which is most fitting? Learn it! 

Router is a "classifier over experts" 


![](references/papers/transformers/Lucas_Beyer_Intro_to_Transformers_images/Lucas_Beyer_Intro_to_Transformers.pdf-0028-04.png)


**----- Start of picture text -----**<br>
GeLU GeLU GeLU GeLU GeLU<br>Routing weights a<br>Router<br>**----- End of picture text -----**<br>


## Modern extensions that stuck: 1. Mixture of Experts 

**2021** William Fedus, Barret Zoph, Noam Shazeer: Switch Transformers: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity 

Pass the token through the highest-scored MLP only. Multiply output by score. 


![](references/papers/transformers/Lucas_Beyer_Intro_to_Transformers_images/Lucas_Beyer_Intro_to_Transformers.pdf-0029-03.png)


**----- Start of picture text -----**<br>
x<br>GeLU GeLU GeLU GeLU GeLU<br>Routing weights a<br>Router<br>**----- End of picture text -----**<br>


## Modern extensions that stuck: 1. Mixture of Experts 

**2021** William Fedus, Barret Zoph, Noam Shazeer: Switch Transformers: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity 

Modern variant: smaller experts, but more of them, and activate multiple (usually two) 


![](references/papers/transformers/Lucas_Beyer_Intro_to_Transformers_images/Lucas_Beyer_Intro_to_Transformers.pdf-0030-03.png)


**----- Start of picture text -----**<br>
x + x =<br>GeLU GeLU GeLU GeLU GeLU GeLU GeLU GeLU GeLU GeLU<br>Routing weights a<br>Router<br>**----- End of picture text -----**<br>


## Modern extensions that stuck: 2. Interleaved global-local 

**2019** Rewon Child, Scott Gray, Alec Radford, Ilya Sutskever: Generating Long Sequences with Sparse Transformers 

Global Attention Global Attention Global Attention Global Attention Global Attention Global Attention 


![](references/papers/transformers/Lucas_Beyer_Intro_to_Transformers_images/Lucas_Beyer_Intro_to_Transformers.pdf-0031-03.png)


Global Attention Local / RNN Local / RNN Global Attention Local / RNN Local / RNN 


![](references/papers/transformers/Lucas_Beyer_Intro_to_Transformers_images/Lucas_Beyer_Intro_to_Transformers.pdf-0031-05.png)


Used by a lot of models now, either with “SWA” (Sliding Window Attention) or RNN-revivals (“Linear Attention”) 

# **The first (1.5[th] ) big takeover: Language Modeling / NLP** 

## **Decoder-only Encoder-only GPT BERT** 

## **Enc-Dec** 

## **T5** 

Das ist gut. 

**[sat_]** 


![](references/papers/transformers/Lucas_Beyer_Intro_to_Transformers_images/Lucas_Beyer_Intro_to_Transformers.pdf-0033-05.png)



![](references/papers/transformers/Lucas_Beyer_Intro_to_Transformers_images/Lucas_Beyer_Intro_to_Transformers.pdf-0033-06.png)


[START] [The_] [cat_] 

[*]    [*] **[sat_]** [*] **[the_]** [*] 


![](references/papers/transformers/Lucas_Beyer_Intro_to_Transformers_images/Lucas_Beyer_Intro_to_Transformers.pdf-0033-09.png)


[The_] [cat_] **[MASK]** [on_] **[MASK]** [mat_] 

A storm in Attala caused 6 victims. This is not toxic. 


![](references/papers/transformers/Lucas_Beyer_Intro_to_Transformers_images/Lucas_Beyer_Intro_to_Transformers.pdf-0033-12.png)


Translate EN-DE: This is good. Summarize: state authorities dispatched… Is this toxic: You look beautiful today! 

Transformer image source: "Attention Is All You Need" paper 

# **The second big takeover: Computer Vision** 

Many prior works attempted to introduce self-attention at the pixel level. Previous approaches For 224px², that's 50k sequence length, too much! 

## 1. On pixels, but locally or factorized 


![](references/papers/transformers/Lucas_Beyer_Intro_to_Transformers_images/Lucas_Beyer_Intro_to_Transformers.pdf-0035-02.png)


Usually replaces 3x3 conv in ResNet: 


![](references/papers/transformers/Lucas_Beyer_Intro_to_Transformers_images/Lucas_Beyer_Intro_to_Transformers.pdf-0035-04.png)



![](references/papers/transformers/Lucas_Beyer_Intro_to_Transformers_images/Lucas_Beyer_Intro_to_Transformers.pdf-0035-05.png)


Results: 

Are usually "meh", nothing to call home about Do not justify increased complexity Do not justify slowdown over convolutions 

Examples: 

Non-local NN (Wang et.al. 2017) SASANet (Stand-Alone Self-Attention in Vision Models) HaloNet (Scaling Local Self-Attn for Parameter Efficient...) LR-Net (Local Relation Networks for Image Recognition) SANet (Exploring Self-attention for Image Recognition) 

Image credit: Stand-Alone Self-Attention in Vision Models by Ramachandran et.al. Image credit: Local Relation Networks for Image Recognition by Hu et.al. 

... 

## Previous approaches 

## 2. Globally, after/inside a full-blown CNN, or even detector/segmenter! 


![](references/papers/transformers/Lucas_Beyer_Intro_to_Transformers_images/Lucas_Beyer_Intro_to_Transformers.pdf-0036-02.png)



![](references/papers/transformers/Lucas_Beyer_Intro_to_Transformers_images/Lucas_Beyer_Intro_to_Transformers.pdf-0036-03.png)


Cons: 

result is highly complex, often multi-stage trained architecture. not from pixels, i.e. transformer can't "learn to fix" the (often frozen!) CNN's mistakes. Examples: DETR (Carion, Massa et.al. 2020)          Visual Transformers (Wu et.al. 2020) UNITER (Chen, Li, Yu et.al. 2019)            ViLBERT (Lu et.al. 2019)                                              etc... VisualBERT (Li et.al. 20190) 

Image credit: UNITER: UNiversal Image-TExt Representation Learning by Chen et.al. Image credit: Visual Transformers: Token-based Image Representation and Processing for Computer Vision by Wu et.al. 

## An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale 

**2020** , A Dosovitskiy, L Beyer, A Kolesnikov, D Weissenborn, X Zhai, T Unterthiner, M Dehghani, M Minderer, G Heigold, S Gelly, J Uszkoreit, N Houlsby 

## **Vision Transformer** 

Many prior works attempted to introduce self-attention at the pixel level. 

**(ViT)** 

For 224px², that's 50k sequence length, too much! Thus, most works restrict attention to local pixel neighborhoods, or as high-level mechanism on top of detections. 

The **key breakthrough** in using the full Transformer architecture, standalone, was to **"tokenize" the image** by **cutting it into patches** of 16px², and treating each patch as a token, e.g. embedding it into input space. 


![](references/papers/transformers/Lucas_Beyer_Intro_to_Transformers_images/Lucas_Beyer_Intro_to_Transformers.pdf-0037-07.png)


## Side-note: MLP-Mixer 

**2020** , I Tolstikhin, N Houlsby, A Kolesnikov, L Beyer, X Zhai, T Unterthiner, J Yung, A Steiner, D Keysers, J Uszkoreit, M Lucic, A Dosovitskiy 

After ViT answered the question "Are convolutions really needed to process images?" with NO... We wondered if self-attention is really needed? The role of self-attention is to "mix" information across tokens. Another simple way to achieve this, is to "transpose" tokens and run that through an MLP: 


![](references/papers/transformers/Lucas_Beyer_Intro_to_Transformers_images/Lucas_Beyer_Intro_to_Transformers.pdf-0038-03.png)



![](references/papers/transformers/Lucas_Beyer_Intro_to_Transformers_images/Lucas_Beyer_Intro_to_Transformers.pdf-0038-04.png)


# **The third big takeover: Speech** 

Conformer: Convolution-augmented Transformer for Speech Recognition **2020** , A Gulati, J Qin, C-C Chiu, N Parmar, Y Zhang, J Yu, W Han, S Wang, Z Zhang, Y Wu, R Pang [igat] Largely the same story as in computer vision. But with spectrograms instead of images. 


![](references/papers/transformers/Lucas_Beyer_Intro_to_Transformers_images/Lucas_Beyer_Intro_to_Transformers.pdf-0040-01.png)



![](references/papers/transformers/Lucas_Beyer_Intro_to_Transformers_images/Lucas_Beyer_Intro_to_Transformers.pdf-0040-02.png)



![](references/papers/transformers/Lucas_Beyer_Intro_to_Transformers_images/Lucas_Beyer_Intro_to_Transformers.pdf-0040-03.png)


Conformer adds a third type of block using convolutions, and slightly reorder blocks, but overall very transformer-like. Exists as encoder-decoder variant, or as encoder-only variant with CTC loss. 

UPDATE Sept 2022: Plain transformer model in “Whisper” by A Radford, J W Kim, T Xu, G Brockman, C McLeavey, I Sutskever 


![](references/papers/transformers/Lucas_Beyer_Intro_to_Transformers_images/Lucas_Beyer_Intro_to_Transformers.pdf-0040-06.png)



![](references/papers/transformers/Lucas_Beyer_Intro_to_Transformers_images/Lucas_Beyer_Intro_to_Transformers.pdf-0040-07.png)



![](references/papers/transformers/Lucas_Beyer_Intro_to_Transformers_images/Lucas_Beyer_Intro_to_Transformers.pdf-0040-08.png)



![](references/papers/transformers/Lucas_Beyer_Intro_to_Transformers_images/Lucas_Beyer_Intro_to_Transformers.pdf-0040-09.png)



![](references/papers/transformers/Lucas_Beyer_Intro_to_Transformers_images/Lucas_Beyer_Intro_to_Transformers.pdf-0040-10.png)


**----- Start of picture text -----**<br>
[The_] [detective_] [invest]<br>**----- End of picture text -----**<br>


Transformer image source: "Attention Is All You Need" paper 

Whisper: Robust Speech Recognition via Large-Scale Weak Supervision **2022** , A Radford, J W Kim, T Xu, G Brockman, C McLeavey, I Sutskever 

[igat] 

Largely the same story as in computer vision. But with spectrograms instead of images. 


![](references/papers/transformers/Lucas_Beyer_Intro_to_Transformers_images/Lucas_Beyer_Intro_to_Transformers.pdf-0041-03.png)



![](references/papers/transformers/Lucas_Beyer_Intro_to_Transformers_images/Lucas_Beyer_Intro_to_Transformers.pdf-0041-04.png)



![](references/papers/transformers/Lucas_Beyer_Intro_to_Transformers_images/Lucas_Beyer_Intro_to_Transformers.pdf-0041-05.png)


Conformer adds a third type of block using convolutions, and slightly reorder blocks, but overall very transformer-like. Exists as encoder-decoder variant, or as encoder-only variant with CTC loss. 

UPDATE Sept 2022: Plain transformer model in “Whisper” by A Radford, J W Kim, T Xu, G Brockman, C McLeavey, I Sutskever 


![](references/papers/transformers/Lucas_Beyer_Intro_to_Transformers_images/Lucas_Beyer_Intro_to_Transformers.pdf-0041-08.png)



![](references/papers/transformers/Lucas_Beyer_Intro_to_Transformers_images/Lucas_Beyer_Intro_to_Transformers.pdf-0041-09.png)


**----- Start of picture text -----**<br>
[The_] [detective_] [invest]<br>**----- End of picture text -----**<br>



![](references/papers/transformers/Lucas_Beyer_Intro_to_Transformers_images/Lucas_Beyer_Intro_to_Transformers.pdf-0041-10.png)



![](references/papers/transformers/Lucas_Beyer_Intro_to_Transformers_images/Lucas_Beyer_Intro_to_Transformers.pdf-0041-11.png)



![](references/papers/transformers/Lucas_Beyer_Intro_to_Transformers_images/Lucas_Beyer_Intro_to_Transformers.pdf-0041-12.png)



![](references/papers/transformers/Lucas_Beyer_Intro_to_Transformers_images/Lucas_Beyer_Intro_to_Transformers.pdf-0041-13.png)


Transformer image source: "Attention Is All You Need" paper 

# **The fourth big takeover: Reinforcement Learning** 

## Decision Transformer: Reinforcement Learning via Sequence Modeling 

**2021** , L Chen, K Lu, A Rajeswaran, K Lee, A Grover, M Laskin, P Abbeel, A Srinivas, I Mordatch 

Cast the (supervised/offline) RL problem into a sequence ("language") modeling task: 


![](references/papers/transformers/Lucas_Beyer_Intro_to_Transformers_images/Lucas_Beyer_Intro_to_Transformers.pdf-0043-03.png)


Can generate/decode sequences of actions with desired return (eg skill) The trick is prompting: "The following is a trajectory of an expert player: [obs] ..." 

Slide credit: Igor Mordatch 

# **The Transformer's Unification of communities** 


![](references/papers/transformers/Lucas_Beyer_Intro_to_Transformers_images/Lucas_Beyer_Intro_to_Transformers.pdf-0045-00.png)


**----- Start of picture text -----**<br>
Anything you can tokenize, you can feed to Transformer Encoder<br>ca 2021  and onwards [2]<br>Tokenize different modalities each in their<br>own way (some kind of "patching"), and send<br>them all jointly into a Transformer...<br>Seems to just work...<br>Currently an explosion of works doing this!<br>[3]<br>[1]<br>Images from:<br>[1] LIMoE by B Mustafa, C Riquelme, J Puigcerver, R Jenatton, N Houlsby<br>[2] MERLOT Reserve by R Zellers, J Lu, X Lu, Y Yu, Y Zhao, M Salehi, A Kusupati, J Hessel, A Farhadi, Y Choi<br>**----- End of picture text -----**<br>


[2] MERLOT Reserve by R Zellers, J Lu, X Lu, Y Yu, Y Zhao, M Salehi, A Kusupati, J Hessel, A Farhadi, Y Choi [3] VATT by H Akbari, L Yuan, R Qian, W-H Chuang, S-F Chang, Y Cui, B Gong 

## Anything you can discretize, you can Generate! 

## 

**ca 2023** and onwards 

Encode inputs 

Decode outputs 

[1] 

For outputs: learn discrete code (VQ-VAE, FSQ) that represents the output, append to vocab. 


![](references/papers/transformers/Lucas_Beyer_Intro_to_Transformers_images/Lucas_Beyer_Intro_to_Transformers.pdf-0046-08.png)


**----- Start of picture text -----**<br>
[2]<br>**----- End of picture text -----**<br>


[3] 

[1] PaliGemma by L. Beyer, A. Steiner, A. S. Pinto, A. Kolesnikov, X. Wang, …, X. Zhai 

[2] VQ-VAE by A. van den Oord, O. Vinyals, K. Kavukcuoglu 

- [3] PaliGemma 2 by Andreas Steiner, A. S. Pinto, M. Tschannen, …, L. Beyer, X. Zhai 

# **A note on Efficient Transformers** 

## A note on Efficient Transformers 

The self-attention operation complexity is O(N²) for sequence length N. 


![](references/papers/transformers/Lucas_Beyer_Intro_to_Transformers_images/Lucas_Beyer_Intro_to_Transformers.pdf-0048-02.png)


We'd like to use large N: 

Whole articles or books Full video movies High resolution images 

Many O(N) approximations to the full self-attention have been proposed in the past two years. 

Unfortunately, none provides a clear improvement. They always trade-off between speed and quality. 

Based on "Efficient Transformers: A Survey" by Yi Tay, Mostafa Dehghani, Dara Bahri, Donald Metzler and "Long Range Arena: A Benchmark for Efficient Transformers" by Y Tay, M Dehghani, S Abnar, Y Shen, D Bahri, P Pham, J Rao, L Yang, S Ruder, D Metzler 

# **Thanks for your... Attention** 

Transformer image source: "Attention Is All You Need" paper 

