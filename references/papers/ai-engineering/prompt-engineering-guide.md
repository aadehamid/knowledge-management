title: Prompt Engineering and Zero-Shot/Few-Shot Learning \[Guide\] - inovex GmbH
description: This blog post gives you an overview of prompt engineering, the definition of zero-shot and few-shot learning, and provides a practical guide.
author: Suteera Seeha

# Prompt Engineering and Zero-Shot/Few-Shot Learning \[Guide\] - inovex GmbH

In this blog post, I will give you an overview of prompt engineering, talk about its fascinating capabilities, the definition of zero-shot and few-shot learning, and provide a practical guide on how to adopt prompt engineering for your task of interest.

Prompt engineering or prompt learning is a novel approach for leveraging pre-trained language models (LMs) to perform NLP tasks without fine-tuning. In this approach, the model is informed about the target task directly through a natural language task description which is integrated into the actual input sentence in some way. The task description is called a *prompt* as it prompts the model to perform a specific task. Prompt engineering is often implemented in a zero-shot or few-shot setting, which means no or only a few labeled examples are used.

## **From Fine-Tuning to Prompt Learning**

Fine-tuning is a widely used, powerful transfer learning technique that can turn a pre-trained LM into a task-specific model. It has demonstrated remarkable performance for many downstream tasks including natural language inference, question answering, and text classification ( [Devlin et al., 2019](https://aclanthology.org/N19-1423/) ; [ Sun et al., 2019](https://arxiv.org/abs/1905.05583) ; [ Howard and Ruder, 2018](https://arxiv.org/abs/1801.06146) ; [ He et al., 2021](https://arxiv.org/abs/2006.03654) ). However, to achieve such high performance, the model usually requires sufficiently large annotated training data which is often costly and can be hard to obtain. Due to this constraint, methods that require less annotated data are attracting much interest from industry and researchers. Prompt engineering is one of the works in this direction.

The idea is to ask a pre-trained language model to perform the target task by giving it a task description such as “Translate this French sentence into an English sentence“,  followed by the French sentence we want to translate. The model is supposed to understand the task description and return the English translation of the input sentence.

**Prompt:**               Translate this French sentence into an English sentence.

 **J’aime la pizza.**

**Model output:**     I like pizza

This may seem impossible, but large-scale LMs like [ GPT3](https://arxiv.org/abs/2005.14165) and its predecessor [ GPT2](https://www.semanticscholar.org/paper/Language-Models-are-Unsupervised-Multitask-Learners-Radford-Wu/9405cc0d6169988371b2755e573cc28650d14dfe?p2df) have shown that they are capable of performing a range of NLP tasks using prompting. For instance, machine translation, question answering, cloze questions, reasoning tasks, and domain adaptation. This achievement has been regarded as a paradigm shift for NLP [ (Liu et al.,2021)](https://www.semanticscholar.org/paper/Pre-train%2C-Prompt%2C-and-Predict%3A-A-Systematic-Survey-Liu-Yuan/28692beece311a90f5fa1ca2ec9d0c2ce293d069) , where we only need to train a single LM to be powerful enough, design prompts that the LM can understand, and we can have it perform arbitrary tasks without or with only a few labeled data. After GPT3, there is an increasing trend in research on prompt learning. Some work focuses on finding optimal prompt templates for large LMs. Other work tries to enable this technique for smaller LMs in a few-shot setting or combining fine-tuning with prompt learning.

## **Zero-Shot and Few-Shot Learning**

To begin, I would like to clarify the definition of zero-shot and few-shot learning, as the terms are used differently in varying contexts and domains.

### Zero-Shot Learning

In the prompt engineering literature, the term “zero-shot“ often refers to **a setting where zero-labeled data is used in model training or inference.**  In a broader sense, “zero-shot“ refers to “ **teaching a model to do something it has not been explicitly trained to do“.**

Zero-shot learning originates from the field of computer vision. It refers to the problem setting where the goal is  **to train a classifier on a labeled dataset so that it can classify objects of unseen classes** [ (Wang et al., 2019)](https://www.semanticscholar.org/paper/A-Survey-of-Zero-Shot-Learning-Wang-Zheng/dafa29f1f0534448d205365796d68873a0068c6b) . Since samples from unseen classes are not available during training, solving the zero-shot problem requires some auxiliary information that connects the learned knowledge from the training phase with the unseen classes.

Zero-shot learning in this sense has also been applied to NLP tasks like text summarization and machine translation. [Liu et al., (2019)](https://arxiv.org/pdf/1910.00998.pdf) proposed a denoising autoencoder for text summarization trained only on source paragraphs. The model encodes the training paragraphs and each sentence of the paragraph in a shared space. It generates the output summary by decoding the encoded paragraph. The denoising objective plays an important role here. It works in a self-supervised way and serves as data augmentation. In machine translation, zero-shot learning can help create example pairs for language pairs that do not have parallel corpora ( [Firat et al., 2016](https://arxiv.org/abs/1606.04164) ; [Johnson et al., 2016](https://arxiv.org/abs/1611.04558) ). Zero-shot learning is also used in semantic utterance classification [(Dauphin et al., 2014)](https://arxiv.org/abs/1401.0509) to classify new semantic classes.

### Few-Shot Learning

Few-shot learning is **a setting where the system is given only a very small number of supervised examples** [ (Wang et al., 2019)](https://arxiv.org/abs/1904.05046) . Few-shot usually means two to five examples per class, but it can also be up to 100 examples [(Wang et al., 2021)](https://dl.acm.org/doi/abs/10.1145/3447548.3467235) . When only one example is given, it is called **one-shot learning.**  Typically, systems in a few-shot setting would require some prior knowledge (e.g. a pre-trained language model) to compensate for the small number of training examples. In prompt engineering, examples are often added directly to the prompt. Other applications in few-shot settings include parsing [(Joshi et al., 2018)](https://aclanthology.org/P18-1110) , translation [(Kaiser et al., 2017)](https://arxiv.org/abs/1703.03129) , question answering [(Chada and Natarajan, 2021)](https://arxiv.org/abs/2109.01951) , and relation classification [(Han et al., 2018)](https://aclanthology.org/D18-1514) .

## **A Guide to Utilizing Prompt Engineering**

There are many factors that could affect the performance of a prompt-based system, such as the choice of the language model, how the prompt is formulated, and whether the language model parameters are tuned or frozen. I will discuss them in this section.

Suppose you want to solve an NLP task using the prompt engineering approach. How can you get started? First, let’s categorize NLP tasks into  **text classification**  and **text generation** tasks. It will help you later when selecting the other components.

- **Text classification** is, for example, topic labeling, sentiment analysis, named entity recognition, and natural language inference.
- **Text generation** is, for example, translation, text summarization, and open-domain question answering.

### **Choosing the Language Model**

There are a number of LMs that have been proposed so far. They differ in their structure, training objective, domain, and language. Which one should you choose? Here are the three popular types of LMs, categorized by their training method and directionality.

1.  **Left-to-right Language Models:**   Left-to-right LMs are trained to predict the next token given a sequence of tokens, from left to right, one token at a time. Language models trained in this way are also known as[ autoregressive models](https://huggingface.co/docs/transformers/model_summary). Left-to-right LMs language models have been dominant until recently with the introduction of masked language models.

**Models:** [ GPT-3](https://arxiv.org/abs/2005.14165) , [ GPT-2](https://www.semanticscholar.org/paper/Language-Models-are-Unsupervised-Multitask-Learners-Radford-Wu/9405cc0d6169988371b2755e573cc28650d14dfe?p2df) , [ GPT-Neo](https://github.com/EleutherAI/gpt-neo)

**Application:** text classification & text generation

2.  **Masked Language Models:** A masked language model (MLM) is given a text as input in which several tokens are masked. It is then trained to correctly predict these masked positions. MLM is a variant of [ autoencoding models](https://huggingface.co/docs/transformers/model_summary) , which refer to models that have been trained with corrupted input sequences and attempt to reconstruct the original sequence. One of the most popular models of this type is BERT, which is based on bidirectional transformers. In general, MLMs are better suited for text classification tasks than left-to-right LMs. The reason is that text classification tasks can often be formulated as cloze texts, which aligns with the training objective of MLMs. BERT-based models are not suitable for text generation tasks due to their training objective, bidirectionality, and output format, which are not optimized for generating texts. However, several works have shown ways to use BERT for text generation, such as the works of [ Chen et al. (2019)](https://arxiv.org/abs/1911.03829) and [ Wang and Cho (2019)](https://arxiv.org/abs/1902.04094) .

**Models:** [ BERT](https://aclanthology.org/N19-1423/) , [ RoBERTa](https://arxiv.org/abs/1907.11692) , [ ERNIE](https://arxiv.org/abs/1904.09223) , and their variants.

**Application:** text classification

3.  **Encoder-Decoder Language Models:**  Encoder-decoder models (also known as sequence-to-sequence models) are a common architecture for conditional text generation tasks such as machine translation or text summarization, where the output is not a direct mapping of the input [ (Jurafsky and Martin, 2009)](https://web.stanford.edu/~jurafsky/slp3/10.pdf) . Encoder-decoder language models can be naturally used for text generation tasks. They also work for non-generation tasks that can be reformulated as generation problems in the form of prompts. For example, [ information extraction](https://arxiv.org/abs/2106.01760) and [ question answering](https://arxiv.org/abs/2005.00700) **.**

**Encoder-decoder LMs:** [ UniLM 1](https://arxiv.org/abs/1905.03197) , [ UniLM 2](https://arxiv.org/abs/2002.12804) , [ ERNIE-M](https://arxiv.org/abs/2012.15674) , [ T5](https://jmlr.org/papers/v21/20-074.html) , [ BART](https://aclanthology.org/2020.acl-main.703/) , [ MASS](https://arxiv.org/abs/1905.02450)

**Application:** text classification & text generation

### **Designing the Prompt (Prompt Engineering)**

After choosing the LM, the next step is to design the prompt. Depending on their shape, text prompts can be categorized into cloze prompts and prefix prompts.

- **Cloze prompts** are prompts in which one or more positions are hidden (masked) from the LM. The task of the LM is to fill these masked positions with text strings.

 I don’t like this movie at all. It was such a \[MASK\] movie. I would \[MASK\] recommend it.

- **Prefix prompts** are prompts that do not contain masked positions. The prompt is formulated as a text that should be continued by the LM.

 The translation of “Ich arbeite von zu Hause aus“ is \_\_\_\_\_\_\_\_\_\_\_\_\_\_\_

When choosing the prompt, we consider both the target task and the LM. For text generation tasks with a left-to-right autoregressive LM, prefix prompts are a good choice because they align with the left-to-right nature of the model and how the model is trained. Bidirectional models often underperform in text generation [ (Mangal et al., 2019)](https://arxiv.org/abs/1908.04332) . For classification tasks with a masked LM, cloze prompts are a good solution as they match the pre-training objective of the LM. Encoder-decoder LMs with original text reconstruction objectives are more versatile and can be used with both cloze and prefix prompts. You may want to take a look at the paper from [ Liu et al. (2021)](https://www.semanticscholar.org/paper/Pre-train%2C-Prompt%2C-and-Predict%3A-A-Systematic-Survey-Liu-Yuan/28692beece311a90f5fa1ca2ec9d0c2ce293d069) . They have provided a very useful and comprehensive list of language models and their applicable tasks (page 46).

There are two ways to obtain the prompts: create them manually, or use an algorithm to compute them automatically. Typically, a prompt consists of three components: actual input, task description, and optionally some demonstrations.

The service was rude. This review is negative. \
 The room was clean and beautifully decorated. This review is positive. \
 I love this product! This review is  \_\_\_\_\_\_\_

#### **Manually designed prompts**

One way to obtain the prompts is to design them manually. This process can require a lot of trial and error. You may use your intuition to formulate the task description, arrange the components, and see if the prompt works well with your language model. Below are some commonly used prompts taken from [ Liu et al. (2021).](https://www.semanticscholar.org/paper/Pre-train%2C-Prompt%2C-and-Predict%3A-A-Systematic-Survey-Liu-Yuan/28692beece311a90f5fa1ca2ec9d0c2ce293d069)  You can try the prompt with OpenAI’s GPT3 playground and Huggingface’s [BERT API](https://huggingface.co/bert-base-uncased).

#### **Text classification tasks**

Sentiment analysis

I love this product! Is this review positive?  \[MASK\]

**I love this product!  It was \[MASK\].**

For text classification tasks, the answer generated by the language model is later mapped to the actual class label. See the section “Answer Engineering“ below for more details.

#### **Text generation tasks**

Text Summarization

Text: \[input text\] Summary: \_\_\_\_\_\_\_\_\_\_\_

\[input text\] TL;DR: \_\_\_\_\_\_\_\_\_\_\_

\[input text\] In summary, \_\_\_\_\_\_\_\_\_\_\_

Machine Translation

French: \[French sentence\] English: \_\_\_\_\_\_\_\_\_\_\_

A French sentence is provided: \[French sentence\]

The French translator translates the sentence into English: \_\_\_\_\_\_\_\_\_\_\_

If you have some labeled data, you may want to add them as **demonstrations** to the prompt. We usually use a couple of demonstrations per class. 

German: Der Himmel ist blau          English: The sky is blue      

German: Heute ist es sonnig           English: Today it is sunny   

German: Ich liebe meinen Hund English:  \_\_\_\_\_\_\_\_\_\_\_           

The order and content of each component can greatly affect the model prediction ( [Lu et al., 2021](https://arxiv.org/abs/2104.08786) ; [ Rubin et al., 2022](https://arxiv.org/abs/2112.08633#:~:text=In-context) ; [ Gao et al., 2020](https://arxiv.org/abs/2012.15723) ). Giving more textual context does not always lead to better performance. Sometimes a simple prompt could also yield better performance than a complex one ( [Petroni et al., 2020](https://arxiv.org/abs/2005.04611) ; [ Reynolds and McDonell, 2021](https://arxiv.org/abs/2102.07350) ). Moreover, it is not always helpful to add demonstrations ( [Reynolds and McDonell, 2021](https://arxiv.org/abs/2102.07350) ). Check out the paper from [ Mishra et al. (2021)](https://arxiv.org/abs/2104.08773) for a guideline on how to construct natural language prompts and things to avoid.

There is also an **ensemble approach** that makes use of multiple prompts. The input is applied to several different prompt templates. All these prompts are then passed to the model one by one. The final output can generally be obtained by averaging the results from all the prompts ( [Jiang et al., 2020](https://direct.mit.edu/tacl/article/doi/10.1162/tacl_a_00324/96460/How-Can-We-Know-What-Language-Models-Know) ; [ Schick and Schütze, 2020](https://arxiv.org/abs/2012.11926) , [  2021a](https://arxiv.org/abs/2001.07676) , [ 2021b](https://arxiv.org/abs/2009.07118) ; [ Liu et al., 2021](https://arxiv.org/abs/2104.07210) ).

To give you some ideas, here are works that provide prompt templates for a variety of tasks that you can try: [ Brown et al. (2020)](https://arxiv.org/abs/2005.14165) , [Petroni et al. (2019)](https://aclanthology.org/D19-1250/) , [Schick and Schütze (2020](https://arxiv.org/abs/2012.11926) , [ 2021a](https://arxiv.org/abs/2001.07676) , [ 2021b),](https://arxiv.org/abs/2009.07118) and more in the further reading section at the end of this article.

#### **Automatic prompt search**

As you can see, manually designing the prompts is not an easy task and could require a lot of experimentation and expertise. To address this problem, many methods to automate the template design process have been proposed. Example works on  **automatic textual prompts**  (also called  **discrete prompt / hard prompt)** include prompt mining ( [Jiang et al., 2020](https://arxiv.org/abs/1911.12543) ), prompt paraphrasing ( [Jiang et al. 2020](https://arxiv.org/abs/1911.12543) ; [ Yuan et al., 2021](https://arxiv.org/abs/2106.11520) ; [ Haviv et al., 2021](https://aclanthology.org/2021.eacl-main.316/) ), and gradient-based search [ ( Wallace et al., 2019](https://aclanthology.org/D19-1221/) ; [ Shin et al., 2020](https://arxiv.org/abs/2010.15980) ). Note that most approaches require a large amount of annotated data to find the prompts, which arguably may not be considered true zero-shot or few-shot.

There is another form of prompts, namely **continuous prompts (soft prompts),** where the prompt is expressed directly in the language model’s embedding space. The advantage is that the prompt has its own parameters in the model that can be tuned on the training data of the target task, rather than simply being represented like other input tokens. An example of a continuous prompt approach is prefix tuning. **Prefix tuning**  ( [Li and Liang, 2021](https://arxiv.org/abs/2101.00190) ) can be viewed as a lightweight alternative to fine-tuning. Here, some randomized word vectors (called the prefix) are prepended to the input vector and only these prefix vectors will be trained, by using a small amount of training data. The rest of the LM parameters are frozen. In general, initializing the prefix with embeddings of some real words results in better performance than initializing it with purely random vectors. Other methods include [ Lester et al., 2021](https://arxiv.org/abs/2104.08691) ; [ Zhong et al., 2021](https://arxiv.org/abs/2104.05240) ; [ Qin and Eisner, 2021](https://aclanthology.org/2021.naacl-main.410/) ; [ Hambardzumyan et al., 2021](https://arxiv.org/abs/2101.00121) ; [ Liu et al., 2021](https://arxiv.org/abs/2103.10385) ; [ Han et al., 2021](https://arxiv.org/abs/2105.11259) .

#### **Prompt-based fine-tuning**

Another interesting approach that has shown competitive performance, particularly in few-shot scenarios, is the combination of fine-tuning and prompting. This method is similar to the standard fine-tuning but it transforms the input samples into prompts before using them in fine-tuning. Examples of these methods include [ LM-BFF](https://arxiv.org/abs/2012.15723) , [ PET for text classification](https://arxiv.org/abs/2001.07676) , [ and PET for text generation](https://arxiv.org/abs/2012.11926) . This kind of fine-tuning allows the language model to better understand the task that the prompt is asking it to perform.

### **Answer Engineering**

For text generation, the output of the language model can usually be used directly as the final output, e.g., in machine translation or text summarization. However, for tasks that aim to classify the input into a specific class, we need an additional step to assign the LM output to the target class. For example, to fill in a masked token in a cloze-prompt, the BERT model basically calculates the probability that each word in the vocabulary occurs at that position and selects the word with the highest probability as the answer. To derive a class label from this answer, we need to define a mapping from the model answer to the class label. This process is also called **label mapping** and the mapping is called **the verbalizer** .

For example, sentiment analysis with 3 classes

class positive \= { “great“, “good“, “nice“ }

class negative \= { “terrible, “bad“, worse“ }

class neutral \= { “OK“, “fine“, “acceptable“}

The class that contains the word with the highest probability will be selected as the final class. For example, if “ *good“*  gets the highest probability, we assign “ *positive“*  as the final class. This also means that the user will need access to the model output probabilities, which could be difficult if the model is not open-source. Fortunately, for GPT3, there is a [ way](https://medium.com/edge-analytics/getting-the-most-out-of-gpt-3-based-text-classifiers-part-three-77305628f472) to achieve this.

The design of the label mapping is as important as the design of the prompt. The choice of representative words for each class obviously influences the performance. Works on answer engineering include, for example, [ Schick et al., 2021](https://arxiv.org/abs/2001.07676) ; [ Schick and Schütze, 2020](https://arxiv.org/abs/2012.11926) ; [ Gao et al., 2020](https://arxiv.org/abs/2012.15723) .

## **Conclusion**

Prompt engineering is a powerful technique that allows us to employ a pre-trained language model for a variety of NLP tasks without fine-tuning it. Instead of fine-tuning, the model is given a natural language task description directly along with the input. This technique is particularly useful for large LMs such as GPT3, where the model is so large that fine-tuning becomes difficult or very expensive. It is also applicable to smaller language models such as BERT or RoBERTa in a few-shot setting. The biggest challenge, however, lies in designing the prompt so that the model can understand. How to choose the appropriate language model and deriving the final class prediction are also tricky decisions. We hope that our guide to selecting these components will give you a good overview of the topic and help you get started with your prompt engineering project.

**Further readings:**

- Try out GPT3
- [Try out BERT](https://huggingface.co/bert-base-uncased)
- Example prompts for GPT3 from OpenAI 
- Timeline of Prompt Engineering Progress
- [GPT-3 Creative Fiction](https://www.gwern.net/GPT-3)
- [OpenPrompt](https://github.com/thunlp/OpenPrompt) : open-source prompt learning toolkit
- [Must-read papers on prompt learning](https://github.com/thunlp/PromptPapers) organized by topics
- Pretrain, Prompt, Predict : a collection of prompt learning resources such as frequent updates of the latest research, relevant slides, etc.

- [What Makes Good In-Context Examples for GPT-3?](https://arxiv.org/abs/2101.06804) (2021)
- [How Many Data Points is a Prompt Worth?](https://arxiv.org/abs/2103.08493) (2021)
- [Surface Form Competition-Why the Highest Probability Answer Isn’t Always Right](https://arxiv.org/abs/2104.08315) (2021)
