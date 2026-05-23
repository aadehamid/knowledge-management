[![Hugging Face's logo](/front/assets/huggingface_logo-noborder.svg) Hugging Face](/)

* [Models](/models)
* [Datasets](/datasets)
* [Spaces](/spaces)
* [Buckets new](/storage)
* [Docs](/docs)
* [Enterprise](/enterprise)
* [Pricing](/pricing)
* + Website

    - [Tasks](/tasks)
    - [HuggingChat](/chat)
    - [Collections](/collections)
    - [Languages](/languages)
    - [Organizations](/organizations)
  + Community

    - [Blog](/blog)
    - [Posts](/posts)
    - [Daily Papers](/papers)
    - [Learn](/learn)
    - [Discord](/join/discord)
    - [Forum](https://discuss.huggingface.co/)
    - [GitHub](https://github.com/huggingface)
  + Solutions

    - [Team & Enterprise](/enterprise)
    - [Hugging Face PRO](/pro)
    - [Enterprise Support](/support)
    - [Inference Providers](/inference/models)
    - [Inference Endpoints](/inference-endpoints)
    - [Storage Buckets](/storage)
* ---
* [Log In](/login)
* [Sign Up](/join)

[Back to Articles](/blog)

# KV Caching Explained: Optimizing Transformer Inference Efficiency

[Community Article](/blog/community) Published
January 30, 2025

[[ ]   Upvote

334](/login?next=%2Fblog%2Fnot-lain%2Fkv-caching)

* [![](https://cdn-avatars.huggingface.co/v1/production/uploads/5eb609664e876668a0c377b7/abN9STfU6biiOTYY3-nbK.jpeg)](/salti "salti")
* [![](https://cdn-avatars.huggingface.co/v1/production/uploads/1608146735109-5fcfb7c407408029ba3577e2.png)](/sbrandeis "sbrandeis")
* [![](https://cdn-avatars.huggingface.co/v1/production/uploads/1617264212503-603d25b75f9d390ab190b777.jpeg)](/pcuenq "pcuenq")
* [![](/avatars/e9e7a5ce25531f7a3d2cdc100401883d.svg)](/RishuD7 "RishuD7")
* [![](https://cdn-avatars.huggingface.co/v1/production/uploads/608aabf24955d2bfc3cd99c6/-YxmtpzEmf3NKOTktODRP.jpeg)](/ariG23498 "ariG23498")
* [![](https://cdn-avatars.huggingface.co/v1/production/uploads/60a551a34ecc5d054c8ad93e/dhcBFtwNLcKqqASxniyVw.jpeg)](/mishig "mishig")
 * +328

[![Not Lain's avatar](https://cdn-avatars.huggingface.co/v1/production/uploads/6527e89a8808d80ccff88b7a/CuGNmF1Et8KMQ0mCd1NEJ.jpeg)](/not-lain)

[Not Lain

not-lain

Follow](/not-lain)

## * [Introduction](#introduction "Introduction") * [Prerequisites](#prerequisites "Prerequisites") * [Standard Inference and the Rise of KV Caching](#standard-inference-and-the-rise-of-kv-caching "Standard Inference and the Rise of KV Caching") * [How Does KV Caching Work?](#how-does-kv-caching-work "How Does KV Caching Work?") + [Step-by-Step Process](#step-by-step-process "Step-by-Step Process") * [Comparison: KV Caching vs. Standard Inference](#comparison-kv-caching-vs-standard-inference "Comparison: KV Caching vs. Standard Inference") * [Practical Implementation](#practical-implementation "Practical Implementation") * [Conclusion](#conclusion "Conclusion") * [References & Further Reading](#references--further-reading "References &amp; Further Reading") Introduction

When AI models generate text, they often repeat many of the same calculations, which can slow things down. **Key-Value caching** is a technique that helps speed up this process by remembering important information from previous steps. Instead of recomputing everything from scratch, the model reuses what it has already calculated, making text generation much faster and more efficient.

In this blogpost, we’ll break down KV caching in an easy-to-understand way, explain why it’s useful, and show how it helps AI models work faster.

![](https://cdn-uploads.huggingface.co/production/uploads/6527e89a8808d80ccff88b7a/ZiRajz9XfXiPT3NIM05FS.png)

## Prerequisites

To fully grasp the content, readers should be familiar with:

1. **Transformer Architecture**: Familiarity with components such as the attention mechanism.
2. **Autoregressive Modeling**: Understanding of how models like GPT generate sequences.
3. **Linear Algebra Basics**: Concepts like matrix multiplication and transposition, which are essential for understanding attention computations.

This 👉 [**BLOG**](https://huggingface.co/blog/not-lain/tensor-dims) should cover up most of the prerequisites needed for this article.

click here for some of the most essential takeaways.

* attention weight has a shape of [batch,h,Seqlen,Seqlen] [\text{batch}, h, \mathrm{Seq}\_{\mathrm{len}}, \mathrm{Seq}\_{\mathrm{len}}] [batch,h,Seqlen​,Seqlen​]
* masked multi-head attention allows each token to be represented by itself and all the previous tokens.
* to generate a new token the model needs to look at all the previous tokens and their representations by their preceding tokens

[![](https://cdn-uploads.huggingface.co/production/uploads/6527e89a8808d80ccff88b7a/9n4ttDGvMkcZKF8puUBz0.png)
![](https://cdn-uploads.huggingface.co/production/uploads/6527e89a8808d80ccff88b7a/zPsMCUsd_ohKun4r2axV0.png)](https://huggingface.co/blog/not-lain/tensor-dims)

[](https://cdn-uploads.huggingface.co/production/uploads/6527e89a8808d80ccff88b7a/RsRm-SLIpIXdRwALshIB-.mp4)

## Standard Inference and the Rise of KV Caching

When a model generates text, it **looks at all the previous tokens** to predict the next one. Normally, it would *repeat the same calculations* for every new token, which can slow things down.

[](https://cdn-uploads.huggingface.co/production/uploads/6527e89a8808d80ccff88b7a/PWI-EwqizVLInztmiI7Eo.mp4)

> KV caching solves compute overlap by **remembering these calculations** from previous steps, this can be achieved by storing the intermediate states of attention layers during inference.

[](https://cdn-uploads.huggingface.co/production/uploads/6527e89a8808d80ccff88b7a/HnzDhoJdAbJhSassYjzEy.mp4)

## How Does KV Caching Work?

### Step-by-Step Process

1. **First Generation**: When the model sees the first input, it calculates and stores its keys and values in the cache.
   ⇓ \Downarrow ⇓
2. **Next Words**: For each new word, the model retrieves the stored keys and values and adds the new ones instead of starting over.
3. **Efficient Attention Computation**: calculate attention using the cached KKK and VVV along with the new QQQ (query) to compute the output.
4. **Update Input**: add the newly generated token to the input and go back to step 2\texttt{go back to step 2} go back to step 2 until we finish generating.

![](https://cdn-uploads.huggingface.co/production/uploads/6527e89a8808d80ccff88b7a/DbL2RbXFRoMWA5CrOaGB8.png)

The process is illustrated below:

```
Token 1: [K1, V1] ➔ Cache: [K1, V1]
Token 2: [K2, V2] ➔ Cache: [K1, K2], [V1, V2]
...
Token n: [Kn, Vn] ➔ Cache: [K1, K2, ..., Kn], [V1, V2, ..., Vn]
```

| KV Caching | Standard Inference |
| --- | --- |
|  |  |

In the table above we used a dk=5d\_k = 5 dk​=5 for better visuals, note that this number can be significantly bigger than what we have presented.

## Comparison: KV Caching vs. Standard Inference

Here’s how KV caching compares to the regular generations :

| **Feature** | **Standard Inference** | **KV Caching** |
| --- | --- | --- |
| **Computation per Word** | The model repeats the same calculations for every word. | The model reuses past calculations for faster results. |
| **Memory Usage** | Uses less memory at each step, but memory grows with longer texts. | Uses extra memory to store past information, but keeps things efficient. |
| **Speed** | Gets slower as the text gets longer because it repeats work. | Stays fast even with longer texts by avoiding repeated work. |
| **Efficiency** | High computational cost and slower response times. | Faster and more efficient since the model remembers past work. |
| **Handling Long Texts** | Struggles with long texts due to repeated calculations. | Perfect for long texts as it remembers past steps. |

KV caching makes a big difference in **speed** and **efficiency**, especially for long texts. By saving and reusing past calculations, it avoids the need to start over each time, making it much faster than the regular way of generating text.

## Practical Implementation

This is a simplified example of implementing KV caching in PyTorch:

```
# Pseudocode for KV Caching in PyTorch
class KVCache:
    def __init__(self):
        self.cache = {"key": None, "value": None}

    def update(self, key, value):
        if self.cache["key"] is None:
            self.cache["key"] = key
            self.cache["value"] = value
        else:
            self.cache["key"] = torch.cat([self.cache["key"], key], dim=1)
            self.cache["value"] = torch.cat([self.cache["value"], value], dim=1)

    def get_cache(self):
        return self.cache
```

When using the transformers library this behavior is enabled by default through the `use_cache` parameter, you can also access multiple caching methods through the [`cache_implementation`](https://huggingface.co/docs/transformers/main_classes/text_generation#transformers.GenerationConfig.cache_implementation) parameter, here's a minimalistic code :

```
from transformers import AutoModelForCausalLM, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained('HuggingFaceTB/SmolLM2-1.7B')
model = AutoModelForCausalLM.from_pretrained('HuggingFaceTB/SmolLM2-1.7B').cuda()

tokens = tokenizer.encode("The red cat was", return_tensors="pt").cuda()
output = model.generate(
    tokens, max_new_tokens=300, use_cache = True # by default is set to True
)
output_text = tokenizer.batch_decode(output, skip_special_tokens=True)[0]
```

We benchmarked the code above with/without kv caching on a T4 GPU we got the following results :

| with KV Caching | Standard Inference | Speedup |
| --- | --- | --- |
| 11.7 s | 1min 1s | ~5.21x times faster |

## Conclusion

KV caching is a simple but powerful technique that helps AI models generate text faster and more efficiently. By remembering past calculations instead of repeating them, it reduces the time and effort needed to predict new words. While it does require extra memory, this method is especially useful for long conversations ensuring fast and efficient generation.

Understanding KV caching can help developers and AI enthusiasts build faster, smarter, and more scalable language models for real-world applications.

I would like to extend my sincerest gratitude to [Aritra Roy Gosthipaty](https://hf.co/ariG23498) 🤗 for his invaluable support, feedback, and dedication in developing this blog post.

## References & Further Reading

1. [Transformers KV Caching Explained](https://medium.com/%40joaolages/kv-caching-explained-276520203249)
2. [Transformers Key-Value Caching Explained](https://neptune.ai/blog/transformers-key-value-caching)
3. [Mastering LLM Techniques: Inference Optimization](https://developer.nvidia.com/blog/mastering-llm-techniques-inference-optimization/)
4. [Hugging Face Documentation - KV Caching in Transformers](https://huggingface.co/docs/transformers/main/en/generation_strategies#kv-caching)

More from this author

[![](https://cdn-uploads.huggingface.co/production/uploads/6527e89a8808d80ccff88b7a/gCBPUSQAqS-D9uhEoVCn4.jpeg)

## Visualizing How VLMs Work

* ![](https://cdn-avatars.huggingface.co/v1/production/uploads/6527e89a8808d80ccff88b7a/CuGNmF1Et8KMQ0mCd1NEJ.jpeg)

54

 October 7, 2025](/blog/not-lain/vlms)

[## Mastering Tensor Dimensions in Transformers

* ![](https://cdn-avatars.huggingface.co/v1/production/uploads/6527e89a8808d80ccff88b7a/CuGNmF1Et8KMQ0mCd1NEJ.jpeg)

174

 January 12, 2025](/blog/not-lain/tensor-dims)

### Community

![](/avatars/b34c1d0bdd87b3a091b730b7e9a4f628.svg)

 [ryg81](/ryg81)

       [Jan 31, 2025](#679cd0243cc265a444a09923)

Can this be similar for image generation models? (I am not a programmer :- or expert in AI))

See translation

* [![](/avatars/abcd73d03219f4455f605f3c8f119df0.svg)](/olegGerbylev "olegGerbylev")
* [![](/avatars/513f184423a3e5e87f271957c104e5d9.svg)](/stepanogil "stepanogil")
 * 2 replies

 ·

🔥

2

2

🚀

2

2

+

![](/avatars/abcd73d03219f4455f605f3c8f119df0.svg)

 [olegGerbylev](/olegGerbylev)

       [Apr 1, 2025](#67ec368b0d37308e10786d89)

This comment has been hidden (marked as Spam)

  Expand 1
reply

![](https://cdn-avatars.huggingface.co/v1/production/uploads/6848fcbb2f229c24e5b8c60d/DDhXu6w-0U6sfnrqMPf7I.jpeg)

 [emilibennett](/emilibennett)

       [Jun 11, 2025](#6848fd432b1c7fe843b0c203)

This comment has been hidden (marked as Spam)

![](https://cdn-avatars.huggingface.co/v1/production/uploads/63434b14e2647466b42dbae1/0HnjTfKycdKIzk7Kzh5ty.jpeg)

 [mbcool](/mbcool)

       [Jun 23, 2025](#6858ada7601774b43e2ba061)

Great reference, thanks for posting.

See translation

* [![](https://cdn-avatars.huggingface.co/v1/production/uploads/6527e89a8808d80ccff88b7a/CuGNmF1Et8KMQ0mCd1NEJ.jpeg)](/not-lain "not-lain")
 * 1 reply

 ·

🤗

2

2

+

![](https://cdn-avatars.huggingface.co/v1/production/uploads/6527e89a8808d80ccff88b7a/CuGNmF1Et8KMQ0mCd1NEJ.jpeg)

 [not-lain](/not-lain)

  Article author     [Jul 16, 2025](#6877dfaf11ff0202663a9c79)

Thanks a lot for the kind words (≧∇≦)ﾉ✨

See translation

![](/avatars/f91e81bc551e36002c2ffc18fe740895.svg)

 [dutta18](/dutta18)

       [Sep 13, 2025](#68c4f66d50b2167a8bfe34ad)

I really appreciate the effort that HF team puts in to create these easy-to-digest blogs. Thanks a ton !

See translation

* [![](https://cdn-avatars.huggingface.co/v1/production/uploads/6527e89a8808d80ccff88b7a/CuGNmF1Et8KMQ0mCd1NEJ.jpeg)](/not-lain "not-lain")
 * 1 reply

 ·

🤗

2

2

+

![](https://cdn-avatars.huggingface.co/v1/production/uploads/6527e89a8808d80ccff88b7a/CuGNmF1Et8KMQ0mCd1NEJ.jpeg)

 [not-lain](/not-lain)

  Article author     [Sep 13, 2025](#68c59dad9e06a8db73e025b5)

Very grateful for the kind words [@dutta18](/dutta18)  🤗

See translation

![](/avatars/07d65e53f3937c520b870aa215963253.svg)

 [jonathon1964](/jonathon1964)

       [Sep 26, 2025](#68d60ff7a65200ac91b4101d)

very clear ex!

🤗

3

3

+

Reply

![](https://cdn-avatars.huggingface.co/v1/production/uploads/noauth/4UmxFrc_TEiXcnm3RewZM.jpeg)

 [Student-Xiaoji](/Student-Xiaoji)

       [Oct 26, 2025](#68fd85d5c35dcb59c382a379)

love this simple, easy understood and straight forward explanation❤
thanks for you effort☺

See translation

❤️

2

2

+

Reply

![](https://cdn-avatars.huggingface.co/v1/production/uploads/6527e89a8808d80ccff88b7a/CuGNmF1Et8KMQ0mCd1NEJ.jpeg)

 [not-lain](/not-lain)

  Article author     [Oct 27, 2025](#69000217d65f71e8227fe941)

thanks for the kind feedback 🤗

See translation

Reply

![](/avatars/9ca9aaa27bd82a0ebcdabd26136eeffe.svg)

 [seldn](/seldn)

       [Oct 30, 2025](#6903890155e099299ba66ef1)

This was a great read. Thanks for making this.

See translation

* [![](https://cdn-avatars.huggingface.co/v1/production/uploads/6527e89a8808d80ccff88b7a/CuGNmF1Et8KMQ0mCd1NEJ.jpeg)](/not-lain "not-lain")
 * 1 reply

 ·

👍

3

3

+

![](https://cdn-avatars.huggingface.co/v1/production/uploads/6527e89a8808d80ccff88b7a/CuGNmF1Et8KMQ0mCd1NEJ.jpeg)

 [not-lain](/not-lain)

  Article author     [Nov 16, 2025](#69191d497d3d488fd28cdc10)

don't mention it (\*/ω＼\*)

See translation

![](/avatars/0bb437ac61ea1e24a199af179b9071d1.svg)

 [chunlinyang](/chunlinyang)

       [Nov 7, 2025](#690d5210bdd77022732b6c09)

It helps me understand KV cache better. Ty.

See translation

* [![](https://cdn-avatars.huggingface.co/v1/production/uploads/6527e89a8808d80ccff88b7a/CuGNmF1Et8KMQ0mCd1NEJ.jpeg)](/not-lain "not-lain")
 * 1 reply

 ·

🤗

1

1

+

![](https://cdn-avatars.huggingface.co/v1/production/uploads/6527e89a8808d80ccff88b7a/CuGNmF1Et8KMQ0mCd1NEJ.jpeg)

 [not-lain](/not-lain)

  Article author     [Nov 16, 2025](#69191d1759f85b91230c4eff)

thanks a lot for the kind warm words 🤗

See translation

![](https://cdn-avatars.huggingface.co/v1/production/uploads/no-auth/SxBTz8ntyQNawHWVqqVUN.png)

 [KANGKKANG](/KANGKKANG)

       [Nov 14, 2025](#6916c70222626558d6dd306d)

maybe i can use this job on ACT model?

See translation

👍

1

1

+

Reply

![](https://cdn-avatars.huggingface.co/v1/production/uploads/no-auth/JlLx6zb-QcMSyfOhz_QmZ.png)

 [kyars](/kyars)

       [Dec 7, 2025](#6935ff250712d88b9e3ce84e)

I didn't understand the explanation

See translation

* [![](https://cdn-avatars.huggingface.co/v1/production/uploads/6527e89a8808d80ccff88b7a/CuGNmF1Et8KMQ0mCd1NEJ.jpeg)](/not-lain "not-lain")
* [![](https://cdn-avatars.huggingface.co/v1/production/uploads/no-auth/JlLx6zb-QcMSyfOhz_QmZ.png)](/kyars "kyars")
 * 2 replies

 ·

👍

1

1

+

![](https://cdn-avatars.huggingface.co/v1/production/uploads/6527e89a8808d80ccff88b7a/CuGNmF1Et8KMQ0mCd1NEJ.jpeg)

 [not-lain](/not-lain)

  Article author     [Dec 10, 2025](#6938f9be4ad36d1320fbdc81)

Hi [@kyars](/kyars)  is there any part that you think i can improve upon or is it everything?
would appreciate any feedback!

See translation

  Expand 1
reply

![](/avatars/a54c6176d741354436b823c1c9d5b1b1.svg)

 [talrejaa8](/talrejaa8)

       [Dec 13, 2025](#693d9b325e5078b5f90a6d81)

I really appreciate your effort to explaining this so well. Just one doubt I have, what exactly is being cached?

1. The QK^t dot product results and the Value vectors of the already generated tokens
   or
2. The just the key vectors and the value vectors of already generated tokens?

Also, is this done for each transformer block in an LLM?

See translation

* [![](https://cdn-avatars.huggingface.co/v1/production/uploads/no-auth/JlLx6zb-QcMSyfOhz_QmZ.png)](/kyars "kyars")
 * 1 reply

 ·

![](https://cdn-avatars.huggingface.co/v1/production/uploads/no-auth/JlLx6zb-QcMSyfOhz_QmZ.png)

 [kyars](/kyars)

       [Dec 13, 2025](#693deeb8c7eb6b6bbe489408)

Yes, it's done for each transformer block in an LM because each transformer block has different attention heads. If you do it for only one transformer block across all blocks, then you don't get the same representation.

See translation

![](/avatars/10db736338840f129caf1778dc600549.svg)

 [TimHH](/TimHH)

       [Apr 19](#69e49977c4e5b93dfc051416)

•

[edited Apr 19](#69e49977c4e5b93dfc051416 "Edited 2 times by TimHH")

What dose that "<bos>" mean? Some element, special character, start of message/text (0x02/STX), or what?

Edit: replaced < / > with &lt; / &gt; show it dose show...

See translation

Reply

EditPreview

Upload images, audio, and videos by dragging in the text input, pasting, or clicking here.

Tap or paste here to upload images

Comment

· [Sign up](/join?next=%2Fblog%2Fnot-lain%2Fkv-caching) or [log in](/login?next=%2Fblog%2Fnot-lain%2Fkv-caching) to comment

[[ ]   Upvote

334](/login?next=%2Fblog%2Fnot-lain%2Fkv-caching)

* [![](https://cdn-avatars.huggingface.co/v1/production/uploads/5eb609664e876668a0c377b7/abN9STfU6biiOTYY3-nbK.jpeg)](/salti "salti")
* [![](https://cdn-avatars.huggingface.co/v1/production/uploads/1608146735109-5fcfb7c407408029ba3577e2.png)](/sbrandeis "sbrandeis")
* [![](https://cdn-avatars.huggingface.co/v1/production/uploads/1617264212503-603d25b75f9d390ab190b777.jpeg)](/pcuenq "pcuenq")
* [![](/avatars/e9e7a5ce25531f7a3d2cdc100401883d.svg)](/RishuD7 "RishuD7")
* [![](https://cdn-avatars.huggingface.co/v1/production/uploads/608aabf24955d2bfc3cd99c6/-YxmtpzEmf3NKOTktODRP.jpeg)](/ariG23498 "ariG23498")
* [![](https://cdn-avatars.huggingface.co/v1/production/uploads/60a551a34ecc5d054c8ad93e/dhcBFtwNLcKqqASxniyVw.jpeg)](/mishig "mishig")
* [![](https://cdn-avatars.huggingface.co/v1/production/uploads/1624526712199-60d3257cd7e9cf17e5265abf.jpeg)](/realjanpaulus "realjanpaulus")
* [![](/avatars/d0d726558e0ab9a0acf29a0611d33d0b.svg)](/ltc "ltc")
* [![](https://cdn-avatars.huggingface.co/v1/production/uploads/611bb0fed5038f101c087d64/xxUtx_Z9CHEwcknW6vg_a.png)](/Programmer-RD-AI "Programmer-RD-AI")
* [![](https://cdn-avatars.huggingface.co/v1/production/uploads/6141a88b3a0ec78603c9e784/DJsxSmWV39M33JFheLobC.jpeg)](/merve "merve")
* [![](/avatars/39cc15c0a70e0d2b1f1ef1c7a98e7db8.svg)](/ianyeung "ianyeung")
* [![](https://cdn-avatars.huggingface.co/v1/production/uploads/1633207894360-noauth.jpeg)](/clepelaars "clepelaars")
 * +322

System theme

Company

[TOS](/terms-of-service) [Privacy](/privacy) [About](/huggingface) [Careers](https://apply.workable.com/huggingface/)

Website

[Models](/models) [Datasets](/datasets) [Spaces](/spaces) [Pricing](/pricing) [Docs](/docs)