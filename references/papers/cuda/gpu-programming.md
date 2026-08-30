[![From Scratch](https://substackcdn.com/image/fetch/$s_!-HWp!,w_40,h_40,c_fill,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2Fbe5ab36b-c782-420e-ac35-c7599a1f77ad_976x976.png)](/)

# [From Scratch](/)

SubscribeSign in

# GPU Programming

### Writing code for massively parallel processors

[![Michal Pitr's avatar](https://substackcdn.com/image/fetch/$s_!f0vx!,w_36,h_36,c_fill,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F73a6e774-038e-424d-afee-ce5041c3e7e0_1080x1080.jpeg)](https://substack.com/%40michalpitr)

[Michal Pitr](https://substack.com/%40michalpitr)

May 04, 2024

50

1

3

Share

One of the great joys of software engineering is dispelling magic. I’ve written code that executed on a GPU using frameworks like PyTorch or TensorFlow, but I never understood the “how”. It’s time to dispel the magic of GPU programming and learn how it works under the hood.

Thanks for reading Michal’s Substack! Subscribe for free to receive new posts and support my work.

Subscribe

## C CUDA basics

C CUDA is Nvidia’s extension of ANSI C. For the most part, it is the same as C with some added syntax and built-in functions. C CUDA gives us control over what parts of our code are executed on the CPU and the GPU. We call code executed on the CPU host code and GPU code device code. Procedures that run on the GPU are for historical reasons called kernels.

Instead of focusing on CUDA itself, let’s write a simple program that blurs an image. I’ll try to fill in the details as needed.

## Blurring images

We want to write code to blur an image on a GPU.

[![](https://substackcdn.com/image/fetch/$s_!ksVf!,w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F60744dcf-3477-4aa5-9cdb-f020a0ba874c_2472x888.png)](https://substackcdn.com/image/fetch/%24s_%21ksVf%21%2Cf_auto%2Cq_auto%3Agood%2Cfl_progressive%3Asteep/https%3A//substack-post-media.s3.amazonaws.com/public/images/60744dcf-3477-4aa5-9cdb-f020a0ba874c_2472x888.png)

Blurring an image with a GPU

Here’s roughly what our code needs to do:

* Load the image in the host code
* Allocate memory on the GPU
* Copy over the input image to the GPU
* Blur the image with a kernel
* Copy over the output image to the CPU
* Save the output image to the disk

First, we need to know how an image is represented in memory and how to blur it. An RGB image is usually thought of as a 3-dimensional matrix of shape (channels, height, width). In memory, it’s usually represented as a flat array in row-major order. Our GPU code will assume this format.

[![](https://substackcdn.com/image/fetch/$s_!--Nz!,w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F124766e9-d68d-4638-a87d-3ee650fe8327_1600x339.png)](https://substackcdn.com/image/fetch/%24s_%21--Nz%21%2Cf_auto%2Cq_auto%3Agood%2Cfl_progressive%3Asteep/https%3A//substack-post-media.s3.amazonaws.com/public/images/124766e9-d68d-4638-a87d-3ee650fe8327_1600x339.png)

RGB image represented in row-major order

To access the (n, row, col) pixel in a 3-channel image, we can use the following expression.

```
i = (row*WIDTH + column)*3 + n
```

To blur an image, we calculate the value of each pixel as the average of surrounding pixels and write the result into the output image.

[![](https://substackcdn.com/image/fetch/$s_!2dOp!,w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F5448b7b4-f384-48d8-8acb-f8b9ffa28bd9_2155x752.png)](https://substackcdn.com/image/fetch/%24s_%212dOp%21%2Cf_auto%2Cq_auto%3Agood%2Cfl_progressive%3Asteep/https%3A//substack-post-media.s3.amazonaws.com/public/images/5448b7b4-f384-48d8-8acb-f8b9ffa28bd9_2155x752.png)

Image blurring with a 3x3 blur filter

Can we parallelize this? Of course! Each output pixel only depends on the input image, but has no dependencies on other outputs. If we had a processor with width\*height cores, we could process every output pixel in parallel. Turns out, that’s pretty much what GPUs are!

## Writing a kernel

Let’s finally write our kernel. It will closely follow the 2D example above, but generalize it to n-channel images.

A kernel in execution is called a thread. Each thread will compute the RGB channels for a single pixel in the image.

To tell each thread which pixel to compute, CUDA automatically injects variables blockIdx, blockDim, and threadIdx into the kernel. We use these to determine which pixel a given thread should process.

Once we know which pixel we are processing, we iterate over the neighboring pixels and accumulate their red, green, and blue values in pixVarR, pixVarG, and pixVarB. We also count the number of pixels we’ve iterated over, to handle cases where the blur-radius reaches beyond the edges of the image.

[![](https://substackcdn.com/image/fetch/$s_!01gH!,w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F1d3246d4-f6e3-409d-b86c-afa998fda546_1622x754.png)](https://substackcdn.com/image/fetch/%24s_%2101gH%21%2Cf_auto%2Cq_auto%3Agood%2Cfl_progressive%3Asteep/https%3A//substack-post-media.s3.amazonaws.com/public/images/1d3246d4-f6e3-409d-b86c-afa998fda546_1622x754.png)

Applying the blur filter in edge-cases

Note that the coordinate calculation might feel unnatural since the image is flattened as discussed earlier.

[![](https://substackcdn.com/image/fetch/$s_!o0WV!,w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F0ecc1921-0240-43c7-8345-b925ab5adc2a_1780x1412.png)](https://substackcdn.com/image/fetch/%24s_%21o0WV%21%2Cf_auto%2Cq_auto%3Agood%2Cfl_progressive%3Asteep/https%3A//substack-post-media.s3.amazonaws.com/public/images/0ecc1921-0240-43c7-8345-b925ab5adc2a_1780x1412.png)

Blur kernel source code

You might notice the special \_\_global\_\_ identifier before the kernel name. This is how we specify that a procedure is a kernel and should be compiled to run on the GPU. It’s also how the c cuda compiler (NVCC) knows to inject the blockIdx, blockDim, and threadIdx variables.

Now that we have the kernel, let’s briefly write the main function to set things up and run it. The main function closely follows the setup steps outlined earlier. The cuda-prefixed functions are automatically included by the NVCC compiler. These mostly copy built-in C functions to provide similar functionality but for GPUs.

[![](https://substackcdn.com/image/fetch/$s_!eUGJ!,w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F24675370-6885-46e7-a440-a1fd494db5a2_2048x2046.png)](https://substackcdn.com/image/fetch/%24s_%21eUGJ%21%2Cf_auto%2Cq_auto%3Agood%2Cfl_progressive%3Asteep/https%3A//substack-post-media.s3.amazonaws.com/public/images/24675370-6885-46e7-a440-a1fd494db5a2_2048x2046.png)

Main function source code. Main sets up GPU to run blur kernel.

The interesting part is when we call the kernel.

[![](https://substackcdn.com/image/fetch/$s_!uME_!,w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F98b262d9-d717-4eb8-9379-644ae6dc63ce_1494x518.png)](https://substackcdn.com/image/fetch/%24s_%21uME_%21%2Cf_auto%2Cq_auto%3Agood%2Cfl_progressive%3Asteep/https%3A//substack-post-media.s3.amazonaws.com/public/images/98b262d9-d717-4eb8-9379-644ae6dc63ce_1494x518.png)

We designed our kernel to process all channels for a given pixel with a single thread.

To make it concrete, our lenna.png is of shape 512 \* 512, so we need that many threads to process the whole image. My GPU, however, only has 1920 CUDA cores. That’s fine - GPUs are much faster at context switching than CPUs, so having more threads than physical cores is desirable to maximize throughput.

To do this, we use the <<<dimGrid, dimBlock>>> syntax. Each argument is a struct with 3 fields {x, y, z}. The configuration generally follows the shape of the input data. Since each thread corresponds to a single pixel, a natural division is using 2 dimensions. We can group threads into 16x16 blocks, meaning each block will process a 16x16 patch of the image. Since we are not using multiple threads per the z-axis, we leave it at 1.

The grid dimension tells us how many blocks per dimension to create. Since we need to cover the whole image using 16x16 patches, we need width/16 blocks in the *x* direction and height/16 blocks in the *y* direction. In case our image dimensions don’t divide evenly by 16, we round up.

This rounding means that we might need to spawn some extra blocks where only some threads are utilized. To make sure these unused threads behave correctly, we added the conditional check in our kernel. Only threads that have a corresponding pixel will do some work!

Oof, that’s a lot of low-level details!

So, why 16x16 blocks? In our case, it is pretty arbitrary. We could’ve used 8x8 blocks or 32x32. There’s a hard limit on the number of threads per block, which is usually 1024 on recent cards. As far as I understand, properly organizing threads into blocks and threads can improve performance thanks to memory locality.

Finally, let’s compile and run our code. I’m using BLUR\_SIZE=31 for an extra blurry effect.

We can compile this with NVCC to yield the blurred image.

[![](https://substackcdn.com/image/fetch/$s_!2odI!,w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2Fc20e4d37-a08b-4a19-aad9-831df1ed260a_1158x370.png)](https://substackcdn.com/image/fetch/%24s_%212odI%21%2Cf_auto%2Cq_auto%3Agood%2Cfl_progressive%3Asteep/https%3A//substack-post-media.s3.amazonaws.com/public/images/c20e4d37-a08b-4a19-aad9-831df1ed260a_1158x370.png)

Compilation command

[![](https://substackcdn.com/image/fetch/$s_!drTc!,w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2Fe7280cf8-1c95-4f77-b0ab-b9fbd0a41ff1_982x982.png)](https://substackcdn.com/image/fetch/%24s_%21drTc%21%2Cf_auto%2Cq_auto%3Agood%2Cfl_progressive%3Asteep/https%3A//substack-post-media.s3.amazonaws.com/public/images/e7280cf8-1c95-4f77-b0ab-b9fbd0a41ff1_982x982.png)

Blurred Lenna with blur\_size=31. You can learn more about the history of Lenna at [lenna.org](http://lenna.org)

You might wonder if we can run code directly on the GPU without any intervention of the CPU. As far as I know, that’s not possible. Our executable makes calls to cuda runtime API, which in turn communicates with the GPU drivers. However, it’s possible to chain kernels to keep as much of the computation on the GPU without CPU intervention.

If you would like to learn more about this area, consider checking out the book Programming Massively Parallel Processors and the official CUDA C++ Programming guide from Nvidia.

Thanks for reading Michal’s Substack! Subscribe for free to receive new posts and support my work.

Subscribe

---

Thanks for reading! If you enjoyed this write-up, you might enjoy my previous one where I explain [how MapReduce works by building it from scratch](https://michalpitr.substack.com/p/mapreduce-from-scratch)!

Researching and writing these articles takes a lot of time and effort. To ensure you don’t miss the next one, consider subscribing or following me on [LinkedIn](https://www.linkedin.com/in/michal-pitr-a7156b127/).

Do you know someone who might be interested in GPU programming? Consider sharing the post with them.

[Share](https://michalpitr.substack.com/p/gpu-programming?utm_source=substack&utm_medium=email&utm_content=share&action=share)

50

1

3

Share

PreviousNext

#### Discussion about this post

CommentsRestacks

![User's avatar](https://substackcdn.com/image/fetch/$s_!TnFC!,w_32,h_32,c_fill,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack.com%2Fimg%2Favatars%2Fdefault-light.png)

[![Fukitol's avatar](https://substackcdn.com/image/fetch/$s_!mHHn!,w_32,h_32,c_fill,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F4d0c149e-def9-409b-ae6a-835230643e0c_144x144.png)](https://substack.com/profile/11917457-fukitol?utm_source=comment)

[Fukitol](https://substack.com/profile/11917457-fukitol?utm_source=substack-feed-item)

[Apr 8, 2025](https://michalpitr.substack.com/p/gpu-programming/comment/107120171 "Apr 8, 2025, 6:44 PM")

Liked by Michal Pitr

Seems pretty similar conceptually to working with shaders, which is not surprising.

Notably there's some weird stripey artifacts in your output that I would guess are the edge pixels of each block being blurred with the void. But I didn't go over your code in detail.

ReplyShare

TopLatestDiscussions

No posts

### Ready for more?

Subscribe

© 2026 Michal Pitr · [Privacy](https://substack.com/privacy) ∙ [Terms](https://substack.com/tos) ∙ [Collection notice](https://substack.com/ccpa#personal-data-collected)

[Start your Substack](https://substack.com/signup?utm_source=substack&utm_medium=web&utm_content=footer)[Get the app](https://substack.com/app/app-store-redirect?utm_campaign=app-marketing&utm_content=web-footer-button)

[Substack](https://substack.com) is the home for great culture

This site requires JavaScript to run correctly. Please [turn on JavaScript](https://enable-javascript.com/) or unblock scripts