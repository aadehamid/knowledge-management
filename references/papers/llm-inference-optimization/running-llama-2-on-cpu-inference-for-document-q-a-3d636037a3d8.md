title: Running Llama 2 on CPU Inference Locally for Document Q&A | Towards Data Science
description: Clearly explained guide for running quantized open-source LLM applications on CPUs using LLama 2, C Transformers, GGML, and LangChain
author: Kenneth Leung

# Running Llama 2 on CPU Inference Locally for Document Q&A | Towards Data Science

After running the Python script above, the vector store will be generated and saved in the local directory named `'vectorstore/db_faiss'`, and is ready for semantic search and retrieval.

***Note:*** The relatively smaller LLMs, like the 7B model, appear particularly sensitive to formatting. For instance, I got slightly different outputs when I altered the whitespaces and indentation of the prompt template.

We can define a host of [configuration settings](https://github.com/marella/ctransformers#config) for the LLM, such as maximum new tokens, top k value, temperature, and repetition penalty.

***Note*** *:* I set the temperature as 0.01 instead of 0 because I got odd responses (e.g., a long repeated string of the letter E) when the temperature was exactly zero.

### Step 7 - Running a sample query {#step-7---running-a-sample-query}

It is now time to put our application to the test. Upon loading the virtual environment from the project directory, we can run a command in the command line interface (CLI) that comprises our user query.

For example, we can ask about the value of the minimum guarantee payable by Adidas (Manchester United's global technical sponsor) with the following command:

bash

```
poetry run python main.py "How much is the minimum guarantee payable by adidas?"
```

***Note:*** If we are not using Poetry, we can omit the prepended `poetry run`.

### Results {#results}

![Output from user query passed into document Q&amp;A application | Image by author](https://assets.insightmediagroup.io/media/wp-content/uploads/2023/07/1vm9U1WsWIPtYz0fy56oCzg.png "Output from user query passed into document Q&A application | Image by author")

The output shows that we successfully obtained the correct response for our user query (i.e., £750 million), along with the relevant document chunks that are semantically similar to the query.

The total time of 31 seconds for launching the application and generating a response is pretty good, given that we are running it locally on an AMD Ryzen 5600X (which is a good CPU but by no means the best in the market currently).

The result is even more impressive given that running LLM inference on GPUs (e.g., directly on HuggingFace) can also take double-digit seconds.

### Your Mileage May Vary {#your-mileage-may-vary}

Depending on your CPU, the time taken to obtain a response may vary. For example, when I test it out on my laptop, it could go into the range of several minutes.

The thing to note is that getting LLMs to fit into consumer hardware is still in the early stages, so we cannot expect speeds that are on par with OpenAI APIs (which are driven by loads of computing power).

For now, one can certainly consider running this on a more powerful CPU instance, or switching to using GPU instances (such as free ones on Google Colab).

## (5) Next Steps {#5-next-steps}

Now that we have built a document Q&A backend LLM application that runs on CPU inference, there are many exciting steps we can take to bring this project forward.

- Build a frontend chat interface with Streamlit, especially since it has made two major announcements recently: [Integration of Streamlit with LangChain](https://blog.streamlit.io/langchain-streamlit/), and the [launch of Streamlit ChatUI](https://docs.streamlit.io/knowledge-base/tutorials/build-conversational-apps) to build powerful chatbot interfaces easily.
- Dockerize and deploy the application on a cloud CPU instance. While we have explored local inference, the application can easily be ported to the cloud. We can also leverage more powerful CPU instances on the cloud to speed up inference (e.g., [compute-optimized](https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/compute-optimized-instances.html) AWS EC2 instances like c5.4xlarge)
- Experiment with slightly larger LLMs like the [Llama 13B Chat](https://huggingface.co/TheBloke/Llama-2-13B-chat-GGML) model. Since we have worked with 7B models, assessing the performance of slightly larger ones is a good idea since they should theoretically be more accurate and still fit within memory.
- Experiment with smaller quantized formats like the 4-bit and 5-bit (including those with the new k-quant method) to objectively evaluate the differences in inference speed and response quality.
- Leverage local GPU to speed up inference. If we want to test the use of [GPUs on the C Transformers models](https://github.com/marella/ctransformers#gpu), we can do so by running some of the model layers on the GPU. It is useful because Llama is the only model type that has GPU support currently.
- Evaluate the use of [vLLM](https://vllm.readthedocs.io/en/latest/), a high-throughput and memory-efficient inference and serving engine for LLMs. However, utilizing vLLM requires the use of GPUs.

I will work on articles and projects addressing the above ideas in the upcoming weeks, so stay tuned for more insightful generative AI content!

## Before you go {#before-you-go}

I welcome you to **join me on a journey of data science discovery!** Follow this [Medium](https://kennethleungty.medium.com/) page and visit my [GitHub](https://github.com/kennethleungty) to stay updated with more engaging and practical content. Meanwhile, have fun running open-source LLMs on CPU inference!

> [**arXiv Keyword Extraction and Analysis Pipeline with KeyBERT and Taipy**](https://towardsdatascience.com/arxiv-keyword-extraction-and-analysis-pipeline-with-keybert-and-taipy-2972e81d9fa4)
> [**How to Dockerize Machine Learning Applications Built with H2O, MLflow, FastAPI, and Streamlit**](https://towardsdatascience.com/how-to-dockerize-machine-learning-applications-built-with-h2o-mlflow-fastapi-and-streamlit-a56221035eb5)
> [**Micro, Macro & Weighted Averages of F1 Score, Clearly Explained**](https://towardsdatascience.com/micro-macro-weighted-averages-of-f1-score-clearly-explained-b603420b292f)
