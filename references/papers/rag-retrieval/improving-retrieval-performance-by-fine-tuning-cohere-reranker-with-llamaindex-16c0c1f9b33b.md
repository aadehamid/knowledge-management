title: Fine-Tune Cohere Reranker for Better RAG | LlamaIndex
description: Fine-tune the Cohere Reranker with LlamaIndex to boost your RAG retrieval performance. Master data curation and custom model evaluation. Discover how.

# Introduction:

Achieving an efficient Retrieval-Augmented-Generation (RAG) pipeline is heavily dependent on robust retrieval performance. As we explored in our previous [blog post](https://medium.com/llamaindex-blog/boosting-rag-picking-the-best-embedding-reranker-models-42d079022e83), rerankers have a significant impact on boosting retrieval performance. But what if we could take it a step further? What if our reranker was not just any reranker, but one tuned specifically to our domain or dataset? Could this specialization enhance the retrieval performance even more?

##  Ready to get started with LlamaParse? 

 Explore our free and paid plans today. 

To answer these questions, we turn to CohereAI’s beta release of fine-tuning reranker(Custom reranker) models. By integrating these with LlamaIndex, we now offer the ability to build your very own Cohere custom reranker using our streamlined process.

In this blog post, we’ll guide you through the steps to create a Cohere custom reranker with LlamaIndex and evaluate the retrieval performance.

For a hands-on walkthrough, you can follow the tutorial on [Google Colab Notebook](https://colab.research.google.com/github/run-llama/llama_index/blob/main/docs/examples/finetuning/rerankers/cohere_custom_reranker.ipynb).

Let’s start fine-tuning a Cohere reranker (custom reranker) with LlamaIndex.

> NOTE: This is a guide for fine-tuning a Cohere reranker (custom reranker). The results presented at the end of this tutorial are unique to the chosen dataset and parameters. We suggest experimenting with your dataset and various parameters before deciding to incorporate it into your RAG pipeline.

# Setting Up the Environment

```
!pip install llama-index cohere pypdf
```

# Setting Up the Keys

```
openai_api_key = 'YOUR OPENAI API KEY'
cohere_api_key = 'YOUR COHEREAI API KEY'

import os

os.environ["OPENAI_API_KEY"] = openai_api_key
os.environ["COHERE_API_KEY"] = cohere_api_key
```

# Download the Data

We will use Lyft 2021 10K SEC Filings for training and Uber 2021 10K SEC Filings for evaluation.

```
!mkdir -p 'data/10k/'
!wget 'https://raw.githubusercontent.com/run-llama/llama_index/main/docs/examples/data/10k/uber_2021.pdf' -O 'data/10k/uber_2021.pdf'
!wget 'https://raw.githubusercontent.com/run-llama/llama_index/main/docs/examples/data/10k/lyft_2021.pdf' -O 'data/10k/lyft_2021.pdf'
```

# Load the Data

```
lyft_docs = SimpleDirectoryReader(input_files=['./data/10k/lyft_2021.pdf']).load_data()
uber_docs = SimpleDirectoryReader(input_files=['./data/10k/uber_2021.pdf']).load_data()
```

# Data Curation

**Create Nodes.**

The [documentation](https://docs.cohere.com/docs/rerank-models) mentions that Query \+ Relevant Passage/ Query \+ Hard Negatives should be less than 510 tokens. To accommodate that we limit `chunk_size`to 400 tokens. (Each chunk will eventually be treated as a Relevant Passage/ Hard Negative)

```
# Limit chunk size to 400
node_parser = SimpleNodeParser.from_defaults(chunk_size=400)

# Create nodes
lyft_nodes = node_parser.get_nodes_from_documents(lyft_docs)
uber_nodes = node_parser.get_nodes_from_documents(uber_docs)
```

We will use gpt-4 to create questions from chunks.

```
llm = OpenAI(api_key=openai_api_key, temperature=0, model='gpt-4')
```

Prompt to generate questions from each Node/ chunk.

```
# Prompt to generate questions
qa_generate_prompt_tmpl = """\
Context information is below.

---------------------
{context_str}
---------------------

Given the context information and not prior knowledge.
generate only questions based on the below query.

You are a Professor. Your task is to setup \
{num_questions_per_chunk} questions for an upcoming \
quiz/examination. The questions should be diverse in nature \
across the document. The questions should not contain options, not start with Q1/ Q2. \
Restrict the questions to the context information provided.\
"""
```

It expects a minimum of 256 (Query \+ Relevant passage) pairs with or without hard negatives for training and 64 pairs for validation. Please note that the validation is optional.

**Training:** We use the first 256 nodes from Lyft for creating training pairs.

**Validation:** We will use the next 64 nodes from Lyft for validation.

**Testing:** We will use the first 150 nodes from Uber.

```
# Training dataset
qa_dataset_lyft_train = generate_question_context_pairs(
    lyft_nodes[:256], llm=llm, num_questions_per_chunk=1, qa_generate_prompt_tmpl=qa_generate_prompt_tmpl
)

# Save [Optional]
qa_dataset_lyft_train.save_json("lyft_train_dataset.json")

# Validation dataset
qa_dataset_lyft_val = generate_question_context_pairs(
    lyft_nodes[257:321], llm=llm, num_questions_per_chunk=1, qa_generate_prompt_tmpl=qa_generate_prompt_tmpl
)

# Save [Optional]
qa_dataset_lyft_val.save_json("lyft_val_dataset.json")

# Testing dataset
qa_dataset_uber_val = generate_question_context_pairs(
    uber_nodes[:150], llm=llm, num_questions_per_chunk=1, qa_generate_prompt_tmpl=qa_generate_prompt_tmpl
)

# Save [Optional]
qa_dataset_uber_val.save_json("uber_val_dataset.json")
```

Now that we have compiled questions from each chunk, we will format the data according to the specifications required for training and validation.

# Data Format and Requirements

For both training and validation, it currently accepts data in the format of triplets, every row should have the following

**query:** This represents the question or target.

**relevant\_passages:** This represents a list of documents or passages that contain information that answers the query. For every query, there must be at least one relevant\_passage

**hard\_negatives:** This represents chunks or passages that don’t contain answers for the query. It should be noted that Hard negatives are optional but providing at least \~5 hard negatives will lead to meaningful improvement.

You can check the [documentation](https://docs.cohere.com/docs/rerank-models) for more details.

We need to have an embedding model for creating hard negatives with a cosine similarity approach.

```
# Initialize the Cohere embedding model which we use it for creating Hard Negatives.
embed_model = CohereEmbedding(
    cohere_api_key=cohere_api_key,
    model_name="embed-english-v3.0",
    input_type="search_document",
)
```

Let’s create 3 datasets.

1. Dataset without hard negatives.
2. Dataset with hard negatives selected at random.
3. Dataset with hard negatives selected based on cosine similarity.

```
# Train and val datasets without hard negatives.
generate_cohere_reranker_finetuning_dataset(
    qa_dataset_lyft_train,
    finetune_dataset_file_name = "train.jsonl"
)

generate_cohere_reranker_finetuning_dataset(
    qa_dataset_lyft_val,
    finetune_dataset_file_name = "val.jsonl"
)

# Train and val datasets with hard negatives selected at random.
generate_cohere_reranker_finetuning_dataset(
    qa_dataset_lyft_train,
    num_negatives = 5,
    hard_negatives_gen_method = "random",
    finetune_dataset_file_name = "train_5_random.jsonl",
    embed_model = embed_model,
)

generate_cohere_reranker_finetuning_dataset(
    qa_dataset_lyft_val,
    num_negatives = 5,
    hard_negatives_gen_method = "random",
    finetune_dataset_file_name = "val_5_random.jsonl",
    embed_model = embed_model,
)

# Train and val datasets with hard negatives selected based on cosine similarity.
generate_cohere_reranker_finetuning_dataset(
    qa_dataset_lyft_train,
    num_negatives = 5,
    hard_negatives_gen_method = "cosine_similarity",
    finetune_dataset_file_name = "train_5_cosine_similarity.jsonl",
    embed_model = embed_model,
)

generate_cohere_reranker_finetuning_dataset(
    qa_dataset_lyft_val,
    num_negatives = 5,
    hard_negatives_gen_method = "cosine_similarity",
    finetune_dataset_file_name = "val_5_cosine_similarity.jsonl",
    embed_model = embed_model,
)
```

# Fine-tuning Reranker (Custom Reranker)

With our training and validation datasets ready, we’re set to proceed with the training process. Be aware that this training is expected to take approximately 25 to 45 minutes.

```
# Reranker model with 0 hard negatives.
finetune_model_no_hard_negatives = CohereRerankerFinetuneEngine(
    train_file_name="train.jsonl",
    val_file_name="val.jsonl",
    model_name="lyft_reranker_0_hard_negatives1",
    model_type="RERANK",
    base_model="english",
    api_key = cohere_api_key
)
finetune_model_no_hard_negatives.finetune()

# Reranker model with 5 hard negatives selected at random
finetune_model_random_hard_negatives = CohereRerankerFinetuneEngine(
    train_file_name="train_5_random.jsonl",
    val_file_name="val_5_random.jsonl",
    model_name="lyft_reranker_5_random_hard_negatives1",
    model_type="RERANK",
    base_model="english",
)
finetune_model_random_hard_negatives.finetune()

# Reranker model with 5 hard negatives selected based on cosine similarity
finetune_model_cosine_hard_negatives = CohereRerankerFinetuneEngine(
    train_file_name="train_5_cosine_similarity.jsonl",
    val_file_name="val_5_cosine_similarity.jsonl",
    model_name="lyft_reranker_5_cosine_hard_negatives1",
    model_type="RERANK",
    base_model="english",
)
finetune_model_cosine_hard_negatives.finetune()
```

Once the jobs are submitted, you can check the training status in the `models` section of the [dashboard](https://dashboard.cohere.com/models). You can check the status of the job in the dashboard and you should see an image something similar to the following one.

![](https://blog.llamaindex.ai/blog/images/1*CFAtQH8dwzjnUrzVktqfbA.png){width=700 height=552}

You then need to get the Cohere Reranker model for testing.

```
reranker_base = CohereRerank(top_n=5)
reranker_model_0 = finetune_model_no_hard_negatives.get_finetuned_model(
    top_n=5
)
reranker_model_5_random = (
    finetune_model_random_hard_negatives.get_finetuned_model(top_n=5)
)
reranker_model_5_cosine = (
    finetune_model_cosine_hard_negatives.get_finetuned_model(top_n=5)
)
```

# Testing

We will conduct tests on the first 150 nodes from Uber using the following different rerankers.

1. Without Reranker.
2. Cohere Reranker.
3. Fine-tuned reranker (Custom reranker) without hard negatives.
4. Fine-tuned reranker (Custom reranker) with hard negatives selected at random.
5. Fine-tuned reranker (Custom reranker) with hard negatives selected based on cosine similarity.

Let’s define the rerankers.

```
RERANKERS = {
    "WithoutReranker": "None",
    "CohereRerank": reranker_base,
    "CohereRerank_0": reranker_model_0,
    "CohereRerank_5_random": reranker_model_5_random,
    "CohereRerank_5_cosine": reranker_model_5_cosine,
}
```

Create an Index and Retriever for evaluation purposes.

```
# Initialize the Cohere embedding model, `input_type` is different for indexing and retrieval.
index_embed_model = CohereEmbedding(
    cohere_api_key=cohere_api_key,
    model_name="embed-english-v3.0",
    input_type="search_document",
)

query_embed_model = CohereEmbedding(
    cohere_api_key=cohere_api_key,
    model_name="embed-english-v3.0",
    input_type="search_query",
)

service_context_index = ServiceContext.from_defaults(llm=None, embed_model=index_embed_model)
service_context_query = ServiceContext.from_defaults(llm=None, embed_model=query_embed_model)

vector_index = VectorStoreIndex(uber_nodes[:150], service_context=service_context_index)
vector_retriever = VectorIndexRetriever(index=vector_index, similarity_top_k=10, service_context=service_context_query)
```

Define a function to display the results

```
def display_results(embedding_name, reranker_name, eval_results):
    """Display results from evaluate."""

    metric_dicts = []
    for eval_result in eval_results:
        metric_dict = eval_result.metric_vals_dict
        metric_dicts.append(metric_dict)

    full_df = pd.DataFrame(metric_dicts)

    hit_rate = full_df["hit_rate"].mean()
    mrr = full_df["mrr"].mean()

    metric_df = pd.DataFrame(
        {"Embedding": [embedding_name], "Reranker": [reranker_name], "hit_rate": [hit_rate], "mrr": [mrr]}
    )

    return metric_df
```

Loop over different rerankers and evaluate retrieval performance using Custom Retriever.

```
results_df = pd.DataFrame()

embed_name = 'CohereEmbedding'

# Loop over rerankers
for rerank_name, reranker in RERANKERS.items():

    print(f"Running Evaluation for Reranker: {rerank_name}")

    # Define Retriever
    class CustomRetriever(BaseRetriever):
        """Custom retriever that performs both Vector search and Knowledge Graph search"""

        def __init__(
            self,
            vector_retriever: VectorIndexRetriever,
        ) -&gt; None:
            """Init params."""

            self._vector_retriever = vector_retriever

        def _retrieve(self, query_bundle: QueryBundle) -&gt; List[NodeWithScore]:
            """Retrieve nodes given query."""

            retrieved_nodes = self._vector_retriever.retrieve(query_bundle)

            if reranker != 'None':
                retrieved_nodes = reranker.postprocess_nodes(retrieved_nodes, query_bundle)
            else:
                retrieved_nodes = retrieved_nodes[:5]

            return retrieved_nodes

        async def _aretrieve(self, query_bundle: QueryBundle) -&gt; List[NodeWithScore]:
            """Asynchronously retrieve nodes given query.
            """
            return self._retrieve(query_bundle)

        async def aretrieve(self, str_or_query_bundle: QueryType) -&gt; List[NodeWithScore]:
            if isinstance(str_or_query_bundle, str):
                str_or_query_bundle = QueryBundle(str_or_query_bundle)
            return await self._aretrieve(str_or_query_bundle)

    custom_retriever = CustomRetriever(vector_retriever)

    retriever_evaluator = RetrieverEvaluator.from_metric_names(
        ["mrr", "hit_rate"], retriever=custom_retriever
    )
    eval_results = await retriever_evaluator.aevaluate_dataset(qa_dataset_uber_val)

    current_df = display_results(embed_name, rerank_name, eval_results)
    results_df = pd.concat([results_df, current_df], ignore_index=True)
```

# Results:

![](https://blog.llamaindex.ai/blog/images/1*Jv2U5pxMZWZrXjhfuXAqxg.png){width=700 height=142}

From the above table (1- without reranker, 2 — with base cohere reranker, 3–5: Fine-tuned rerankers (Custom rerankers)), we can see that the Fine-tuned rerankers (custom rerankers) have resulted in performance improvements. It’s crucial to note that the choice of the optimal number of hard negatives, as well as the decision between random or cosine sampling, should be grounded in empirical evidence. This guide offers a structured approach for improving retrieval systems through the fine-tuning of the Cohere re-ranker.

# Summary:

In this blog post, we’ve demonstrated fine-tuning a Cohere reranker (custom reranker) using LlamaIndex, which has improved retrieval performance metrics. We eagerly anticipate the community’s use of these abilities to boost their retrieval efficiency within RAG pipelines. Additionally, there is room for advancement in selecting hard negatives, and we invite the community to contribute.
