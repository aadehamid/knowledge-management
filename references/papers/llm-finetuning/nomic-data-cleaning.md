title: Data exploration and filtering with Nomic Atlas
description: A Blog post by Alexander Visheratin on Hugging Face

# Data exploration and filtering with Nomic Atlas

##  tl;dr 

In this post, I show how you can easily visualize a multimodal dataset in [Nomic Atlas](https://atlas.nomic.ai/). After that, I explore how to combine multiple projections to show them on one map. Finally, I look into how to detect duplicates across multiple fields, like images and questions. The notebook with the code is available [here](https://colab.research.google.com/drive/1LdcHkpzDzjikQaNGVakTVz8rrczl0Xm9?usp=sharing).

##  Task 

Let’s say you want to finetune a VLM (e.g., [MC-LLaVA](https://huggingface.co/blog/visheratin/vlm-resolution-curse)) on some new data. But if this is brand new data, the first thing needed is to analyze the data and find possible issues in it - incorrect data, duplicates, etc. To do it, you need to be able to go through the data both manually and programmatically. You can write your own processing pipeline and visualization tools. I can tell you from experience that building it from scratch is hard and very time-consuming. Some time ago, I discovered a platform that is a perfect fit for such tasks - Nomic Atlas. Today, I will show how you can easily explore and process a large multimodal dataset in the Atlas to make it more suitable for training.

##  Dataset 

###  Images 

As a source of our images, we will be using the lite version of the [Unsplash dataset](https://github.com/unsplash/datasets), which contains almost 25,000 high-resolution images available for commercial use. These images perfectly fit our needs because the average width of the images is 4400 pixels, and the average height is 4200 pixels.

###  Text 

We can’t train a VLM without the text related to the images. Unfortunately, the dataset itself doesn’t contain much useful textual information. But we live in 2024, a year of powerful closed models! And yes, you can laugh at my notion of “powerful” if you read it in 2025. Or late 2024. Or in a couple of months after publishing this post. Anyway, we will be using Gemini Vision 1.0 API to generate captions and question-answer pairs for every image in the dataset. The initial unprocessed version of the resulting dataset is available [here](https://huggingface.co/datasets/visheratin/unsplash-caption-questions-init).

##  Processing 

###  Text processing 

First of all, let’s load the data into Atlas. This part will be a piece of cake because both the Hugging Face datasets and the Nomic library are compatible with Pandas.

Convert dataset to the dataframe:

```python
from datasets import load_dataset

dataset = load_dataset("visheratin/unsplash-caption-questions-init")
df = dataset["train"].to_pandas()
```

Load the dataframe to Atlas:

```python
from nomic import atlas

nomic_dataset = atlas.map_data(
    data=df,
    id_field='id',
    indexed_field='question',
    identifier="unsplash-synthetic-data-init",
    topic_model=False, # we don't need topics
    duplicate_detection=True, # it is True by default, but we still set it for visibility
)
```

After that, we need to wait for a couple of minutes while everything is indexed and mapped. And then, we can start exploring the data! You can see that there are many small clusters and some weird rings across the map.

[![image/png](https://cdn-uploads.huggingface.co/production/uploads/609ede05121df5de54007033/aPjwz2Ja7qVANGTmMbaK3.png)](https://cdn-uploads.huggingface.co/production/uploads/609ede05121df5de54007033/aPjwz2Ja7qVANGTmMbaK3.png)

Color denotes duplicate classes. If you filter by duplicate class, you will see that there are almost 17,000 deletion candidates based on the generated question! What is going on?

Upon manually inspecting individual clusters, we see that, in many cases, the questions are indeed identical. However, the answers are not always the same because the images have different contents. And captions are always unique. We need to investigate our data from different angles.

So far, we created only one projection for the data based on the question. Let’s add two more indexes for answers and captions.

```python
nomic_dataset.create_index(
    indexed_field='caption',
    topic_model=False,
    duplicate_detection=True,
)
nomic_dataset.create_index(
    indexed_field='answer',
    topic_model=False,
    duplicate_detection=True,
)
```

We processed all our text data, but if you open the map, you can see only the latest generated projection in the UI. We will fix it very soon. Before that, we need embeddings for the images.

###  A note on the environment 

There are two standard types of development environments for the kind of data processing we are doing here. The first one is your own computer. This gives full control over the process and an absolute persistence of your data (unless your hard drive dies). But to perform complex tasks (wrangling huge datasets, running large models), you may need more power than is available locally. This is when people start using cloud-based solutions like Google Colab. In a couple of clicks, you can get a machine with a lot of memory and GPU attached to fit all your needs. But these environments are not persistent, and if you leave it idle for too long, it will disconnect, and you’ll have to re-do everything. Yes, you can attach Google Drive to store your data, but it is still suboptimal. Recently, [Lightning](https://lightning.ai/) released its cloud environment to solve this specific problem. When you create a studio, you get a persistent environment that runs on the CPU by default. However, if you need a GPU or more memory, you can quickly switch your environment and continue working with almost no interruptions! That is why I used Lightning Studio for data processing for this post. It took several days (more like nights) of experimenting and figuring things out. I really enjoyed having an environment that is persistent, doesn’t clutter my machine, and I can use GPU only when I need one.

My main personal use for Lightning Studio is for data processing and training experiments. Full multi-GPU or multi-node training is too expensive for me compared to dedicated GPU cloud providers (e.g., Lambda), but I’m sure Lightning will solve this in the near future.

###  Image processing 

Back to data processing! In 2023, Google released SigLIP models - a family of CLIP models that have small sizes but produce superb text and image embeddings. We will use a vision encoder from one of these models - ViT-B-16-SigLIP - to embed our images. I used the model from the OpenCLIP library. But you can use the implementation from [Transformers](https://huggingface.co/google/siglip-base-patch16-224) if you like. You can check out the code in the [notebook](https://colab.research.google.com/drive/1LdcHkpzDzjikQaNGVakTVz8rrczl0Xm9?usp=sharing).

We generated embeddings, and to display them on the map, we need to do one more step - reduce dimensions from 768 to 2. We will do it in two steps. First, we apply principal components analysis to reduce the embedding dimensions from 768 to 20. Second, we apply t-SNE to convert 30 dimensions to 2.

```python
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

pca_embeddings = PCA(n_components=20).fit_transform(embeddings)
tsne_embeddings = TSNE(n_components=2, perplexity=30).fit_transform(pca_embeddings)
```

There is no clear consensus on what dimensionality reduction methods and their combinations work best. I find this two-step method very good for generating visualizations.

Before proceeding, we can check out how good our embeddings are. Let’s find the image pairs that are very similar to each other. For that, we first calculate the cosine similarity matrix and then find index pairs for which the score is higher than the threshold (0.95 in this case).

```python
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

similarity_matrix = cosine_similarity(embeddings)
threshold = 0.95

similar_pairs = np.column_stack(np.where(similarity_matrix > threshold))
unique_pairs = set(tuple(sorted(pair)) for pair in similar_pairs)
filtered_pairs = np.array([list(pair) for pair in unique_pairs if pair[0] != pair[1]])
```

After that, we can visualize some of the resulting image pairs:

[![image/png](https://cdn-uploads.huggingface.co/production/uploads/609ede05121df5de54007033/QyiG7YklrAB039bPgU2UH.png)](https://cdn-uploads.huggingface.co/production/uploads/609ede05121df5de54007033/QyiG7YklrAB039bPgU2UH.png)

[![image/png](https://cdn-uploads.huggingface.co/production/uploads/609ede05121df5de54007033/PlW2xivpLNWzOb4iJFhwM.png)](https://cdn-uploads.huggingface.co/production/uploads/609ede05121df5de54007033/PlW2xivpLNWzOb4iJFhwM.png)

Yes, they are not identical, but palms are palms, and clouds are clouds. And in the absence of other notable details these images are not very information-dense.

The last part of image processing is duplicate detection. This one is more for display purposes, as we will get more creative with duplicates later. We can use standard DBSCAN to group t-SNE embeddings and then treat the point closest to the cluster center as the “best” point (“retention candidate” in the Atlas terminology), the rest of the points in the cluster as “deletion candidates” and non-clustered points as “singletons”.

```python
from sklearn.cluster import DBSCAN
import numpy as np
from scipy.spatial import distance

dbscan = DBSCAN(eps=0.5, min_samples=3).fit(tsne_embeddings)

closest_point_indices = set()
for cluster in set(dbscan.labels_):
    if cluster == -1:
        continue
    indices_in_cluster = np.where(dbscan.labels_ == cluster)[0]
    points_in_cluster = tsne_embeddings[indices_in_cluster]
    centroid = points_in_cluster.mean(axis=0)
    distances = distance.cdist([centroid], points_in_cluster, 'euclidean')[0]
    closest_point_index = np.argmin(distances)
    global_index = indices_in_cluster[closest_point_index]
    closest_point_indices.add(global_index)
 
image_classes = []

for i in range(len(tsne_embeddings)):
    if i in closest_point_indices:
        image_classes.append("retention candidate")
    elif dbscan.labels_[i] == -1:
        image_classes.append("singleton")
    else:
        image_classes.append("deletion candidate")
```

To make it easier to work with the data later, we compile embeddings and classes into the data frame:

```python
image_items = []
for i in range(len(image_ids)):
    image_items.append([image_ids[i], tsne_embeddings[i,0], tsne_embeddings[i,1], image_classes[image_ids[i]]])

image_df = pd.DataFrame.from_records(image_items, columns=["id", "x_image", "y_image", "duplicate_class_image"])
```

###  Embeddings display 

Alright, we have all our embeddings. Now it’s time to put everything together on one map! One nice feature of the Atlas is that you have access to all coordinates and embeddings that were generated. Here is how to extract coordinates for all text projections into Pandas data frames:

```python
answer_embeddings_df = nomic_dataset.maps[0].embeddings.df
caption_embeddings_df = nomic_dataset.maps[1].embeddings.df
question_embeddings_df = nomic_dataset.maps[2].embeddings.df
```

After that, we merge all data frames together:

```python
import pandas as pd

merged_embeddings_df = answer_embeddings_df.merge(caption_embeddings_df, on='id', suffixes=('_answer', '_caption'))
merged_embeddings_df = merged_embeddings_df.merge(question_embeddings_df, on='id')
merged_embeddings_df.rename(columns={'x': 'x_question', 'y': 'y_question'}, inplace=True)
```

Then, we extract info about duplicates from each text projection:

```python
answer_duplicates_df = nomic_dataset.maps[0].duplicates.df
caption_duplicates_df = nomic_dataset.maps[1].duplicates.df
question_duplicates_df = nomic_dataset.maps[2].duplicates.df
```

Rename the columns to make them more readable:

```python
answer_duplicates_df = answer_duplicates_df.rename(
    columns={
        "duplicate_class_at_0,100": "duplicate_class",
        "cluster_id_at_0,100": "cluster_id"
    }
)
caption_duplicates_df = caption_duplicates_df.rename(
    columns={
        "duplicate_class_at_0,100": "duplicate_class",
        "cluster_id_at_0,100": "cluster_id"
    }
)
question_duplicates_df = question_duplicates_df.rename(
    columns={
        "duplicate_class_at_0,100": "duplicate_class",
        "cluster_id_at_0,100": "cluster_id"
    }
)
```

Merge duplicate data frames:

```python
merged_duplicates_df = answer_duplicates_df.merge(caption_duplicates_df, on='id', suffixes=('_answer', '_caption'))
merged_duplicates_df = merged_duplicates_df.merge(question_duplicates_df, on='id')
merged_duplicates_df.rename(columns={'duplicate_class': 'duplicate_class_question', 'cluster_id': 'cluster_id_question'}, inplace=True)
```

And now, merge all data frames into one final data frame:

```python
final_df = pd.merge(df, merged_embeddings_df, on='id')
final_df = pd.merge(final_df, merged_duplicates_df, on='id')
final_df = pd.merge(final_df, image_df, on='id')
```

We can finally extract everything we need for display:

```python
objects_list = final_df.apply(lambda row: {
    "id": row["id"],
    "answer.x": row["x_answer"],
    "answer.y": row["y_answer"],
    "caption.x": row["x_caption"],
    "caption.y": row["y_caption"],
    "question.x": row["x_question"],
    "question.y": row["y_question"],
    "image.x": row["x_image"],
    "image.y": row["y_image"],
    ".url": row["url"], # dots are needed to make sure these fields are displayed at the top
    ".question": row["question"],
    ".answer": row["answer"],
    ".caption": row["caption"],
    "answer_duplicate_class": row["duplicate_class_answer"],
    "caption_duplicate_class": row["duplicate_class_caption"],
    "question_duplicate_class": row["duplicate_class_question"],
    "image_duplicate_class": row["duplicate_class_image"]
}, axis=1).tolist()
```

And load the data into the new dataset:

```python
combined_dataset = AtlasDataset(
    "unsplash-data-combined",
    unique_id_field="id",
)
combined_dataset.add_data(data=objects_list)
combined_dataset.create_index(
    indexed_field='.question',
    topic_model=False,
    duplicate_detection=True,
)
```

We can go to the web UI to see the fruits of our work. To be able to change visualization coordinates, you first need to enable beta features and then enable “Custom Mapping” in the map web interface.

\<video controls autoplay src\="

Beautiful, isn’t it? The UI allows us to go through the data easily, find some patterns, and look at the duplicates. It’s clear that there are repeated questions and very similar images in the dataset. However, similar images are not really a big problem if they are accompanied by different captions or question-answer pairs. But how do we search for such cases?

###  Joint embeddings 

To find the duplicates based on multiple properties, we can simply concatenate the embeddings for these properties and analyze them. For example, to get the joint embeddings of questions and images, we can use this code:

```python
import numpy as np

question_image_embeddings = final_df.apply(lambda row: np.concatenate((row["embeddings_question"], row["embeddings_image"]), axis=0), axis=1).tolist()
```

And now we get to one of the latest Atlas features - embedding display and duplicate detection. We don’t need to perform dimensionality reduction and duplicate detection ourselves. We can simply load the embeddings to the new dataset and have them analyzed in the same way as the original texts!

```python
from nomic import AtlasDataset

combined_dataset = AtlasDataset(
    "unsplash-data-joint-embeddings",
    unique_id_field="id",
)
combined_dataset.add_data(data=objects_list, embeddings=np.array(image_question_embeddings))
combined_dataset.create_index(
    indexed_field='embeddings',
    topic_model=False,
    duplicate_detection=True,
)
```

That’s it! Now we only have to wait a couple of minutes for the data to be processed, and we can access the new map.

[![image/png](https://cdn-uploads.huggingface.co/production/uploads/609ede05121df5de54007033/aiNwiA3zqTaZFhIV9G5c8.png)](https://cdn-uploads.huggingface.co/production/uploads/609ede05121df5de54007033/aiNwiA3zqTaZFhIV9G5c8.png)

It is clear that there are much fewer duplicates than before, 735, to be exact. These samples are definitely worth removing as they don’t contribute anything new to the model. You can try different combinations of fields (e.g., image\+caption) or duplication detection parameters.

After that, removing the duplicates is easy:

```python
duplicates_df = combined_dataset.maps[0].duplicates.df
filter_df = duplicates_df[duplicates_df['duplicate_class_at_0,100'] == "deletion candidate"]
mask = final_df['id'].isin(filter_df['id'])
filtered_df = final_df[~mask]
```

##  Conclusion 

Nomic Atlas provides great tools for data exploration and analysis. The ability to access the data both programmatically and through the fast and responsive UI is marvelous. I didn't cover other features, like tagging, which can help you remove large chunks of bad data in a couple of clicks. But I hope that this post will help you get started with Atlas and that you'll be able to make your datasets even better.
