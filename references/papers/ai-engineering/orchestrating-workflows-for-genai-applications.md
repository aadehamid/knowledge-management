title: Orchestrating Workflows for GenAI Applications
description: Turn your GenAI prototype into an automated pipeline using Apache Airflow

# Orchestrating Workflows for GenAI Applications

Learn to build and orchestrate a RAG pipeline in **Orchestrating Workflows for GenAI Applications**, built in partnership with Astronomer and taught by Kenten Danas (DevRel Senior Manager) and Tamara Fingerlin (Developer Advocate).

When building generative AI applications, it’s common to start in a Jupyter notebook or a Python script but to move into production, your AI workflows need to run reliably, adapt to changing data, and gracefully recover from failures.

In this course, you’ll learn how to turn a Retrieval Augmented Generation (RAG) prototype into a robust, automated pipeline using Airflow 3.0, a leading open-source orchestration tool.

You’ll build two workflows for a typical RAG application: one that ingests and embeds book description texts into a vector database, and another that queries that database to recommend books.

Along the way, you’ll learn how to break workflows into discrete tasks, schedule pipelines using both time-based and data-aware triggers, and process tasks in parallel with dynamic task mapping. You’ll also add retries, alerts, and backfills to handle failure scenarios. Lastly, you’ll explore how you can apply these best practices to other real-world GenAI applications such as batch inference. All of this is done using Airflow dags, which are pipelines, made up of tasks that run in a specific order, each with clear code logic and task dependencies.

In detail, you’ll:

- Apply the best practices to transform a GenAI prototype into automated and maintainable pipelines using Apache Airflow 3.0.
- Build and run a RAG prototype for book recommendation, inside a Jupyter notebook, which ingests and embeds book description texts, and recommends books based on user’s queries.
- Explore the underlying architecture of Airflow 3.0, build your first dags or pipelines, and track their status in the airflow UI.
- Convert your notebook-based RAG system into two standalone Airflow dags, one that ingests, embeds, and loads book descriptions to your vector database, and the other that queries the database.
- Schedule your first dag using time-based scheduling so that it runs automatically at a regular cadence, and your second dag using event-based triggers so that it runs when new data becomes available in the database.
- Use dynamic task mapping to create parallel task instances, making your pipeline more efficient and easier to troubleshoot.
- Add automatic retries to tasks to protect against transient failures (such as API rate limits) and learn how to add notifications in case a dag or task fails.
- Apply GenAI orchestration principles from real-world applications to scale your pipelines and improve performance over time.

By the end of this course, you’ll be able to design, build, and automate GenAI workflows using Airflow 3.0, ready for production.
