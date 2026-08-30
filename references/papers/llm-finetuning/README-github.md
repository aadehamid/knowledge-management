title: text2cypher/finetuning/unsloth-llama3/README.md at main · neo4j-labs/text2cypher
description: collection of text2cypher datasets, evaluations, and finetuning instructions - neo4j-labs/text2cypher

## Navigation Menu

- 
AI CODE CREATION[GitHub Copilot Write better code with AI](https://github.com/features/copilot)[GitHub Copilot app Direct agents from issue to merge](https://github.com/features/ai/github-app)[MCP Registry Integrate external tools](https://github.com/mcp)DEVELOPER WORKFLOWS[Actions Automate any workflow](https://github.com/features/actions)[Codespaces Instant dev environments](https://github.com/features/codespaces)[Issues Plan and track work](https://github.com/features/issues)[Code Review Manage code changes](https://github.com/features/code-review)[Code Quality Enforce quality at merge](https://github.com/features/code-quality)APPLICATION SECURITY[GitHub Advanced Security Find and fix vulnerabilities](https://github.com/security/advanced-security)[Code security Secure your code as you build](https://github.com/security/advanced-security/code-security)[Secret protection Stop leaks before they start](https://github.com/security/advanced-security/secret-protection)EXPLORE[Why GitHub](https://github.com/why-github)[Documentation](https://docs.github.com)[Blog](https://github.blog)[Changelog](https://github.blog/changelog)[Marketplace](https://github.com/marketplace)[View all features](https://github.com/features)- 
BY COMPANY SIZE[Enterprises](https://github.com/enterprise)[Small and medium teams](https://github.com/team)[Startups](https://github.com/enterprise/startups)[Nonprofits](https://github.com/solutions/industry/nonprofits)BY USE CASE[App Modernization](https://github.com/solutions/use-case/app-modernization)[DevSecOps](https://github.com/solutions/use-case/devsecops)[DevOps](https://github.com/solutions/use-case/devops)[CI/CD](https://github.com/solutions/use-case/ci-cd)[View all use cases](https://github.com/solutions/use-case)BY INDUSTRY[Healthcare](https://github.com/solutions/industry/healthcare)[Financial services](https://github.com/solutions/industry/financial-services)[Manufacturing](https://github.com/solutions/industry/manufacturing)[Government](https://github.com/solutions/industry/government)[View all industries](https://github.com/solutions/industry)[View all solutions](https://github.com/solutions)- 
EXPLORE BY TOPIC[Software Development](https://github.com/resources/articles?topic=software-development)[DevOps](https://github.com/resources/articles?topic=devops)[Security](https://github.com/resources/articles?topic=security)[View all topics](https://github.com/resources/articles)EXPLORE BY TYPE[Customer stories](https://github.com/customer-stories)[Events & webinars](https://github.com/resources/events)[Ebooks & reports](https://github.com/resources/whitepapers)[Business insights](https://github.com/solutions/executive-insights)[GitHub Skills](https://skills.github.com)SUPPORT & SERVICES[Documentation](https://docs.github.com)[Customer support](https://support.github.com)[Community forum](https://github.com/orgs/community/discussions)[Trust center](https://github.com/trust-center)[Partners](https://github.com/partners)[View all resources](https://github.com/resources)- 
    - 
COMMUNITY[GitHub Sponsors Fund open source developers](https://github.com/open-source/sponsors)    - 
PROGRAMS[Security Lab](https://securitylab.github.com)[Maintainer Community](https://maintainers.github.com)[GitHub Stars](https://stars.github.com)[Archive Program](https://archiveprogram.github.com)    - 
REPOSITORIES[Topics](https://github.com/topics)[Trending](https://github.com/trending)[Collections](https://github.com/collections)- 
    - 
ENTERPRISE SOLUTIONS[Enterprise platform AI-powered developer platform](https://github.com/enterprise)    - 
AVAILABLE ADD-ONS[GitHub Advanced Security Enterprise-grade security features](https://github.com/security/advanced-security)[Copilot for Business Enterprise-grade AI features](https://github.com/features/copilot/copilot-business)[Premium Support Enterprise-grade 24/7 support](https://github.com/enterprise/premium-support)- [Pricing](https://github.com/pricing)

[ neo4j-labs ](https://github.com/neo4j-labs)  /  **[text2cypher](https://github.com/neo4j-labs/text2cypher)**  Public

-  [ Notifications ](https://github.com/login?return_to=%2Fneo4j-labs%2Ftext2cypher) 
-  [ Fork 32 ](https://github.com/login?return_to=%2Fneo4j-labs%2Ftext2cypher) 
- [  Star  244 ](https://github.com/login?return_to=%2Fneo4j-labs%2Ftext2cypher)

-  [  Code  ](https://github.com/neo4j-labs/text2cypher)
-  [  Issues 2 ](https://github.com/neo4j-labs/text2cypher/issues)
-  [  Pull requests 1 ](https://github.com/neo4j-labs/text2cypher/pulls)
-  [  Actions  ](https://github.com/neo4j-labs/text2cypher/actions)
-  [  Projects  ](https://github.com/neo4j-labs/text2cypher/projects)
-  [  Security and quality  ](https://github.com/neo4j-labs/text2cypher/security)
-  [  Insights  ](https://github.com/neo4j-labs/text2cypher/pulse)

## 

1. [text2cypher](https://github.com/neo4j-labs/text2cypher/tree/main)
2. [finetuning](https://github.com/neo4j-labs/text2cypher/tree/main/finetuning)
3. [unsloth-llama3](https://github.com/neo4j-labs/text2cypher/tree/main/finetuning/unsloth-llama3)

# README.md {#file-name-id}

[History](https://github.com/neo4j-labs/text2cypher/commits/main/finetuning/unsloth-llama3/README.md)

128 lines (107 loc) · 4.07 KB

1. [text2cypher](https://github.com/neo4j-labs/text2cypher/tree/main)
2. [finetuning](https://github.com/neo4j-labs/text2cypher/tree/main/finetuning)
3. [unsloth-llama3](https://github.com/neo4j-labs/text2cypher/tree/main/finetuning/unsloth-llama3)

# README.md {#sticky-file-name-id}

128 lines (107 loc) · 4.07 KB

[Raw](https://github.com/neo4j-labs/text2cypher/raw/refs/heads/main/finetuning/unsloth-llama3/README.md)

# Finetuning Llama3 using Unsloth

We have two notebooks here:

## Using simple prompt template

- Filename: `llama3_text2cypher_simple.ipynb`
- Contributed by: [Geraldus Wilsen](https://github.com/projectwilsen/)
- Dataset: synthetic\_gpt4turbo\_demodbs
- Originally published: [https://github.com/projectwilsen/KnowledgeGraphLLM](https://github.com/projectwilsen/KnowledgeGraphLLM)

This notebook uses simple prompt completion template to finetune Llama3 to construct Cypher statements on a single database (recommendations).

For more information, you could watch this video tutorial: [https://www.youtube.com/watch?v\=7VU-xWJ39ng](https://www.youtube.com/watch?v=7VU-xWJ39ng)

## Using chat prompt template

- Filename: `llama3_text2cypher_chat.ipynb`
- Contributed by: [Tomaz Bratanic](https://github.com/tomasonjo)
- Dataset: [https://huggingface.co/datasets/tomasonjo/text2cypher-gpt4o-clean](https://huggingface.co/datasets/tomasonjo/text2cypher-gpt4o-clean)
- HuggingFace model: [https://huggingface.co/collections/tomasonjo/llama3-text2cypher-demo-6647a9eae51e5310c9cfddcf](https://huggingface.co/collections/tomasonjo/llama3-text2cypher-demo-6647a9eae51e5310c9cfddcf)
- Ollama model: [https://ollama.com/tomasonjo/llama3-text2cypher-demo](https://ollama.com/tomasonjo/llama3-text2cypher-demo)

This notebook uses chat prompt template (system,user,assistant) to finetune Llama3 to construct Cypher statements on 16 different graph databases available on demo server.

You can load and use the model in LangChain or LlamaIndex. First load the model using Ollama

```
ollama pull tomasonjo/llama3-text2cypher-demo
```

### LangChain

Now you can use the following code to generate Cypher statements with LangChain:

```
pip install langchain langchain-community neo4j
from langchain_community.graphs import Neo4jGraph
from langchain_community.chat_models import ChatOllama
from langchain_core.prompts import ChatPromptTemplate

DEMO_URL = "neo4j+s://demo.neo4jlabs.com"
DATABASE = "recommendations"

graph = Neo4jGraph(
    url=DEMO_URL,
    database=DATABASE,
    username=DATABASE,
    password=DATABASE,
    enhanced_schema=True,
    sanitize=True,
)
llm = ChatOllama(model="tomasonjo/llama3-text2cypher-demo")
prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "Given an input question, convert it to a Cypher query. No pre-amble.",
        ),
        (
            "human",
            (
                "Based on the Neo4j graph schema below, write a Cypher query that would answer the user's question: "
                "\n{schema} \nQuestion: {question} \nCypher query:"
            ),
        ),
    ]
)
chain = prompt | llm

question = "How many movies did Tom Hanks play in?"
response = chain.invoke({"question": question, "schema": graph.schema})
print(response.content)
```

### LlamaIndex

Now you can use the following code to generate Cypher statements with LlamaIndex:

```
pip install llama-index llama-index-embeddings-openai llama-index-graph-stores-neo4j
from llama_index.graph_stores.neo4j import Neo4jPGStore
from llama_index.llms.openai import OpenAI
from llama_index.core.retrievers import TextToCypherRetriever
from llama_index.llms.ollama import Ollama
from llama_index.core import PropertyGraphIndex
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core.indices.property_graph import (
    ImplicitPathExtractor,
    SimpleLLMPathExtractor,
)
from llama_index.core.query_engine import RetrieverQueryEngine

DEMO_URL = "neo4j+s://demo.neo4jlabs.com"
DATABASE = "recommendations"
graph_store = Neo4jPGStore(
    username=DATABASE,
    password=DATABASE,
    database=DATABASE,
    url=DEMO_URL,
)
llm = Ollama(model="tomasonjo/llama3-text2cypher-demo", request_timeout=60.0)
cypher_retriever = TextToCypherRetriever(
    graph_store,
    llm=llm
)
# run this if index is already loaded
index = PropertyGraphIndex.from_existing(
    graph_store,
    embed_model=OpenAIEmbedding(model_name="text-embedding-3-small"),
    kg_extractors=[
        ImplicitPathExtractor(),
        SimpleLLMPathExtractor(
            llm=OpenAI(model="gpt-3.5-turbo", temperature=0.3),
            num_workers=4,
            max_paths_per_chunk=10,
        ),
    ],
    show_progress=True,
)
query_engine = RetrieverQueryEngine.from_args(
    index.as_retriever(sub_retrievers=[cypher_retriever]), llm=llm
)

response = query_engine.query("Who played in Pulp Fiction?")
print(str(response))
```

 © 2026 GitHub, Inc. 

-  [Terms](https://docs.github.com/site-policy/github-terms/github-terms-of-service) 
-  [Privacy](https://docs.github.com/site-policy/privacy-policies/github-privacy-statement) 
-  [Security](https://github.com/security) 
-  [Status](https://www.githubstatus.com/) 
-  [Community](https://github.community/) 
-  [Docs](https://docs.github.com/) 
-  [Contact](https://support.github.com?tags=dotcom-footer) 
