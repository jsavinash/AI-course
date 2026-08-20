# Retrieval-Augmented Generation (RAG)

Retrieval-Augmented Generation (RAG) is a way to make AI answers more reliable by combining searching for relevant information and then generating a response. Instead of guessing based only on old training data, it first finds useful data from external sources (like documents or databases) and then uses it to give a better answer.

- Fetches up-to-date data and reduces incorrect or made-up answers
- Works well with specialized data like medical or legal content
- No need to retrain the model every time new data comes in
- Can use user-specific data to give more relevant responses

## Network Type
Retrieval-Augmented Generation pipeline combining information retrieval with language generation.

## Architecture

### Key Components

1. **External Knowledge Source**: Stores domain specific or general information like documents, APIs or databases.
2. **TextChunker**: Breaks large text into smaller, manageable chunks and cleans it for consistency.
3. **TFIDFEmbedding**: Converts text into numerical vectors that capture semantic importance via TF-IDF.
4. **VectorDatabase**: Stores embeddings and enables similarity search for fast information retrieval.
5. **Retriever**: Transforms the user's query into a vector for comparison with stored embeddings and returns the most relevant chunks.
6. **PromptAugmenter**: Combines retrieved chunks with the user's query to provide context to the generator.
7. **SimpleRAGGenerator**: Generates a grounded response using both the query and retrieved knowledge.
8. **RAGModel**: Main orchestrator that ties all components together.
9. **Updater (Optional)**: Refreshes and re-embeds data to keep the knowledge base up to date.

### Core Technologies

- **TF-IDF Embeddings**: Term Frequency-Inverse Document Frequency vectors capture word importance across the corpus.
- **Cosine Similarity**: Measures the angle between query and chunk vectors for semantic retrieval.
- **Overlapping Chunks**: Sliding window chunking preserves context across chunk boundaries.
- **Prompt Augmentation**: Injects retrieved context into the LLM prompt to ground responses.

## How RAG Works

1. **Creating External Data**: External data from APIs, databases or documents is chunked, converted into embeddings and stored in a vector database to build a knowledge library.
2. **Retrieving Relevant Information**: User queries are converted into vectors and matched against stored embeddings to fetch the most relevant data ensuring accurate responses.
3. **Augmenting the Prompt**: Retrieved content is added to the user's query giving the system extra context to work with.
4. **Answer Generation**: System uses both the query and retrieved data to generate a factually accurate, context aware response.
5. **Keeping Data Updated**: External data and embeddings are refreshed regularly in real time or scheduled so the system always retrieves latest information.

## What Problems Does RAG Solve

1. **Hallucinations**: Traditional generative models can produce incorrect information. RAG reduces this risk by retrieving verified, external data to ground responses in factual knowledge.
2. **Outdated Information**: Static models rely on training data that may become outdated. It dynamically retrieves latest information ensuring relevance and accuracy in real time.
3. **Contextual Relevance**: Generative models often struggle with maintaining context in complex or multi turn conversations. RAG retrieves relevant documents to enrich the context improving coherence and relevance.
4. **Domain Specific Knowledge**: Generic models may lack expertise in specialized fields. It integrates domain specific external knowledge for tailored and precise responses.
5. **Cost and Efficiency**: Fine tuning large models for specific tasks is expensive. It eliminates the need for retraining by dynamically retrieving relevant data reducing costs and computational load.
6. **Scalability Across Domains**: It is adaptable to diverse industries from healthcare to finance without extensive retraining making it highly scalable.

## Challenges

1. **Complexity**: Combining retrieval and generation adds complexity to the model requires careful tuning and optimization to ensure both components work seamlessly together.
2. **Latency**: The retrieval step can introduce latency making it challenging to deploy RAG models in real time applications.
3. **Quality of Retrieval**: The overall performance heavily depends on the quality of the retrieved documents. Poor retrieval can lead to suboptimal generation, undermining the model's effectiveness.
4. **Bias and Fairness**: It can inherit biases present in the training data or retrieved documents, necessitating ongoing efforts to ensure fairness and mitigate biases.

## Applications

1. **Question-Answering Systems**: It enables chatbots or virtual assistants to pull information from a knowledge base or documents and generate accurate, context aware answers.
2. **Content Creation and Summarization**: It can gather information from multiple sources and generate concise, simplified summaries or articles.
3. **Conversational Agents and Chatbots**: It enhances chatbots by grounding their responses in reliable data making interactions more informative and personalized.
4. **Information Retrieval**: Goes beyond traditional search by retrieving documents and generating meaningful summaries of their content.
5. **Educational Tools and Resources**: Provides students with explanations, diagrams or multimedia references tailored to their queries.

## Alternatives

| Method | Description | When to Use |
|---|---|---|
| Prompt Engineering | Adjusts the input prompt to guide model behavior without changing its training. | When you need a quick and simple solution for specific tasks or queries. |
| Retrieval-Augmented Generation (RAG) | Combines retrieval and generation to use external data for more factual and context-aware responses. | When you want the model's responses to include real-time, relevant information from external sources. |
| Fine-Tuning | Retrains the model on a smaller, domain-specific dataset. | When you need better performance on a particular topic or industry data. |
| Pre-Training | Trains the model from scratch using a large and diverse dataset. | When you want to build a strong foundation for later customization and adaptation. |

## Training

```bash
rag-train --model-dir ./artifacts/models --n-docs 50 --chunk-size 200 --overlap 40 --top-k 5
```

## Serving API

```bash
uvicorn retrieval_augmented_generation.api:app --host 0.0.0.0 --port 8016
```

### Endpoints
- `GET /` - Service info
- `GET /health` - Health check
- `POST /query` - Ask a question and receive a grounded answer
- `POST /index` - Index new documents into the knowledge base
- `GET /stats` - RAG model statistics
- `POST /refresh` - Refresh the vector index
- `GET /metrics` - Prometheus metrics

### Query Parameters
- `query`: User question to answer via RAG

### Index Parameters
- `documents`: List of documents with `id`, `title`, and `content`
- `chunk_size`: Number of tokens per chunk
- `overlap`: Number of overlapping tokens between chunks

## Dependencies
- Python >= 3.11
- NumPy, FastAPI, Pydantic
- mlops-shared
