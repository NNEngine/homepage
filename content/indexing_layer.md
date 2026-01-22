# <div align = "center">Indexing Layer of a Vector Database</div>

First of all,lets understand what a vector database is:
> A vector database is a specialized database designed to store, manage, and search high-dimensional vector embeddings, which are numerical representation of complex data like text, images, audio or video.


A Vector database is a system and it consists of the following components each with a single responsibility.
- API / Query Interface Layer
- Storage Layer (Data + MetaData)
- Indexing Layer (Logical Layer)
- ANN (Approximate Nearest Neighbour) Search
- Mapping (Resolution Layer)

We are going to study specifically about indexing layer.

## <div align = "center">Indexing </div>

**Indexing**: Indexing is the process of organizing vector embeddings into a data structure that allows fast similarity search without comparing against every vector.

*Example*
> You have 10M books. You need to find a book. Without indexing, you compare the new book with every book which is slow --> O(N)
> With Indexing, organize books into a smart shelves, jump only to likely shelves which is fast --> O(logN).
