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
> That "smart shelving is indexing"

Many Indexing techniques already exists, like:
- Flat Index
- IVF [Inverted File Index]
- HNSW [Graph-based Index]
- PQ [Product Quantization]

In this project, we will be building an IVF style Indexing layer from scratch.

---

But before builing, let's understand, do we even need IVF indexing or we can work with flat indexing(without indexing).

Let's first understand what we are trying to optimize with indexing, whether it's memory space, search time, computation cost etc.

We will run a small experiment where we will search a given vector in a given set of vectors.

Let's setup the following problem:
- Number of vectors (k) = 1,000,0000
- Dimension (n) = 729
- Datatype = float32 (4 bytes)
- Operation = cosine similarity

**Memory Calculation**: 729 x 4 x 1000000 = ~2.9 GB of memory to store embeddings.

**FLOPs Calculation**: Im cosine similarity, we perform the following operations per vector

| Operation | Count |
|-----------|--------|
|Multiplication|2n + n + 1 = (dor + 2 norms)|
|Additions | 3(n-1)|
|Square roots | 2|
| Divisions| 1|

for n = 729

|Type | Count|
|------|------|
|Multiplication| 2, 188|
|Additions | 2, 184|
|Square roots| 2|
Divisions | 1|

for per vector FLOPs = 2188 + 2184 + 2 + 1 = 4375 FLOPs per vector

Total Flops for 1M vectors

4375 x 1000000 = 4.37 x 10<sup>9</sup>

A CPU with ~1.7 GHz with FP32 can have 3-6 GFLOPs/sec

So, compute-only time = ~1 second.

So, **we can see that the compute is not the main killer -- memory is**, you need to read ~2.9 GB per query. If the sustem has ~10-20GB/s (best case) then memory time: 2.9 / 15 = 0.19 seconds. Even with zero computation, you already lose ~200ms.

Thats's where the indexing wins by reducing memory touched.
- IVF with 1% clusters -> scan 10k vectors
- Memory touched: 30 MB
- Time < 10 ms

`That's a 50x-100x speedup`!
