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

Now, let's understand this from code.

First, let's build some vectors. Because of the system limitations, we are going with `n = 729` and `k = 100000`

```
import numpy as np

embeddings = np.random.rand(100000, 729).astype(np.float32)

np.save("embeddings.npy", embeddings)
print("Embeddings saved to embeddings.npy")
```

Now, let's perform the per vector search (brute force) to find the compute and space used.

```
import numpy as np
import time

# -----------------------------
# Load and normalize embeddings
# -----------------------------
embeddings = np.load("embeddings.npy").astype(np.float32)
embeddings /= np.linalg.norm(embeddings, axis=1, keepdims=True)

rows, dims = embeddings.shape

# -----------------------------
# Prepare query
# -----------------------------
query = np.random.rand(dims).astype(np.float32)
query /= np.linalg.norm(query)

# -----------------------------
# Brute-force cosine similarity
# -----------------------------
start = time.perf_counter()
scores = embeddings @ query
end = time.perf_counter()

elapsed = end - start

# -----------------------------
# MEMORY TRAFFIC ANALYSIS
# -----------------------------
bytes_per_float = 4
bytes_per_vector = dims * bytes_per_float
total_bytes_read = rows * bytes_per_vector   # embeddings only

mb_read = total_bytes_read / (1024 ** 2)
bandwidth = (mb_read / elapsed) / (1024)   # MB/s

# -----------------------------
# COMPUTE REALITY CHECK
# -----------------------------
# Dot product FLOPs per vector = (mul + add) = dims + (dims - 1)
flops_per_vector = 2 * dims - 1
total_flops = flops_per_vector * rows

gflops = total_flops / elapsed / 1e9

# -----------------------------
# OUTPUT
# -----------------------------
print(f"Shape: {rows} x {dims}")
print(f"Time (ms): {elapsed * 1000:.3f}")

print("\n--- Memory Traffic Analysis ---")
print(f"Total data read: {mb_read:.2f} MB")
print(f"Effective memory bandwidth: {bandwidth:.2f} GB/s")

print("\n--- Compute Reality Check ---")
print(f"Total FLOPs: {total_flops:,}")
print(f"Effective compute throughput: {gflops:.3f} GFLOPs/s")
```

**Output**
```
Shape: 100000 x 729
Time (ms): 29.053

--- Memory Traffic Analysis ---
Total data read: 278.09 MB
Effective memory bandwidth: 9.35 GB/s

--- Compute Reality Check ---
Total FLOPs: 145,700,000
Effective compute throughput: 5.015 GFLOPs/s
```
