# <div align = "center">Indexing Layer of a Vector Database</div>

<img width="474" height="148" alt="IVF_image" src="https://github.com/user-attachments/assets/ba5aeba5-24bc-4e0a-97ba-62cba03df3e0" />


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

---

Now, let's code an IVF Style Indexing layer (with n = 256, k = 10000)

- Load the embeddings
- Normalize the embeddings
- Decide number of clusters
- Run K-means (you can use other algorithm as well)
- Extract centroids
- Save IVF index files

```
import numpy as np
from sklearn.cluster import KMeans
import math

# -----------------------------
# Load embeddings
# -----------------------------
embeddings = np.load("embeddings.npy").astype(np.float32)
N, D = embeddings.shape

print(f"Loaded embeddings: {N} x {D}")

# -----------------------------
# Normalize embeddings (VERY IMPORTANT)
# -----------------------------
norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
embeddings = embeddings / norms

# -----------------------------
# Decide number of clusters
# -----------------------------
K = int(math.sqrt(N))
print(f"Number of clusters (K): {K}")

# -----------------------------
# Run K-means
# -----------------------------
kmeans = KMeans(
    n_clusters=K,
    random_state=42,
    n_init=10,
    max_iter=300,
    verbose=1
)

print("Running K-means...")
cluster_ids = kmeans.fit_predict(embeddings)

# -----------------------------
# Extract centroids
# -----------------------------
centroids = kmeans.cluster_centers_.astype(np.float32)

# Normalize centroids as well (important for cosine search)
centroids /= np.linalg.norm(centroids, axis=1, keepdims=True)

# -----------------------------
# Save IVF index files
# -----------------------------
np.save("cluster_ids.npy", cluster_ids.astype(np.int32))
np.save("centroids.npy", centroids)

print("Saved files:")
print(" - cluster_ids.npy")
print(" - centroids.npy")
```

You will get to see something like this:

```
Loaded embeddings: 10000 x 256
Number of clusters (K): 100
Running K-means...
Initialization complete
Iteration 0, inertia 3994.23193359375.
Iteration 1, inertia 2404.404052734375.
Iteration 2, inertia 2394.2685546875.
Iteration 3, inertia 2389.8642578125.
Iteration 4, inertia 2387.59765625.
Iteration 5, inertia 2386.28857421875.
Iteration 6, inertia 2385.387451171875.
Iteration 7, inertia 2384.763427734375.
Iteration 8, inertia 2384.38525390625.
Iteration 9, inertia 2384.082763671875.
Iteration 10, inertia 2383.88671875.
Iteration 11, inertia 2383.732421875.
Iteration 12, inertia 2383.607421875.
Iteration 13, inertia 2383.50146484375.
Iteration 14, inertia 2383.395751953125.
Iteration 15, inertia 2383.31201171875.
Iteration 16, inertia 2383.239501953125.
Iteration 17, inertia 2383.1962890625.
Iteration 18, inertia 2383.164306640625.
Iteration 19, inertia 2383.139892578125.
Iteration 20, inertia 2383.125.
Iteration 21, inertia 2383.1103515625.
Iteration 22, inertia 2383.0830078125.
Iteration 23, inertia 2383.057861328125.
Iteration 24, inertia 2383.03515625.
Iteration 25, inertia 2383.01904296875.
Iteration 26, inertia 2383.0048828125.
Iteration 27, inertia 2382.98779296875.
Iteration 28, inertia 2382.97705078125.
Iteration 29, inertia 2382.966796875.
Iteration 30, inertia 2382.95166015625.
Iteration 31, inertia 2382.932861328125.
Iteration 32, inertia 2382.923828125.
Iteration 33, inertia 2382.91162109375.
Iteration 34, inertia 2382.90185546875.
Iteration 35, inertia 2382.8876953125.
Iteration 36, inertia 2382.8740234375.
Iteration 37, inertia 2382.86572265625.
Converged at iteration 37: strict convergence.
Initialization complete
Iteration 0, inertia 3962.75830078125.
Iteration 1, inertia 2404.5732421875.
Iteration 2, inertia 2394.144775390625.
Iteration 3, inertia 2389.58935546875.
Iteration 4, inertia 2387.109375.
Iteration 5, inertia 2385.52880859375.
Iteration 6, inertia 2384.55908203125.
.
.
.

Converged at iteration 37: strict convergence

Saved Files:
- cluster_ids.npy
- centroids.npy
```

- The shape of cluster_ids.npy --> (10000, 256)
- The shape of centroids.npy --> (100, 256)
