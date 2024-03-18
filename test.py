import faiss
import numpy as np
import time
#import faiss.contrib.torch_utils
import torch
import pandas as pd 

device = torch.device("cuda" if torch.cuda.is_available() else "mps")

df = pd.read_pickle('DATA/Dataset/wikiart_full_combined_try.pkl')
feature = 'image_features'
tensors = torch.stack(df[feature].tolist()).to(device)
tensor_dim = tensors.shape[1]
nb_tensors = tensors.shape[0]
k = 11

query_tensors = tensors.cpu().numpy()


# tensor_dim = 512
# nb_tensors = 29000
# query_tensors = 100
# k = 10
# torch.manual_seed(42)
# tensors = torch.randn(nb_tensors, tensor_dim, device=device)
# query_tensors = tensors[:100].cpu().numpy()

# Build index
nlist = 100  # adjust based on experimentation
quantizer = faiss.IndexFlatL2(tensor_dim)
index = faiss.IndexIVFFlat(quantizer, tensor_dim, nlist)
#index = faiss.index_cpu_to_all_gpus(index)  # If you're using multiple GPUs
index.train(tensors.cpu().numpy())
index.add(tensors.cpu().numpy())



# Perform search
start_time = time.time()
D, I = index.search(query_tensors, k)
end_time = time.time()
print("Standard search time: {:.4f} seconds".format(end_time - start_time))

df['index_vector_similarity'] = I.tolist()
print(df['index_vector_similarity'])
#df.to_csv('DATA/Dataset/wikiart_full_combined_index.csv', index=False)
# Optimizing parameters
# index.nprobe = 10  # increase the number of clusters to explore during search
# start_time = time.time()
# D, I = index.search(query_tensors, k)
# end_time = time.time()
# print("Optimized search time: {:.4f} seconds".format(end_time - start_time))


# D contains distances, I contains indices of nearest neighbors
#print("Distances:", D)










# Generate random data
# d = 64  # dimension
# nb = 100000  # number of vectors
# np.random.seed(42)
# xb = np.random.random((nb, d)).astype('float32')

# # Parameters for FAISS index
# nlist = 100  # number of clusters
# quantizer = faiss.IndexFlatL2(d)  # the other index
# index = faiss.IndexIVFFlat(quantizer, d, nlist)

# # Train the index
# index.train(xb)
# index.add(xb)

# # Query parameters
# k = 10  # number of nearest neighbors to search for
# n_queries = 1000

# # Generate random query vectors
# xq = np.random.random((n_queries, d)).astype('float32')

# # Timing standard search
# start_time = time.time()
# D, I = index.search(xq, k)
# end_time = time.time()
# print("Standard search time: {:.4f} seconds".format(end_time - start_time))

# # Optimizing parameters
# index.nprobe = 10  # increase the number of clusters to explore during search
# start_time = time.time()
# D, I = index.search(xq, k)
# end_time = time.time()
# print("Optimized search time: {:.4f} seconds".format(end_time - start_time))

# # Check if GPU resources are available
# if faiss.get_num_gpus() > 0:
#     res = faiss.StandardGpuResources()
#     gpu_index = faiss.index_cpu_to_gpu(res, 0, index)
#     start_time = time.time()
#     D, I = gpu_index.search(xq, k)
#     end_time = time.time()
#     print("GPU search time: {:.4f} seconds".format(end_time - start_time))
# else:
#     print("GPU not available")

# Other optimizations
# - Preprocess data to reduce dimensionality if possible
# - Choose appropriate index types (IVF, IVFADC, etc.)
# - Tune parameters such as nprobe, nlist, etc.
