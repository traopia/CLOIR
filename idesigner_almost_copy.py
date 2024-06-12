import pandas as pd
import torch
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

df = pd.read_pickle('DATA/Dataset/iDesigner/idesigner_influences_cropped_features.pkl')
features = df['image_features']
features = features.to_numpy()
features = np.stack(features)
features = torch.tensor(features)



similarity_matrix = cosine_similarity(features)
n = len(similarity_matrix)
similarity_matrix = similarity_matrix - np.identity(n)

id_map = df.index.to_list()
all_similar_images = []
for idx, sample in enumerate(df.iterrows()):
    #for each image, find the most similar images above a certain threshold
    threshold = 0.96
    similar_images = np.where(similarity_matrix[idx] > threshold)[0]
    similar_images = [id_map[x] for x in similar_images]
    if len(similar_images) > 1:
        all_similar_images.append(similar_images)

indexes_to_remove = []
for similar_images in all_similar_images:
    indexes_to_remove.extend(similar_images[1:])
indexes_to_remove = list(set(indexes_to_remove))
df = df.drop(indexes_to_remove)
df = df.reset_index(drop=True)
df.to_pickle('DATA/Dataset/iDesigner/idesigner_influences_cropped_features_no_duplicate.pkl')