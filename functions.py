from matplotlib import pyplot as plt
from PIL import Image

def plot_examples(dataset_name, query, positive_indexes, df):
    # Plot single image
    if dataset_name == 'wikiart':
        general_image_path = '/home/tliberatore2/Reproduction-of-ArtSAGENet/wikiart/'
    elif dataset_name == 'fashion':
        general_image_path = 'DATA/Dataset/iDesigner/designer_image_train_v2_cropped/'
    plt.figure(figsize=(10, 5))
    plt.imshow(Image.open(general_image_path + df.loc[query].relative_path))
    plt.axis('off')
    plt.title(str(df.loc[query].artist_name + ', influencers: ' + str(df.loc[query].influenced_by)))
    plt.show()

    # Plot grid of images
    fig, axes = plt.subplots(3,3, figsize=(20, 20))  # 5 rows, 2 columns
    for i, ax in enumerate(axes.flatten()):
        if i < len(positive_indexes):
            image_path = general_image_path + df.iloc[positive_indexes[i]].relative_path
            image = Image.open(image_path)
            ax.imshow(image)
            ax.axis('off')
            ax.set_title(str(i+1)+" "+ df.iloc[positive_indexes[i]].artist_name)
        else:
            ax.axis('off')  # Hide unused subplots

    plt.tight_layout()  # Adjust spacing between subplots
    plt.show()
