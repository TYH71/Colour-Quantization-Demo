import streamlit as st
import numpy as np
from skimage import io, color, img_as_ubyte
from PIL import Image
from sklearn.cluster import MiniBatchKMeans

import numpy as np
import matplotlib.pyplot as plt

class plot_utils:
    def __init__(self, img_data, title, num_pixels=10000, colors=None, centroids=None):
        self.img_data = img_data
        self.title = title
        self.num_pixels = num_pixels
        self.colors = colors
        self.centroids = centroids

    def colorSpace(self):
        if self.colors is None:
            self.colors = self.img_data

        rand = np.random.RandomState(42)
        index = rand.permutation(self.img_data.shape[0])[:self.num_pixels]
        colors = self.colors[index]
        R, G, B = self.img_data[index].T

        fig, ax = plt.subplots(1, 2, figsize=(15, 9), tight_layout=True)
        ax[0].scatter(R, G, color=colors, marker='.')
        ax[0].set(xlabel='Red', ylabel='Green', xlim=(0, 1), ylim=(0, 1))
        ax[1].scatter(R, B, color=colors, marker='.')
        ax[1].set(xlabel='Red', ylabel='Blue', xlim=(0, 1), ylim=(0, 1))
        fig.suptitle(self.title, size=28)

        if self.centroids is not None:
            ax[0].scatter(self.centroids[:, 0], self.centroids[:, 1], color='white', 
            marker='o', s=400, edgecolor='black')
            for i, c in enumerate(self.centroids):
                ax[0].scatter(c[0], c[1], marker='$%d$' % i, s=100, edgecolor='k')
            ax[1].scatter(self.centroids[:, 0], self.centroids[:, 2], color='white', 
            marker='o', s=400, edgecolor='black')
            for i, c in enumerate(self.centroids):
                ax[1].scatter(c[0], c[2], marker='$%d$' % i, s=100, edgecolor='k')

        return fig

    def colorSpace3d(self):
        if self.colors is None:
            self.colors = self.img_data

        rand = np.random.RandomState(42)
        index = rand.permutation(self.img_data.shape[0])[:self.num_pixels]
        colors = self.colors[index]
        R, G, B = self.img_data[index].T

        fig = plt.figure(figsize=(15, 9), tight_layout=True)
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter3D(R, G, B, color=colors, marker='.')
        ax.set_xlabel('Red')
        ax.set_ylabel('Green')
        ax.set_zlabel('Blue')
        fig.suptitle(self.title, size=28)


# Header
st.title("Colour Quantization")
st.info(
'''
Conduct Colour-Based Segmentation of an Image based on the chromaticity plane leveraging over Mini-Batch K-Means Clustering.
''')

# sidebar
st.sidebar.title("Colour Quantization")
st.sidebar.info("Control the number of distinct colours for Quantization.")
cluster_parameters = st.sidebar.slider('Number of Clusters', 4, 256, 64)

# Buffer File Upload
buffer_file = st.sidebar.file_uploader('Choose a File', type=['jpg', 'png', 'jpeg'], accept_multiple_files=False)

@st.cache
def predict(img_data):
    return MiniBatchKMeans(
        n_clusters=cluster_parameters, 
        max_iter=500, 
        batch_size=1536,
        tol=0.01 
    ).fit((img_data/255.0).reshape(-1, 3)) 

# Checking State of File Upload
if buffer_file is not None:
    img = Image.open(buffer_file)
    img_data = np.asarray(img)

    col1, col2 = st.columns(2)

    # Show Image
    col1.image(img_data, caption='Uploaded Image', use_column_width=True)

    # Generating Image
    km = predict(img_data)

    # Generating Quantized Image
    k_colors = km.cluster_centers_[km.labels_]
    k_img = img_as_ubyte(np.reshape(k_colors,(img_data.shape)))

    # Show Quantized Image
    col2.image(k_img, caption=f'Compressed Colour ({cluster_parameters} Distinct Colours)', use_column_width=True)


    colourspace = st.expander('Colourspace')
    
    colourspace.image(plot_utils(img_data, title=None).colorSpace(), caption='Colour Space', use_column_width=True)

else:
    st.markdown(">No Image Uploaded")