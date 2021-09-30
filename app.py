import streamlit as st
import numpy as np
from skimage import io, color, img_as_ubyte
from PIL import Image
from sklearn.cluster import MiniBatchKMeans

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

# Checking State of File Upload
if buffer_file is not None:
    img = Image.open(buffer_file)
    img_data = np.asarray(img)

    # Show Image
    st.image(img_data, caption='Uploaded Image', use_column_width=True)

    # Generating Image
    st.write("Received Image. Generating Compressed Image")
    km = MiniBatchKMeans(
        n_clusters=cluster_parameters, 
        max_iter=500, 
        batch_size=1536,
        tol=0.01 
    ).fit((img_data/255.0).reshape(-1, 3))

    # Generating Quantized Image
    k_colors = km.cluster_centers_[km.labels_]
    k_img = img_as_ubyte(np.reshape(k_colors,(img_data.shape)))

    # Show Quantized Image
    st.image(k_img, caption=f'Compressed Colour ({cluster_parameters} Distinct Colours)', use_column_width=True)

else:
    st.write("No Image Uploaded")