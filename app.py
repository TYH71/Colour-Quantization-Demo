import streamlit as st
import numpy as np
from skimage import io, color, img_as_ubyte
from PIL import Image
from sklearn.cluster import MiniBatchKMeans

# 
st.title("Colour Quantization")
st.info(
'''
Conduct Colour-Based Segmentation of an Image based on the chromaticity plane leveraging over Mini-Batch K-Means Clustering.
''')

# Buffer File Upload
buffer_file = st.file_uploader('Choose a File', type=['jpg', 'png', 'jpeg'], accept_multiple_files=False)

# Checking State of File Upload
if buffer_file is not None:
    img = Image.open(buffer_file)
    img_data = np.asarray(img)

    # Show Image
    st.image(img_data, caption='Uploaded Image', use_column_width=True)

    cluster_parameters = 4
    km = MiniBatchKMeans(
        n_clusters=cluster_parameters, 
        max_iter=500, 
        batch_size=3072
    ).fit((img_data/255.0).reshape(-1, 3))

    st.write("Received Image. Generating Compressed Image")

    k_colors = km.cluster_centers_[km.labels_]
    k_img = img_as_ubyte(np.reshape(k_colors,(img_data.shape)))

    st.image(k_img, caption='Compressed Colour', use_column_width=True)

# # reading image
# image = io.imread('Original.jpg')
# st.image(image, caption='Original Image', use_column_width=True)

# # preprocessing
# image_data = (image / 255).reshape(-1, 3)

# from assets.plot_utils import plot_utils
# x = plot_utils(image_data, title="Input colour space")
# st.pyplot(x.colorSpace())

# st.pyplot(x.colorSpace3d())

# km = MiniBatchKMeans(n_clusters=4, max_iter=500, batch_size=3072).fit(image_data)
# k_colors = km.cluster_centers_[km.labels_]

# y = plot_utils(image_data, colors=k_colors, title="Reduced color space: 4 colours", 
#     centroids=km.cluster_centers_)
# st.pyplot(y.colorSpace())

# k_img = img_as_ubyte(np.reshape(k_colors,(image.shape)))
# st.image(k_img, caption='Reduced color space: 4 colours', use_column_width=True)
