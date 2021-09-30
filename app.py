import streamlit as st
import numpy as np
from skimage import io, color, img_as_ubyte
from PIL import Image
from sklearn.cluster import MiniBatchKMeans
from assets.plot_utils import plot_utils
import matplotlib.pyplot as plt

plt.style.use('fivethirtyeight')

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

code = st.expander('Algorithm in Action')
with code.container():
    with st.echo():
        ## MiniBatchKMeans Clustering Algorithm 
        ## Colour Quantization / Colour Space Segmentation

        @st.cache # caching mechanism to stay performant in big computation
        def predict(img_data, clusters):
            return MiniBatchKMeans(
                n_clusters=clusters, 
                max_iter=500, 
                batch_size=3072,
                tol=0.01 
            ).fit(img_data, clusters) 
            

# Checking State of File Upload
if buffer_file is not None:
    img = Image.open(buffer_file)
    img_data = np.asarray(img) # original image
    img_preprocess = (img_data/255.0).reshape(-1, 3) # image after normalization and reshape

    col1, col2 = st.columns(2)

    # Show Image
    col1.image(img, caption='Uploaded Image', use_column_width=True)

    # Generating Image
    km = predict(img_preprocess, cluster_parameters)

    # Generating Quantized Image
    k_colors = km.cluster_centers_[km.labels_]
    k_img = img_as_ubyte(np.reshape(k_colors,(img_data.shape)))

    # Show Quantized Image
    col2.image(k_img, caption=f'Compressed Image ({cluster_parameters} Distinct Colours)', use_column_width=True)

    # Colour Space as a Scatterplot
    colourspace = st.expander('Colour Space')

    colourspace.info('Colour Space: A plane containing all the possible colours that can be produced by mixing the three colour channels.')
    x = plot_utils(img_preprocess, title="Original Image Colour Space")
    colourspace.pyplot(x.colorSpace())

    y = plot_utils(img_preprocess, colors=k_colors, title=f"Reduced color space: {cluster_parameters} colours", 
    centroids=km.cluster_centers_)
    colourspace.pyplot(y.colorSpace())

    st.success('Python Code Finished Running')

else:
    st.markdown(">No Image Uploaded")