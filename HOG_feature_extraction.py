# importing required libraries
from io import BytesIO

import matplotlib.pyplot as plt
import numpy as np
import requests
from PIL import Image
from skimage import exposure
from skimage.feature import hog

# URL for a random image from LoremFlickr
url = "https://loremflickr.com/g/320/240/cup/all"
response = requests.get(url)
image = np.array(Image.open(BytesIO(response.content)))

# creating HOG features
fd, hog_image = hog(image, orientations=8, pixels_per_cell=(16, 16),
                    cells_per_block=(1, 1), visualize=True, channel_axis=-1)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4), sharex=True, sharey=True)

ax1.axis('off')
ax1.imshow(image, cmap="gray")
ax1.set_title('Input image')

# Rescale histogram for better display
zoom_hog_cell(27, 19)
hog_image_rescaled = exposure.rescale_intensity(hog_image, in_range=(0, 10))

ax2.axis('off')
ax2.imshow(hog_image_rescaled, cmap="gray")
ax2.set_title('Histogram of Oriented Gradients')
plt.show()
