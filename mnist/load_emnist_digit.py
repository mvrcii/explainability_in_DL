from emnist import extract_training_samples
import matplotlib.pyplot as plt
images, labels = extract_training_samples('digits')
print(images.shape)

plt.imshow(images[0], cmap='gray', interpolation='none')
plt.axis('off')
plt.subplots_adjust(left=0, right=1, top=1, bottom=0)  # Remove padding
plt.savefig('first_digit.png', bbox_inches='tight', pad_inches=0)
