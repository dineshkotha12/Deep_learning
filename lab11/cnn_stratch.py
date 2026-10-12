
import numpy as np

# Create a 32x32 input image with random values
image = np.random.randint(0, 10, (32, 32))

# Define a 3x3 kernel
kernel = np.array([
    [1, 0, -1],
    [1, 0, -1],
    [1, 0, -1]
])

# Get image dimensions
image_height, image_width = image.shape
kernel_height, kernel_width = kernel.shape

# Calculate output dimensions
output_height = image_height - kernel_height + 1
output_width = image_width - kernel_width + 1

# Create output matrix
output = np.zeros((output_height, output_width))

# Apply convolution operation
for i in range(output_height):
    for j in range(output_width):

        # Select a 3x3 region from the image
        region = image[i:i+3, j:j+3]

        # Multiply region with kernel and add the values
        output[i, j] = np.sum(region * kernel)


print("Input image shape:", image.shape)
print("Kernel shape:", kernel.shape)
print("Output image shape:", output.shape)
print("\nInput image:\n", image)
print("\nKernel:\n", kernel)
print("\nOutput image:\n", output)