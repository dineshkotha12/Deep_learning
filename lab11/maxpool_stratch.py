import numpy as np

# Create a 32x32 input image
image = np.random.randint(0, 10, (32, 32))


pool_size = 2
stride = 2


output_height = (32 - pool_size) // stride + 1
output_width = (32 - pool_size) // stride + 1


output = np.zeros((output_height, output_width))

# Apply max pooling
for i in range(0, 32 - pool_size + 1, stride):
    for j in range(0, 32 - pool_size + 1, stride):

        # Select a 2x2 region
        region = image[i:i+pool_size, j:j+pool_size]

        # Find the maximum value
        output[i//stride, j//stride] = np.max(region)

print("Input image shape:", image.shape)
print("Output image shape:", output.shape)
print("\nInput image:\n", image)
print("\nMax pooling output:\n", output)