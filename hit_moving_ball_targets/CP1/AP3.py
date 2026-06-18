import numpy as np
import cv2
import matplotlib.pyplot as plt
import os

def get_pixel_value(img, x, y, channel=None):
    height, width = img.shape[:2]
    x = np.clip(x, 0, width - 1)
    y = np.clip(y, 0, height - 1)
    
    if channel is not None:
        return img[int(y), int(x), channel]
    return img[int(y), int(x)]

def frobenius_norm(matrix):                
    return np.sqrt(np.sum(matrix ** 2))

def infinity_norm(matrix):
    return np.divide(np.sum(abs(matrix)), np.max(abs(matrix)))

def cubic_kernel(x):
    absx = abs(x)
    absx2 = absx * absx
    absx3 = absx2 * absx
    
    return ((1.5 * absx3 - 2.5 * absx2 + 1) * (absx <= 1) + 
            (-0.5 * absx3 + 2.5 * absx2 - 4 * absx + 2) * ((1 < absx) & (absx <= 2)))

def quadratic_kernel(x):
    absx = abs(x)
    return (0.75 - absx * absx) * (absx <= 0.5) + \
           (0.5 * (1.5 - absx) * (1.5 - absx)) * ((0.5 < absx) & (absx <= 1.5))

def cubic_kernel_vectorized(x):
    x = np.abs(x)
    x2 = x * x
    x3 = x2 * x
    
    kernel = np.zeros_like(x, dtype=np.float32)
    mask1 = x <= 1
    mask2 = (x > 1) & (x <= 2)
    
    kernel[mask1] = 1.5 * x3[mask1] - 2.5 * x2[mask1] + 1
    kernel[mask2] = -0.5 * x3[mask2] + 2.5 * x2[mask2] - 4 * x[mask2] + 2
    
    return kernel

def quadratic_kernel_vectorized(x):
    x = np.abs(x)
    kernel = np.zeros_like(x, dtype=np.float32)
    
    mask1 = x <= 0.5
    mask2 = (x > 0.5) & (x <= 1.5)
    
    kernel[mask1] = 0.75 - x[mask1] * x[mask1]
    kernel[mask2] = 0.5 * (1.5 - x[mask2]) * (1.5 - x[mask2])
    
    return kernel

def nearest_neighbor_interpolation(image, new_height, new_width):
    height, width = image.shape[:2]
    
    # Create coordinate matrices for mapping
    x_ratio = width / new_width
    y_ratio = height / new_height
    
    # Generate coordinates arrays
    y_coords = np.floor(np.arange(new_height).reshape(-1, 1) * y_ratio).astype(np.int32)
    x_coords = np.floor(np.arange(new_width).reshape(1, -1) * x_ratio).astype(np.int32)
    
    # Ensure coordinates are within bounds
    y_coords = np.clip(y_coords, 0, height - 1)
    x_coords = np.clip(x_coords, 0, width - 1)
    
    return image[y_coords, x_coords, :]

def bilinear_interpolation(image, new_height, new_width):
    height, width = image.shape[:2]
    
    # Calculate position ratios
    x_ratio = (width - 1) / (new_width - 1) if new_width > 1 else 0
    y_ratio = (height - 1) / (new_height - 1) if new_height > 1 else 0
    
    # Generate coordinate matrices
    y = np.arange(new_height)
    x = np.arange(new_width)
    
    # Calculate sample positions
    y_floor = np.floor(y * y_ratio).astype(np.int32)
    y_ceil = np.minimum(y_floor + 1, height - 1)
    x_floor = np.floor(x * x_ratio).astype(np.int32)
    x_ceil = np.minimum(x_floor + 1, width - 1)
    
    # Calculate interpolation weights
    y_weight = (y * y_ratio - y_floor).reshape(-1, 1)
    x_weight = (x * x_ratio - x_floor).reshape(1, -1)
    
    # Get pixel values for all four corners
    top_left = image[y_floor[:, np.newaxis], x_floor, :]
    top_right = image[y_floor[:, np.newaxis], x_ceil, :]
    bottom_left = image[y_ceil[:, np.newaxis], x_floor, :]
    bottom_right = image[y_ceil[:, np.newaxis], x_ceil, :]        

    # Interpolate along x axis
    top = top_left * (1 - x_weight)[:, :, np.newaxis] + top_right * x_weight[:, :, np.newaxis]
    bottom = bottom_left * (1 - x_weight)[:, :, np.newaxis] + bottom_right * x_weight[:, :, np.newaxis]
        
    # Interpolate along y axis
    return (top * (1 - y_weight)[:, :, np.newaxis] + bottom * y_weight[:, :, np.newaxis]).astype(np.uint8)

def calculate_error(original, resized):
    # Ensure same size for comparison
    if original.shape != resized.shape:
        original = cv2.resize(original, (resized.shape[1], resized.shape[0]))
    
    # Convert to float for calculations
    original = original.astype(float)
    resized = resized.astype(float)
    
    # Calculate difference matrix
    diff_matrix = original - resized
    
    # Calculate different matrix norms using custom implementations
    frobnorm = frobenius_norm(diff_matrix)
    infnorm = infinity_norm(diff_matrix)
    
    return {
        'Frobenius': frobnorm,
        'Infinity': infnorm,
    }

def bicubic_interpolation(image, new_height, new_width):
    height, width = image.shape[:2]
    
    # Calculate scale factors
    scale_y = (height - 1) / (new_height - 1) if new_height > 1 else 0
    scale_x = (width - 1) / (new_width - 1) if new_width > 1 else 0
    
    # Generate coordinate matrices
    y = np.arange(new_height)
    x = np.arange(new_width)
    
    # Calculate sample positions
    y_pos = y * scale_y
    x_pos = x * scale_x
    
    y_floor = np.floor(y_pos).astype(np.int32)
    x_floor = np.floor(x_pos).astype(np.int32)
    
    # Calculate kernel weights
    y_kernel_coords = np.expand_dims(y_pos - y_floor, 1) + np.arange(-1, 3)
    x_kernel_coords = np.expand_dims(x_pos - x_floor, 1) + np.arange(-1, 3)
    
    y_weights = cubic_kernel_vectorized(y_kernel_coords)
    x_weights = cubic_kernel_vectorized(x_kernel_coords)
    
    # Create output array
    output = np.zeros((new_height, new_width) + image.shape[2:], dtype=np.float32)
    
    # Interpolate for each channel
    for i in range(-1, 3):
        for j in range(-1, 3):
            y_idx = np.clip(y_floor + i, 0, height - 1)
            x_idx = np.clip(x_floor + j, 0, width - 1)
            
            weight = np.outer(y_weights[:, i+1], x_weights[:, j+1])
            
            output += image[y_idx[:, np.newaxis], x_idx, :] * weight[:, :, np.newaxis]
    
    return np.clip(output, 0, 255).astype(np.uint8)

def biquadratic_interpolation(image, new_height, new_width):
    height, width = image.shape[:2]
    
    # Calculate scale factors
    scale_y = (height - 1) / (new_height - 1) if new_height > 1 else 0
    scale_x = (width - 1) / (new_width - 1) if new_width > 1 else 0
    
    # Generate coordinate matrices
    y = np.arange(new_height)
    x = np.arange(new_width)
    
    # Calculate sample positions
    y_pos = y * scale_y
    x_pos = x * scale_x
    
    y_floor = np.floor(y_pos).astype(np.int32)
    x_floor = np.floor(x_pos).astype(np.int32)
    
    # Calculate kernel weights
    y_kernel_coords = np.expand_dims(y_pos - y_floor, 1) + np.arange(-1, 2)
    x_kernel_coords = np.expand_dims(x_pos - x_floor, 1) + np.arange(-1, 2)
    
    y_weights = quadratic_kernel_vectorized(y_kernel_coords)
    x_weights = quadratic_kernel_vectorized(x_kernel_coords)
    
    # Create output array
    output = np.zeros((new_height, new_width) + image.shape[2:], dtype=np.float32)
      
    # Interpolate for each channel
    for i in range(-1, 2):
        for j in range(-1, 2):
            y_idx = np.clip(y_floor + i, 0, height - 1)
            x_idx = np.clip(x_floor + j, 0, width - 1)
            
            weight = np.outer(y_weights[:, i+1], x_weights[:, j+1])
            
            output += image[y_idx[:, np.newaxis], x_idx, :] * weight[:, :, np.newaxis]
            
    return np.clip(output, 0, 255).astype(np.uint8)