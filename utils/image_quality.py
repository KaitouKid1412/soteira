"""
Image quality assessment utilities.
"""

import cv2
import numpy as np


def calculate_blur_score(image):
    """
    Calculate blur score using Laplacian variance.
    Higher score = sharper image.
    
    Args:
        image: Input image (BGR or grayscale)
        
    Returns:
        float: Blur score (higher = sharper)
    """
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    
    # Calculate Laplacian variance
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    variance = laplacian.var()
    
    return variance


def is_image_sharp(image, blur_threshold=100.0):
    """
    Check if image is sharp enough based on blur score.
    
    Args:
        image: Input image
        blur_threshold: Minimum blur score for sharp image
        
    Returns:
        bool: True if image is sharp enough
    """
    blur_score = calculate_blur_score(image)
    return blur_score >= blur_threshold


def calculate_brightness(image):
    """
    Calculate average brightness of image.
    
    Args:
        image: Input image (BGR)
        
    Returns:
        float: Average brightness (0-255)
    """
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    
    return np.mean(gray)


def is_image_well_lit(image, min_brightness=30, max_brightness=225):
    """
    Check if image has good lighting (not too dark or overexposed).
    
    Args:
        image: Input image
        min_brightness: Minimum average brightness
        max_brightness: Maximum average brightness
        
    Returns:
        bool: True if well lit
    """
    brightness = calculate_brightness(image)
    return min_brightness <= brightness <= max_brightness


def calculate_contrast(image):
    """
    Calculate contrast using standard deviation.
    
    Args:
        image: Input image
        
    Returns:
        float: Contrast score
    """
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    
    return np.std(gray)


def is_image_quality_good(image, blur_threshold=100.0, min_brightness=30, 
                         max_brightness=225, min_contrast=15.0):
    """
    Comprehensive image quality check.
    
    Args:
        image: Input image
        blur_threshold: Minimum blur score
        min_brightness: Minimum brightness
        max_brightness: Maximum brightness  
        min_contrast: Minimum contrast
        
    Returns:
        tuple: (is_good, quality_metrics)
    """
    blur_score = calculate_blur_score(image)
    brightness = calculate_brightness(image)
    contrast = calculate_contrast(image)
    
    is_sharp = blur_score >= blur_threshold
    is_well_lit = min_brightness <= brightness <= max_brightness
    has_contrast = contrast >= min_contrast
    
    quality_metrics = {
        'blur_score': blur_score,
        'brightness': brightness,
        'contrast': contrast,
        'is_sharp': is_sharp,
        'is_well_lit': is_well_lit,
        'has_contrast': has_contrast
    }    
    is_good = is_sharp and is_well_lit and has_contrast
    
    return is_good, quality_metrics
