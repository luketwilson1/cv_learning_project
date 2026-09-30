import cv2
import numpy as np


def resize_image(image, width, height):
    return cv2.resize(image, (width, height))

def resize_keep_aspect(image, target_width, target_height):
    """
    Resize an image while preserving its aspect ratio.

    Args:
        image: Input image as a NumPy array.
        target_width: Desired output width in pixels.
        target_height: Desired output height in pixels.

    Returns:
        A resized and padded NumPy image array.
    """
    height, width = image.shape[:2]

    scale = min(
        target_width / width,
        target_height / height
    )

    new_width = int(width * scale)
    new_height = int(height * scale)

    resized = cv2.resize(
        image,
        (new_width, new_height)
    )

    pad_width = target_width - new_width
    pad_height = target_height - new_height

    left = pad_width // 2
    right = pad_width - left

    top = pad_height // 2
    bottom = pad_height - top

    padded = cv2.copyMakeBorder(
        resized,
        top,
        bottom,
        left,
        right,
        cv2.BORDER_CONSTANT,
        value=(114, 114, 114)
    )

    return padded

def normalize_image(image):
    """
    Normalize image pixel values from 0-255 to 0.0-1.0.

    Args:
        image: Input image as a NumPy array.

    Returns:
        A float32 NumPy array with normalized pixel values.
    """
    return image.astype("float32") / 255.0

def bgr_to_rgb(image):
    """
    Convert an OpenCV image from BGR color order to RGB.

    Args:
        image: Input image as a NumPy array in BGR format.

    Returns:
        A NumPy array in RGB format.
    """
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def hwc_to_chw(image):
    """
    Rearrange an image from height-width-channels to channels-height-width.

    Args:
        image: Input image as a NumPy array with shape (H, W, C).

    Returns:
        A NumPy array with shape (C, H, W).
    """
    return image.transpose(2, 0, 1)

def convert_to_grayscale(image):
    """
    Convert a BGR image to grayscale.

    Args:
        image: Input image as a NumPy array in BGR format.

    Returns:
        A single-channel grayscale NumPy array.
    """
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

def apply_gaussian_blur(image, kernel_size=5):
    """
    Apply Gaussian blur to reduce high-frequency detail and noise.

    Args:
        image: Input image as a NumPy array.
        kernel_size: Size of the square blur kernel. Must be a positive odd integer.

    Returns:
        A blurred NumPy image array.
    """
    return cv2.GaussianBlur(
        image,
        (kernel_size, kernel_size),
        0
    )

def apply_motion_blur(image, kernel_size=15):
    """
    Apply horizontal motion blur to an image.

    Args:
        image: Input image as a NumPy array.
        kernel_size: Length of the motion blur kernel.

    Returns:
        A motion-blurred NumPy image array.
    """
    kernel = np.zeros((kernel_size, kernel_size))
    kernel[kernel_size // 2, :] = 1.0

    kernel /= kernel_size

    return cv2.filter2D(image, -1, kernel)

def add_glare(image, center, radius=80, strength=0.6):
    """
    Add synthetic glare to an image.

    Args:
        image: Input image as a NumPy array.
        center: Glare center as an (x, y) tuple.
        radius: Radius of the glare region in pixels.
        strength: Glare opacity from 0.0 to 1.0.

    Returns:
        A NumPy image array with simulated glare.
    """
    overlay = image.copy()

    cv2.circle(
        overlay,
        center,
        radius,
        (255, 255, 255),
        -1
    )

    return cv2.addWeighted(
        overlay,
        strength,
        image,
        1 - strength,
        0
    )

def rectify_perspective(image, src_points, output_width, output_height):
    """
    Rectify a perspective-distorted region into a front-facing rectangle.

    Args:
        image: Input image as a NumPy array.
        src_points: Four source points ordered as top-left, top-right,
            bottom-right, and bottom-left.
        output_width: Width of the rectified output image.
        output_height: Height of the rectified output image.

    Returns:
        A perspective-corrected NumPy image array.
    """
    src = np.float32(src_points)

    dst = np.float32([
        [0, 0],
        [output_width - 1, 0],
        [output_width - 1, output_height - 1],
        [0, output_height - 1]
    ])

    transform = cv2.getPerspectiveTransform(src, dst)

    rectified = cv2.warpPerspective(
        image,
        transform,
        (output_width, output_height)
    )

    return rectified

