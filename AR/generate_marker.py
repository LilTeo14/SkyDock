import cv2
import numpy as np

def generate():
    print(f"OpenCV Version: {cv2.__version__}")

    # Use 4x4 dictionary with 50 markers
    dict_type = cv2.aruco.DICT_4X4_50
    if hasattr(cv2.aruco, 'getPredefinedDictionary'):
        dictionary = cv2.aruco.getPredefinedDictionary(dict_type)
    else:
        dictionary = cv2.aruco.Dictionary_get(dict_type)

    # Dimensions for the landing pad image
    board_size = 800
    padded_image = np.ones((board_size, board_size), dtype=np.uint8) * 255

    # Marker sizes (must keep exact 5:1 ratio for physical scale consistency)
    large_size = 300  # ID 0 (15cm equivalent)
    small_size = 60   # ID 1 (3cm equivalent)

    # Generate Large Marker (ID 0)
    if hasattr(cv2.aruco, 'generateImageMarker'):
        large_marker = cv2.aruco.generateImageMarker(dictionary, 0, large_size)
    else:
        large_marker = cv2.aruco.drawMarker(dictionary, 0, large_size)

    # Generate Small Marker (ID 1)
    if hasattr(cv2.aruco, 'generateImageMarker'):
        small_marker = cv2.aruco.generateImageMarker(dictionary, 1, small_size)
    else:
        small_marker = cv2.aruco.drawMarker(dictionary, 1, small_size)

    # Placement coordinates:
    # Small marker: Centered exactly at (400, 400)
    s_half = small_size // 2
    s_center = board_size // 2
    padded_image[s_center - s_half : s_center + s_half, s_center - s_half : s_center + s_half] = small_marker

    # Large marker: Centered horizontally at 400, vertically offset upwards to y=170 (range 20 to 320)
    l_half = large_size // 2
    l_cx = board_size // 2
    l_cy = 170
    padded_image[l_cy - l_half : l_cy + l_half, l_cx - l_half : l_cx + l_half] = large_marker

    # Save file
    filename = "marker_0.png"
    cv2.imwrite(filename, padded_image)
    print(f"Successfully generated dual landing pad '{filename}':")
    print(f"  - Large Marker (ID 0): size {large_size}px, centered at (400, 170)")
    print(f"  - Small Marker (ID 1): size {small_size}px, centered at (400, 400)")
    print(f"  - Center-to-center offset: {s_center - l_cy}px (Y-axis)")

if __name__ == "__main__":
    generate()
