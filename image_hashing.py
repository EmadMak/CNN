import numpy as np
from PIL import Image

def get_image_hash(img: Image.Image) -> str:
    """
    Calculates a simple, unique hash for an image array.
    
    This is used to detect exact pixel duplicates among geometrically transformed images.
    Since the images are small (19x19) and binarized (0 or 255), this hash is fast
    and robust for exact match detection.
    """
    try:
        # Convert PIL Image to NumPy array
        arr = np.array(img, dtype=np.uint8)
        # Flatten the array and convert it to a tuple for hashing
        # This tuple represents the sequence of pixel values (19 * 19 = 361 values)
        return hash(arr.tobytes())
    except Exception as e:
        print(f"Error hashing image: {e}")
        # Fallback hash if array conversion fails
        return f"error_{id(img)}"

def remove_duplicates(images: list[Image.Image]) -> list[Image.Image]:
    """
    Takes a list of PIL images, removes duplicates based on the image hash,
    and returns a new list containing only unique images.
    """
    unique_images = []
    seen_hashes = set()
    total_input = len(images)

    for img in images:
        img_hash = get_image_hash(img)
        
        if img_hash not in seen_hashes:
            seen_hashes.add(img_hash)
            unique_images.append(img)
            
    num_removed = total_input - len(unique_images)
    print(f"    [Deduplication]: Removed {num_removed} duplicate images ({len(unique_images)} unique images remain).")
    
    return unique_images
