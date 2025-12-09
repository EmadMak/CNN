import numpy as np
import numpy.random as npr
from PIL import Image, ImageFilter
import os
import json
import random
import glob
import itertools 
from image_hashing import remove_duplicates 

# --- config and setup ---

# Define the base directory structure
cwd = os.getcwd()
BASE_DIR = os.path.abspath(os.path.join(cwd, "..", "data"))
OUTPUT_DIR = os.path.join(BASE_DIR, "augmented")
INPUT_DIR = os.path.join(BASE_DIR, "base")

# The required fixed size of the images
IMAGE_SIZE = 19 
# Number of augmentations to generate per unique chain type (e.g., how many R-S-Z sequences to make)
NUM_AUGMENTATIONS_PER_CHAIN_TYPE = 50 
# Maximum number of effects in a single chain (1=R, 2=R-S, 3=R-S-Z, 4=R-S-Z-M)
MAX_CHAIN_LENGTH = 4 
# Number of independent noise images to generate per base image
NUM_NOISE_AUGMENTATIONS = 15

# NEW CONFIG: List of chain lengths that should receive the Phase 2 noise application.
# Setting to [1, 2] means noise is applied to single-step and two-step unique transforms.
NOISE_APPLY_CHAIN_LENGTHS = [1, 2] 

# Ensure the output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

print(f"Base data path: {BASE_DIR}")
print(f"Output path: {OUTPUT_DIR}")
print(f"All images will be processed at {IMAGE_SIZE}x{IMAGE_SIZE}.")

# Load limits
try:
    with open(os.path.join(BASE_DIR, "limits.json"), "r") as f:
        limits = json.load(f)
except FileNotFoundError:
    print("Warning: limits.json not found. Using internal default limits.")
    limits = {
        "default": {
            "max_rotation": 5,
            "max_shift_x": [-2, 2],
            "max_shift_y": [-2, 2],
            "noise_amount": 0.05,
            "morph_kernel_size": 1, # Fixed to 1 for subtle morphology on 19x19
            "scale_factors": [0.9, 1.1], 
            "max_shear_x": 0.2,
            "max_shear_y": 0.2
        }
    }
except json.JSONDecodeError as e:
    print(f"Error reading limits.json: {e}")
    # Fallback to default if JSON is unreadable
    limits = {"default": {"morph_kernel_size": 1}} 

# --- Utility Functions ---

def binarize(img: Image.Image) -> Image.Image:
    """Ensures the image is strictly black (0) and white (255)."""
    return img.point(lambda x: 255 if x > 128 else 0)

def load_all_base_images(input_path):
    """
    Loads all base images from the input directory.
    Attaches a 'base_name' (e.g., '0_1') and 'class_id_key' (e.g., '0') attribute for tracking.
    """
    base_images = []
    search_pattern = os.path.join(input_path, '*.png')
    image_paths = sorted(glob.glob(search_pattern))
    
    if not image_paths:
        print(f"Error: No base images found matching '*.png' in {input_path}")
        return []

    for filepath in image_paths:
        filename = os.path.basename(filepath)
        # base_name_prefix is '0_1'
        base_name_prefix = filename.split('.')[0]
        # class_id is '0'
        class_id = base_name_prefix.split('_')[0]
        
        try:
            img = Image.open(filepath).convert("L")
            if img.size != (IMAGE_SIZE, IMAGE_SIZE):
                img = img.resize((IMAGE_SIZE, IMAGE_SIZE), Image.Resampling.NEAREST)
            img = binarize(img)
            
            # Attach custom metadata
            img.base_name = base_name_prefix
            img.class_id_key = class_id # Used for limits lookup
            base_images.append(img)
        except Exception as e:
            print(f"Error loading {filename}: {e}")
            
    print(f"Successfully loaded {len(base_images)} unique base images for augmentation.")
    return base_images

def save_image(photo: Image.Image, base_name: str, aug_type: str, number: str, destination_dir: str):
    """Saves the augmented image to a subdirectory inside the main output directory."""
    sub_dir = os.path.join(OUTPUT_DIR, destination_dir)
    os.makedirs(sub_dir, exist_ok=True) 

    # Ensure number part is clean for filename (e.g., replaces leading hyphen with 'n')
    clean_number = str(number).replace('-', 'n')
    
    # aug_type is now just a single letter identifier like 'R', 'S', 'c', etc.
    filename = f"{base_name}_{aug_type}{clean_number}.png"
    path = os.path.join(sub_dir, filename)
    
    try:
        photo.save(path)
    except Exception as e:
        print(f"Error saving {filename}: {e}")
        

# --- Atomic Augmentation Functions (applied with specific parameters) ---

def apply_shift(img: Image.Image, dx: int, dy: int) -> Image.Image:
    """Applies a specific x and y shift."""
    w, h = img.size
    shifted = Image.new("L", (w, h), 255)
    shifted.paste(img, (dx, dy))
    return shifted

def apply_rotate(img: Image.Image, angle: int) -> Image.Image:
    """Applies a specific rotation angle."""
    rot = img.rotate(
        angle,
        expand=False,
        resample=Image.Resampling.NEAREST,
        fillcolor=255
    )
    return binarize(rot)

def apply_morphology(img: Image.Image, kernel_size: int, type: str) -> Image.Image:
    """Applies a specific morphology operation (erode or dilate)."""
    # Note: kernel_size is fixed at 1 for subtle morphology on 19x19
    if type == "erode":
        morphed = img.filter(ImageFilter.MinFilter(kernel_size))
    elif type == "dilate":
        morphed = img.filter(ImageFilter.MaxFilter(kernel_size))
    else:
        return img 
    return binarize(morphed)


def apply_scale(img: Image.Image, factor: float) -> Image.Image:
    """Applies a specific scaling factor (zoom in/out)."""
    w, h = img.size
    new_w, new_h = int(w * factor), int(h * factor)
    
    scaled = img.resize((new_w, new_h), Image.Resampling.NEAREST)
    scaled = binarize(scaled)
    
    if factor < 1.0:
        # Zoom Out (Scale Down and Pad to center)
        new_img = Image.new("L", (w, h), 255)
        x_offset = (w - new_w) // 2
        y_offset = (h - new_h) // 2
        new_img.paste(scaled, (x_offset, y_offset))
        return new_img
        
    elif factor > 1.0:
        # Zoom In (Scale Up and Crop center)
        temp_img = Image.new("L", (new_w, new_h), 255)
        temp_img.paste(scaled, (0, 0))

        # Calculate the crop box (center 19x19 region)
        left = (new_w - w) // 2
        top = (new_h - h) // 2
        right = left + w
        bottom = top + h
        
        cropped = temp_img.crop((left, top, right, bottom))
        return cropped
        
    return img

def apply_shear(img: Image.Image, sx: float, sy: float) -> Image.Image:
    """Applies a specific horizontal (sx) and vertical (sy) shear."""
    w, h = img.size
    
    transform_matrix = [
        1, sx, -sx * h / 2,
        sy, 1, -sy * w / 2
    ]
    
    sheared_img = img.transform(
        (w, h), 
        Image.Transform.AFFINE, 
        transform_matrix, 
        resample=Image.Resampling.NEAREST, 
        fillcolor=255
    )
    
    return binarize(sheared_img)

def apply_pepper_noise(img: Image.Image, amount: float) -> Image.Image:
    """Adds black dots (pepper) noise, speckling the white background."""
    arr = np.array(img).copy()
    h, w = arr.shape
    n_pixels = int(h * w * amount)

    # Note: These are random, scattered coordinates across the entire image
    ys = npr.randint(0, h, n_pixels)
    xs = npr.randint(0, w, n_pixels)

    arr[ys, xs] = 0  # set to black (0) -> Pepper noise
    return Image.fromarray(arr, 'L')

def apply_salt_noise(img: Image.Image, amount: float) -> Image.Image:
    """Randomly turns existing black pixels to white (salt) noise, creating holes."""
    arr = np.array(img).copy()
    
    # Find all current black pixels (the character itself)
    black_coords = np.argwhere(arr == 0)
    
    n_pixels = int(len(black_coords) * amount)
    if n_pixels == 0:
        return Image.fromarray(arr, 'L')
    
    # Choose a random subset of black pixels to erase (turn white)
    erase_indices = npr.choice(len(black_coords), n_pixels, replace=False)
    for idx in erase_indices:
        y, x = black_coords[idx]
        arr[y, x] = 255  # set to white (255) -> Salt noise
    
    return Image.fromarray(arr, 'L')


# --- Augmentation Group Definitions ---

def create_transformation_map(config: dict):
    """
    Defines the parameters and functions for each augmentation group 
    based on the loaded limits config. The list stores (function, tag_string) tuples.
    """
    max_rot = config.get('max_rotation', 5)
    shift_x_range = config.get('max_shift_x', [-2, 2])
    shift_y_range = config.get('max_shift_y', [-2, 2])
    morph_k = config.get('morph_kernel_size', 1) 
    scale_f = config.get('scale_factors', [0.9, 1.1])
    max_shear_x = config.get('max_shear_x', 0.2)
    max_shear_y = config.get('max_shear_y', 0.2)
    
    # R: Rotation 
    rotations = [
        (lambda img, a=a: apply_rotate(img, a), f"r{a}") 
        for a in range(-max_rot, max_rot + 1, 2) if a != 0
    ]

    # S: Shift - Randomly choose 5 combinations of shifts 
    shifts = []
    for i in range(5):
        dx = random.randint(shift_x_range[0], shift_x_range[1])
        dy = random.randint(shift_y_range[0], shift_y_range[1])
        shifts.append((lambda img, dx=dx, dy=dy: apply_shift(img, dx, dy), f"s{dx}y{dy}"))
    
    # Z: Scale/Shear (Zoomies)
    scale_shear = []
    scale_shear.append((lambda img, f=scale_f[0]: apply_scale(img, f), f"z{int(scale_f[0]*100)}"))
    scale_shear.append((lambda img, f=scale_f[1]: apply_scale(img, f), f"z{int(scale_f[1]*100)}"))
    scale_shear.append((lambda img, sx=max_shear_x, sy=0: apply_shear(img, sx, sy), f"sh{int(max_shear_x*100)}v0"))
    scale_shear.append((lambda img, sx=0, sy=max_shear_y: apply_shear(img, sx, sy), f"sh0v{int(max_shear_y*100)}"))

    # M: Morphology (Erode/Dilate)
    morphology_ops = [
        (lambda img, k=morph_k: apply_morphology(img, k, "erode"), f"me{morph_k}"),
        (lambda img, k=morph_k: apply_morphology(img, k, "dilate"), f"md{morph_k}"),
    ]
    
    # Map transformation groups to labels
    transformation_map = {
        'R': rotations,
        'S': shifts,
        'Z': scale_shear, 
        'M': morphology_ops
    }
    
    return transformation_map


# --- Unified Augmentation Generator (Chains of length 1, 2, 3, and 4) ---

def generate_augmentations(base_img: Image.Image, config: dict, num_images_per_chain: int) -> tuple[int, list[Image.Image]]:
    """
    Generates augmentations by applying random sequences of 1, 2, 3, or 4 effects.
    
    Returns the count of generated images and a list of the generated PIL Image objects.
    Each image object now stores its chain_length metadata for later filtering.
    """
    total_generated = 0
    base_name = base_img.base_name
    generated_images = []

    # Get the transformation functions and tags based on the current configuration limits
    transformation_map = create_transformation_map(config)
    group_keys = list(transformation_map.keys())
    
    # Generate all unique chains of lengths 1 through MAX_CHAIN_LENGTH (allowing repetition)
    all_chains = []
    for length in range(1, MAX_CHAIN_LENGTH + 1):
        # itertools.product generates ordered sequences with repetition (e.g., (R, R) is included)
        chains_of_length = itertools.product(group_keys, repeat=length)
        all_chains.extend(list(chains_of_length))
        
    print(f"    - Generating augmentations for {len(all_chains)} unique chain types (lengths 1 to {MAX_CHAIN_LENGTH}).")
    
    for combo_keys in all_chains:
        # Create a descriptive directory name for this chain type (e.g., 'R_S_Z', or just 'R' for length 1)
        chain_name = "_".join(combo_keys) 
        
        # Determine the directory for saving
        length = len(combo_keys)
        destination_dir = f"chained_L{length}/{chain_name}"

        # Get the lists of specific (function, tag) tuples for the current combo
        group_function_tags = [transformation_map[key] for key in combo_keys]
        
        # Guard against empty groups (shouldn't happen with current setup but good practice)
        if any(not group for group in group_function_tags):
             print(f"        Warning: Skipping chain {chain_name} due to empty transformation group.")
             continue

        for i in range(num_images_per_chain):
            
            current_img = base_img
            chain_tags = []
            
            # Build the chain: randomly pick one specific (function, tag) from each of the required groups
            chain = [random.choice(group) for group in group_function_tags]
            
            # Apply the chain sequentially and capture the tags
            for transform_func, tag in chain:
                current_img = transform_func(current_img)
                chain_tags.append(tag)
            
            # Concatenate the tags to form the parameter string for the filename (e.g., r-4_s2y1_z110)
            aug_tag = "_".join(chain_tags)
            
            # Save the result: filename contains parameter tags
            # Use 'c' type for all chained/combined effects, regardless of length
            save_image(current_img, base_name, "c", aug_tag, destination_dir)
            
            # Re-attach base metadata lost during transformation
            current_img.base_name = base_img.base_name
            current_img.class_id_key = base_img.class_id_key
            
            # Keep the image in memory and attach the tag and length for the noise phase
            current_img.chained_tag = f"c{aug_tag}" # Prefix 'c' for chained effects
            current_img.chain_length = length # Store chain length for filtering
            
            generated_images.append(current_img)
            total_generated += 1
            
        print(f"      -> Chain type {chain_name} (L={length}) generated {num_images_per_chain} images.")

    return total_generated, generated_images

# --- Noise on ALL Transformed Generator ---

def generate_noise_on_transformed(transformed_images: list[Image.Image], config: dict, num_noise_variants: int) -> int:
    """
    Generates Salt and Pepper noise variations for a list of ANY already transformed images (single or chained).
    Uses the image's parameter tag (chained_tag) in the output filename for full traceability.
    """
    total_generated = 0
    
    # Define the range and bell curve parameters
    min_noise = 0.05
    max_noise = 0.20
    mean = 0.10 
    std_dev = 0.05 
    
    # We will generate a distinct noise amount for every Salt/Pepper application
    num_total_noise_applications = len(transformed_images) * num_noise_variants * 2 # *2 for Salt + Pepper
    noise_amounts = npr.normal(loc=mean, scale=std_dev, size=num_total_noise_applications)
    clamped_amounts = np.clip(noise_amounts, min_noise, max_noise)
    
    # Track the index of the noise amount array
    amount_index = 0

    print(f"    - Generating {num_noise_variants * 2} noise variants per {len(transformed_images)} transformed image.")
    
    for i, img in enumerate(transformed_images):
        base_name = img.base_name
        
        # Retrieve the full parameter tag from the previous phase (e.g., 'Rr-4' or 'cr-4_s2y1_z90')
        transform_tag = getattr(img, 'chained_tag', f"unknown{i}") # Fallback for safety
        
        for j in range(num_noise_variants):
            if amount_index + 1 >= len(clamped_amounts):
                 print("Warning: Ran out of pre-generated noise amounts.")
                 break 
            
            # Use two consecutive amounts from the clamped array for Salt and Pepper
            pepper_amount = clamped_amounts[amount_index]
            salt_amount = clamped_amounts[amount_index + 1]
            amount_index += 2 
            
            # Generate clean string representation of the amount (e.g., 0.123 -> 123)
            pepper_amount_str = f"{pepper_amount:.3f}".replace('.', '')
            salt_amount_str = f"{salt_amount:.3f}".replace('.', '')
            
            # 1. Apply Pepper Noise (Black dots on white background)
            img_pepper = apply_pepper_noise(img, pepper_amount)
            # Tag format: [transform_tag]_nP[noise_amount] (e.g., Rr-4_nP123)
            final_tag_pepper = f"{transform_tag}_nP{pepper_amount_str}"
            save_image(img_pepper, base_name, "ncP", final_tag_pepper, "noise_on_transformed")
            total_generated += 1

            # 2. Apply Salt Noise (White holes in the black character)
            img_salt = apply_salt_noise(img, salt_amount)
            # Tag format: [transform_tag]_nS[noise_amount] (e.g., Rr-4_nS130)
            final_tag_salt = f"{transform_tag}_nS{salt_amount_str}"
            save_image(img_salt, base_name, "ncS", final_tag_salt, "noise_on_transformed")
            total_generated += 1
            
    return total_generated


# --- Noise Generator (Original, applied only to base images) ---

def generate_noise_augmentations(base_img: Image.Image, config: dict, num_images: int) -> int:
    """
    Generates a set of noisy images using a bell curve for noise percentage, 
    favoring lower noise amounts. Applied only to the original base image.
    Uses the noise amount in the filename.
    """
    total_generated = 0
    base_name = base_img.base_name
    
    # Define the range and bell curve parameters
    min_noise = 0.05
    max_noise = 0.20
    mean = 0.10 
    std_dev = 0.05 
    
    # Generate noise amounts using a normal distribution
    noise_amounts = npr.normal(loc=mean, scale=std_dev, size=num_images)
    
    # Clamp values to the desired range [0.05, 0.20]
    clamped_amounts = np.clip(noise_amounts, min_noise, max_noise)
    
    print(f"    - Noise percentages generated (min/max): {np.min(clamped_amounts):.3f} / {np.max(clamped_amounts):.3f}")

    for i, amount in enumerate(clamped_amounts):
        # Generate clean string representation of the amount (e.g., 0.123 -> 123)
        amount_str = f"{amount:.3f}".replace('.', '')
        
        # 1. Apply Pepper Noise (Black dots on white background)
        img_pepper = apply_pepper_noise(base_img, amount)
        # Filename now includes the noise amount string (e.g., 0_1_nP123.png)
        save_image(img_pepper, base_name, "nP", amount_str, "noise_pepper")
        total_generated += 1

        # 2. Apply Salt Noise (White holes in the black character)
        img_salt = apply_salt_noise(base_img, amount)
        # Filename now includes the noise amount string (e.g., 0_1_nS123.png)
        save_image(img_salt, base_name, "nS", amount_str, "noise_salt")
        total_generated += 1
        
    return total_generated


# --- Main Augmentation Pipeline ---

def augment_pipeline(base_images):
    """
    Runs the full augmentation pipeline on all base images.
    """
    total_images_generated = 0
    
    # Calculate the total number of unique chain types that will receive noise
    chains_receiving_noise = 0
    for length in NOISE_APPLY_CHAIN_LENGTHS:
        if length <= MAX_CHAIN_LENGTH:
            chains_receiving_noise += (4 ** length)
    
    # Calculate the expected number of images passing to Phase 2 (before deduplication)
    expected_inputs_for_noise = len(base_images) * chains_receiving_noise * NUM_AUGMENTATIONS_PER_CHAIN_TYPE
    
    print(f"\n--- Augmentation Pipeline Configuration ---")
    print(f"Total Unique Chains (L1-L{MAX_CHAIN_LENGTH}): {340}")
    print(f"Noise Applied To Chain Lengths (L): {NOISE_APPLY_CHAIN_LENGTHS}")
    print(f"Expected Geometric Inputs for Noise (before dedupe): {expected_inputs_for_noise}")
    print(f"Expected Final Noise Images: {expected_inputs_for_noise * NUM_NOISE_AUGMENTATIONS * 2}")
    print(f"-----------------------------------------\n")


    for base_img in base_images:
        
        # List to hold ALL geometrically transformed images for potential noise application
        all_transformed_images = []
        
        # Get configuration using the class ID (e.g., '0' for '0_1.png')
        class_id_key = base_img.class_id_key
        config = limits.get(class_id_key, limits.get("default", {}))

        print(f"\nProcessing base image {base_img.base_name} (Class ID: {class_id_key})")
        
        # --- PHASE 1: Unified Augmentations (Chains of length 1, 2, 3, and 4) ---
        print(f" -> Generating unified augmentations (Chains L=1 to L={MAX_CHAIN_LENGTH})...")
        combined_count, combined_images = generate_augmentations(base_img, config, NUM_AUGMENTATIONS_PER_CHAIN_TYPE)
        total_images_generated += combined_count
        all_transformed_images.extend(combined_images)
        print(f"    - Generated {combined_count} combined augmentations. ({len(all_transformed_images)} transformed images total)")

        # --- DEDUPLICATION STEP ---
        # Filter the geometrically transformed images to ensure uniqueness before applying noise
        unique_transformed_images = remove_duplicates(all_transformed_images)
        
        # --- PHASE 2 INPUT FILTER: Use only configured Chain Lengths for Noise ---
        # Filter the unique list based on the globally defined NOISE_APPLY_CHAIN_LENGTHS
        filtered_for_noise_images = [
            img for img in unique_transformed_images 
            if getattr(img, 'chain_length', 0) in NOISE_APPLY_CHAIN_LENGTHS
        ]
        
        # --- PHASE 2: Noise on Filtered Transformed Images ---
        chain_lengths_str = ", ".join(map(str, NOISE_APPLY_CHAIN_LENGTHS))
        print(f" -> Generating noise on {len(filtered_for_noise_images)} unique, L={chain_lengths_str} transformed images...")
        noisy_transformed_count = generate_noise_on_transformed(filtered_for_noise_images, config, NUM_NOISE_AUGMENTATIONS)
        total_images_generated += noisy_transformed_count
        print(f"    - Generated {noisy_transformed_count} total noisy L={chain_lengths_str} augmentations. Saved to 'noise_on_transformed'.")
        
        # --- PHASE 3: Independent Noise Generation (Bell Curve Distribution on BASE images) ---
        print(f" -> Generating {NUM_NOISE_AUGMENTATIONS * 2} independent noise images (Salt & Pepper) on BASE...")
        noise_count = generate_noise_augmentations(base_img, config, NUM_NOISE_AUGMENTATIONS)
        total_images_generated += noise_count
        print(f"    - Generated {noise_count} base noise augmentations. Saved to 'noise_pepper'/'noise_salt'.")
            
    print(f"\nPipeline finished. Total augmentations saved: {total_images_generated}")

# --- Execution ---

if __name__ == "__main__":
    # 1. Load the base images
    base_images = load_all_base_images(INPUT_DIR)
    
    if base_images:
        # 2. Run the pipeline
        augment_pipeline(base_images)
    else:
        print("No base images loaded. Please check your 'base' directory.")
