import numpy as np
import numpy.random as npr
from PIL import Image, ImageFilter
import os
import json
import random
import glob
import itertools 

# --- config and setup ---

# Define the base directory structure
cwd = os.getcwd()
BASE_DIR = os.path.abspath(os.path.join(cwd, "..", "data"))
OUTPUT_DIR = os.path.join(BASE_DIR, "augmented")
INPUT_DIR = os.path.join(BASE_DIR, "base")

# The required fixed size of the images
IMAGE_SIZE = 19 
# Number of chained (combinatorial) images to generate per base image
NUM_CHAINED_AUGMENTATIONS = 50
# Number of independent single-effect images to generate per group (R, S, Z, M)
NUM_SINGLE_AUGMENTATIONS = 10 
# Number of independent noise images to generate per base image
NUM_NOISE_AUGMENTATIONS = 15

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
            # FIX: Changed default kernel size from 2 (even) to 3 (odd)
            "morph_kernel_size": 3, 
            "scale_factors": [0.9, 1.1], 
            "max_shear_x": 0.2,
            "max_shear_y": 0.2
        }
    }
except json.JSONDecodeError as e:
    print(f"Error reading limits.json: {e}")
    # Fallback to default if JSON is unreadable
    # Changed default here too to 3 for safety
    limits = {"default": {"morph_kernel_size": 3}} 

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
    # Note: In the new system, 'number' is now the parameter tag (e.g., 'r-4', 's2y-1')
    clean_number = str(number).replace('-', 'n')
    
    # We now use the parameter string directly as the file suffix.
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
    # Note: kernel_size must be an odd integer (e.g., 3, 5, 7)
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
    based on the loaded limits config. The list now stores (function, tag_string) tuples.
    """
    max_rot = config.get('max_rotation', 5)
    shift_x_range = config.get('max_shift_x', [-2, 2])
    shift_y_range = config.get('max_shift_y', [-2, 2])
    morph_k = config.get('morph_kernel_size', 3) # Using default 3
    scale_f = config.get('scale_factors', [0.9, 1.1])
    max_shear_x = config.get('max_shear_x', 0.2)
    max_shear_y = config.get('max_shear_y', 0.2)
    
    # R: Rotation (Range from -max to +max, step 2) -> (function, tag)
    rotations = [
        (lambda img, a=a: apply_rotate(img, a), f"r{a}") 
        for a in range(-max_rot, max_rot + 1, 2) if a != 0
    ]

    # S: Shift - Randomly choose 5 combinations of shifts -> (function, tag)
    shifts = []
    for i in range(5):
        dx = random.randint(shift_x_range[0], shift_x_range[1])
        dy = random.randint(shift_y_range[0], shift_y_range[1])
        shifts.append((lambda img, dx=dx, dy=dy: apply_shift(img, dx, dy), f"s{dx}y{dy}"))
    
    # Z: Scale/Shear (Zoomies) - (function, tag)
    scale_shear = []
    # Scales
    scale_shear.append((lambda img, f=scale_f[0]: apply_scale(img, f), f"z{int(scale_f[0]*100)}"))
    scale_shear.append((lambda img, f=scale_f[1]: apply_scale(img, f), f"z{int(scale_f[1]*100)}"))
    # Shears
    scale_shear.append((lambda img, sx=max_shear_x, sy=0: apply_shear(img, sx, sy), f"sh{int(max_shear_x*100)}v0"))
    scale_shear.append((lambda img, sx=0, sy=max_shear_y: apply_shear(img, sx, sy), f"sh0v{int(max_shear_y*100)}"))

    # M: Morphology (Erode/Dilate) - (function, tag)
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


# --- Single Effect Generator ---

def generate_single_augmentations(base_img: Image.Image, config: dict, num_images: int) -> int:
    """
    Generates single-effect augmentations (Rotation only, Shift only, etc.)
    by randomly sampling one function and its associated parameter tag from each group.
    """
    total_generated = 0
    base_name = base_img.base_name

    transformation_map = create_transformation_map(config)
    
    print(f"    - Generating {num_images} images for each of the {len(transformation_map)} single-effect groups.")

    for key, group_functions in transformation_map.items():
        if not group_functions:
            print(f"        Warning: Skipping single-effect group {key} due to empty transformation list.")
            continue
            
        print(f"      -> Single effect type: {key}")
            
        for i in range(num_images):
            # Pick one function AND its tag randomly from the group
            transform_func, tag = random.choice(group_functions)
            current_img = transform_func(base_img)
            
            # Save the result: destination_dir is now 'single/R', filename contains tag (e.g., 0_1_Rr-4.png)
            # The 'number' argument in save_image is now the parameter tag itself
            save_image(current_img, base_name, key, tag, f"single/{key}")
            total_generated += 1

    return total_generated


# --- Combinatorial Generator ---

def generate_chained_augmentations(base_img: Image.Image, config: dict, num_images: int) -> tuple[int, list[Image.Image]]:
    """
    Generates complex augmentations by applying a random sequence of 3 effects.
    Images are saved into subdirectories based on the combination type (e.g., R_S_Z).
    
    Returns the count of generated images and a list of the generated PIL Image objects,
    with the final parameter tag attached as an attribute.
    """
    total_generated = 0
    base_name = base_img.base_name
    generated_images = []

    # Get the transformation functions and tags based on the current configuration limits
    transformation_map = create_transformation_map(config)
    
    # Define all unique 3-step chains (e.g., R, S, Z)
    chain_combinations = list(itertools.combinations(transformation_map.keys(), 3))
    
    print(f"    - Generating {num_images} images for each of the {len(chain_combinations)} unique 3-step chains.")

    for combo_keys in chain_combinations:
        # Create a descriptive directory name for this chain type (e.g., 'R_S_Z')
        chain_name = "_".join(combo_keys) 
        
        print(f"      -> Chain type: {chain_name}")

        group_function_tags = [transformation_map[key] for key in combo_keys]
        
        if any(not group for group in group_function_tags):
             print(f"        Warning: Skipping chain {chain_name} due to empty transformation group.")
             continue

        for i in range(num_images):
            
            current_img = base_img
            chain_tags = []
            
            # Build the chain: randomly pick one specific (function, tag) from each of the three groups
            chain = [random.choice(group) for group in group_function_tags]
            
            # Apply the chain sequentially and capture the tags
            for transform_func, tag in chain:
                current_img = transform_func(current_img)
                chain_tags.append(tag)
            
            # Concatenate the tags to form the parameter string for the filename (e.g., r-4_s2y1_z110)
            aug_tag = "_".join(chain_tags)
            
            # Save the result: destination_dir is now 'chained/R_S_Z', filename contains parameter tags
            save_image(current_img, base_name, "c", aug_tag, f"chained/{chain_name}")
            
            # Keep the image in memory and attach the tag for the noise phase
            current_img.chained_tag = aug_tag
            generated_images.append(current_img)
            total_generated += 1

    return total_generated, generated_images

# --- Noise on Chained Generator (NEW) ---

def generate_noise_on_chained(chained_images: list[Image.Image], config: dict, num_noise_variants: int) -> int:
    """
    Generates Salt and Pepper noise variations for a list of already transformed images.
    Uses the chained image's parameter tag in the output filename for full traceability.
    """
    total_generated = 0
    
    # Define the range and bell curve parameters
    min_noise = 0.05
    max_noise = 0.20
    mean = 0.10 
    std_dev = 0.05 
    
    # We will generate a distinct noise amount for every Salt/Pepper application
    num_total_noise_applications = len(chained_images) * num_noise_variants * 2 # *2 for Salt + Pepper
    noise_amounts = npr.normal(loc=mean, scale=std_dev, size=num_total_noise_applications)
    clamped_amounts = np.clip(noise_amounts, min_noise, max_noise)
    
    # Track the index of the noise amount array
    amount_index = 0

    print(f"    - Generating {num_noise_variants * 2} noise variants per chained image.")
    
    for i, img in enumerate(chained_images):
        base_name = img.base_name
        
        # Retrieve the full parameter tag from the previous phase
        chained_tag = getattr(img, 'chained_tag', f"c{i}") # Fallback for safety
        
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
            # Tag format: c[chained_params]_p[noise_amount] (e.g., cR5_S-2y1_Z90_p123)
            final_tag_pepper = f"{chained_tag}_p{pepper_amount_str}"
            save_image(img_pepper, base_name, "ncP", final_tag_pepper, "noise_on_chained")
            total_generated += 1

            # 2. Apply Salt Noise (White holes in the black character)
            img_salt = apply_salt_noise(img, salt_amount)
            # Tag format: c[chained_params]_s[noise_amount] (e.g., cR5_S-2y1_Z90_s130)
            final_tag_salt = f"{chained_tag}_s{salt_amount_str}"
            save_image(img_salt, base_name, "ncS", final_tag_salt, "noise_on_chained")
            total_generated += 1
            
    # Note: Each chained image gets 2 * NUM_NOISE_AUGMENTATIONS (Salt + Pepper) applied to it
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
    
    for base_img in base_images:
        
        # Get configuration using the class ID (e.g., '0' for '0_1.png')
        class_id_key = base_img.class_id_key
        config = limits.get(class_id_key, limits.get("default", {}))

        print(f"\nProcessing base image {base_img.base_name} (Class ID: {class_id_key})")
        
        # --- PHASE 1a: Single Augmentations (Rotation only, Shift only, etc.) ---
        print(f" -> Generating single-effect augmentations (R, S, Z, M)...")
        single_count = generate_single_augmentations(base_img, config, NUM_SINGLE_AUGMENTATIONS)
        total_images_generated += single_count
        print(f"    - Generated {single_count} single-effect augmentations. Filenames now show parameters (e.g., 0_1_Rr-4.png)")

        # --- PHASE 1b: Combinatorial Augmentations (Rotation + Shift + Scale, etc.) ---
        print(f" -> Generating complex augmentations across different chain types (3-step chains)...")
        # NOTE: This now returns the list of images for use in the next phase
        chained_count, chained_images = generate_chained_augmentations(base_img, config, NUM_CHAINED_AUGMENTATIONS)
        total_images_generated += chained_count
        print(f"    - Generated {chained_count} complex augmentations. Filenames now show chained parameters (e.g., 0_1_cr-4_s2y1_z90.png)")
        
        # --- PHASE 1c: Noise on Chained Augmentations (NEW) ---
        print(f" -> Generating noise on {chained_count} chained images...")
        # Multiplied by 2 because Salt (White) and Pepper (Black) noise are generated per amount
        noisy_chained_count = generate_noise_on_chained(chained_images, config, NUM_NOISE_AUGMENTATIONS)
        total_images_generated += noisy_chained_count
        print(f"    - Generated {noisy_chained_count} noisy chained augmentations. Filenames fully traceable (e.g., 0_1_ncPr-4_s2y1_z90_p123.png)")
        
        # --- PHASE 2: Independent Noise Generation (Bell Curve Distribution on BASE images) ---
        print(f" -> Generating {NUM_NOISE_AUGMENTATIONS * 2} independent noise images (Salt & Pepper) on BASE...")
        noise_count = generate_noise_augmentations(base_img, config, NUM_NOISE_AUGMENTATIONS)
        total_images_generated += noise_count
        print(f"    - Generated {noise_count} base noise augmentations. Filenames now include amount (e.g., 0_1_nP123.png)")
            
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
