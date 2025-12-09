import numpy as np
from PIL import Image, ImageFilter
import os
import json
import glob
import random

# This file took a long time to make. I have no idea what I'm writing anymore.

# --- config ---

# current working directory
cwd = os.getcwd()
print("current directory: ", cwd)

# go back one directory
BASE_DIR = os.path.join(cwd, "..", "data")
BASE_DIR = os.path.abspath(BASE_DIR)
OUTPUT_DIR = os.path.join(BASE_DIR, "augmented")
INPUT_DIR = os.path.join(BASE_DIR, "base")

IMAGE_SIZE = 19

# Number of chained (combinatorial) images to generate per base image
NUM_CHAINED_AUGMENTATIONS = 50
# Number of independent single-effect images to generate per group (R, S, Z, M)
NUM_SINGLE_AUGMENTATIONS = 10 
# Number of independent noise images to generate per base image
NUM_NOISE_AUGMENTATIONS = 15

# Extra insurance
os.makedirs(OUTPUT_DIR, exist_ok=True)

print(f"Base data path: {BASE_DIR}")
print(f"Output path: {OUTPUT_DIR}")
print(f"All images will be processed at {IMAGE_SIZE}x{IMAGE_SIZE}.")

# Load limits
try:
    with open(os.path.join(BASE_DIR, "limits.json"), "r") as f:
        limits = json.load(f)
except FileNotFoundError:
    print("Warning: limits.json not found. Using default limits.")


# with open(images_path + "/limits.json", "r") as f:
#    limits = json.load(f)

print(limits)

# --- Load Base Images ---

# This should make shit easier for me. I decided to
# use glob library because of the * symbol thing...
# This should generalize the base images names. It's
# great because I'll add hand written data too. It will
# look like {base}_{version}.png.

def binarize(img: Image.Image) -> Image.Image:
    """Ensures the image is strictly black (0) and white (255)."""
    return img.point(lambda x: 255 if x > 128 else 0)

def load_all_base_images(input_path):
    """
    Loads all base images from the input directory using a wildcard search for all .png files.
    This supports the flexible naming convention [ClassId]_[VersionId].png.
    """
    base_images = []
    
    # Search for all .png files in the base directory
    search_pattern = os.path.join(input_path, '*.png')
    image_paths = sorted(glob.glob(search_pattern))
    
    if not image_paths:
        print(f"Error: No base images found matching '*.png' in {input_path}")
        return []

    for filepath in image_paths:
        # Extract filename (e.g., "0_1.png")
        filename = os.path.basename(filepath)
        # Extract base_name_prefix (e.g., "0_1") for limits lookup and naming augmented files
        base_name_prefix = filename.split('.')[0]

        class_id = base_name_prefix.split('_')[0]
        
        try:
            # Use convert("L") for grayscale
            img = Image.open(filepath).convert("L")
            # Enforce the target size
            if img.size != (IMAGE_SIZE, IMAGE_SIZE):
                print(f"Resizing {filename} from {img.size} to {IMAGE_SIZE}x{IMAGE_SIZE}...")
                img = img.resize((IMAGE_SIZE, IMAGE_SIZE), Image.Resampling.NEAREST)
            img = binarize(img) # Ensure base images are also strictly B/W
            
            # Attach base_name metadata for tracking purposes
            img.base_name = base_name_prefix
            img.class_id_key = class_id
            base_images.append(img)
        except Exception as e:
            print(f"Error loading {filename}: {e}")
            
    print(f"Successfully loaded {len(base_images)} unique base images for augmentation.")
    return base_images

# --- Augmentation ---

# img_PIL = Image.open(new_path + "/base/0_0.png").convert("L")
# img = (np.array(img_PIL) > 128).astype(np.uint8)

# print(img)

# Shifting
#def shift_image(img, dx_range, dy_range, base_name = None):
#    h, w = img.shape
#    gen_photos = []
#    img = img * 255
#
#    for dx in range(dx_range[0], dx_range[1] + 1):
#        for dy in range(dy_range[0], dy_range[1] + 1):
#
#            # Create a NEW blank image every time
#            shifted = np.ones_like(img) * 255
#
#            # Compute safe bounds
#            src_x_start = max(0, -dx)
#            src_x_end   = min(w, w - dx)
#            dst_x_start = max(0, dx)
#            dst_x_end   = min(w, w + dx)
#
#            src_y_start = max(0, -dy)
#            src_y_end   = min(h, h - dy)
#            dst_y_start = max(0, dy)
#            dst_y_end   = min(h, h + dy)
#            
#            # Copy valid region from source to destination
#            shifted[dst_y_start:dst_y_end, dst_x_start:dst_x_end] = img[src_y_start:src_y_end, src_x_start:src_x_end]
#
#            gen_photos.append(shifted.copy())
#    return gen_photos

# shifting
def shift(img: Image.Image, dx_range: int, dy_range: int, base_name_prefix: str, destination: str):
    """
    img: PIL Image in mode 'L' (0=black, 255=white)
    """
    w, h = img.size
    results = []
    # base_name = img.filename.split('.')[0]

    for dx in range(dx_range[0], dx_range[1] + 1):
        for dy in range(dy_range[0], dy_range[1] + 1):

            # white background
            shifted = Image.new("L", (w, h), 255)

            # paste the original at an offset
            shifted.paste(img, (dx, dy))
            
            # save
            save(shifted, base_name_prefix, 's', f"{dx}_{dy}", destination)

            # store (image, basename, dx, dy) for tracking
            # results.append((shifted, base_name_prefix, dx, dy))
            results.append(shifted)

    return results

# rotation
def rotate(img: Image.Image, max_angle: int, base_name_prefix: str, destination: str):
    results = []
    # current_image = img.filename.split('.')[0]

    for angle in range(-max_angle, max_angle + 1):
        rot = img.rotate(
                angle,
                expand = False,
                resample=Image.NEAREST,
                fillcolor=255
            )

        rot = rot.point(lambda x: 255 if x > 128 else 0) # Is this more efficient?
        
        save(rot, base_name_prefix, "r", angle, destination)
        
        # I think serves no purpose. This is an artifact of old code. Should look into later...
        arr = np.array(rot)

        # store for tracking
        results.append(arr)

    return results

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

# Scaling
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

def apply_morphology(img: Image.Image, kernel_size: int, type: str) -> Image.Image:
    """Applies a specific morphology operation (erode or dilate)."""
    if type == "erode":
        morphed = img.filter(ImageFilter.MinFilter(kernel_size))
    elif type == "dilate":
        morphed = img.filter(ImageFilter.MaxFilter(kernel_size))
    else:
        return img 
    return binarize(morphed)

def add_black_noise(img: Image.Image, amount=0.05):
    """
    Add black dots to a white-background arrow.
    img: PIL grayscale image (0=black, 255=white)
    amount: fraction of pixels to turn black
    """
    arr = np.array(img).copy()
    h, w = arr.shape
    n_pixels = int(h * w * amount)

    # choose random pixel coordinates
    ys = np.random.randint(0, h, n_pixels)
    xs = np.random.randint(0, w, n_pixels)

    arr[ys, xs] = 0  # set to black
    return Image.fromarray(arr, 'L')

def add_white_noise(img: Image.Image, amount=0.05):
    """
    Randomly turn black pixels to white to simulate erasing.
    img: PIL grayscale image (0=black, 255=white)
    amount: fraction of black pixels to turn white
    """
    arr = np.array(img).copy()
    
    # find all black pixels
    black_coords = np.argwhere(arr == 0)
    
    n_pixels = int(len(black_coords) * amount)
    if n_pixels == 0:
        return Image.fromarray(arr)
    
    # randomly choose pixels to erase
    erase_indices = np.random.choice(len(black_coords), n_pixels, replace=False)
    for idx in erase_indices:
        y, x = black_coords[idx]
        arr[y, x] = 255  # turn black pixel to white
    
    return Image.fromarray(arr, 'L')

# --- augmentation process ---

def augment_pipeline(base_images):
    total_images_generated = 0

    for img in base_images:
        
        shape_key = img.base_name.split('_')[0]
        config = limits.get(shape_key, limits.get("default", {}))

        print("here is limits: ", limits)
        print("here is config: ", config)
        print("here is the shape_key: ", shape_key)
        print("here is img.base_name", img.base_name)

        max_rot = config.get('max_rotation')
        shift_x = config.get('max_shift_x')
        shift_y = config.get('max_shift_y')
        
        print("here is max_rot: ", max_rot)
        print(f"Processing base image {img.base_name} with configuration: {config}")

        # 1. ROTATION on base images
        print(" -> Rotating base image...")
        rotations = rotate(img, max_rot, img.base_name, "rotate")
        total_images_generated += len(rotations)

        # 2. SHIFTING
        print(" -> Shifting base image...")
        shifts = shift(img, shift_x, shift_y, img.base_name, "shift")
        total_images_generated += len(shifts)

        print("Total_images_generated is: ", total_images_generated)


def augment_pipeline2(base_images):
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

# --- Augmentation Group ----

# Called by augment_pipeline to create a transformation map. Essential for chaining.
def create_transformation_map(config: dict):
    """
    Defines the parameters and functions for each augmentation group 
    based on the loaded limits config.
    """
    max_rot = config.get('max_rotation', 5)
    shift_x_range = config.get('max_shift_x', [-2, 2])
    shift_y_range = config.get('max_shift_y', [-2, 2])
    morph_k = config.get('morph_kernel_size', 2)
    scale_f = config.get('scale_factors', [0.9, 1.1])
    max_shear_x = config.get('max_shear_x', 0.2)
    max_shear_y = config.get('max_shear_y', 0.2)
    
    # R: Rotation (Range from -max to +max, step 2)
    rotations = [lambda img, a=a: apply_rotate(img, a) for a in range(-max_rot, max_rot + 1, 2) if a != 0]

    # S: Shift - Randomly choose 5 combinations of shifts
    shifts = []
    for _ in range(5):
        dx = random.randint(shift_x_range[0], shift_x_range[1])
        dy = random.randint(shift_y_range[0], shift_y_range[1])
        shifts.append(lambda img, dx=dx, dy=dy: apply_shift(img, dx, dy))
    
    # Z: Scale/Shear (Zoomies) - 2 scales + 2 shears (one for X, one for Y)
    scale_shear = []
    scale_shear.append(lambda img, f=scale_f[0]: apply_scale(img, f))
    scale_shear.append(lambda img, f=scale_f[1]: apply_scale(img, f))
    scale_shear.append(lambda img, sx=max_shear_x, sy=0: apply_shear(img, sx, sy))
    scale_shear.append(lambda img, sx=0, sy=max_shear_y: apply_shear(img, sx, sy))

    # M: Morphology (Erode/Dilate)
    morphology_ops = [
        lambda img, k=morph_k: apply_morphology(img, k, "erode"),
        lambda img, k=morph_k: apply_morphology(img, k, "dilate"),
    ]
    
    # Map transformation groups to labels
    transformation_map = {
        'R': rotations,
        'S': shifts,
        'Z': scale_shear, 
        'M': morphology_ops
    }
    
    return transformation_map


# --- single effect generator ---

def generate_single_augmentations(base_img: Image.Image, config: dict, num_images: int) -> int:
    """
    Generates single-effect augmentations (Rotation only, Shift only, etc.)
    by randomly sampling one function from each group.
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
            # Pick one function randomly from the group (e.g., rotate 3 degrees or rotate -5 degrees)
            transform_func = random.choice(group_functions)
            current_img = transform_func(base_img)
            
            # Save the result: destination_dir is now 'single/R'
            aug_tag = f"{i}"
            save_image(current_img, base_name, key, aug_tag, f"single/{key}")
            total_generated += 1

    return total_generated

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

# --- noise generator ---

# Single
def generate_noise_augmentations(base_img: Image.Image, config: dict, num_images: int) -> int:
    """
    Generates a set of noisy images using a bell curve for noise percentage, 
    favoring lower noise amounts.
    """
    total_generated = 0
    base_name = base_img.base_name
    
    # Define the range and bell curve parameters
    min_noise = 0.05
    max_noise = 0.20
    # Mean of the bell curve, chosen to bias toward the low end (0.05)
    # The key word here is bias, the actual mean is 0.125, but I set it to
    # 0.10 to bias the images towards lower noise percentages
    mean = 0.10 
    # Standard deviation to control spread. 0.05 ensures most are between 0.05 and 0.15
    # Look up 68-95-99.7 rule. Basically: mean +- 1 * std_dev = 68%, then solve for std_dev
    std_dev = 0.05 
    
    # Generate noise amounts using a normal distribution
    noise_amounts = npr.normal(loc=mean, scale=std_dev, size=num_images)
    
    # Clamp values to the desired range [0.05, 0.20]
    clamped_amounts = np.clip(noise_amounts, min_noise, max_noise)
    
    print(f"    - Noise percentages generated (min/max): {np.min(clamped_amounts):.3f} / {np.max(clamped_amounts):.3f}")

    for i, amount in enumerate(clamped_amounts):
        # 1. Apply Black Noise (Pepper)
        img_salt = add_pepper_noise(base_img, amount)
        save_image(img_salt, base_name, "nP", f"{amount:.3f}".replace('.', ''), "noise_pepper")
        total_generated += 1

        # 2. Apply White Noise (Salt - simulating erasure)
        img_pepper = add_salt_noise(base_img, amount)
        save_image(img_pepper, base_name, "nS", f"{amount:.3f}".replace('.', ''), "noise_salt")
        total_generated += 1
        
    return total_generated

# chained
def generate_noise_on_chained(chained_images: list[Image.Image], config: dict, num_noise_variants: int) -> int:
    """
    Generates Salt and Pepper noise variations for a list of already transformed images.
    Uses the chained image's parameter tag in the output filename for full traceability.
    """
    total_generated = 0
    # base_name = chained_images.base_name
    
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

# photos = shift_image(img, limits[arrow]["max_shift_x"], limits[arrow]["max_shift_y"])
# photos = rotate(img_PIL, limits[arrow]["max_rotation"])

# print(photos)
def save_image(photo: Image.Image, base_name: str, aug_type: str, number: str, destination: str):
    sub_dir = os.path.join(OUTPUT_DIR, destination)
    os.makedirs(sub_dir, exist_ok=True)

    filename = f"{base_name}_{aug_type}_{number}.png"
    path = os.path.join(sub_dir, filename)

    try:
        photo.save(path)
    except Exception as e:
        print(f"Error saving {filename}: {e}")


if __name__ == "__main__":
    # 1. Load the base images
    base_images = load_all_base_images(INPUT_DIR)
    
    if base_images:
        # 2. Run the pipeline
        augment_pipeline2(base_images)
    else:
        print("No base images loaded. Please check your 'base' directory.")
