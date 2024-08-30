from PIL import Image
import subprocess
import os

# Set base path
base_path = "C:/uploads/lama"
input_image_path = "C:/uploads/lama/ii.png"
input_mask_path = "C:/uploads/lama/ii_mask.jpg"

def generate_cleanup(init_image: Image, mask_image: Image):
    """
    Generate a cleaned-up image using IOPaint.
    Args:
        init_image (Image): The initial image to clean up.
        mask_image (Image): The mask image indicating areas to clean up.
    Returns:
        Image: The cleaned-up image or None if an error occurred.
    """
    try:
        # Save the images temporarily to disk
        temp_input_path = os.path.join(base_path, "temp_input.png")
        temp_mask_path = os.path.join(base_path, "temp_mask.png")
        output_path = os.path.join(base_path, "output")

        # Save input and mask images
        init_image.save(temp_input_path)
        mask_image.save(temp_mask_path)

        # Prepare the IOPaint command
        cmd = [
            "iopaint", "run",
            "--model=lama",
            "--device=cuda",
            f"--image={temp_input_path}",
            f"--mask={temp_mask_path}",
            f"--output={output_path}"
        ]

        # Start subprocess using cmd and wait for it to finish
        result = subprocess.run(cmd, shell=True)

        if result.returncode == 0:
            print("Cleanup process completed successfully.")
            # Load the output image
            final_image_path = os.path.join(output_path, "temp_input.png")  # Adjust the output filename as needed
            final_image = Image.open(final_image_path)
            return final_image
        else:
            print(f"Cleanup process failed with return code: {result.returncode}")
            return None

    except Exception as e:
        print(f"Error during cleanup generation: {e}")
        return None
    finally:
        # Clean up temporary files
        if os.path.exists(temp_input_path):
            os.remove(temp_input_path)
        if os.path.exists(temp_mask_path):
            os.remove(temp_mask_path)

if __name__ == "__main__":
    # Example usage
    print("Starting batch processing...")

    # Load input image and mask
    init_image = Image.open(input_image_path).convert("RGB")
    mask_image = Image.open(input_mask_path).convert("RGB")

    # Perform cleanup
    cleaned_image = generate_cleanup(init_image, mask_image)

    if cleaned_image:
        cleaned_image.save(os.path.join(base_path, "cleaned_output.png"))
        print("Cleaned image saved successfully.")
    else:
        print("Failed to generate cleaned image.")