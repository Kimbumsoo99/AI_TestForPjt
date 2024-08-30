import subprocess

# Set base path
base_path = "C:/uploads/lama"
input_image_path = "C:/uploads/lama/ii.png"
input_mask_path = "C:/uploads/lama/ii_mask.jpg"

if __name__ == "__main__":
    print("Starting batch processing...")

    cmd = [
        "iopaint", "run",
        "--model=lama",
        "--device=cuda",
        f"--image={input_image_path}",
        f"--mask={input_mask_path}",
        f"--output={base_path}"
    ]

    # start subprocess using cmd and wait for it to finish
    result = subprocess.run(cmd, shell=True)

    if result.returncode == 0:
        print("Batch processing completed successfully.")
    else:
        print(f"Batch processing failed with return code: {result.returncode}")
