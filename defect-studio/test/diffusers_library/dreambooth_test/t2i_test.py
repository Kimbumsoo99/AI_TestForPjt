from diffusers import StableDiffusionPipeline
import torch
from dotenv import load_dotenv
from datetime import datetime
import os

load_dotenv()

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="1"

current_time = datetime.now().strftime("%Y%m%d_%H%M%S")

model_id = "/home/j-j11s001/project/bumsoo/dataset/dreambooth_output/20240906225504_dreambooth"
pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16).to("cuda")

seed = torch.seed()
#seed = 4111916325600
print(seed)

generator = torch.Generator("cuda").manual_seed(seed)
prompt = "A photo of sks cat in a bucket"

# image = pipe(prompt, num_inference_steps=50, guidance_scale=7.5).images[0]
images = pipe(
    prompt=prompt,
    # negative_prompt="ugly, deformed, disfigured, poor details, bad anatomy",
    generator=generator,
    height=512, width=512,
    num_images_per_prompt=10,
    guidance_scale=7.5).images


model_dir = os.getenv('IMAGE_OUTPUT_DIR')

current_directory = os.path.dirname(os.path.abspath(__file__))
# output_folder = os.path.join(model_dir, "output")
output_folder = os.path.join(current_directory, "output")
os.makedirs(output_folder, exist_ok=True)

for i, img in enumerate(images):
    image_path = os.path.join(output_folder, f"generated_image_{seed}_{current_time}_{i+1}.png")
    img.save(image_path)
    print(f"Image {i+1} saved at {image_path}")