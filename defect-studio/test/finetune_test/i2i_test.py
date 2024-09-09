from diffusers import StableDiffusionImg2ImgPipeline
import torch
from dotenv import load_dotenv
from datetime import datetime
import os
from PIL import Image

load_dotenv()

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="1"

current_time = datetime.now().strftime("%Y%m%d_%H%M%S")

model_id = "/home/j-j11s001/project/bumsoo/dataset/dreambooth_output/20240908013252_cat3_dreambooth"
pipe = StableDiffusionImg2ImgPipeline.from_pretrained(model_id, torch_dtype=torch.float16).to("cuda")

seed = torch.seed()
#seed = 4111916325600
print(seed)

generator = torch.Generator("cuda").manual_seed(seed)
prompt = "A photo of sks cat on a man's hand"

image_path = "./temp/image.jpg"
init_image = Image.open(image_path).convert("RGB")
init_image = init_image.resize((512, 512))

# image = pipe(prompt, num_inference_steps=50, guidance_scale=7.5).images[0]
images = pipe(
    prompt=prompt,
    image=init_image,
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
    