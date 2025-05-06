import os
import argparse

import torch
from diffusers import DiffusionPipeline

def parse_args():
    parser = argparse.ArgumentParser(description='Generate images using Stable Diffusion with LoRA weights')
    parser.add_argument(
        'prompt',
        type=str,
        help='Text prompt for image generation'
    )
    parser.add_argument(
        '--lora_weights',
        type=str,
        default="TristanJLegg/MinecraftStyleStableDiffusion",
        help='Path to safetensors file or HuggingFace repo ID for LoRA weights (default: TristanJLegg/MinecraftStyleStableDiffusion)'
    )
    parser.add_argument(
        '--output_file',
        type=str,
        default="output.png",
        help='Output filename for the generated image (default: output.png)'
    )
    parser.add_argument(
        '--seed', 
        type=int, 
        default=None, 
        help='Random seed for reproducible image generation (default: None)'
    )

    args = parser.parse_args()
    return args

def main(args):    
    prompt = args.prompt
    lora_weights = args.lora_weights
    
    print(f"Generating image with prompt: {prompt}")
    print(f"Using LoRA weights from: {lora_weights}")

    if args.seed is not None:
        print(f"Using seed: {args.seed}")
        generator = torch.Generator().manual_seed(args.seed)
    else:
        generator = None
    
    pipe = DiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5").to("cuda")

    if os.path.isfile(lora_weights):
        pipe.load_lora_weights(".", weight_name=lora_weights)
    else:
        pipe.load_lora_weights(lora_weights)

    img = pipe(prompt, generator=generator).images[0]

    # Check and create output directory
    if not os.path.exists("outputs"):
        os.makedirs("outputs")
    
    # Create a filename based on the weights being used
    output_file = f"outputs/{args.output_file}"
    img.save(output_file)
    print(f"Image saved to {output_file}")

if __name__ == "__main__":
    main(parse_args())