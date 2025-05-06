import math
import os
import argparse

import datasets
import numpy as np
import torch
from transformers import CLIPTextModel, CLIPTokenizer
from peft import LoraConfig
from peft.utils import get_peft_model_state_dict
from torchvision import transforms
from diffusers import AutoencoderKL, DDPMScheduler, StableDiffusionPipeline, UNet2DConditionModel
from diffusers.optimization import get_scheduler
from diffusers.utils import convert_state_dict_to_diffusers

def parse_args():
    parser = argparse.ArgumentParser(description='Train a LoRA adapter for Stable Diffusion')
    
    # Output parameters
    parser.add_argument('--weights_name', type=str, default="pytorch_lora_weights.safetensors",
                        help='Filename for the output weights (default: pytorch_lora_weights.safetensors)')
    
    # Training parameters
    parser.add_argument('--lora_rank', type=int, default=4,
                        help='Rank of the LoRA adapter (default: 4)')
    parser.add_argument('--lr', type=float, default=5e-4,
                        help='Learning rate (default: 5e-4)')
    parser.add_argument('--betas', type=float, nargs=2, default=[0.9, 0.999],
                        help='Adam optimizer betas (default: 0.9 0.999)')
    parser.add_argument('--decay', type=float, default=1e-2,
                        help='Weight decay (default: 1e-2)')
    parser.add_argument('--eps', type=float, default=1e-8,
                        help='Adam optimizer epsilon (default: 1e-8)')
    parser.add_argument('--resolution', type=int, default=512,
                        help='Image resolution for training (default: 512)')
    parser.add_argument('--max_train_steps', type=int, default=2000,
                        help='Maximum number of training steps (default: 2000)')
    parser.add_argument('--batch_size', type=int, default=1,
                        help='Batch size for training (default: 1)')
    parser.add_argument('--log_interval', type=int, default=50,
                        help='Interval for logging training progress (default: 50)')
    
    args = parser.parse_args()
    return args

def main(args):
    # Output parameters
    weights_name = args.weights_name
    
    # Fixed parameters
    model = "runwayml/stable-diffusion-v1-5"
    dataset_name = "TristanJLegg/MinecraftGameplayCaptioned"
    weights_dir = "weights"
    
    # Training parameters
    lora_rank = args.lora_rank
    lr = args.lr
    betas = tuple(args.betas)
    decay = args.decay
    eps = args.eps
    resolution = args.resolution
    max_train_steps = args.max_train_steps
    batch_size = args.batch_size
    log_interval = args.log_interval

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {torch.cuda.get_device_name() if torch.cuda.is_available() else 'CPU'}")

    noise_scheduler = DDPMScheduler.from_pretrained(model, subfolder="scheduler")
    tokenizer = CLIPTokenizer.from_pretrained(model, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(model, subfolder="text_encoder")
    vae = AutoencoderKL.from_pretrained(model, subfolder="vae")
    unet = UNet2DConditionModel.from_pretrained(model, subfolder="unet")

    for param in text_encoder.parameters():
        param.requires_grad_(False)
    unet.requires_grad_(False)
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)

    unet_lora_config = LoraConfig(
        r=lora_rank,
        lora_alpha=lora_rank,
        init_lora_weights="gaussian",
        target_modules=["to_k", "to_q", "to_v", "to_out.0"]
    )

    # Send to device
    unet.to(device, dtype=torch.float32)
    vae.to(device, dtype=torch.float32)
    text_encoder.to(device, dtype=torch.float32)

    # Add adapter
    unet.add_adapter(unet_lora_config)

    # Create optimizer
    lora_layers = filter(lambda p: p.requires_grad, unet.parameters())
    optimizer = torch.optim.AdamW(
        lora_layers,
        lr=lr,
        betas=betas,
        eps=eps,
        weight_decay=decay
    )

    # Load Dataset
    train_dataset = datasets.load_dataset(dataset_name, split="train")

    def tokenize_captions(examples):
        input_tokenizer = tokenizer(
            examples["text"],
            return_tensors="pt",
            padding="max_length",
            max_length=tokenizer.model_max_length,
            truncation=True
        )
        return input_tokenizer.input_ids

    # Preprocessing the datasets.
    train_transforms = transforms.Compose(
        [
            transforms.Resize(resolution, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(resolution),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ]
    )

    def preprocess_training_dataset(examples):
        images = [image.convert("RGB") for image in examples["image"]]
        examples["pixel_values"] = [train_transforms(image) for image in images]
        examples["input_ids"] = tokenize_captions(examples)
        return examples

    def collate(examples):
        pixels = torch.stack([example["pixel_values"] for example in examples])
        pixels = pixels.to(memory_format=torch.contiguous_format).float()
        input_ids = torch.stack([example["input_ids"] for example in examples])
        return {"pixel_values": pixels, "input_ids": input_ids}

    processed_train_dataset = train_dataset.with_transform(preprocess_training_dataset)

    train_dataloader = torch.utils.data.DataLoader(
        processed_train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        collate_fn=collate
    )

    lr_scheduler = get_scheduler(
        "constant",
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=max_train_steps
    )

    print("Training...")

    num_train_epochs = math.ceil(max_train_steps / len(train_dataloader))
    for epoch in range(num_train_epochs):
        unet.train()

        for step, batch in enumerate(train_dataloader):
            # Convert images to latent space
            latents = vae.encode(batch["pixel_values"].to(device=device, dtype=torch.float32)).latent_dist.sample()
            latents = latents * vae.config.scaling_factor

            # Sample noise to add to latents
            noise = torch.randn_like(latents)
            bsz = latents.shape[0]

            # Sample a random timestep for each images
            timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,), device=latents.device).long()

            # Add noise to the latents according to the noise magnitude at each timstep (forward diffusion process)
            noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

            # Get the text embedding for conditioning
            encoder_hidden_states = text_encoder(batch['input_ids'].to(device=device), return_dict=False)[0]

            # Calculate the loss
            target = noise
            model_prediction = unet(noisy_latents, timesteps, encoder_hidden_states, return_dict=False)[0]
            loss = torch.nn.functional.mse_loss(model_prediction.float(), target.float(), reduction="mean")

            # Backpropogate
            loss.backward()

            # clip grad norm
            params_to_clip = lora_layers
            torch.nn.utils.clip_grad_norm_(params_to_clip, 1.0)

            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()

            # Logging
            if epoch == 0 and step == 0:
                print(f"Step {step+1 + (epoch * len(train_dataloader))}, Loss: {loss.item()}")
            elif (step+1) % log_interval == 0:
                print(f"Step {step+1 + (epoch * len(train_dataloader))}, Loss: {loss.item()}")

        # Create weights directory if it doesn't exist
        if not os.path.exists(weights_dir):
            os.makedirs(weights_dir)

        # Full path for the weights file
        weights_path = os.path.join(weights_dir, weights_name)
            
        print(f"Saving model to {weights_path}...")
            
        # Save the model
        unet = unet.to(torch.float32)
        unet_lora_state_dict = convert_state_dict_to_diffusers(get_peft_model_state_dict(unet))
        StableDiffusionPipeline.save_lora_weights(
            save_directory=weights_dir,
            unet_lora_layers=unet_lora_state_dict,
            safe_serialization=True,
            weight_name=weights_name
        )
        
        print(f"Model saved successfully to {weights_path}")

if __name__ == "__main__":
    main(parse_args())