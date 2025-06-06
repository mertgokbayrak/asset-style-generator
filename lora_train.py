import argparse
import os
import torch
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from dotenv import load_dotenv

from diffusers import StableDiffusionPipeline, UNet2DConditionModel, DDPMScheduler, AutoencoderKL
from transformers import CLIPTextModel, CLIPTokenizer
from accelerate import Accelerator
from accelerate.utils import set_seed
from peft import LoraConfig
from transformers import get_scheduler

# Load environment variables from a .env file
load_dotenv()

# special token for the style and the base model
SPECIAL_TOKEN = "<GameStyle>"
STABLE_DIFFUSION_MODEL = "runwayml/stable-diffusion-v1-5"



class StyleDataset(Dataset):
    def __init__(self, folder, tokenizer, resolution=512, caption_token=SPECIAL_TOKEN):
        self.tokenizer = tokenizer
        self.image_paths = [os.path.join(folder, f) for f in os.listdir(folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        self.transform = transforms.Compose([
            transforms.Resize(resolution, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(resolution),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.1, contrast=0.1),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ])
        # The caption includes our special token, now passed as an argument.
        self.caption = f"a photo in the style of {caption_token}"

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert("RGB")
        pixel_values = self.transform(image)
        
        # Tokenizing the caption that includes our special token <GameStyle>.
        input_ids = self.tokenizer(
            self.caption,
            padding="max_length",
            truncation=True,
            max_length=self.tokenizer.model_max_length, 
            return_tensors="pt",
        ).input_ids

        return {"pixel_values": pixel_values, "input_ids": input_ids.squeeze(0)}

def generate_preview(unet, text_encoder, tokenizer, output_dir, base_model, step, device, prompt):
    """Generates and saves a preview image for a given prompt."""
    preview_dir = os.path.join(output_dir, "previews")
    os.makedirs(preview_dir, exist_ok=True)
    
    # On MPS (Metal Performance Shaders by Apple), float32 seems safer for pipelines.
    dtype = torch.float16 if device.type == 'cuda' else torch.float32
    
    pipeline = StableDiffusionPipeline.from_pretrained(
        base_model,
        unet=unet,
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        torch_dtype=dtype,
        safety_checker=None, 
    ).to(device)
    
    with torch.no_grad():
        image = pipeline(prompt, num_inference_steps=25, guidance_scale=7.5).images[0]
    
    # Making the filename safe for the filesystem
    safe_prompt = "".join(c for c in prompt if c.isalnum() or c in (' ', '_')).rstrip()
    safe_prompt = safe_prompt.replace(" ", "_")
    
    image.save(os.path.join(preview_dir, f"preview_step_{step}_{safe_prompt[:40]}.png"))
    print(f"Saved preview image for prompt: '{prompt}'")
    
    # Cleaning up to free memory
    del pipeline
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    elif device.type == 'mps':
        torch.mps.empty_cache()

def setup_training_components(base_model_id):
    """Loads all necessary models and tokenizer from the base model."""
    tokenizer = CLIPTokenizer.from_pretrained(base_model_id, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(base_model_id, subfolder="text_encoder")
    vae = AutoencoderKL.from_pretrained(base_model_id, subfolder="vae")
    unet = UNet2DConditionModel.from_pretrained(base_model_id, subfolder="unet")
    noise_scheduler = DDPMScheduler.from_pretrained(base_model_id, subfolder="scheduler")
    return tokenizer, text_encoder, vae, unet, noise_scheduler

def add_and_initialize_new_token(tokenizer, text_encoder, token_str):
    """Adds a new token to the tokenizer and initializes its embedding."""
    if token_str not in tokenizer.get_vocab():
        tokenizer.add_tokens([token_str])
        text_encoder.resize_token_embeddings(len(tokenizer))
    
    token_id = tokenizer.convert_tokens_to_ids(token_str)
    
    # Initializing the new token embedding with random noise
    embedding_layer = text_encoder.get_input_embeddings()
    embedding_layer.weight.data[token_id] = torch.randn(
        1, embedding_layer.weight.size(1)
    )

def setup_lora(unet, rank, alpha):
    """Configures and adds LoRA adapter to the UNet."""
    lora_config = LoraConfig(
        r=rank,
        lora_alpha=alpha,
        target_modules=["to_q", "to_v"],
        lora_dropout=0.1,
        bias="none",
    )
    unet.add_adapter(lora_config)
    return lora_config

def save_lora_model(accelerator, unet, lora_config, output_dir):
    """Saves the LoRA weights."""
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        print("Training finished. Saving LoRA weights...")
        unwrapped_unet = accelerator.unwrap_model(unet)
        
        save_path = os.path.join(output_dir, "unet_lora")
        unwrapped_unet.save_pretrained(save_path)
        lora_config.save_pretrained(save_path)

        print(f"LoRA weights saved to: {save_path}")
        
def train(args):
    # Initializing the Accelerator, logging with TensorBoard, and setting the project directory.
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with="tensorboard",
        project_dir=os.path.join(args.output_dir, "logs")
    )

    # MPS doesn't support fp16, so we disable it here.
    if 'mps' in str(accelerator.device):
        args.mixed_precision = 'no'
        
    set_seed(args.seed)

    if accelerator.is_main_process:
        os.makedirs(args.output_dir, exist_ok=True)

    # Getting the device from the accelerator for consistent placement
    device = accelerator.device

    # Loading the models
    tokenizer, text_encoder, vae, unet, noise_scheduler = setup_training_components(args.base_model)

    # It's crucial to freeze the models BEFORE modifying them.
    # This ensures that only the newly added LoRA layers and token embeddings are trainable.
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    unet.requires_grad_(False)

    # Adding a new token for our game style. The new embedding will be trainable by default.
    add_and_initialize_new_token(tokenizer, text_encoder, SPECIAL_TOKEN)

    # Adding LoRA adapters to UNet. The new adapter layers will be trainable.
    lora_config = setup_lora(unet, args.lora_rank, args.lora_alpha)
    
    # We want to train the LoRA parameters and the new token embedding
    params_to_train = [
        *filter(lambda p: p.requires_grad, unet.parameters()),
        *filter(lambda p: p.requires_grad, text_encoder.parameters()),
    ]
    
    if not params_to_train:
        raise ValueError("No trainable parameters found. Check model freezing and LoRA/token setup.")

    optimizer = torch.optim.AdamW(params_to_train, lr=args.learning_rate)

    # Adding a learning rate scheduler to adjust the learning rate during training
    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * args.gradient_accumulation_steps,
        num_training_steps=args.max_train_steps * args.gradient_accumulation_steps,
    )

    # Building the dataset and dataloader
    dataset = StyleDataset(args.train_data_dir, tokenizer, args.resolution, caption_token=SPECIAL_TOKEN)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)

    # Preparing everything with Accelerator
    unet, optimizer, dataloader, lr_scheduler = accelerator.prepare(unet, optimizer, dataloader, lr_scheduler)
    
    # Explicitly moving models to the accelerator's device
    text_encoder_dtype = torch.float16 if args.mixed_precision == 'fp16' and device.type == 'cuda' else torch.float32
    text_encoder.to(device, dtype=text_encoder_dtype)
    vae.to(device)

    # Training loop
    global_step = 0
    for epoch in range(args.num_epochs):
        unet.train()
        for step, batch in enumerate(dataloader):
            with accelerator.accumulate(unet):
                # Getting the text embeddings
                with torch.no_grad():
                    encoder_hidden_states = text_encoder(batch["input_ids"])[0]
                
                # Preparing the noisy latents
                pixel_values = batch["pixel_values"].to(accelerator.device)
                
                # Converting images to latent space
                with torch.no_grad():
                    latents = vae.encode(pixel_values).latent_dist.sample()
                    latents = latents * vae.config.scaling_factor

                # Adding noise to the latents
                noise = torch.randn_like(latents)
                timesteps = torch.randint(
                    0, noise_scheduler.config.num_train_timesteps, (latents.shape[0],), device=latents.device
                ).long()
                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)
                
                # Predicting the noise residual
                noise_pred = unet(noisy_latents, timesteps, encoder_hidden_states).sample
                loss = torch.nn.functional.mse_loss(noise_pred.float(), noise.float(), reduction="mean")
                
                # Backpropagating
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    # Adding gradient clipping
                    accelerator.clip_grad_norm_(params_to_train, 1.0)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
            
            global_step += 1
            if accelerator.is_main_process:
                print(f"[Epoch {epoch+1}/{args.num_epochs}, Step {global_step}] Loss: {loss.item():.4f}")

            if global_step % args.preview_steps == 0:
                if accelerator.is_main_process:
                    print("Generating previews...")
                    unwrapped_unet = accelerator.unwrap_model(unet)
                    
                    # Generating previews for multiple prompts
                    preview_prompts = [
                        f"a sheep in the style of {SPECIAL_TOKEN}",
                        f"a cute robot in the style of {SPECIAL_TOKEN}",
                        f"a bowl of soup, in the style of {SPECIAL_TOKEN}",
                        f"a photo of a wooden chair in the style of {SPECIAL_TOKEN}"
                    ]
                    
                    for prompt in preview_prompts:
                        generate_preview(unwrapped_unet, text_encoder, tokenizer, args.output_dir, args.base_model, global_step, device, prompt)

            if global_step >= args.max_train_steps:
                break
        if global_step >= args.max_train_steps:
            break

    # Saving the LoRA weights
    save_lora_model(accelerator, unet, lora_config, args.output_dir)


# Main function to run the training process with command line arguments
# This allows us to run the script with different parameters from the command line.
# 10 epochs, 1 batch size, 4 gradient accumulation steps, 16 LoRA rank, 32 LoRA alpha, 100 learning rate warmup steps, and 2540 max train steps.
# 2540 steps = 10 epochs * 254 steps per epoch
def main():
    parser = argparse.ArgumentParser(description="Fine-tune a Stable Diffusion model with LoRA.")
    
    # Path and model configurations
    path_group = parser.add_argument_group("Paths and models")
    path_group.add_argument("--base_model", type=str, default=STABLE_DIFFUSION_MODEL, help="Base model ID.")
    path_group.add_argument("--train_data_dir", type=str, default="./assets/", help="Directory with training images.")
    path_group.add_argument("--output_dir", type=str, default="./lora_output/game_style/unet_lora/", help="Directory to save LoRA weights.")

    # Training parameter configurations
    training_group = parser.add_argument_group("Training parameters")
    training_group.add_argument("--resolution", type=int, default=512, help="Resolution of the training images.")
    training_group.add_argument("--batch_size", type=int, default=1, help="Batch size (per device).")
    training_group.add_argument("--gradient_accumulation_steps", type=int, default=4, help="Number of steps to accumulate gradients.")
    training_group.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate for the optimizer.")
    training_group.add_argument("--num_epochs", type=int, default=10, help="Number of epochs to train for.")
    training_group.add_argument("--max_train_steps", type=int, default=2540, help="Overrides num_epochs.")
    training_group.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    training_group.add_argument("--mixed_precision", type=str, default="no", choices=["no", "fp16", "bf16"], help="Mixed precision training. 'no' for MPS.")
    
    # Learning rate scheduler configurations
    lr_group = parser.add_argument_group("Learning rate scheduler")
    lr_group.add_argument("--lr_scheduler", type=str, default="cosine", help="Learning rate scheduler type.")
    lr_group.add_argument("--lr_warmup_steps", type=int, default=100, help="Number of warmup steps for the learning rate scheduler.")
    
    # LoRA configurations
    lora_group = parser.add_argument_group("LoRA parameters")
    lora_group.add_argument("--lora_rank", type=int, default=16, help="Rank of LoRA matrices.")
    lora_group.add_argument("--lora_alpha", type=int, default=32, help="Alpha for LoRA scaling.")
    
    # Preview generation configurations
    preview_group = parser.add_argument_group("Preview generation")
    preview_group.add_argument("--preview_steps", type=int, default=254, help="Generate preview every N steps.")
    
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()