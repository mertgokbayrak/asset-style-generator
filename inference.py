import argparse
import torch
from diffusers import StableDiffusionPipeline
from safetensors.torch import load_file
import os
import re
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Constant for the special style token and the base model
SPECIAL_TOKEN = "<GameStyle>"
STABLE_DIFFUSION_MODEL = "runwayml/stable-diffusion-v1-5"

def setup_pipeline(base_model_id, lora_path, device, dtype):
    """
    Loads the base Stable Diffusion pipeline, applies the LoRA weights,
    and adds the special token for the fine-tuned style.
    This function should be called only once at the start of the script.
    """
    print("Loading base pipeline...")
    pipe = StableDiffusionPipeline.from_pretrained(
        base_model_id,
        torch_dtype=dtype,
    )
    pipe.to(device)

    # Manually loading the LoRA weights from the specified path
    lora_file_path = os.path.join(lora_path, "diffusion_pytorch_model.safetensors")
    if not os.path.exists(lora_file_path):
        raise FileNotFoundError(f"LoRA weights not found at {lora_file_path}")
        
    print(f"Loading LoRA weights from: {lora_file_path}")
    state_dict = load_file(lora_file_path, device=device)

    pipe.unet.load_attn_procs(state_dict)
    print("LoRA weights loaded and applied to U-Net.")

    # Adding the special token to the tokenizer if it doesn't exist
    if SPECIAL_TOKEN not in pipe.tokenizer.get_vocab():
        pipe.tokenizer.add_tokens([SPECIAL_TOKEN])
        pipe.text_encoder.resize_token_embeddings(len(pipe.tokenizer))
    
    return pipe

def generate_and_save_image(pipe, prompt, args):
    """
    Generates an image based on a user prompt and saves it.
    A safe filename is created from the prompt.
    """
    # Crafting the full prompt with the style token placeholder
    final_prompt = prompt.format(style_token=SPECIAL_TOKEN)
    print(f"\nGenerating image with prompt: '{final_prompt}'")
    
    # Use a specific seed for reproducibility if one is provided.
    # Otherwise, the generator will be initialized with a random seed.
    generator = torch.Generator(device=pipe.device)
    if args.seed is not None:
        generator.manual_seed(args.seed)

    with torch.no_grad():
        image = pipe(
            final_prompt, 
            num_inference_steps=args.steps, 
            guidance_scale=args.guidance,
            generator=generator
        ).images[0]

    # Generating a safe filename from the user's original prompt (without the token)
    safe_prompt_text = prompt.replace("{style_token}", "").strip()
    safe_filename = re.sub(r'[^\w\s-]', '', safe_prompt_text.lower()).strip().replace(' ', '_')[:60]
    output_path = os.path.join(args.output_dir, f"{safe_filename}.png")
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    image.save(output_path)
    print(f"Image saved to {output_path}")

def main():
    """
    Main function to handle argument parsing and run the interactive generation loop.
    """
    parser = argparse.ArgumentParser(description="Generate images interactively using a fine-tuned LoRA model.")
    
    # Arguments now pull their defaults from the environment variables
    parser.add_argument("--base_model", type=str, default=STABLE_DIFFUSION_MODEL, help="Base Stable Diffusion model ID.")
    parser.add_argument("--lora_path", type=str, default="./lora_output/game_style/unet_lora/", help="Path to the directory with LoRA weights.")
    parser.add_argument("--output_dir", type=str, default="./generated_images/", help="Directory to save generated images.")
    parser.add_argument("--steps", type=int, default=30, help="Number of inference steps.")
    parser.add_argument("--guidance", type=float, default=7.5, help="Guidance scale.")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility. Omit for random seed.")
    
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32

    # Set up the pipeline once to avoid reloading the model repeatedly.
    try:
        pipe = setup_pipeline(args.base_model, args.lora_path, device, dtype)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please check the --lora_path argument to make sure it points to the correct directory.")
        return

    # Start the interactive loop for prompting.
    print("\n--- Interactive Image Generation ---")
    print(f"Model and LoRA weights loaded successfully.")
    print("Type a prompt and press Enter to generate an image.")
    print("IMPORTANT: Include '{style_token}' in your prompt to apply the custom style.")
    print("IMPORTANT: The style token in our case is <GameStyle>")
    print("Example: 'a beautiful castle in the mountains, in the style of {style_token}'")
    print("Type 'quit' to end the session.")
    
    while True:
        try:
            user_prompt = input("\nEnter your prompt > ")
            if user_prompt.lower() in ['quit']:
                break
            
            if not user_prompt.strip():
                continue
            
            generate_and_save_image(pipe, user_prompt, args)

        except (EOFError, KeyboardInterrupt):
            # Handle Ctrl+D or Ctrl+C to exit gracefully.
            break

    print("\nSession ended. Goodbye!")


if __name__ == "__main__":
    main() 