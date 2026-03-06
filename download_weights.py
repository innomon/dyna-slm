import os
from huggingface_hub import snapshot_download

def download_t5gemma():
    model_id = "google/t5gemma-2-270m-270m"
    local_dir = "models/t5gemma-2-270m"
    
    print(f"🚀 Starting download for {model_id}...")
    print(f"📂 Target directory: {os.path.abspath(local_dir)}")

    # Ensure the directory exists
    os.makedirs(local_dir, exist_ok=True)

    try:
        # We download safetensors, config, and tokenizer files
        path = snapshot_download(
            repo_id=model_id,
            local_dir=local_dir,
            local_dir_use_symlinks=False,
            allow_patterns=[
                "*.safetensors", 
                "*.json", 
                "*.txt",
                "tokenizer.model"
            ],
            # If you haven't run 'huggingface-cli login', 
            # you can pass your token here:
            # token="your_hf_token_here"
        )
        print(f"\n✅ Download complete!")
        print(f"📍 Files are located at: {path}")
        
    except Exception as e:
        print(f"\n❌ Error downloading model: {e}")
        print("\n💡 Tip: Ensure you have accepted the license on Hugging Face")
        print("💡 Tip: Run 'pip install huggingface_hub' and 'huggingface-cli login' first.")

if __name__ == "__main__":
    download_t5gemma()
