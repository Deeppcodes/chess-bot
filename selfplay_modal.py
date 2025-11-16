"""
Modal deployment script for self-play data generation.
"""

import modal

# Persistent volume for datasets
volume = modal.Volume.from_name("selfplay-vol", create_if_missing=True)

# FIXED: Pin numpy BEFORE installing torch (Torch 2.2.2 requires numpy < 2.0)
# DO NOT use requirements.txt - it has numpy>=2.0.0 which breaks torch!
image = (
    modal.Image.debian_slim(python_version="3.10")
    .pip_install("numpy==1.26.4")                  # ← Must be 1.x for torch 2.2.2
    .pip_install("torch==2.2.2", "python-chess")   # ← Now torch sees NumPy 1.26
    .add_local_dir(".", remote_path="/root")       # ← Keep your project files inside image
)

app = modal.App("selfplay-generator")


@app.function(
    image=image,
    gpu="A10G",
    timeout=3600,
    volumes={"/data": volume},
)
def run_selfplay(
    games: int = 200,
    mcts_sims: int = 50,
    max_length: int = 200,
    output_name: str = "selfplay_phase2.npz",
    model_path: str = "chess_model_best.pth",
    use_volume_model: bool = False,
):
    import os, sys

    os.chdir("/root")
    sys.path.insert(0, "/root")

    from selfplay_generator import generate_selfplay_dataset

    print("="*70)
    print(f"🚀 PHASE 2: SELF-PLAY GENERATION ON MODAL GPU")
    print(f"Games: {games}, MCTS sims: {mcts_sims}")
    print(f"Model: {model_path}")
    print(f"Output: {output_name}")
    print("="*70)

    output_path = f"/data/{output_name}"

    # Check if model is in volume or local directory
    if use_volume_model:
        full_model_path = f"/data/{model_path}"
        print(f"Loading model from volume: {full_model_path}")
    else:
        full_model_path = f"/root/{model_path}"
        print(f"Loading model from local: {full_model_path}")

    if not os.path.exists(full_model_path):
        print("❌ Model not found:", full_model_path)
        if use_volume_model:
            print("Available files in /data:", os.listdir("/data"))
        else:
            print("Available files in /root:", os.listdir("/root"))
        raise FileNotFoundError(full_model_path)

    generate_selfplay_dataset(
        num_games=games,
        mcts_sims=mcts_sims,
        max_length=max_length,
        output_path=output_path,
        model_path=full_model_path,
        device="cuda",
    )

    volume.commit()

    print(f"✅ Saved to Modal volume: {output_path}")
    return output_path


@app.function(image=image, volumes={"/data": volume})
def list_datasets():
    import os
    files = [
        f for f in os.listdir("/data")
        if f.endswith(".npz")
    ]
    print("\n📦 Datasets in volume:")
    for f in files:
        print("  -", f)
    return files


@app.function(image=image, volumes={"/data": volume})
def download_dataset(filename: str):
    import os
    path = "/data/" + filename
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    return open(path, "rb").read()


@app.local_entrypoint()
def main(
    games: int = 200,
    mcts_sims: int = 50,
    max_length: int = 200,
    output_name: str = "selfplay_phase2.npz",
    model_path: str = "chess_model_best.pth",
    use_volume_model: bool = False,
    download: bool = False,
    list_files: bool = False,
):
    if list_files:
        return list_datasets.remote()

    if download:
        data = download_dataset.remote(output_name)
        with open(output_name, "wb") as f:
            f.write(data)
        print("Downloaded:", output_name)
        return

    print("▶ Running self-play on Modal GPU…")
    path = run_selfplay.remote(
        games=games,
        mcts_sims=mcts_sims,
        max_length=max_length,
        output_name=output_name,
        model_path=model_path,
        use_volume_model=use_volume_model,
    )
    print("Dataset saved:", path)
