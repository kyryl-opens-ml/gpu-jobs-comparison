import os

import modal
from modal import Image

app = modal.App("gpu-jobs")
env = {
    "WANDB_PROJECT": os.getenv("WANDB_PROJECT"),
    "WANDB_API_KEY": os.getenv("WANDB_API_KEY"),
    "HF_TOKEN": os.getenv("HF_TOKEN"),
    "RUN_NAME": os.getenv("RUN_NAME"),
}
custom_image = Image.from_registry("ghcr.io/kyryl-opens-ml/gpu-jobs-comparison:latest").env(env)


@app.function(image=custom_image, gpu="a10g", timeout=10 * 60 * 60)
def run_training():
    from text2sql_training import main
    main()


def run_from_python():
    fn = modal.Function.lookup("gpu-jobs", "run_training")
    fn_id = fn.spawn()
    print(f"Run training object: {fn_id}")


if __name__ == "__main__":
    run_from_python()
