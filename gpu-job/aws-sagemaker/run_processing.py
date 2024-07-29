from sagemaker.processing import Processor
import os


env = {
    "WANDB_PROJECT": os.getenv("WANDB_PROJECT"),
    "WANDB_API_KEY": os.getenv("WANDB_API_KEY"),
    "HF_TOKEN": os.getenv("HF_TOKEN"),
    "RUN_NAME": os.getenv("RUN_NAME"),
}

sagemaker_role_arn = f"arn:aws:iam::{os.getenv('AWS_ACCOUNT_ID')}:role/sagemaker-execution-role"
iamge_uri = f"{os.getenv('AWS_ACCOUNT_ID')}.dkr.ecr.{os.getenv('AWS_DEFAULT_REGION')}.amazonaws.com/gpu-jobs-comparison:latest"

processor = Processor(
    role=sagemaker_role_arn,
    image_uri=iamge_uri,
    instance_count=1,
    instance_type='ml.g5.2xlarge',
    env=env
)

# Define processing inputs and outputs (if any)
processing_inputs = []
processing_outputs = []

processor.run(
    inputs=processing_inputs,
    outputs=processing_outputs,
    arguments=['python', 'text2sql_training.py']
)
