import boto3
from sagemaker.processing import Processor, ProcessingInput, ProcessingOutput

# Initialize the SageMaker session
sagemaker_session = boto3.Session().client('sagemaker')

# Define the processor
processor = Processor(
    role='YourSageMakerRoleArn',  # Replace with your SageMaker role ARN
    image_uri='ghcr.io/kyryl-opens-ml/gpu-jobs-comparison:pr-1',
    instance_count=1,
    instance_type='ml.p3.2xlarge',  # Choose an appropriate instance type
    env={
        'HF_TOKEN': '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',
        'WANDB_PROJECT': 'gpu-jobs-comparison',
        'WANDB_API_KEY': 'cb86168a2e8db7edb905da69307450f5e7867d66',
        'RUN_NAME': 'phi-3-mini-lora-text2sql-sagemaker'
    }
)

# Define processing inputs and outputs (if any)
processing_inputs = []
processing_outputs = []

# Run the processing job
processor.run(
    inputs=processing_inputs,
    outputs=processing_outputs,
    arguments=['python', 'text2sql_training.py']
)
