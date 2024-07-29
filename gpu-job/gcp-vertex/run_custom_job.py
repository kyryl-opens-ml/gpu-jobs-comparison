import os
from google.cloud import aiplatform

# Set your project ID and location
PROJECT_ID = os.getenv('GOOGLE_CLOUD_PROJECT', 'gothic-doodad-323015')
LOCATION = os.getenv('GOOGLE_CLOUD_LOCATION', 'us-central1')
STAGING_BUCKET = 'gs://gpu-jobs-comparison'  # Replace with your staging bucket

# Initialize Vertex AI
aiplatform.init(project=PROJECT_ID, location=LOCATION, staging_bucket=STAGING_BUCKET)

# Initialize Vertex AI
# aiplatform.init(project=PROJECT_ID, location=LOCATION)

# Define environment variables
env = {
    "WANDB_PROJECT": os.getenv("WANDB_PROJECT"),
    "WANDB_API_KEY": os.getenv("WANDB_API_KEY"),
    "HF_TOKEN": os.getenv("HF_TOKEN"),
    "RUN_NAME": os.getenv("RUN_NAME"),
}

# Define the container image URI from Artifact Registry
image_uri = "us-central1-docker.pkg.dev/gothic-doodad-323015/gpu-jobs-comparison/gpu-jobs-comparis"

# Define the worker pool specification
worker_pool_specs = [
    {
        "machine_spec": {
            "machine_type": "n1-standard-4",
            "accelerator_type": "NVIDIA_TESLA_T4",
            "accelerator_count": 1,
        },
        "replica_count": 1,
        "container_spec": {
            "image_uri": image_uri,
            "command": ["python", "text2sql_training.py"],  # Command to run the training script
            "args": [],  # Additional arguments can be added here
            "env": [{"name": key, "value": value} for key, value in env.items()]
        },
    }
]

# Create the CustomJob
job = aiplatform.CustomJob(
    display_name='text2sql-training',
    worker_pool_specs=worker_pool_specs,
    labels={'env': 'production'},  # Example label
)

# Run the job
job.run(sync=True)