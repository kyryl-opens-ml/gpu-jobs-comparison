# gpu-jobs-comparison

## Run SSH in VM

```bash
export HF_TOKEN=****
export WANDB_PROJECT=gpu-jobs-comparison
export WANDB_API_KEY==****
export RUN_NAME=phi-3-text2sql-ssh 

pip install -r requirements.txt
python text2sql_training.py
```

## Kubernetes

```bash
minikube start --driver docker --container-runtime docker --gpus all

export HF_TOKEN=****
export WANDB_API_KEY==****

kubectl create secret generic gpu-job-secrets --from-literal=HF_TOKEN=$HF_TOKEN --from-literal=WANDB_API_KEY=$WANDB_API_KEY
kubectl create -f gpu-job/kubernetes/job-app-ml.yaml
```

## Modal

```bash
export HF_TOKEN=****
export WANDB_PROJECT=gpu-jobs-comparison
export WANDB_API_KEY==****
export RUN_NAME=phi-3-text2sql-modal

pip install modal

modal setup
modal deploy ./gpu-job/modal/run_training_job.py

python ./gpu-job/modal/run_training_job.py
```

## AWS SageMaker

```bash
export AWS_ACCESS_KEY_ID=****
export AWS_SECRET_ACCESS_KEY=****
export AWS_DEFAULT_REGION=us-east-1
export AWS_ACCOUNT_ID=****

pip install boto3 sagemaker awscli

aws iam create-role --role-name sagemaker-execution-role --assume-role-policy-document file://gpu-job/aws-sagemaker/trust-policy.json
aws iam attach-role-policy --role-name sagemaker-execution-role --policy-arn arn:aws:iam::aws:policy/AmazonSageMakerFullAccess
aws iam attach-role-policy --role-name sagemaker-execution-role --policy-arn arn:aws:iam::aws:policy/AmazonEC2ContainerRegistryFullAccess


aws ecr get-login-password --region $AWS_DEFAULT_REGION | docker login --username AWS --password-stdin $AWS_ACCOUNT_ID.dkr.ecr.$AWS_DEFAULT_REGION.amazonaws.com
aws ecr create-repository --repository-name gpu-jobs-comparison
docker pull ghcr.io/kyryl-opens-ml/gpu-jobs-comparison:latest
docker tag ghcr.io/kyryl-opens-ml/gpu-jobs-comparison:latest $AWS_ACCOUNT_ID.dkr.ecr.$AWS_DEFAULT_REGION.amazonaws.com/gpu-jobs-comparison:latest
docker push $AWS_ACCOUNT_ID.dkr.ecr.$AWS_DEFAULT_REGION.amazonaws.com/gpu-jobs-comparison:latest



export HF_TOKEN=****
export WANDB_PROJECT=gpu-jobs-comparison
export WANDB_API_KEY=****
export RUN_NAME=phi-3-text2sql-sagemaker
python ./gpu-job/aws-sagemaker/run_processing.py
```