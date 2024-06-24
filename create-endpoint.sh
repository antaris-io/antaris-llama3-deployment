# doc: https://docs.aws.amazon.com/sagemaker/latest/dg/serverless-endpoints-create.html

# create a ECR repository and follow the instructions to push the image to the repository

# upload model checkpoint to S3
aws s3 cp Meta-Llama-3-8B-hf/ s3://253333439226-app-registry/llama3/Meta-Llama-3-8B-hf/ --recursive
aws s3 cp Meta-Llama-3-8B-hf-finetuned/ s3://253333439226-app-registry/llama3/Meta-Llama-3-8B-hf-finetuned/ --recursive

# login, this is an other command as in the doc
sudo docker login -u AWS -p $(aws ecr get-login-password --region eu-central-1) 253333439226.dkr.ecr.eu-central-1.amazonaws.com

# create docker image
sudo docker build -t llama3-deployment .

# tag docker image
sudo docker tag llama3-deployment:latest 253333439226.dkr.ecr.eu-central-1.amazonaws.com/llama3-deployment:latest

# push docker image
sudo docker push 253333439226.dkr.ecr.eu-central-1.amazonaws.com/llama3-deployment:latest

# create a model (using the console)
# create an endpoint configuration (using the console), instancwe type: ml.g4dn.12xlarge
# create an endpoint (using the console)