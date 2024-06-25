# doc: https://docs.aws.amazon.com/sagemaker/latest/dg/serverless-endpoints-create.html

# upload model checkpoint to S3
#echo "Uploading model checkpoint to S3"
#aws s3 cp Meta-Llama-3-8B-hf/ s3://253333439226-app-registry/llama3/Meta-Llama-3-8B-hf/ --recursive
#aws s3 cp Meta-Llama-3-8B-hf-finetuned/ s3://253333439226-app-registry/llama3/Meta-Llama-3-8B-hf-finetuned/ --recursive

# login, this is an other command as in the doc
sudo docker login -u AWS -p $(aws ecr get-login-password --region eu-central-1) 253333439226.dkr.ecr.eu-central-1.amazonaws.com

# create docker image
echo "Building docker image"
sudo docker build -t llama3-deployment .

# tag docker image
echo "Tagging docker image"
sudo docker tag llama3-deployment:latest 253333439226.dkr.ecr.eu-central-1.amazonaws.com/llama3-deployment:latest

# push docker image
echo "Pushing docker image"
sudo docker push 253333439226.dkr.ecr.eu-central-1.amazonaws.com/llama3-deployment:latest

# remove old model
echo "Deleting old model"
aws sagemaker delete-model \
    --model-name llama3-html-to-json \
    --region eu-central-1

# remove old endpoint configuration
echo "Deleting old endpoint configuration"
aws sagemaker delete-endpoint-config \
    --endpoint-config-name llama3-html-to-json \
    --region eu-central-1

# remove old endpoint
echo "Deleting old endpoint"
aws sagemaker delete-endpoint \
    --endpoint-name llama3-html-to-json \
    --region eu-central-1

# create a mode
echo "Creating model"
aws sagemaker create-model \
    --model-name llama3-html-to-json \
    --primary-container Image=253333439226.dkr.ecr.eu-central-1.amazonaws.com/llama3-deployment:latest \
    --execution-role-arn arn:aws:iam::253333439226:role/service-role/SageMaker-ExecutionRole-20240219T160673 \
    --region eu-central-1

# create an endpoint configuration
echo "Creating endpoint configuration"
aws sagemaker create-endpoint-config \
    --endpoint-config-name llama3-html-to-json \
    --production-variants VariantName=AllTraffic,ModelName=llama3-html-to-json,InitialInstanceCount=1,InstanceType=ml.g4dn.12xlarge,InitialVariantWeight=1.0 \
    --region eu-central-1

# create an endpoint
# echo "Creating endpoint"
# aws sagemaker create-endpoint \
#     --endpoint-name llama3-html-to-json \
#     --endpoint-config-name llama3-html-to-json \
#     --region eu-central-1