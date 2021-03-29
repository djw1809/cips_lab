import boto3
import json
#%%

account = 736072302811
region = 'us-east-2'
ecr_registry = 'sage_testing'
image_name = 'test_fixed'
role = 'arn:aws:iam::736072302811:role/sage_do_stuff'

hosting_container = {
        'Image': '{}.dkr.ecr.{}.amazonaws.com/{}:{}'.format(account, region, ecr_registry, image_name),
        'ModelDataUrl': 'https://sagemaker-dilawn-test.s3.us-east-2.amazonaws.com/model.tar.gz'
        }

#%%

sm = boto3.client('sagemaker')
create_model_response = sm.create_model(
                                        ModelName = 'batch-test-093020',
                                        ExecutionRoleArn=role,
                                        PrimaryContainer = hosting_container)


modelarn = create_model_response['ModelArn']


endpoint_config = 'batch-test-093020-endpoint-config'

create_endpoint_config_response = sm.create_endpoint_config(
    EndpointConfigName= endpoint_config,
    ProductionVariants=[{
        'InstanceType': 'ml.m4.xlarge',
        'InitialInstanceCount': 1,
        'ModelName': 'batch-test-093020',
        'VariantName': 'AllTraffic'}])

endpoint = 'batch-test-093020-endpoint'

create_endpoint_response = sm.create_endpoint(
                            EndpointName = endpoint,
                            EndpointConfigName = endpoint_config

)

endpointarn = create_endpoint_response['EndpointArn']

resp = sm.describe_endpoint(EndpointName = endpoint)runtime = boto3.Session().client('runtime.sagemaker')
payload = json.dumps({"keywords":["insurance-"], "prompt":"I hate", "max_len":50, "top_k":200, "top_p":.5, "num_return_sequences":2})

runtime = boto3.Session().client('runtime.sagemaker')

response = runtime.invoke_endpoint(EndpointName = 'batch-test-093020-endpoint',
                                    ContentType='application/json',
                                    Body = payload)
response['Body'].read()
