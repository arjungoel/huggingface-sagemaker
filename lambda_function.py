import boto3

# The name of the endpoint. The name must be unique within an AWS Region in your AWS account. 
ENDPOINT_NAME = "xxxxxx"
runtime = boto3.client("sagemaker-runtime", region_name='ca-central-1')


def lambda_handler(event, context):
    inputs = event['data']
    result = []
    for input in inputs:
        serialized_input = ','.join(map(str, input))
        # After you deploy a model into production using SageMaker hosting 
        # services, your client applications use this API to get inferences 
        # from the model hosted at the specified endpoint.
        response = runtime.invoke_endpoint(
                    EndpointName=ENDPOINT_NAME,
                    ContentType='image/jpeg',
                    Body=serialized_input) # Replace with your own data.
    # Optional - Print the response body and decode it so it is human read-able.
        result.append(response['Body'].read().decode())
        return result


# test event
data = {
    'inputs': {
        'data': 'My name is Arjun Goel'
    }
}