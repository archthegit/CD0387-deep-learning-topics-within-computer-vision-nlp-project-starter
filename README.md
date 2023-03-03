# Image Classification using AWS SageMaker

Use AWS Sagemaker to train a pretrained model that can perform image classification by using the Sagemaker profiling, debugger, hyperparameter tuning and other good ML engineering practices. This can be done on either the provided dog breed classication data set or one of your choice.

## Project Set Up and Installation
Enter AWS through the gateway in the course and open SageMaker Studio. 
Download the starter files.
Download/Make the dataset available. 

## Dataset
The provided dataset is the dogbreed classification dataset which can be found in the classroom.
The project is designed to be dataset independent so if there is a dataset that is more interesting or relevant to your work, you are welcome to use it to complete the project.

### Access
Upload the data to an S3 bucket through the AWS Gateway so that SageMaker has access to the data. 

## Hyperparameter Tuning
What kind of model did you choose for this experiment and why? Give an overview of the types of parameters and their ranges used for the hyperparameter search

I used a resnet18 model as it deals with the task of identifying images quite well.
I tuned 3 hyperparameters referenced from the udacity lesson: 
- `lr`:  (0.01 - 0.1)
- 'batch size`: [32, 64, 128, 256, 512]
- `epochs`: (2, 4)
Completed training jobs and hyperparameter tuning jobs:

<img width="1231" alt="Screenshot 2023-03-03 at 9 59 30 PM" src="https://user-images.githubusercontent.com/22144490/222738833-e79e4c73-0e43-47e9-8536-ccd6ad2d38d0.png">

<img width="1259" alt="Screenshot 2023-03-03 at 9 57 37 PM" src="https://user-images.githubusercontent.com/22144490/222738467-cd73418b-c8ce-4168-97cc-a75f6a165251.png">


Remember that your README should:
- Include a screenshot of completed training jobs
- Logs metrics during the training process
- Tune at least two hyperparameters
- Retrieve the best best hyperparameters from all your training jobs

## Debugging and Profiling
Give an overview of how you performed model debugging and profiling in Sagemaker

Debugging and profiling was done by setting rules and setting up the profiler and debugger through their custom configurations. For proper debugging, we used the `train_model.py` script which was based off of `hpo.py`

### Results
What are the results/insights did you get by profiling/debugging your model?
```
2023-03-03 10:31:35 Uploading - Uploading generated training model
2023-03-03 10:31:35 Completed - Training job completed
VanishingGradient: NoIssuesFound
Overfit: InProgress
Overtraining: NoIssuesFound
PoorWeightInitialization: IssuesFound
Training seconds: 564
Billable seconds: 564
```

## Model Deployment
Give an overview of the deployed model and instructions on how to query the endpoint with a sample input.
To deploy the model we use the `inference.py` script which will transform the input image in order to fit what the model accepts.

In order to query the endpoint:
```
from PIL import Image
from IPython.display import Image as ImageDisplay
import io
from os import listdir

def identify_dog(image_path):    
    buf = io.BytesIO()
    Image.open(image_path).save(buf, format="JPEG")
    response = predictor.predict(buf.getvalue())
    
    folders = list(listdir("./dogImages/train"))
    folders.sort()
    return folders[list(response[0]).index(max(response[0]))]
   ````
   use this script and call identify_dog(path_to_dog_img)
Remember to provide a screenshot of the deployed active endpoint in Sagemaker.
<img width="1297" alt="Screenshot 2023-03-03 at 10 08 10 PM" src="https://user-images.githubusercontent.com/22144490/222741190-f1d2b3b8-0308-472c-9bd8-9e7ced09e911.png">

