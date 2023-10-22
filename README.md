### Deploy LLM as a model within CML
This project walks through a deployment and hosting of a Large Languge Model (LLM) within CML. 

Deploy the model by:
- Navigate to  Model Deployments
- Click `New Model`
- Give it a Name and Description
- Disable Authentication (for convenience)
- Select File `launch_model_*.py`
- Set Function Name `api_wrapper`
  - This is the function implemented in the python script which wraps text inference with the llama2-chat model
- Set sample json payload
   ```
    {
    "prompt": "test prompt hello"
    }
   ```
- Pick Runtime
  - Workbench -- Python 3.9 -- Nvidia GPU -- 2023.08
- Set Resource Profile
  - At least 4CPU / 16MEM
  - 1 GPU (this will be a v100 in the shared hackathon cluster)
- Click `Deploy Model`
- Wait until it is Deployed

Test the Model