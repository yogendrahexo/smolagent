from dotenv import load_dotenv
_ = load_dotenv()

import os
from smolagents import AzureOpenAIServerModel, CodeAgent

model_o1 = AzureOpenAIServerModel(
    model_id = os.getenv("AZURE_o1_DEPLOYMENT_NAME"),
    azure_endpoint=os.getenv("AZURE_o1_ENDPOINT_URL"),
    api_key=os.getenv("AZURE_o1_SUBSCRIPTION_KEY_1"),
    api_version=os.getenv("AZURE_o1_API_VERSION", "2024-12-01-preview")
)

model_4o = AzureOpenAIServerModel(
    model_id = os.getenv("AZURE_GPT4o_DEPLOYMENT_NAME"),
    azure_endpoint=os.getenv("AZURE_GPT4o_ENDPOINT_URL"),
    api_key=os.getenv("AZURE_GPT4o_SUBSCRIPTION_KEY_1"),
    api_version=os.getenv("AZURE_GPT4o_API_VERSION", "2024-12-01-preview")
)

agent = CodeAgent(
    tools=[],
    model=model_o1,
    max_steps=4,
    verbosity_level=1,
    name="example_agent",
    executor_type="unrestricted_local",
    description="This is an example agent that runs code directly.",
    additional_authorized_imports=["*"] 
)

agent.run("Use PyTorch to check GPU availability and display detailed system information")