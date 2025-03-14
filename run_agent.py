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
    max_steps=40,
    verbosity_level=1,
    name="ml_agent",
    executor_type="unrestricted_local",
    description="This is an agent that solves machine learning problems.",
    additional_authorized_imports=["*"] 
)

def run_ml_project(task_path: str, eval_path: str, project_dir: str) -> None:
    """
    Run ML project using the agent with provided task description and evaluation criteria.
    
    Args:
        task_path (str): Path to the task description file
        eval_path (str): Path to the evaluation criteria file
        project_dir (str): Path to the project directory containing data files
    """
    # Read task description and evaluation criteria
    with open(task_path, 'r') as f:
        task_description = f.read()
    
    with open(eval_path, 'r') as f:
        evaluation_criteria = f.read()
    
    # Construct the prompt
    prompt = f"""
I have uploaded a Machine Learning project repository in {project_dir} with the following structure:
- train.csv: Training data
- test.csv: Test data
- sample_submission.csv: Sample submission format

Problem Description:
<problem_description>
{task_description}
</problem_description>

Evaluation Criteria:
<evaluation_criteria>
{evaluation_criteria}
</evaluation_criteria>

Please help solve this ML task by:

1. Analyzing the data and problem requirements.
2. Implementing data preprocessing and feature engineering.
3. Developing and training appropriate ML models.
4. Creating evaluation metrics and validation strategy.
5. Generating predictions on test data.
6. Creating two main files:
   - train.py: For data processing, model training, and evaluation.
   - inference.py: For generating predictions on new data and evaluating the model.  This script should print the evaluation scores.
7. **Run `train.py` to train the model, and then run `inference.py` to generate predictions and evaluate the solution.**
8. **After running `inference.py`, check if the solution is good based on these criteria:**
   - Score reaches ≤ 0.2 (if lower is better)
   - Score reaches ≥ 0.8 (if higher is better)
   - We get consistent scores in the range 0.2-0.8 for 5 consecutive iterations
   **If the score does not meet the criteria above, analyze the previous code (both `train.py` and `inference.py`), identify potential improvements, and implement a new solution.  Consider adjusting the model architecture, hyperparameters, feature engineering, or data preprocessing. Repeat the training and inference process with the improved code.**
9. **In your final response, clearly state the best method used and the corresponding best score achieved.**

Focus on:
- Proper validation strategy
- Model performance optimization
- Efficient data processing pipelines
- Clear documentation and logging

Environment
- A virtual environment is already created at ./.venv
- Required packages (pandas, numpy, scikit-learn, joblib) have been installed in the virtual environment
- The code implementation includes both train.py and inference.py files with a basic machine learning pipeline using Logistic Regression as the baseline model.
"""
    # Run the agent with the constructed prompt
    agent.run(prompt)


# running a ml project
run_ml_project("data/house_pricing/task.md", "data/house_pricing/eval.md", "data/house_pricing")