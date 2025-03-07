import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.language_models import BaseChatModel
from langchain.schema import Document
from langchain_community.document_loaders import DataFrameLoader
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_experimental.utilities import PythonREPL
from typing import Dict, List, Any, Optional, Tuple
from langchain_core.messages import BaseMessage
from langchain_core.runnables import RunnableConfig, RunnableSerializable
from langchain_core.runnables.config import run_in_executor
from langchain_core.output_parsers import StrOutputParser
from typing import Annotated, List, Sequence, TypedDict, Union, Dict, Any
from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain.tools import BaseTool, StructuredTool
from langchain_core.tools import tool
import json
import io
import base64
from operator import itemgetter
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langgraph.graph import StateGraph, END
# from langchain_community.llms import OpenAI
from langchain_openai import ChatOpenAI, OpenAI
# Configure the LLM - using LM Studio with local model
llm = ChatOpenAI(
    model="qwen2.5-7b-instruct",
    base_url="http://127.0.0.1:1234",
    api_key="qwen2.5-7b-instruct",
    temperature=0
)


# Define state schema
class AgentState(TypedDict):
    user_input: str
    data: Optional[pd.DataFrame]
    data_summary: Optional[Dict[str, Any]]
    preprocessing_recommendation: Optional[Dict[str, Any]]
    visualization_recommendation: Optional[Dict[str, Any]]
    processed_data: Optional[pd.DataFrame]
    generated_visualizations: Optional[List[Dict[str, str]]]
    final_report: Optional[str]


# Tool for data analysis and summary
@tool
def analyze_data(df_json: str) -> str:
    """Analyze the dataframe and provide a summary of its characteristics."""
    try:
        # Convert the JSON string back to a dataframe
        df = pd.read_json(io.StringIO(df_json))

        # Get basic information
        shape = df.shape
        dtypes = df.dtypes.astype(str).to_dict()
        missing_values = df.isna().sum().to_dict()

        # Get numeric statistics if applicable
        numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
        numeric_stats = {}
        if numeric_columns:
            numeric_stats = df[numeric_columns].describe().to_dict()

        # Get categorical statistics if applicable
        categorical_columns = df.select_dtypes(include=['object', 'category']).columns.tolist()
        categorical_stats = {}
        if categorical_columns:
            for col in categorical_columns:
                categorical_stats[col] = df[col].value_counts().to_dict()

        # Prepare the summary
        summary = {
            "shape": shape,
            "dtypes": dtypes,
            "missing_values": missing_values,
            "numeric_stats": numeric_stats,
            "categorical_stats": categorical_stats
        }

        return json.dumps(summary)
    except Exception as e:
        return f"Error analyzing data: {str(e)}"


# Tool for data preprocessing
@tool
def preprocess_data(preprocessing_steps: str, df_json: str) -> str:
    """Apply preprocessing steps to the dataframe and return the processed dataframe."""
    try:
        # Convert the JSON string back to a dataframe
        df = pd.read_json(io.StringIO(df_json))

        # Parse preprocessing steps
        steps = json.loads(preprocessing_steps)

        # Apply preprocessing steps
        processed_df = df.copy()
        applied_steps = []

        for step in steps:
            step_type = step.get("type")

            if step_type == "handle_missing_values":
                strategy = step.get("strategy", "drop")
                columns = step.get("columns", processed_df.columns.tolist())

                if strategy == "drop":
                    processed_df = processed_df.dropna(subset=columns)
                    applied_steps.append(f"Dropped rows with missing values in columns: {columns}")
                elif strategy == "fill_mean":
                    for col in columns:
                        if pd.api.types.is_numeric_dtype(processed_df[col]):
                            processed_df[col] = processed_df[col].fillna(processed_df[col].mean())
                    applied_steps.append(f"Filled missing values with mean in columns: {columns}")
                elif strategy == "fill_median":
                    for col in columns:
                        if pd.api.types.is_numeric_dtype(processed_df[col]):
                            processed_df[col] = processed_df[col].fillna(processed_df[col].median())
                    applied_steps.append(f"Filled missing values with median in columns: {columns}")
                elif strategy == "fill_mode":
                    for col in columns:
                        processed_df[col] = processed_df[col].fillna(processed_df[col].mode()[0])
                    applied_steps.append(f"Filled missing values with mode in columns: {columns}")
                elif strategy == "fill_value":
                    value = step.get("value", 0)
                    for col in columns:
                        processed_df[col] = processed_df[col].fillna(value)
                    applied_steps.append(f"Filled missing values with {value} in columns: {columns}")

            elif step_type == "scale":
                method = step.get("method", "min_max")
                columns = step.get("columns", processed_df.select_dtypes(include=[np.number]).columns.tolist())

                if method == "min_max":
                    for col in columns:
                        if pd.api.types.is_numeric_dtype(processed_df[col]):
                            min_val = processed_df[col].min()
                            max_val = processed_df[col].max()
                            processed_df[col] = (processed_df[col] - min_val) / (max_val - min_val)
                    applied_steps.append(f"Applied min-max scaling to columns: {columns}")
                elif method == "standard":
                    for col in columns:
                        if pd.api.types.is_numeric_dtype(processed_df[col]):
                            mean_val = processed_df[col].mean()
                            std_val = processed_df[col].std()
                            processed_df[col] = (processed_df[col] - mean_val) / std_val
                    applied_steps.append(f"Applied standard scaling to columns: {columns}")

            elif step_type == "encode_categorical":
                method = step.get("method", "one_hot")
                columns = step.get("columns",
                                   processed_df.select_dtypes(include=['object', 'category']).columns.tolist())

                if method == "one_hot":
                    for col in columns:
                        one_hot = pd.get_dummies(processed_df[col], prefix=col)
                        processed_df = pd.concat([processed_df, one_hot], axis=1)
                        processed_df = processed_df.drop(col, axis=1)
                    applied_steps.append(f"Applied one-hot encoding to columns: {columns}")
                elif method == "label":
                    for col in columns:
                        unique_values = processed_df[col].unique()
                        value_map = {value: idx for idx, value in enumerate(unique_values)}
                        processed_df[col] = processed_df[col].map(value_map)
                    applied_steps.append(f"Applied label encoding to columns: {columns}")

            elif step_type == "remove_outliers":
                method = step.get("method", "iqr")
                columns = step.get("columns", processed_df.select_dtypes(include=[np.number]).columns.tolist())

                if method == "iqr":
                    for col in columns:
                        if pd.api.types.is_numeric_dtype(processed_df[col]):
                            Q1 = processed_df[col].quantile(0.25)
                            Q3 = processed_df[col].quantile(0.75)
                            IQR = Q3 - Q1
                            lower_bound = Q1 - 1.5 * IQR
                            upper_bound = Q3 + 1.5 * IQR
                            processed_df = processed_df[
                                (processed_df[col] >= lower_bound) & (processed_df[col] <= upper_bound)]
                    applied_steps.append(f"Removed outliers using IQR method for columns: {columns}")
                elif method == "z_score":
                    for col in columns:
                        if pd.api.types.is_numeric_dtype(processed_df[col]):
                            mean_val = processed_df[col].mean()
                            std_val = processed_df[col].std()
                            z_scores = (processed_df[col] - mean_val) / std_val
                            processed_df = processed_df[abs(z_scores) <= 3]
                    applied_steps.append(f"Removed outliers using Z-score method for columns: {columns}")

            elif step_type == "transform":
                method = step.get("method", "log")
                columns = step.get("columns", processed_df.select_dtypes(include=[np.number]).columns.tolist())

                if method == "log":
                    for col in columns:
                        if pd.api.types.is_numeric_dtype(processed_df[col]) and (processed_df[col] > 0).all():
                            processed_df[col] = np.log(processed_df[col])
                    applied_steps.append(f"Applied log transformation to columns: {columns}")
                elif method == "sqrt":
                    for col in columns:
                        if pd.api.types.is_numeric_dtype(processed_df[col]) and (processed_df[col] >= 0).all():
                            processed_df[col] = np.sqrt(processed_df[col])
                    applied_steps.append(f"Applied square root transformation to columns: {columns}")

            elif step_type == "drop_columns":
                columns = step.get("columns", [])
                processed_df = processed_df.drop(columns=columns)
                applied_steps.append(f"Dropped columns: {columns}")

        # Return the processed dataframe and applied steps
        result = {
            "processed_df": processed_df.to_json(),
            "applied_steps": applied_steps
        }

        return json.dumps(result)
    except Exception as e:
        return f"Error preprocessing data: {str(e)}"


# Tool for generating visualizations
@tool
def generate_visualizations(visualization_specs: str, df_json: str) -> str:
    """Generate visualizations based on the specifications and return them as base64-encoded images."""
    try:
        # Convert the JSON string back to a dataframe
        df = pd.read_json(io.StringIO(df_json))

        # Parse visualization specifications
        specs = json.loads(visualization_specs)

        # Generate visualizations
        visualizations = []

        for spec in specs:
            plt.figure(figsize=(10, 6))

            viz_type = spec.get("type")

            if viz_type == "histogram":
                column = spec.get("column")
                bins = spec.get("bins", 10)
                title = spec.get("title", f"Histogram of {column}")
                sns.histplot(data=df, x=column, bins=bins, kde=spec.get("kde", False))
                plt.title(title)

            elif viz_type == "bar":
                x = spec.get("x")
                y = spec.get("y")
                title = spec.get("title", f"Bar Plot of {y} by {x}")
                sns.barplot(data=df, x=x, y=y)
                plt.title(title)

            elif viz_type == "scatter":
                x = spec.get("x")
                y = spec.get("y")
                hue = spec.get("hue")
                title = spec.get("title", f"Scatter Plot of {y} vs {x}")
                if hue:
                    sns.scatterplot(data=df, x=x, y=y, hue=hue)
                else:
                    sns.scatterplot(data=df, x=x, y=y)
                plt.title(title)

            elif viz_type == "line":
                x = spec.get("x")
                y = spec.get("y")
                hue = spec.get("hue")
                title = spec.get("title", f"Line Plot of {y} vs {x}")
                if hue:
                    sns.lineplot(data=df, x=x, y=y, hue=hue)
                else:
                    sns.lineplot(data=df, x=x, y=y)
                plt.title(title)

            elif viz_type == "box":
                x = spec.get("x")
                y = spec.get("y")
                title = spec.get("title", f"Box Plot of {y} by {x}")
                sns.boxplot(data=df, x=x, y=y)
                plt.title(title)

            elif viz_type == "heatmap":
                columns = spec.get("columns", df.select_dtypes(include=[np.number]).columns.tolist())
                title = spec.get("title", "Correlation Heatmap")
                correlation = df[columns].corr()
                sns.heatmap(correlation, annot=spec.get("annot", True), cmap=spec.get("cmap", "coolwarm"))
                plt.title(title)

            elif viz_type == "pair":
                columns = spec.get("columns", df.select_dtypes(include=[np.number]).columns.tolist()[:4])
                hue = spec.get("hue")
                title = spec.get("title", "Pair Plot")
                if hue:
                    sns.pairplot(df[columns + [hue]], hue=hue)
                else:
                    sns.pairplot(df[columns])
                plt.suptitle(title, y=1.02)

            elif viz_type == "count":
                column = spec.get("column")
                title = spec.get("title", f"Count Plot of {column}")
                sns.countplot(data=df, x=column)
                plt.title(title)

            # Save the figure to a base64-encoded image
            buffer = io.BytesIO()
            plt.tight_layout()
            plt.savefig(buffer, format="png")
            buffer.seek(0)
            img_str = base64.b64encode(buffer.read()).decode("utf-8")
            plt.close()

            visualizations.append({
                "title": spec.get("title", f"Visualization {len(visualizations) + 1}"),
                "type": viz_type,
                "description": spec.get("description", ""),
                "image": img_str
            })

        return json.dumps(visualizations)
    except Exception as e:
        return f"Error generating visualizations: {str(e)}"


# Define node functions
def load_and_analyze_data(state: AgentState) -> AgentState:
    """Load data from user input and perform initial analysis."""
    try:
        # Parse user input to extract dataframe
        user_input = state["user_input"]

        # Check if user input contains data in a format that can be parsed
        if "```" in user_input:
            data_content = user_input.split("```")[1]
            # Try to parse as CSV
            if "," in data_content or "\t" in data_content:
                data_buffer = io.StringIO(data_content)
                df = pd.read_csv(data_buffer, sep=None, engine='python')
            else:
                # Fallback to a simple parsing for structured data
                lines = [line.strip() for line in data_content.strip().split("\n") if line.strip()]
                if lines:
                    header = lines[0].split()
                    data = []
                    for line in lines[1:]:
                        data.append(line.split())
                    df = pd.DataFrame(data, columns=header)
        else:
            # Try to parse JSON-like input
            try:
                # Look for data pattern like [{...}, {...}]
                import re
                json_pattern = r'\[\s*\{.*\}\s*\]'
                match = re.search(json_pattern, user_input, re.DOTALL)
                if match:
                    data_json = match.group(0)
                    # Use StringIO to avoid deprecation warning
                    df = pd.read_json(io.StringIO(data_json))
                else:
                    # If no JSON array pattern found, try to evaluate as Python literal
                    import ast
                    data_list = ast.literal_eval(user_input)
                    if isinstance(data_list, list) and all(isinstance(item, dict) for item in data_list):
                        df = pd.DataFrame(data_list)
                    else:
                        raise ValueError("Unable to parse data from input")
            except:
                # If JSON parsing fails, create a sample dataframe for demo purposes
                df = pd.DataFrame({
                    'A': np.random.normal(0, 1, 100),
                    'B': np.random.normal(0, 1, 100),
                    'C': np.random.choice(['X', 'Y', 'Z'], 100),
                    'D': np.random.randint(0, 100, 100)
                })

        # Analyze the data using .invoke() instead of direct call
        # Use the tool directly since we're not in a chain context
        data_summary_json = analyze_data.func(df.to_json())
        data_summary = json.loads(data_summary_json)

        return {
            **state,
            "data": df,
            "data_summary": data_summary
        }
    except Exception as e:
        return {
            **state,
            "data": pd.DataFrame({
                'Error': [f"Failed to load data: {str(e)}"]
            }),
            "data_summary": {"error": str(e)}
        }


def recommend_preprocessing(state: AgentState) -> AgentState:
    """Recommend preprocessing techniques based on data analysis."""
    data_summary = state["data_summary"]

    # Create a combined prompt for recommendation to work with non-chat LLMs
    prompt_template = f"""You are a data science expert. Based on the data summary provided, 
    recommend appropriate preprocessing techniques. Consider:
    1. Handling missing values
    2. Scaling/normalization needs
    3. Categorical encoding requirements
    4. Outlier detection and removal
    5. Feature transformations

    Data summary: {data_summary}"""


    format = """Format your response as a JSON array of preprocessing steps with the following structure:
    [
        `{
            "type": "handle_missing_values",
            "strategy": "fill_mean",
            "columns": ["column1", "column2"]
        },
        {
            "type": "scale",
            "method": "min_max",
            "columns": ["column1", "column2"]
        }
    ]

    Include a detailed explanation for each recommendation.
    """
    prompt_template = prompt_template + format
    # Format the prompt with actual data
    # formatted_prompt = prompt_template.format(data_summary)
    formatted_prompt = prompt_template
    # print(formatted_prompt )

    # Use direct invocation for the LLM that works with LM Studio
    llm_response = llm.invoke(formatted_prompt)

    # Parse the response to get preprocessing steps and explanation
    try:
        # Extract JSON content if present
        content = llm_response
        import re
        json_pattern = r'\[\s*\{.*\}\s*\]'
        match = re.search(json_pattern, content, re.DOTALL)

        preprocessing_steps = []
        explanation = content

        if match:
            preprocessing_steps_str = match.group(0)
            preprocessing_steps = json.loads(preprocessing_steps_str)
            # Remove the JSON content from the explanation
            explanation = content.replace(preprocessing_steps_str, "")

        recommendation = {
            "steps": preprocessing_steps,
            "explanation": explanation.strip()
        }

        return {
            **state,
            "preprocessing_recommendation": recommendation
        }
    except Exception as e:
        return {
            **state,
            "preprocessing_recommendation": {
                "steps": [],
                "explanation": f"Error generating preprocessing recommendations: {str(e)}"
            }
        }


def recommend_visualizations(state: AgentState) -> AgentState:
    """Recommend visualizations based on data analysis."""
    data_summary = state["data_summary"]

    # Create a combined prompt for visualization recommendations to work with non-chat LLMs
    prompt_template = """You are a data visualization expert. Based on the data summary provided, 
    recommend appropriate visualization techniques. Consider:
    1. Distribution of numeric variables
    2. Relationships between variables
    3. Categorical data visualization
    4. Time-series patterns if applicable
    5. Correlation analysis

    Data summary: {data_summary}

    Format your response as a JSON array of visualization specifications with the following structure:
    [
        {
            "type": "histogram",
            "column": "column1",
            "bins": 10,
            "title": "Distribution of Column1",
            "description": "This histogram shows the distribution of values in Column1."
        },
        {
            "type": "scatter",
            "x": "column1",
            "y": "column2",
            "title": "Column1 vs Column2",
            "description": "This scatter plot shows the relationship between Column1 and Column2."
        }
    ]

    Include a detailed explanation for each recommendation.
    """

    # Format the prompt with actual data
    formatted_prompt = prompt_template.format(data_summary=json.dumps(data_summary, indent=2))

    # Use direct invocation for the LLM that works with LM Studio
    llm_response = llm.invoke(formatted_prompt)

    # Parse the response to get visualization specifications and explanation
    try:
        # Extract JSON content if present
        content = llm_response
        import re
        json_pattern = r'\[\s*\{.*\}\s*\]'
        match = re.search(json_pattern, content, re.DOTALL)

        visualization_specs = []
        explanation = content

        if match:
            visualization_specs_str = match.group(0)
            visualization_specs = json.loads(visualization_specs_str)
            # Remove the JSON content from the explanation
            explanation = content.replace(visualization_specs_str, "")

        recommendation = {
            "specs": visualization_specs,
            "explanation": explanation.strip()
        }

        return {
            **state,
            "visualization_recommendation": recommendation
        }
    except Exception as e:
        return {
            **state,
            "visualization_recommendation": {
                "specs": [],
                "explanation": f"Error generating visualization recommendations: {str(e)}"
            }
        }


def apply_preprocessing(state: AgentState) -> AgentState:
    """Apply recommended preprocessing techniques to the data."""
    try:
        data = state["data"]
        preprocessing_recommendation = state["preprocessing_recommendation"]

        preprocessing_steps = preprocessing_recommendation["steps"]

        # Apply preprocessing - use func directly to avoid using invoke
        preprocessing_result_json = preprocess_data.func(
            json.dumps(preprocessing_steps),
            data.to_json()
        )

        preprocessing_result = json.loads(preprocessing_result_json)

        processed_df = pd.read_json(io.StringIO(preprocessing_result["processed_df"]))

        return {
            **state,
            "processed_data": processed_df
        }
    except Exception as e:
        return {
            **state,
            "processed_data": state["data"],
            "error": f"Error applying preprocessing: {str(e)}"
        }


def create_visualizations(state: AgentState) -> AgentState:
    """Create visualizations based on recommendations."""
    try:
        processed_data = state.get("processed_data", state["data"])
        visualization_recommendation = state["visualization_recommendation"]

        visualization_specs = visualization_recommendation["specs"]

        # Generate visualizations - use func directly to avoid using invoke
        visualizations_json = generate_visualizations.func(
            json.dumps(visualization_specs),
            processed_data.to_json()
        )

        visualizations = json.loads(visualizations_json)

        return {
            **state,
            "generated_visualizations": visualizations
        }
    except Exception as e:
        return {
            **state,
            "generated_visualizations": [],
            "error": f"Error creating visualizations: {str(e)}"
        }


def generate_report(state: AgentState) -> AgentState:
    """Generate a comprehensive data analysis report."""
    data_summary = state["data_summary"]
    preprocessing_recommendation = state["preprocessing_recommendation"]
    visualization_recommendation = state["visualization_recommendation"]
    generated_visualizations = state.get("generated_visualizations", [])

    # Create a prompt template for report generation to work with LM Studio
    prompt_template = """You are a data science report writer. Generate a comprehensive report based on the data analysis performed.
    Include:
    1. Executive summary
    2. Data overview
    3. Data preprocessing recommendations with justification
    4. Visualization insights
    5. Conclusions and next steps

    Format your report in Markdown.

    Data Summary: {data_summary}

    Preprocessing Recommendations:
    {preprocessing_explanation}

    Visualization Recommendations:
    {visualization_explanation}

    Number of Visualizations Generated: {num_visualizations}

    Visualization Types: {visualization_types}
    """

    # Format the prompt with actual data
    formatted_prompt = prompt_template.format(
        data_summary=json.dumps(data_summary, indent=2),
        preprocessing_explanation=preprocessing_recommendation['explanation'],
        visualization_explanation=visualization_recommendation['explanation'],
        num_visualizations=len(generated_visualizations),
        visualization_types=[viz['type'] for viz in generated_visualizations]
    )

    # Use the LLM to generate the report
    report = llm.invoke(formatted_prompt)

    return {
        **state,
        "final_report": report
    }


def format_response(state: AgentState) -> Dict[str, Any]:
    """Format the final response."""
    return {
        "data_summary": state["data_summary"],
        "preprocessing_recommendation": state["preprocessing_recommendation"],
        "visualization_recommendation": state["visualization_recommendation"],
        "generated_visualizations": state.get("generated_visualizations", []),
        "final_report": state.get("final_report", "")
    }


# Build the graph
def build_graph():
    workflow = StateGraph(AgentState)

    # Add nodes
    workflow.add_node("load_and_analyze_data", load_and_analyze_data)
    workflow.add_node("recommend_preprocessing", recommend_preprocessing)
    workflow.add_node("recommend_visualizations", recommend_visualizations)
    workflow.add_node("apply_preprocessing", apply_preprocessing)
    workflow.add_node("create_visualizations", create_visualizations)
    workflow.add_node("generate_report", generate_report)
    workflow.add_node("format_response", format_response)

    # Add edges
    workflow.add_edge("load_and_analyze_data", "recommend_preprocessing")
    workflow.add_edge("recommend_preprocessing", "recommend_visualizations")
    workflow.add_edge("recommend_visualizations", "apply_preprocessing")
    workflow.add_edge("apply_preprocessing", "create_visualizations")
    workflow.add_edge("create_visualizations", "generate_report")
    workflow.add_edge("generate_report", "format_response")
    workflow.add_edge("format_response", END)

    # Set the entry point
    workflow.set_entry_point("load_and_analyze_data")

    return workflow


# Main function to create an interface for the LangGraph agent
def process_data_with_langgraph(user_input: str) -> Dict[str, Any]:
    """Process data using the LangGraph agent."""
    # Build the graph
    workflow = build_graph()

    # Compile the graph
    app = workflow.compile()

    # Initialize the state
    initial_state = {"user_input": user_input}

    # Run the workflow
    result = app.invoke(initial_state)

    return result


# Example usage
if __name__ == "__main__":
    # Example input
    user_input = """
    [
        {"age": 25, "income": 50000, "education": "Bachelor", "job_satisfaction": 7},
        {"age": 30, "income": 65000, "education": "Master", "job_satisfaction": 8},
        {"age": 22, "income": 35000, "education": "Bachelor", "job_satisfaction": 6},
        {"age": 40, "income": 90000, "education": "PhD", "job_satisfaction": 9},
        {"age": 35, "income": 75000, "education": "Master", "job_satisfaction": 7},
        {"age": 45, "income": 120000, "education": "PhD", "job_satisfaction": 8},
        {"age": 28, "income": 55000, "education": "Bachelor", "job_satisfaction": 6},
        {"age": 32, "income": 70000, "education": "Master", "job_satisfaction": 8},
        {"age": 50, "income": 110000, "education": "PhD", "job_satisfaction": 9},
        {"age": 27, "income": 52000, "education": "Bachelor", "job_satisfaction": 7}
    ]
    """

    # Process the data
    result = process_data_with_langgraph(user_input)

    # Print the result
    print("Data Analysis Complete!")
    print("Report:")
    print(result["final_report"])
