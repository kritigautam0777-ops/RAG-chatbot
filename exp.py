from dotenv import load_dotenv
from langsmith import evaluate, Client
from langraph_rag_backend import chatbot

load_dotenv()

client = Client()
dataset_name = "langgraph-eval-project-new" 

# -------------------
# 1. Define your agent function
# -------------------
def run_agent(inputs: dict) -> dict:
    thread_id = f"eval-{inputs['inputs_1'][:20]}"
    config = {
        "configurable": {"thread_id": thread_id},
        "run_name": "eval-run",
        "tags": ["evaluation"]
    }
    result = chatbot.invoke(
        {"messages": [inputs["inputs_1"]]},
        config=config
    )
    last_message = result["messages"][-1]
    return {"answer": last_message.content}


# -------------------
# 2. Define evaluators
# -------------------
def is_non_empty(outputs: dict, reference_outputs: dict) -> bool:
    """Check that the agent returned a non-empty answer."""
    return len(outputs.get("answer", "").strip()) > 10


def exact_match(outputs: dict, reference_outputs: dict) -> bool:
    """Check if output matches expected answer exactly."""
    return outputs.get("answer", "").strip() == reference_outputs.get("outputs_1", "").strip()


# -------------------
# 3. Run the experiment
# -------------------
evaluate(
    run_agent,
    data=dataset_name,
    evaluators=[is_non_empty, exact_match],
    experiment_prefix="langgraph-eval-experiment-v1",
    metadata={"model": "llama-3.1-8b-instant", "version": "v1"}
)

print("✅ Experiment complete! Check LangSmith UI for results.")