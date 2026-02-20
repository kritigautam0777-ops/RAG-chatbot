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


def semantic_similarity(outputs: dict, reference_outputs: dict) -> bool:
    """Pass if answer covers the same meaning, not word for word."""
    answer = outputs.get("answer", "").lower()
    expected = reference_outputs.get("outputs_1", "").lower()
    
    key_words = [w for w in expected.split() if len(w) > 4]
    matches = sum(1 for w in key_words if w in answer)
    return matches / len(key_words) > 0.5


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