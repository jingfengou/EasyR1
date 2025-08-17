import pandas as pd
from typing import Any, Dict, List

def compute_score(
    reward_inputs: List[Dict[str, Any]],
) -> List[Dict[str, float]]:
    """
    A placeholder reward function for the Visual-ARFT task.

    This function simply returns a fixed score of 0.0 for all generations.
    Its purpose is to allow the training pipeline to run without crashing.
    """
    print("--- Visual-ARFT Placeholder Reward Function Called ---")
    
    # Extract the generations from the input list of dictionaries
    generations = [item["response"] for item in reward_inputs]
    num_generations = len(generations)
    
    # Create a placeholder score for each generation
    scores = [0.0] * num_generations
    
    print(f"Processed {num_generations} generations, returning a score of 0.0 for each.")

    # The reward function is expected to return a list of dictionaries
    # with the scores for each generation. The framework expects a key named "overall".
    return [{"overall": score} for score in scores]

