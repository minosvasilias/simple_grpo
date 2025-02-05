#!/usr/bin/env python
import re
import json

from datasets import load_dataset, Dataset
from transformers import AutoTokenizer

from trl import GRPOConfig, GRPOTrainer, ModelConfig, TrlParser



SYSTEM_PROMPT = """Reason about the user request within <|think|> tokens, and only afterwards write your final response.
The expected format for your response is as follows:

<|think|>
Reasoning
<|think|>
Answer
"""


# Extracts thinking content and final model answer
def parse_reasoning_response(text: str) -> dict:
    pattern = r"<\|think\|>\s*(.*?)\s*<\|think\|>\s*(.*)"
    match = re.search(pattern, text, re.DOTALL)
    if not match:
        return {"thinking_content": "", "response": text}
    return {"thinking_content": match.group(1).strip(), "response": match.group(2).strip()}


def get_completion_content(completion: dict) -> str:
    return completion[0]["content"]


def parse_responses(completions: list[dict]) -> list[dict]:
    return [parse_reasoning_response(get_completion_content(c)) for c in completions]

# Score accuracy of answer and log rewards
def accuracy_reward_log(prompts, completions, answer, **kwargs):
    log_rewards(prompts, completions, answer)
    return accuracy_reward(prompts, completions, answer)

# Score accuracy of answer
def accuracy_reward(prompts, completions, answer, **kwargs) -> list[float]:
    parsed_responses = parse_responses(completions)
    rewards = [2.0 if r["response"] == a else 0.0 for r, a in zip(parsed_responses, answer)]
    return rewards

# Score formatting of number (integer expected)
def format_number_reward(prompts, completions, answer, **kwargs) -> list[float]:
    parsed_responses = parse_responses(completions)
    rewards = [0.5 if r["response"].isdigit() else 0.0 for r in parsed_responses]
    return rewards

# Score formatting of reasoning content
def format_reasoning_reward(prompts, completions, answer, **kwargs) -> list[float]:
    parsed_responses = parse_responses(completions)
    rewards = [0.5 if r["thinking_content"] and r["response"] else 0.0 for r in parsed_responses]
    return rewards


# Log rewards and example responses
def log_rewards(prompts, completions, answer, **kwargs):
    rewards = {
        "accuracy": accuracy_reward(prompts, completions, answer),
        "format_number": format_number_reward(prompts, completions, answer),
        "format_reasoning": format_reasoning_reward(prompts, completions, answer),
    }
    example_response = get_completion_content(completions[0])
    example_parsed = parse_reasoning_response(example_response)
    example_answer = answer[0]
    example_prompt = prompts[0][-1]['content']
    print(
        f"-" * 50
        + f"\nExample prompt:\n{example_prompt}\n"
        + f"-" * 10
        + f"\nExample response:\n{example_response}\n"
        + f"-" * 10
        + f"\nExample answer:\n{example_answer}\n"
        + f"-" * 10
        + f"\nExample Correct?: {example_parsed['response'] == example_answer}\n"
        + f"-" * 10
        + f"\nRewards:\n{json.dumps(rewards, indent=2)}"
    )


# Initialize reward functions
reward_funcs = [
    format_reasoning_reward,
    format_number_reward,
    accuracy_reward_log,
]


# Extracts gsm8k answers
def extract_hash_answer(text: str) -> str | None:
    if "####" not in text:
        return None
    return text.split("####")[1].strip()


# Loads GSM8K dataset
def load_data(split="train") -> Dataset:
    data = load_dataset("openai/gsm8k", "main")[split]
    data = data.map(
        lambda x: {
            "prompt": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": x["question"] + "\nAnswer with a number only."},
            ],
            "answer": extract_hash_answer(x["answer"]),
        }
    )
    return data


def main(training_args, model_args):
    # Load the GSM8K dataset.
    data = load_data()

    # Initialize tokenizer (adds pad token).
    tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path)
    tokenizer.pad_token = tokenizer.eos_token

    # Initialize the GRPO trainer with the new dataset and reward functions.
    trainer = GRPOTrainer(
        model=model_args.model_name_or_path,
        processing_class=tokenizer,
        reward_funcs=reward_funcs,
        args=training_args,
        train_dataset=data,
    )

    # Train the model and save/push checkpoints.
    trainer.train()
    trainer.save_model(training_args.output_dir)

if __name__ == "__main__":
    parser = TrlParser((GRPOConfig, ModelConfig))
    training_args, model_args = parser.parse_args_and_config()
    main(training_args, model_args)
