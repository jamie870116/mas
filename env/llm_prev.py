'''
Baseline: SmartLLM + Previous course project - 一次性生成plan, 並執行; 驗證結果+重新生成plan 
'''
import json
import os
import sys
import argparse
from pathlib import Path
from datetime import datetime
import random
import subprocess
import time

from openai import OpenAI
from env.env_b import AI2ThorEnv
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

client = OpenAI(api_key=Path('api_key.txt').read_text())

def get_llm_response(env: AI2ThorEnv, prompt: str, model: str = "gpt-4o-mini", temperature: float = 0.7, max_tokens=1000, frequency_penalty=0.0) -> str:
    response = client.chat.completions.create(model=model, 
                                            messages=prompt, 
                                            max_tokens=max_tokens, 
                                            temperature=temperature, 
                                            frequency_penalty = frequency_penalty)
    print(f"LLM response: {response}")
    return response, response.choices[0].message.content.strip()

def decompose_task(test_tasks, prompt, llm_model, llama_version):
    pass

def allocate_robots(decomposed_plan, prompt, available_robots, objects_ai, llm_model, llama_version):
    pass

def generate_code(decomposed_plan, allocated_plan, prompt, available_robots, llm_model, llama_version):
    pass

def compose_code_plan():
    pass

def verify_results():
    pass

def replan():
    pass

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--floor-plan", type=int, required=True)
    parser.add_argument("--openai-api-key-file", type=str, default="api_key")
    parser.add_argument("--gemini-api-key-file", type=str, default="gemini_api_key")

    parser.add_argument("--llm-model", type=str, default="gpt", 
                        choices=['gpt', 'gemini'])
    parser.add_argument("--gemini-model", type=str, default="gemini-2.0-flash", 
                        choices=['gemini-1.5-flash', 'gemini-2.0-flash', 'gemini-1.5-pro', 'gemini-2.0-flash-lite-preview-02-05'])
    parser.add_argument("--gpt-version", type=str, default="gpt-4", 
                        choices=['gpt-3.5-turbo', 'gpt-4o-mini', 'gpt-4', 'gpt-3.5-turbo-16k'])

    parser.add_argument("--prompt-decompse-set", type=str, default="train_task_decompose", 
                        choices=['train_task_decompose'])

    parser.add_argument("--prompt-allocation-set", type=str, default="train_task_allocation", 
                        choices=['train_task_allocation'])

    parser.add_argument("--test-set", type=str, default="tests", 
                        choices=['final_test', 'tests'])

    parser.add_argument("--log-results", type=bool, default=True)

    return parser.parse_args()

def run_main():
    args = get_args()
    pass

if __name__ == "__main__":
    run_main()
