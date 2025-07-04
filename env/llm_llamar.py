'''
Baseline : Llamar
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

def get_args():
    pass

def get_llm_response(env: AI2ThorEnv, prompt: str, model: str = "gpt-4o-mini", temperature: float = 0.7) -> str:
    pass

def prepare_prompt():
    pass

def prepare_payload():
    pass    

def process_action_llm_output():
    pass

def print_relevant_info():
    pass


def run_main():
    args = get_args()
    pass

if __name__ == "__main__":
    run_main()