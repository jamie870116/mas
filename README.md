# MS project
Aiming to implement a decentralized, LLM-driven multi-robot task planning and coordination system that is tolerant of unexpected events during execution.



## TODO
- [x] Built env with ai2thor    
- [-] Setting up LLM workflow
- [ ] Set up Baselines: Centraulized (One time planning, Replanning loop)
- [ ] Test cases
- [ ] Decentualized planning
- [ ] Experiment

## Setup

### Env
I ran on conda environment with python=3.10.
Other dependecies are shown in `requirements.txt`


### Test
- config file path: `config/config.json`
- log file path: `logs/{task_name}`
- LLM api key file: `api_ley.txt` in root folder

- test env:
```python
python {path}/test_env_b.py
```