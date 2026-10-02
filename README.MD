# SMAT-God

SMAT-God is a multi-agent framework for detection of code smell of AI-generated Code. The system integrates a central decision agent with several specialized sub-agents, each responsible for a distinct aspect of the reasoning and evaluation process. This repository provides the implementation of the agents, the supporting utility modules, and the dataset used in our experiments.

## Repository Structure

### Core Decision Module

- [`decision_agent.py`](./decision_agent.py)  
  Implements the final decision agent. This module aggregates the outputs of the sub-agents and produces the ultimate decision of the system.

### Sub-Agents

- [`metrics_agent.py`](./metrics_agent.py)  
  Implements the metrics sub-agent, which is responsible for computing and evaluating quantitative metrics.

- [`responsibility_agent.py`](./responsibility_agent.py)  
  Implements the responsibility sub-agent, which analyzes and assigns responsibility over the relevant components or entities.

- [`semantic_agent.py`](./semantic_agent.py)  
  Implements the semantic sub-agent, which handles semantic interpretation and semantic-level reasoning.

### Supporting Modules

- [`get_teacher_reason.py`](./get_teacher_reason.py)  
  Provides utilities for obtaining teacher-side reasoning, e.g., reference rationales or explanations used to guide the agents.

- [`process_output_utils.py`](./process_output_utils.py)  
  Contains helper functions for post-processing, formatting, and managing the outputs of the agents.

- [`semantic_decision_combinations.py`](./semantic_decision_combinations.py)  
  Defines the logic for combining semantic analysis with decision outputs, enabling joint reasoning across the semantic and decision layers.

- [`test.py`](./test.py)  
  A testing script for validating agent behavior and running preliminary experiments.

### Data

- [`dataset/`](./dataset)  
  Contains the datasets used by the agents for evaluation and experimentation.

### Auxiliary Files

- [`.gitattributes`](./.gitattributes)  
  Git attributes configuration.

- [`results.jsonl`](./results.jsonl)  
  A sample result file in JSON Lines format, storing agent outputs or experimental results.