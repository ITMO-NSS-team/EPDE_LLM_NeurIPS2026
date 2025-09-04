# EPDE-LLM: Equation Discovery with Large Language Models

We formulate differential equation discovery as a code-generation task for Large Language Models (LLMs). This repository introduces a compact, physics-preserving textual representation for fields and their derivatives, and integrates LLMs as oracles within an Evolutionary PDE Discovery (EPDE) meta-learning loop.

## Overview

Discovering the underlying equations from observed data is a fundamental scientific challenge. Traditional methods can be computationally expensive or require significant expert input. EPDE-LLM combines the power of evolutionary search (via the EPDE framework) with the symbolic reasoning and code-generation capabilities of modern LLMs to efficiently and accurately identify governing equations.

## Features

*   **LLM as an Oracle:** Uses an LLM to propose physically plausible candidate terms for the PDE within an evolutionary algorithm.
*   **Physics-Preserving Representation:** A novel text-based representation for encoding field data and derivatives that guides the LLM towards physically meaningful solutions.
*   **Meta-Learning Loop:** Tight integration with the EPDE framework for robust and efficient equation discovery.
*   **Extensible:** Designed to work with various open-source and proprietary LLMs (e.g., GPT, LLaMA, Claude).

## Prerequisites

*   Python 3.11
*   Conda (recommended) or pip
*   An API key for the LLM service you wish to use (e.g., OpenAI, Anthropic) - *if using a proprietary model*

## Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/ITMO-NSS-team/EPDE_LLM.git
    ```

2.  **Create and activate a Conda environment:**
    ```bash
    conda create -n epde_llm python=3.11 -y
    conda activate epde_llm
    ```

3.  **Install the required dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Set your LLM API key:**
   
    Create creds.py file with API key in it. Use creds_example.py as an example.

## Basic usage

The following example shows how to use EPDE-LLM.

### Step 1: Generate data
First, generate the necessary data for the LLM to process:
```bash
cd data
python data_gen.py
```
### Step 2: Choose your approach

There are three main approaches to experiment with:

#### Option 1. LLM as equation discovery tool:

Uses generative symbolic reasoning where the LLM directly proposes equation structures based on the data representation.

```bash
cd pipeline
python pipeline_main.py
```

#### Option 2. EPDE baseline experiments:

The EPDE framework optimizes equation structures through evolutionary principles, treating each equation as an individual subject to mutation and crossover.

```bash
cd epde_experiments
python burgers.py 
# Other examples: kdv_sindy.py, wave.py, etc.
```

#### Option 3. EPDE-LLM framework experiments:

A hybrid approach where the LLM first generates an initial population of candidate equations, which is then refined by the EPDE algorithm using its evolutionary operations for optimal convergence.

```bash
cd epde_llm
python epde_llm_main.py
```

## License
This project is licensed under the MIT License - see the LICENSE file for details.
