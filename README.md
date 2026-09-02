# Agentic Reinforcement Learning - Final Project

Build a complete agentic AI system using supervised fine-tuning (SFT), agent trajectory generation, and Direct Preference Optimization (DPO) to predict successful conference meeting outcomes.

## Main Improvements from the Starter Code

* **Cleaned and simplified the SFT target space** by clustering labels and standardizing probabilities.
* **Improved autoregressive training** by predicting the reason before the probability.
* **Expanded the dataset safely** with Person A/B swapping while preventing train/validation leakage.
* **Used completion-only supervision** so the model learns only from the JSON answer.
* **Made generation more stable and structured** with deterministic decoding and JSON-aware stopping.
* **Strengthened evaluation and diagnostics** for both SFT and DPO.
* **Improved trace generation** with multiple personas, tool use, and structured rewards.
* **Built higher-quality DPO preference pairs** using same-scenario comparisons and meaningful reward gaps.
* **Improved DPO training and evaluation** with better adapter loading, dynamic training length, and balanced testing.


## Project Summary

### Supervised Fine-Tuning

* Base model: `google/gemma-3-270m-it`
* LoRA: `r=16`, `alpha=32`, `dropout=0.05`
* Learning rate: `1e-5`
* Completion-only training on structured JSON outputs
* Early stopping with best checkpoint at epoch 7
* Validation: MAE `0.2258`, correlation `0.056`, JSON validity `100%`

### Agent Traces

* 7 agent personas
* 30 scenarios per persona
* 210 total traces
* Saved to:
  `traces/agent_traces_20260902_102738.csv`

### Reward Strategy

* Correct + tools: `+1.0`
* Correct without tools: `+0.5`
* Incorrect + tools: `-0.5`
* Invalid/incomplete behavior: `-0.9` to `-1.0`

### DPO Preference Learning

* Same-scenario traces compared by reward
* Minimum reward gap: `0.5`
* Higher-reward trace = `chosen`
* Lower-reward trace = `rejected`
* 66 preference pairs created
* Mean reward gap: `1.5`

### DPO Fine-Tuning

* Base model: `Qwen/Qwen3-1.7B`
* LoRA: `r=8`, `alpha=16`, `dropout=0.05`
* Learning rate: `2e-6`
* DPO beta: `0.1`
* 99 training steps, about 3 dataset passes
* FP16 + gradient checkpointing for GPU efficiency

### Saved Models

```text
SFT adapter:
models/sft_prediction_model_gemma_270m/

DPO adapter:
qwen3-dpo-adapter-v1/
```


-------------------------------------


## Project Overview

In this capstone project, you will apply everything you've learned in the course to build an intelligent agent that:
1. Predicts the likelihood of successful meetings at a tech conference
2. Uses tools to gather contextual information about meeting participants
3. Generates reasoning traces that demonstrate agentic behavior
4. Gets aligned with human preferences through DPO training

The project simulates a real-world scenario where an AI agent helps conference organizers optimize networking by predicting which meetings are likely to be productive based on participant profiles, timing, and context.

---

## Getting Started

### Prerequisites

Before starting this project, you should have completed:
- Exercises 1-6 from the course
- Understanding of SFT, PEFT, LoRA, and DPO concepts
- Familiarity with agentic reasoning loops and tool-calling
- Basic knowledge of the `npcpy` library

### System Requirements

**Minimum:**
- Python 3.8+
- 16GB RAM
- 20GB free disk space
- CPU-based training (slow but functional)

**Recommended:**
- Python 3.9+
- 32GB RAM
- 30GB free disk space
- GPU with 8GB+ VRAM (NVIDIA CUDA-compatible)
- Google Colab or similar cloud platform as alternative

---

### Dependencies

Install all required packages:

```bash
pip install torch>=2.0.0
pip install transformers>=4.30.0
pip install peft>=0.4.0
pip install trl>=0.7.0
pip install datasets>=2.12.0
pip install pandas>=1.5.0
pip install numpy>=1.24.0
pip install npcpy  # Agent framework
```

**For GPU acceleration (recommended):**
```bash
pip install accelerate bitsandbytes
```

**Verify installation:**
```python
import torch
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
```

---

### Installation

**Step 1: Clone or Download the Project**

Ensure you have the project directory with these files:
```
project/
├── README.md (this file)
├── starter_sft.py
├── starter_agentic_traces.py
├── starter_agentic_rlft.py
├── data_classes.py
└── data/
    └── sft_training_data.csv
```

**Step 2: Review the Data**

Examine the training data structure:
```bash
head -n 5 data/sft_training_data.csv
```

The CSV contains:
- `input_text`: Description of two conference attendees and meeting context
- `target_probability`: Target meeting success probability (0.0-1.0)
- `target_reason`: Categorical reason for prediction
- `ground_truth_prob`: Actual ground truth probability

**Step 3: Understand the Domain**

This project simulates a **conference networking assistant**:
- **Person A & B**: Attendees with different roles (founder, engineer, investor, etc.)
- **Context**: Time of day, day of conference, previous meeting fatigue
- **Goal**: Predict if their meeting will be successful
- **Success factors**: Engagement levels, seniority match, energy levels, topic alignment

---

## Testing

### Manual Testing

Test your models at each phase:

**Test each phase as you complete it:**

**Phase 1 - SFT Model:**
After completing `starter_sft.py`, run:
```bash
python starter_sft.py
```
This will train your SFT model and save it to the models directory.

**Phase 2 - Agent Traces:**
After implementing the agent loop in `starter_agentic_traces.py`, run:
```bash
python starter_agentic_traces.py
```
This will generate agent trajectories and save them for DPO training.

**Phase 3 - DPO Alignment:**
After completing `starter_agentic_rlft.py`, run:
```bash
python starter_agentic_rlft.py
```
This will perform DPO training using your generated trajectories.

---

#### Validation Checkpoints

**After Phase 1 (SFT):**
- Check that `models/sft_prediction_model_gemma_270m/` directory was created
- Verify training loss decreased during training
- Confirm model outputs valid probabilities (0.0-1.0)
- Test on a few manual examples to ensure reasonable predictions

**After Phase 2 (Agent Traces):**
- Verify trajectory files were saved with tool call information
- Check that agent called 2-5 tools per trajectory
- Confirm rewards were calculated and saved
- Ensure diverse scenarios were generated (mix of success/failure)

**After Phase 3 (DPO):**
- Check that DPO adapter was saved successfully
- Verify DPO loss decreased during training
- Compare model performance before and after DPO alignment
- Validate that aligned model shows improved reasoning quality

---

## Project Instructions

This section contains all student deliverables for this project.

### Core Deliverables

Complete the project in three phases:

---

### Phase 1: Supervised Fine-Tuning (starter_sft.py)

**Objective:** Train a baseline model that can predict meeting success probability.

**Tasks:**

1. **Configure SFT Hyperparameters** (Lines 36-50)
   - Set `lora_r`: Recommended 8-16
   - Set `lora_alpha`: Typically 2x the rank
   - Set `lora_dropout`: 0.1-0.15
   - Set `num_train_epochs`: 10-30
   - Set `learning_rate`: 1e-5 to 5e-5
   - Set `weight_decay`: 0.01-0.1

2. **Load and Format Training Data**
   - Read `data/sft_training_data.csv`
   - Format each row as a prompt-completion pair
   - Create structured input for the model

3. **Implement Training Pipeline**
   - Set up LoRA configuration
   - Initialize SFT trainer
   - Run training loop
   - Monitor loss values

4. **Evaluate Model**
   - Test on held-out examples
   - Calculate prediction accuracy
   - Analyze error patterns

**Success Criteria:**
- Training loss decreases steadily
- Model achieves >65% accuracy on validation set
- Model outputs valid probability values (0.0-1.0)

---

### Phase 2: Agent Trajectory Generation (starter_agentic_traces.py)

**Objective:** Generate reasoning traces where an agent uses tools to gather information and make predictions.

**Tasks:**

1. **Understand Available Tools**
   - Review the `TOOLS` list (provided)
   - Understand what each tool does for gathering meeting context

2. **Implement Agent Reasoning Loop**
   - Create an agent that observes, thinks, and acts
   - Implement tool selection logic
   - Record complete reasoning traces

3. **Generate Diverse Trajectories**
   - Run agent on multiple scenarios
   - Vary tool usage patterns
   - Create both successful and failed traces

4. **Calculate Rewards**
   - Use the provided `calculate_reward()` function
   - Reward accurate predictions and proper tool usage

**Success Criteria:**
- Generate 50+ diverse agent trajectories
- Trajectories include 2-5 tool calls each
- Agent achieves >60% accuracy on predictions

---

### Phase 3: DPO Alignment (starter_agentic_rlft.py)

**Objective:** Use Direct Preference Optimization to improve agent behavior.

**Tasks:**

1. **Create Preference Pairs**
   - Rank trajectories by reward score
   - Create (chosen, rejected) pairs with clear quality differences

2. **Configure DPO Training**
   - Set `beta`: 0.1-0.5 (KL penalty)
   - Set `learning_rate`: 1e-6 to 5e-6
   - Set `max_steps`: 20-100

3. **Run DPO Training**
   - Train on preference pairs
   - Monitor DPO loss
   - Save aligned model

4. **Evaluate Alignment**
   - Compare base vs DPO-aligned model
   - Measure accuracy improvements

**Success Criteria:**
- DPO loss decreases during training
- Aligned model shows >70% accuracy
- Demonstrates better reasoning quality

---

### Required Submissions

Submit the following:

**1. Trained Models**
- `models/sft_prediction_model_gemma_270m/` - SFT checkpoint
- `models/dpo_aligned_adapter/` - DPO adapter weights

**2. Generated Data**
- `trajectories/agent_traces.json` - All agent trajectories
- `trajectories/preference_pairs.json` - DPO training pairs

**3. Code Files**
- Completed `starter_sft.py` with all TODOs filled
- Completed `starter_agentic_traces.py` with agent implementation
- Completed `starter_agentic_rlft.py` with DPO training

**4. Evaluation Report**
- `results/evaluation_report.md` containing:
  - SFT model accuracy and analysis
  - Agent trajectory examples and statistics
  - DPO alignment improvements
  - Comparison of all three phases
  - Lessons learned and challenges

---

## Built With

* [PyTorch](https://pytorch.org/) - Deep learning framework
* [Transformers](https://huggingface.co/docs/transformers) - Pre-trained model library
* [PEFT](https://huggingface.co/docs/peft) - Parameter-efficient fine-tuning
* [TRL](https://huggingface.co/docs/trl) - Transformer reinforcement learning
* [npcpy](https://github.com/anthropics/npcpy) - Agent framework for tool-calling
* [Gemma](https://huggingface.co/google/gemma-2-2b) - Base language model

---

## Common Issues and Solutions

**Issue: Out of Memory During Training**
- Reduce `per_device_train_batch_size` to 1
- Increase `gradient_accumulation_steps` to 8
- Use smaller model

**Issue: SFT Loss Not Decreasing**
- Increase learning rate to 3e-5
- Check data formatting
- Verify data loader is working

**Issue: Agent Makes Random Tool Calls**
- Add clear decision logic for tool selection
- Implement stopping conditions
- Review system prompts

**Issue: DPO Training Unstable**
- Increase `beta` to 0.3-0.5
- Decrease learning rate to 1e-6
- Use clearer preference pairs

---

## Evaluation Criteria

Your project will be assessed on:

**Technical Implementation (40%)**
- Code runs without errors
- All TODOs completed
- Proper hyperparameter configuration

**Model Performance (30%)**
- SFT model: >65% accuracy
- Agent trajectories: >60% accuracy
- DPO model: >70% accuracy

**Reasoning Quality (20%)**
- Agent uses tools appropriately
- Clear reasoning chains
- Quality preference pairs

**Analysis and Documentation (10%)**
- Thorough evaluation report
- Insightful analysis
- Clear design choices

---

## Tips for Success

1. Start Simple - Get Phase 1 working before Phase 2
2. Test Incrementally - Test each function as you write it
3. Monitor Training - Watch loss curves
4. Save Checkpoints - Save after each phase
5. Document Everything - Keep notes on hyperparameters
6. Analyze Failures - Understand why predictions fail
7. Iterate - Try different configurations

---

## License

[License](../LICENSE.md)
