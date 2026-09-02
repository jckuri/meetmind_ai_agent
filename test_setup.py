#!/usr/bin/env python3
"""
Test Setup Script for Agentic RL Project
Run this to verify your environment is configured correctly before starting the project.
"""

print("="*60)
print("Testing Agentic RL Project Setup")
print("="*60)

# Test 1: Python packages
print("\n[1/5] Checking Python packages...")
try:
    import torch
    import transformers
    import peft
    import trl
    import datasets
    import pandas
    import npcpy
    print("✓ All required packages installed")
    print(f"  - PyTorch: {torch.__version__}")
    print(f"  - Transformers: {transformers.__version__}")
    print(f"  - PEFT: {peft.__version__}")
    print(f"  - TRL: {trl.__version__}")
except ImportError as e:
    print(f"✗ Missing package: {e}")
    print("  Run: pip install torch transformers peft trl datasets pandas npcpy")
    exit(1)

# Test 2: GPU/Device availability
print("\n[2/5] Checking compute device...")
if torch.cuda.is_available():
    print(f"✓ CUDA available - GPU: {torch.cuda.get_device_name(0)}")
elif torch.backends.mps.is_available():
    print("✓ Apple Silicon (MPS) available")
else:
    print("⚠ CPU only (training will be slow)")

# Test 3: HuggingFace authentication
print("\n[3/5] Checking HuggingFace authentication...")
try:
    from huggingface_hub import HfFolder
    token = HfFolder.get_token()
    if token:
        print("✓ HuggingFace token found")
        # Try to access Gemma model
        from transformers import AutoTokenizer
        print("  Testing Gemma model access...")
        tokenizer = AutoTokenizer.from_pretrained("google/gemma-3-270m-it", trust_remote_code=True)
        print("✓ Gemma model accessible")
    else:
        print("✗ No HuggingFace token found")
        print("  Run: huggingface-cli login")
        exit(1)
except Exception as e:
    print(f"✗ HuggingFace authentication failed: {e}")
    print("  Steps to fix:")
    print("  1. Visit https://huggingface.co/google/gemma-3-270m-it")
    print("  2. Click 'Agree and access repository'")
    print("  3. Create a fine-grained token with 'Access public gated repositories'")
    print("  4. Run: huggingface-cli login")
    exit(1)

# Test 4: Ollama
print("\n[4/5] Checking Ollama installation...")
import subprocess
try:
    result = subprocess.run(['ollama', '--version'],
                          capture_output=True,
                          text=True,
                          timeout=5)
    if result.returncode == 0:
        print(f"✓ Ollama installed: {result.stdout.strip()}")

        # Check for required models
        result = subprocess.run(['ollama', 'list'],
                              capture_output=True,
                              text=True,
                              timeout=5)
        models = result.stdout

        if 'qwen3:0.6b' in models:
            print("  ✓ qwen3:0.6b available")
        else:
            print("  ✗ qwen3:0.6b not found - Run: ollama pull qwen3:0.6b")

        if 'qwen3:1.7b' in models:
            print("  ✓ qwen3:1.7b available")
        else:
            print("  ✗ qwen3:1.7b not found - Run: ollama pull qwen3:1.7b")
    else:
        print("✗ Ollama not working correctly")
        exit(1)
except FileNotFoundError:
    print("✗ Ollama not installed")
    print("  Install from: https://ollama.com")
    exit(1)
except Exception as e:
    print(f"⚠ Could not verify Ollama: {e}")

# Test 5: Project files
print("\n[5/5] Checking project files...")
import os
required_files = [
    'data_classes.py',
    'starter_sft.py',
    'starter_agentic_traces.py',
    'starter_agentic_rlft.py',
    'data/sft_training_data.csv'
]

all_present = True
for file in required_files:
    if os.path.exists(file):
        print(f"  ✓ {file}")
    else:
        print(f"  ✗ {file} missing")
        all_present = False

if not all_present:
    print("\n✗ Some project files are missing")
    exit(1)

# Test 6: Import project modules
print("\n[6/6] Testing project imports...")
try:
    from data_classes import TimeSlot, ConferenceSimulator, PersonDescriptor, MeetingPredictor
    print("✓ data_classes imports successfully")
except Exception as e:
    print(f"✗ Failed to import data_classes: {e}")
    exit(1)

print("\n" + "="*60)
print("✓ Setup verification complete!")
print("="*60)
print("\nYou're ready to start the project!")
print("\nNext steps:")
print("1. Open starter_sft.py")
print("2. Fill in the hyperparameters marked 'YOUR CODE HERE'")
print("3. Run: python starter_sft.py")
print("\nSee the project instructions for detailed guidance.")
