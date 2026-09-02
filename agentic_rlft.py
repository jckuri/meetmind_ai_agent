# AGENTIC RL Fine Tuning with DPO
from datasets import Dataset
import glob
import inspect
import json
import math
import os
import random
import re
import time
from typing import Any, Dict, List, Optional

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import numpy as np
import pandas as pd
import torch
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer

from agentic_traces import AgentToolLoop, TOOLS, TRACE_SCHEMA_VERSION, calculate_reward, system_prompt_configurations
from data_classes import ConferenceSimulator, PersonDescriptor, TimeSlot, calculate_accuracy_metrics, get_true_outcome


def _parse_bool(value) -> bool:
    return str(value).strip().lower() in {"true", "1", "yes", "y"}


def _parse_json_object(value):
    if isinstance(value, dict):
        return value
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None
    try:
        parsed = json.loads(str(value))
        return parsed if isinstance(parsed, dict) else None
    except Exception:
        return None


def _row_to_trace(row: pd.Series) -> Dict[str, Any]:
    tools_used = [tool for tool in str(row.get("tools_used", "")).split(",") if tool]
    return {"final_recommendation_parsed": _parse_json_object(row.get("final_recommendation_parsed")), "tools_used": tools_used, "completed_naturally": _parse_bool(row.get("completed_naturally", False)), "ground_truth": float(row["ground_truth_prob"])}


def _completion_json(row: pd.Series) -> str:
    parsed = _parse_json_object(row.get("final_recommendation_parsed")) or {}
    recommendation = str(parsed.get("recommendation", row.get("agent_outcome", ""))).upper().strip()
    reasoning = str(parsed.get("reasoning", row.get("final_recommendation_reasoning", ""))).strip()
    return json.dumps({"recommendation": recommendation, "reasoning": reasoning}, ensure_ascii=False, separators=(",", ":"))


def create_preference_dataset_from_traces(csv_file_path: str, min_reward_gap: float = 0.5, min_pairs_warning: int = 20, max_pairs_per_scenario: int = 3) -> Optional[Dataset]:
    df = pd.read_csv(csv_file_path)
    required = {"trace_schema_version", "scenario_id", "initial_user_prompt", "ground_truth_prob", "tools_used", "completed_naturally", "final_recommendation_parsed"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Trace file is incompatible with the corrected pipeline. Missing columns: {missing}. Regenerate traces with the corrected agentic_traces.py.")
    versions = set(df["trace_schema_version"].dropna().astype(str))
    if versions != {TRACE_SCHEMA_VERSION}:
        raise ValueError(f"Trace schema mismatch: found {sorted(versions)}, expected only {TRACE_SCHEMA_VERSION}. Regenerate traces before DPO training.")
    df = df.dropna(subset=["scenario_id", "initial_user_prompt", "ground_truth_prob", "final_recommendation_parsed"]).copy()
    df["reward"] = df.apply(lambda row: calculate_reward(_row_to_trace(row)), axis=1)
    df = df[df["reward"] > -1.0].copy()
    if len(df) < 2:
        print("Not enough valid traces to create preference pairs.")
        return None
    filtered_pairs = []
    for _, group in df.groupby("scenario_id"):
        group = group.sort_values("reward", ascending=False).reset_index(drop=True)
        scenario_pairs = []
        for high_idx in range(len(group)):
            for low_idx in range(high_idx + 1, len(group)):
                high_trace, low_trace = group.iloc[high_idx], group.iloc[low_idx]
                reward_gap = float(high_trace["reward"]) - float(low_trace["reward"])
                if reward_gap < min_reward_gap:
                    continue
                chosen, rejected = _completion_json(high_trace), _completion_json(low_trace)
                if chosen == rejected:
                    continue
                chosen_rec = json.loads(chosen)["recommendation"]
                rejected_rec = json.loads(rejected)["recommendation"]
                scenario_pairs.append({"prompt": str(high_trace["initial_user_prompt"]).strip(), "chosen": chosen, "rejected": rejected, "chosen_reward": float(high_trace["reward"]), "rejected_reward": float(low_trace["reward"]), "reward_gap": reward_gap, "decision_diff": chosen_rec != rejected_rec})
        scenario_pairs.sort(key=lambda pair: (pair["decision_diff"], pair["reward_gap"]), reverse=True)
        filtered_pairs.extend(scenario_pairs[:max_pairs_per_scenario])
    if not filtered_pairs:
        print("No valid preference pairs created.")
        return None
    if len(filtered_pairs) < min_pairs_warning:
        print(f"Only {len(filtered_pairs)} preference pairs found. Generate more unique trace scenarios before increasing DPO steps.")
    print(f"Created {len(filtered_pairs)} preference pairs.")
    print("Mean reward gap:", float(np.mean([pair["reward_gap"] for pair in filtered_pairs])))
    print("Pairs with different YES/NO decisions:", sum(pair["decision_diff"] for pair in filtered_pairs))
    return Dataset.from_list([{"prompt": pair["prompt"], "chosen": pair["chosen"], "rejected": pair["rejected"]} for pair in filtered_pairs])


def train_model_with_dpo(csv_file_path: str, base_model_id: str, new_adapter_path: str):
    if not os.path.exists(csv_file_path):
        raise FileNotFoundError(f"CSV file not found at {csv_file_path}")
    print("\nStarting DPO Training Process")
    print(f"Loading traces from {csv_file_path}...")
    preference_dataset = create_preference_dataset_from_traces(csv_file_path)
    if preference_dataset is None or len(preference_dataset) == 0:
        print("No valid preference pairs created. Cannot proceed with DPO training.")
        return

    model = AutoModelForCausalLM.from_pretrained(base_model_id, torch_dtype=torch.float16, device_map="auto", low_cpu_mem_usage=True)
    model.config.use_cache = False
    model.enable_input_require_grads()
    model.gradient_checkpointing_enable()

    tokenizer = AutoTokenizer.from_pretrained(base_model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    peft_config = LoraConfig(r=8, lora_alpha=16, lora_dropout=0.05, bias="none", task_type="CAUSAL_LM", target_modules=["q_proj", "k_proj", "v_proj", "o_proj"])

    from trl import DPOConfig, DPOTrainer

    effective_batch_size = 2
    target_epochs = 3.0
    max_steps = min(150, max(30, math.ceil(len(preference_dataset) * target_epochs / effective_batch_size)))
    print(f"DPO preference pairs: {len(preference_dataset)}")
    print(f"DPO max_steps: {max_steps} (~{max_steps * effective_batch_size / len(preference_dataset):.2f} effective dataset passes)")

    training_args = DPOConfig(
        output_dir="./dpo_results",
        per_device_train_batch_size=1,
        gradient_accumulation_steps=2,
        learning_rate=2e-6,
        max_steps=max_steps,
        weight_decay=0.01,
        beta=0.1,
        logging_steps=2,
        save_steps=10,
        remove_unused_columns=False,
        max_length=768,
        max_prompt_length=512,
        dataloader_num_workers=0,
        fp16=True,
        bf16=False,
        gradient_checkpointing=True,
        optim="adamw_torch",
        warmup_steps=2,
        save_strategy="steps",
        save_total_limit=3
    )

    trainer = DPOTrainer(model, args=training_args, train_dataset=preference_dataset, peft_config=peft_config)
    print("Starting DPO training...")
    trainer.train()
    print("DPO training complete.")
    print(f"Saving new LoRA adapter to '{new_adapter_path}'...")
    trainer.save_model(new_adapter_path)
    print("Adapter saved successfully.")


def _python_type_to_json_schema(annotation):
    if annotation in (int,):
        return {"type": "integer"}
    if annotation in (float,):
        return {"type": "number"}
    if annotation in (bool,):
        return {"type": "boolean"}
    return {"type": "string"}


def _build_tool_schemas(tools):
    schemas = []
    for tool in tools:
        signature = inspect.signature(tool)
        properties, required = {}, []
        for name, parameter in signature.parameters.items():
            properties[name] = _python_type_to_json_schema(parameter.annotation)
            if parameter.default is inspect._empty:
                required.append(name)
        schemas.append({"type": "function", "function": {"name": tool.__name__, "description": (tool.__doc__ or "").strip(), "parameters": {"type": "object", "properties": properties, "required": required}}})
    return schemas


class CachedTransformersNPC:
    def __init__(self, name, primary_directive, tools, model, tokenizer, max_new_tokens=256):
        self.name = name
        self.primary_directive = primary_directive
        self.tools = tools
        self.model = model
        self.tokenizer = tokenizer
        self.max_new_tokens = max_new_tokens
        self.tool_map = {tool.__name__: tool for tool in tools}
        self.tool_schemas = _build_tool_schemas(tools)

    def _parse_tool_calls(self, text):
        calls = []
        for match in re.findall(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", text, flags=re.DOTALL):
            try:
                data = json.loads(match)
                name = data.get("name")
                arguments = data.get("arguments", {})
                if isinstance(arguments, str):
                    arguments = json.loads(arguments)
                if name in self.tool_map and isinstance(arguments, dict):
                    calls.append({"name": name, "arguments": arguments})
            except Exception:
                pass
        return calls

    def get_llm_response(self, prompt, messages=None, auto_process_tool_calls=True, **kwargs):
        conversation = list(messages or [])
        if not conversation:
            conversation = [{"role": "system", "content": self.primary_directive}]
        conversation.append({"role": "user", "content": prompt})
        template_kwargs = {"tokenize": False, "add_generation_prompt": True, "tools": self.tool_schemas}
        try:
            rendered = self.tokenizer.apply_chat_template(conversation, enable_thinking=False, **template_kwargs)
        except TypeError:
            rendered = self.tokenizer.apply_chat_template(conversation, **template_kwargs)
        inputs = self.tokenizer(rendered, return_tensors="pt").to(self.model.device)
        input_length = inputs["input_ids"].shape[1]
        with torch.inference_mode():
            generated = self.model.generate(**inputs, max_new_tokens=self.max_new_tokens, do_sample=False, use_cache=True, pad_token_id=self.tokenizer.pad_token_id, eos_token_id=self.tokenizer.eos_token_id)
        response_text = self.tokenizer.decode(generated[0, input_length:], skip_special_tokens=False).strip()
        tool_calls = self._parse_tool_calls(response_text)
        conversation.append({"role": "assistant", "content": response_text})
        returned_tool_calls = []
        if auto_process_tool_calls and tool_calls:
            for call in tool_calls:
                name, arguments = call["name"], call["arguments"]
                returned_tool_calls.append({"function": {"name": name, "arguments": json.dumps(arguments)}})
                try:
                    result = self.tool_map[name](**arguments)
                except Exception as error:
                    result = json.dumps({"status": "error", "message": str(error)})
                if not isinstance(result, str):
                    result = json.dumps(result)
                conversation.append({"role": "tool", "name": name, "content": result})
        return {"response": response_text, "messages": conversation, "tool_calls": returned_tool_calls}


def _load_evaluation_model(base_model_id: str, adapter_path: Optional[str] = None):
    from peft import PeftModel

    dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    print(f"\nLoading evaluation base model ONCE: {base_model_id}")
    model = AutoModelForCausalLM.from_pretrained(base_model_id, torch_dtype=dtype, device_map="auto", low_cpu_mem_usage=True, attn_implementation="eager")
    tokenizer = AutoTokenizer.from_pretrained(base_model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if adapter_path:
        print(f"Loading and merging DPO adapter ONCE: {adapter_path}")
        model = PeftModel.from_pretrained(model, adapter_path).merge_and_unload()
    model.eval()
    model.config.use_cache = True
    model.generation_config.do_sample = False
    model.generation_config.temperature = None
    model.generation_config.top_p = None
    model.generation_config.top_k = None
    return model, tokenizer


def run_local_agent_evaluation(base_model_id: str, test_scenarios: List[Dict], model_type: str, adapter_path: Optional[str] = None, max_iterations: int = 6) -> List[Dict]:
    print(f"\nRunning Full Local Evaluation for {model_type.upper()} Model")
    model, tokenizer = _load_evaluation_model(base_model_id, adapter_path)
    results = []
    try:
        for persona_idx, persona in enumerate(system_prompt_configurations):
            current_agent = CachedTransformersNPC(name=persona["name"].lower(), primary_directive=persona["primary_directive"], tools=TOOLS, model=model, tokenizer=tokenizer, max_new_tokens=256)
            for scenario in test_scenarios:
                scenario_id = f"{scenario['scenario_id']}_p{persona_idx}"
                print(f"Scenario {scenario_id} ({model_type}, {persona['name']})")
                true_outcome = get_true_outcome(scenario["ground_truth"])
                tool_loop = AgentToolLoop(current_agent, max_iterations=max_iterations)
                initial_prompt = f"""Your task is to decide if two people should meet. Use the available tools to gather information step-by-step.\n\nPerson A: {scenario['p1_desc']}\nPerson B: {scenario['p2_desc']}\nTime Slot: {scenario['ts_str']}\n\nBegin your analysis by calling a tool."""
                loop_result = tool_loop.run_tool_loop(initial_prompt)
                final_rec_data = loop_result["final_recommendation"]
                agent_outcome = "FAIL"
                if final_rec_data and "recommendation" in final_rec_data:
                    agent_outcome = str(final_rec_data["recommendation"]).upper().strip()
                is_correct = int(agent_outcome == true_outcome)
                results.append({"scenario_id": scenario_id, "persona": persona["name"], "is_correct": is_correct, "agent_outcome": agent_outcome, "true_outcome": true_outcome})
                print(f"Scenario {scenario_id} complete. GT={true_outcome}, Agent said={agent_outcome}, Correct={bool(is_correct)}")
    finally:
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return results


def _binary_metrics(results: List[Dict]) -> Dict[str, Any]:
    tp = sum(r["agent_outcome"] == "YES" and r["true_outcome"] == "YES" for r in results)
    tn = sum(r["agent_outcome"] == "NO" and r["true_outcome"] == "NO" for r in results)
    fp = sum(r["agent_outcome"] == "YES" and r["true_outcome"] == "NO" for r in results)
    fn = sum(r["agent_outcome"] == "NO" and r["true_outcome"] == "YES" for r in results)
    fail_yes = sum(r["agent_outcome"] not in {"YES", "NO"} and r["true_outcome"] == "YES" for r in results)
    fail_no = sum(r["agent_outcome"] not in {"YES", "NO"} and r["true_outcome"] == "NO" for r in results)
    yes_recall = tp / (tp + fn + fail_yes) if tp + fn + fail_yes else 0.0
    no_recall = tn / (tn + fp + fail_no) if tn + fp + fail_no else 0.0
    precision = tp / (tp + fp) if tp + fp else 0.0
    f1 = 2 * precision * yes_recall / (precision + yes_recall) if precision + yes_recall else 0.0
    return {"accuracy": (tp + tn) / len(results) * 100 if results else 0.0, "balanced_accuracy": (yes_recall + no_recall) / 2 * 100, "yes_recall": yes_recall * 100, "no_recall": no_recall * 100, "f1_yes": f1, "confusion_matrix": {"TP": tp, "TN": tn, "FP": fp, "FN": fn, "FAIL_YES": fail_yes, "FAIL_NO": fail_no}}


def _generate_balanced_scenarios(total_count: int, seed: int = 37) -> List[Dict]:
    if total_count < 2:
        raise ValueError("total_count must be at least 2.")

    target_yes = total_count // 2
    target_no = total_count - target_yes
    scenarios = []
    counts = {"YES": 0, "NO": 0}
    rng = random.Random(seed)
    descriptor = PersonDescriptor(temperature=0.8)
    attempts = 0
    max_attempts = 50000

    while counts["YES"] < target_yes or counts["NO"] < target_no:
        attempts += 1
        if attempts > max_attempts:
            raise RuntimeError(f"Could not create balanced evaluation set after {attempts} attempts: {counts}")

        simulator = ConferenceSimulator(num_attendees=50, seed=rng.randint(20000, 1000000))
        p1_id, p2_id = rng.sample(list(simulator.attendees.keys()), 2)
        person_a = simulator.attendees[p1_id]
        person_b = simulator.attendees[p2_id]
        time_slot = rng.choice(list(TimeSlot))
        _, ground_truth = simulator._calculate_meeting_success(person_a, person_b, time_slot)
        outcome = get_true_outcome(ground_truth)
        target = target_yes if outcome == "YES" else target_no

        if counts[outcome] >= target:
            continue

        # Generate expensive LLM descriptions only after accepting the candidate.
        p1_desc = descriptor.generate_description(person_a, time_slot)
        p2_desc = descriptor.generate_description(person_b, time_slot)
        ts_str = time_slot.value.replace("_", " ").title()
        scenarios.append({"scenario_id": len(scenarios), "p1_desc": p1_desc, "p2_desc": p2_desc, "ts_str": ts_str, "ground_truth": ground_truth})
        counts[outcome] += 1
        print(f"Collected {outcome}: YES={counts['YES']}/{target_yes}, NO={counts['NO']}/{target_no} (attempt {attempts})")

    rng.shuffle(scenarios)
    print(f"Balanced evaluation set created after {attempts} attempts: {counts}")
    return scenarios


def evaluate_model_performance(base_model_id: str, adapter_path: str, test_scenarios_count: int = 20):
    test_scenarios = _generate_balanced_scenarios(test_scenarios_count)
    baseline_results = run_local_agent_evaluation(base_model_id, test_scenarios, "baseline")
    trained_results = run_local_agent_evaluation(base_model_id, test_scenarios, "trained", adapter_path=adapter_path)
    baseline_metrics = _binary_metrics(baseline_results)
    trained_metrics = _binary_metrics(trained_results)
    improvement = trained_metrics["accuracy"] - baseline_metrics["accuracy"]
    print("\nBaseline metrics:", baseline_metrics)
    print("Trained metrics:", trained_metrics)
    print("Accuracy improvement (percentage points):", improvement)
    return {"baseline": baseline_metrics, "trained": trained_metrics, "improvement": improvement}


if __name__ == "__main__":
    csv_pattern = "traces/agent_traces_*.csv"
    existing_csvs = sorted(glob.glob(csv_pattern), key=os.path.getmtime, reverse=True)

    if not existing_csvs:
        raise FileNotFoundError('No "traces/agent_traces_*.csv" file found. Generate fresh traces with the corrected agentic_traces.py first.')

    most_recent_csv = existing_csvs[0]
    file_age_hours = (time.time() - os.path.getmtime(most_recent_csv)) / 3600
    print(f"Found existing trace file: {most_recent_csv} (created {file_age_hours:.1f} hours ago)")

    base_model = "Qwen/Qwen3-1.7B"
    adapter_path = "./qwen3-dpo-adapter-v1"

    train_model_with_dpo(csv_file_path=most_recent_csv, base_model_id=base_model, new_adapter_path=adapter_path)

    print("\n" + "=" * 60)
    print("EVALUATION PHASE")
    print("=" * 60)
    evaluate_model_performance(base_model_id=base_model, adapter_path=adapter_path, test_scenarios_count=4) #20)