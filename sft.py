from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List
import gc
import inspect
import json
import math
import random
import re

import numpy as np
import pandas as pd
import torch
import transformers
from datasets import Dataset
from peft import(LoraConfig, TaskType, get_peft_model,)
from sklearn.metrics import(accuracy_score, balanced_accuracy_score, f1_score,)
from transformers import(AutoModelForCausalLM, AutoTokenizer, DataCollatorForSeq2Seq, EarlyStoppingCallback, StoppingCriteria, StoppingCriteriaList, Trainer, TrainingArguments, set_seed,)


@ dataclass


class SFTConfig:
    base_model_name: str = "google/gemma-3-270m-it"
    output_model_path: str =("models/sft_prediction_model_gemma_270m")
    sft_data_path: str =("data/sft_curated_training_data.csv")
    # LoRA
    lora_r: int = 16 # 32
    lora_alpha: int = 32 # 64
    lora_dropout: float = 0.05
    lora_target_modules: List[str] = field(default_factory = lambda: ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj",])
    # ["q_proj", "k_proj", "v_proj", "o_proj",])
    # Training
    # Train up to 30 epochs, evaluating after every epoch.
    max_epochs: int = 30
    # Early stopping on validation loss.
    # Stop after 3 consecutive epochs without at least a 0.005 improvement.
    early_stopping_patience: int = 4 # 6
    early_stopping_threshold: float = 0.005
    per_device_train_batch_size: int = 2
    per_device_eval_batch_size: int = 2
    gradient_accumulation_steps: int = 4
    learning_rate: float = 1e-5 # 2e-5 # 1e-4
    weight_decay: float = 0.02 # 0.01
    logging_steps: int = 10
    optim: str = "adamw_torch"
    lr_scheduler_type: str = "cosine"
    max_grad_norm: float = 1.0
    # Sequence lengths
    max_length: int = 512
    # Reduced from 48 because targets are ~13–15 tokens
    max_new_tokens: int = 24
    # Numerical setup
    fp16: bool = False
    bf16: bool = False
    # Dataset
    validation_fraction: float = 0.20
    seed: int = 42
    save_total_limit: int = 2


def configure_reproducibility(seed: int,) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    set_seed(seed)


def cleanup_cuda() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def safe_extract_json(text: str,) -> Dict[str, Any]:
    text = text.strip()
    try:
        value = json.loads(text)
        if not isinstance(value, dict):
            raise ValueError("Parsed JSON is not an object.")
        return value
    except Exception:
        pass
    match = re.search(r"\{.*?\}", text, flags = re.DOTALL,)
    if match is None:
        raise ValueError (f"No JSON object found:\n{text }")
    value = json.loads(match.group(0))
    if not isinstance(value, dict):
        raise ValueError("Parsed JSON is not an object.")
    return value


def extract_first_json_text(text: str,) -> str:
    text = text.strip()
    match = re.search(r"\{.*?\}", text, flags = re.DOTALL,)
    if match is None:
        return text
    return match.group(0)


def build_target_json(reason: str, probability: float,) -> str:
    target = {"reason": str(reason).strip(), "probability": round(float(probability), 2,),}
    return json.dumps(target, ensure_ascii = False, separators =(",", ":"),)


def build_prompt(input_text: str,) -> str:
    return ("<start_of_turn>user\n" f"{input_text }" "<end_of_turn>\n"  "<start_of_turn>model\n")


def load_training_dataframe(csv_path: str,) -> pd.DataFrame:
    print("\n========================================")
    print("LOADING SFT DATA")
    print("========================================")
    df = pd.read_csv(csv_path)
    required_columns = {"input_text", "target_reason", "target_probability",}
    missing =(required_columns - set(df.columns))
    if missing:
        raise ValueError ("Missing required columns: " f"{sorted (missing )}")
    df = df[["input_text", "target_reason", "target_probability",]].copy()
    df = df.dropna(subset =["input_text", "target_reason", "target_probability",]).copy()
    df["input_text"] =(df["input_text"].astype(str).str.strip())
    df["target_reason"] =(df["target_reason"].astype(str).str.strip())
    df["target_probability"] = pd.to_numeric(df["target_probability"], errors = "raise",)
    if not ((df["target_probability"] >= 0) &(df["target_probability"] <= 1)).all():
        raise ValueError("target_probability contains " "values outside [0, 1].")
    df = df.reset_index(drop = True)
    print("Rows:", len(df),)
    print("Columns:", list(df.columns),)
    print("\nReason distribution:")
    print(df["target_reason"].value_counts().to_string())
    print("\nNumber of reason clusters:", df["target_reason"].nunique(),)
    print("\nTarget probability statistics:")
    print(df["target_probability"].describe())
    return df


def audit_cluster_probabilities(df: pd.DataFrame,) -> Dict[str, float]:
    print("\n========================================")
    print("CLUSTER / PROBABILITY AUDIT")
    print("========================================")
    mapping = {}
    inconsistencies = {}
    for reason, group in df.groupby("target_reason"):
        probabilities = sorted(set(round(float(value), 8,) for value in group["target_probability"]))
        if len(probabilities) != 1:
            inconsistencies[reason] = probabilities
        else:
            mapping[reason] = probabilities[0]
    if inconsistencies:
        for reason, values in (inconsistencies.items()):
            print(reason, "->", values,)
        raise ValueError("Every reason cluster should map " "to exactly one probability.")
    print("Verified:", len(mapping), "reason clusters.")
    for reason in sorted(mapping):
        print (f"{reason :22s} " f"{mapping [reason ]:.2f}")
    return mapping


def canonical_pair_key(input_text: str) -> str:
    match = re.search(r"^Person A:\s*(.*?)\nPerson B:\s*(.*?)\nTime:\s*(.*?)\n\n", str(input_text).strip(), flags=re.DOTALL)
    if match is None:
        return str(input_text).strip()
    person_a, person_b, time_slot = (part.strip() for part in match.groups())
    first, second = sorted((person_a, person_b))
    return f"Person 1: {first}\nPerson 2: {second}\nTime: {time_slot}"

def split_dataframe(df: pd.DataFrame, validation_fraction: float, seed: int):
    working = df.copy()
    working["_pair_key"] = working["input_text"].map(canonical_pair_key)
    group_keys = list(working["_pair_key"].drop_duplicates())
    rng = random.Random(seed)
    rng.shuffle(group_keys)
    target_validation_rows = max(1, int(round(len(working) * validation_fraction)))
    validation_keys, validation_rows = set(), 0
    for key in group_keys:
        if validation_rows >= target_validation_rows and validation_keys:
            break
        validation_keys.add(key)
        validation_rows += int((working["_pair_key"] == key).sum())
    val_df = working[working["_pair_key"].isin(validation_keys)].drop(columns="_pair_key").copy()
    train_df = working[~working["_pair_key"].isin(validation_keys)].drop(columns="_pair_key").copy()
    train_df = train_df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    val_df = val_df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    train_keys = set(train_df["input_text"].map(canonical_pair_key))
    val_keys = set(val_df["input_text"].map(canonical_pair_key))
    overlap = train_keys & val_keys
    if overlap:
        raise RuntimeError(f"Pair leakage detected across train/validation split: {len(overlap)} groups")
    print("\nTraining examples:", len(train_df))
    print("Validation examples:", len(val_df))
    print("Training pair groups:", len(train_keys))
    print("Validation pair groups:", len(val_keys))
    print("A/B pair-group overlap:", len(overlap))
    return train_df, val_df


def tokenize_completion_only_example(example: Dict[str, Any], tokenizer, max_length: int,) -> Dict[str, Any]:
    prompt_text = build_prompt(str(example["input_text"]))
    target_text =(build_target_json(reason =(example["target_reason"]), probability =(example["target_probability"]),) + "<end_of_turn>")
    prompt_ids = tokenizer(prompt_text, add_special_tokens = True, truncation = False,)["input_ids"]
    target_ids = tokenizer(target_text, add_special_tokens = False, truncation = False,)["input_ids"]
    if not target_ids:
        raise ValueError("Target produced zero tokens.")
    available_prompt_tokens =(max_length - len(target_ids))
    if available_prompt_tokens <= 0:
        raise ValueError("Target is longer than max_length.")
    if len(prompt_ids) >(available_prompt_tokens):
        prompt_ids = prompt_ids[:available_prompt_tokens]
    input_ids =(prompt_ids + target_ids)
    attention_mask =([1] *len(input_ids))
    labels =([- 100] *len(prompt_ids) + target_ids.copy())
    return{"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels,}


def build_tokenized_dataset(df: pd.DataFrame, tokenizer, max_length: int, name: str,) -> Dataset:
    records =[]
    prompt_lengths =[]
    target_lengths =[]
    final_lengths =[]
    truncated_count = 0
    for _, row in df.iterrows():
        prompt_text = build_prompt(row["input_text"])
        original_prompt_ids = tokenizer(prompt_text, add_special_tokens = True, truncation = False,)["input_ids"]
        target_text =(build_target_json(row["target_reason"], row["target_probability"],) + "<end_of_turn>")
        target_ids = tokenizer(target_text, add_special_tokens = False, truncation = False,)["input_ids"]
        tokenized =(tokenize_completion_only_example({"input_text": (row["input_text"]), "target_reason": (row["target_reason"]), "target_probability": float(row["target_probability"]),}, tokenizer, max_length,))
        if (len(original_prompt_ids) + len(target_ids) > max_length):
            truncated_count += 1
        prompt_lengths.append(len(original_prompt_ids))
        target_lengths.append(len(target_ids))
        final_lengths.append(len(tokenized["input_ids"]))
        records.append(tokenized)
    print("\n========================================")
    print (f"TOKEN AUDIT: {name .upper ()}")
    print("========================================")
    print("Examples:", len(records),)
    print ("Mean original prompt length:",f"{np .mean (prompt_lengths ):.2f}",)
    print ("95th percentile prompt length:",f"{np .percentile (prompt_lengths ,95 ):.2f}",)
    print("Maximum original prompt length:", max(prompt_lengths),)
    print ("Mean learnable target tokens:",f"{np .mean (target_lengths ):.2f}",)
    print("Maximum target tokens:", max(target_lengths),)
    print("Maximum final sequence length:", max(final_lengths),)
    print("Prompts truncated:", truncated_count,)
    return Dataset.from_list(records)


def audit_labels(dataset: Dataset, tokenizer, index: int = 0,) -> None:
    example = dataset[index]
    input_ids = example["input_ids"]
    labels = example["labels"]
    ignored_ids =[]
    learned_ids =[]
    for token_id, label in zip(input_ids, labels,):
        if label == - 100:
            ignored_ids.append(token_id)
        else:
            learned_ids.append(token_id)
    print("\n========================================")
    print("LABEL MASK AUDIT")
    print("========================================")
    print("\nIGNORED BY LOSS:")
    print(tokenizer.decode(ignored_ids, skip_special_tokens = False,)[:1500])
    print("\n----------------------------------------")
    print("LEARNABLE / SUPERVISED TOKENS:")
    print(tokenizer.decode(learned_ids, skip_special_tokens = False,))
    print("\nIgnored token count:", sum(label == - 100 for label in labels),)
    print("Learnable token count:", sum(label != - 100 for label in labels),)


def load_model_and_tokenizer(config: SFTConfig,):
    print("\n========================================")
    print("LOADING MODEL")
    print("========================================")
    print("Model:", config.base_model_name,)
    model =(AutoModelForCausalLM.from_pretrained(config.base_model_name, trust_remote_code = True, torch_dtype = torch.float32, attn_implementation = "eager",))
    model.config.use_cache = False
    tokenizer =(AutoTokenizer.from_pretrained(config.base_model_name, trust_remote_code = True,))
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token =(tokenizer.eos_token)
    tokenizer.padding_side =("right")
    return (model, tokenizer,)


def apply_lora(model, config: SFTConfig,):
    peft_config = LoraConfig(r = config.lora_r, lora_alpha =(config.lora_alpha), lora_dropout =(config.lora_dropout), target_modules =(config.lora_target_modules), bias = "none", task_type =(TaskType.CAUSAL_LM),)
    model = get_peft_model(model, peft_config,)
    return model


def audit_trainable_parameters(model,) -> None:
    total = 0
    trainable = 0
    for parameter in (model.parameters()):
        total += parameter.numel()
        if parameter.requires_grad:
            trainable +=(parameter.numel())
    print("\n========================================")
    print("TRAINABLE PARAMETERS")
    print("========================================")
    print ("Total:",f"{total :,}",)
    print ("Trainable:",f"{trainable :,}",)
    print ("Trainable percentage:",f"{100 *trainable /total :.4f}%",)


def configure_greedy_generation(model,) -> None:
    generation_config =(model.generation_config)
    generation_config.do_sample = False
    generation_config.temperature = None
    generation_config.top_p = None
    generation_config.top_k = None
    generation_config.typical_p = None
    generation_config.min_p = None
    generation_config.penalty_alpha = None
    print("\n========================================")
    print("GENERATION CONFIG")
    print("========================================")
    print("do_sample:", generation_config.do_sample,)
    print("temperature:", generation_config.temperature,)
    print("top_p:", generation_config.top_p,)
    print("top_k:", generation_config.top_k,)


class StopAfterJson(StoppingCriteria):
    def __init__(self, tokenizer, prompt_length: int,):
        self.tokenizer =(tokenizer)
        self.prompt_length =(prompt_length)
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs,) -> bool:
        generated_ids = input_ids[0, self.prompt_length:]
        generated_text =(self.tokenizer.decode(generated_ids, skip_special_tokens = True,))
        # ----------------------------------------------------
        # Stop as soon as the first JSON object is complete.
        # ----------------------------------------------------
        return ("{" in generated_text and "}" in generated_text)


def create_training_arguments(config: SFTConfig,) -> TrainingArguments:
    signature = inspect.signature(TrainingArguments.__init__)
    supported = set(signature.parameters)
    kwargs = {"output_dir": (config.output_model_path), "num_train_epochs": (config.max_epochs), "per_device_train_batch_size": (config.per_device_train_batch_size), "per_device_eval_batch_size": (config.per_device_eval_batch_size), "gradient_accumulation_steps": (config.gradient_accumulation_steps), "learning_rate": (config.learning_rate), "weight_decay": (config.weight_decay), "optim": (config.optim), "lr_scheduler_type": (config.lr_scheduler_type), "logging_steps": (config.logging_steps), "logging_strategy": ("steps"), "max_grad_norm": (config.max_grad_norm), "fp16": (config.fp16), "bf16": (config.bf16), "save_strategy": ("epoch"), "save_total_limit": (config.save_total_limit),  # Evaluate and save after every epoch.
    # The best checkpoint is selected by minimum validation loss.
    "load_best_model_at_end": True, "metric_for_best_model": "eval_loss", "greater_is_better": False, "report_to": ("none"), "seed": (config.seed), "data_seed": (config.seed), "label_names": ["labels"], "remove_unused_columns": (False),}
    # Transformers renamed evaluation_strategy -> eval_strategy.
    # Use whichever name exists in the installed version.
    if "eval_strategy" in supported:
        kwargs["eval_strategy"] = "epoch"
    elif "evaluation_strategy" in supported:
        kwargs["evaluation_strategy"] = "epoch"
    else:
        raise RuntimeError("This Transformers version exposes neither " "'eval_strategy' nor 'evaluation_strategy'.")
    if ("save_only_model" in supported):
        kwargs["save_only_model"] = True
    unknown =(set(kwargs) - supported)
    if unknown:
        raise RuntimeError ("Unsupported TrainingArguments: " f"{sorted (unknown )}")
    return TrainingArguments(**kwargs)


def create_data_collator(tokenizer, model,):
    return DataCollatorForSeq2Seq(tokenizer = tokenizer, model = model, padding = True, label_pad_token_id = - 100, return_tensors = "pt",)


def numerical_preflight(trainer: Trainer,):
    print("\n========================================")
    print("NUMERICAL PREFLIGHT")
    print("========================================")
    batch = next(iter(trainer.get_train_dataloader()))
    labels = batch["labels"]
    ignored_tokens = int((labels == - 100).sum())
    learned_tokens = int((labels != - 100).sum())
    print("Ignored prompt/padding tokens:", ignored_tokens,)
    print("Learnable answer tokens:", learned_tokens,)
    device = next(trainer.model.parameters()).device
    batch = {key: (value.to(device) if torch.is_tensor(value) else value) for key, value in batch.items()}
    trainer.model.train()
    trainer.model.zero_grad(set_to_none = True)
    outputs = trainer.model(**batch)
    loss = outputs.loss
    if not torch.isfinite(loss):
        raise RuntimeError (f"Non-finite loss: {loss }")
    loss.backward()
    squared_norm = 0.0
    for name, parameter in (trainer.model.named_parameters()):
        if (not parameter.requires_grad or parameter.grad is None):
            continue
        gradient =(parameter.grad.detach().float())
        if not torch.isfinite(gradient).all():
            raise RuntimeError ("Non-finite gradient in " f"{name }")
        squared_norm += float(gradient.pow(2).sum().cpu())
    gradient_norm = math.sqrt(squared_norm)
    trainer.model.zero_grad(set_to_none = True)
    print("Loss:", float(loss.detach().cpu()),)
    print("Gradient norm:", gradient_norm,)
    cleanup_cuda()
    return{"loss": float(loss.detach().cpu()), "gradient_norm": (gradient_norm), "ignored_tokens": (ignored_tokens), "learned_tokens": (learned_tokens),}


def generate_response(model, tokenizer, input_text: str, max_length: int, max_new_tokens: int,) -> str:
    prompt = build_prompt(input_text)
    inputs = tokenizer(prompt, return_tensors = "pt", truncation = True, max_length = max_length, padding = False,)
    device = next(model.parameters()).device
    input_ids =(inputs["input_ids"].to(device))
    attention_mask =(inputs["attention_mask"].to(device))
    input_length =(input_ids.shape[1])
    stopping_criteria =(StoppingCriteriaList([StopAfterJson(tokenizer = tokenizer, prompt_length =(input_length),)]))
    with torch.no_grad():
        generated = model.generate(input_ids = input_ids, attention_mask =(attention_mask), max_new_tokens =(max_new_tokens), do_sample = False, use_cache = True, pad_token_id =(tokenizer.pad_token_id), eos_token_id =(tokenizer.eos_token_id), stopping_criteria =(stopping_criteria),)
    generated_tokens = generated[0, input_length:]
    raw_response = tokenizer.decode(generated_tokens, skip_special_tokens = True,).strip()
    # --------------------------------------------------------
    # Final safeguard:
    # even if a strange token somehow appears after the JSON,
    # only retain the first complete JSON object.
    # --------------------------------------------------------
    response = extract_first_json_text(raw_response)
    return response


def validate_model(model, tokenizer, val_df: pd.DataFrame, config: SFTConfig,):
    print("\n========================================")
    print("VALIDATION")
    print("========================================")
    model.eval()
    true_reasons =[]
    predicted_reasons =[]
    true_probabilities =[]
    predicted_probabilities =[]
    valid_json_count = 0
    invalid_outputs =[]
    for i, row in (val_df.iterrows()):
        response_text =(generate_response(model = model, tokenizer = tokenizer, input_text =(row["input_text"]), max_length =(config.max_length), max_new_tokens =(config.max_new_tokens),))
        true_reason = str(row["target_reason"]).strip()
        true_probability = float(row["target_probability"])
        try:
            parsed = safe_extract_json(response_text)
            predicted_reason = str(parsed["reason"]).strip()
            predicted_probability = float(parsed["probability"])
            valid_json_count += 1
        except Exception as error:
            predicted_reason =("__INVALID__")
            predicted_probability =(float("nan"))
            invalid_outputs.append({"index": int(i), "error": str(error), "response": (response_text),})
        true_reasons.append(true_reason)
        predicted_reasons.append(predicted_reason)
        true_probabilities.append(true_probability)
        predicted_probabilities.append(predicted_probability)
        if i < 10:
            print (f"\nSample {i }")
            print("Predicted reason:", predicted_reason,)
            print("True reason:", true_reason,)
            print("Predicted probability:", predicted_probability,)
            print("True probability:", true_probability,)
            print("Raw JSON:", response_text,)
    reason_accuracy = float(accuracy_score(true_reasons, predicted_reasons,))
    reason_balanced_accuracy = float(balanced_accuracy_score(true_reasons, predicted_reasons,))
    reason_macro_f1 = float(f1_score(true_reasons, predicted_reasons, average = "macro", zero_division = 0,))
    true_probabilities_array = np.asarray(true_probabilities, dtype = np.float64,)
    predicted_probabilities_array = np.asarray(predicted_probabilities, dtype = np.float64,)
    valid_mask = np.isfinite(predicted_probabilities_array)
    valid_probability_count = int(valid_mask.sum())
    if valid_probability_count > 0:
        probability_mae = float(np.mean(np.abs(predicted_probabilities_array[valid_mask] - true_probabilities_array[valid_mask])))
    else:
        probability_mae = float("nan")
    if (valid_probability_count > 1 and np.std(predicted_probabilities_array[valid_mask]) > 0 and np.std(true_probabilities_array[valid_mask]) > 0):
        probability_correlation = float(np.corrcoef(predicted_probabilities_array[valid_mask], true_probabilities_array[valid_mask],)[0, 1])
    else:
        probability_correlation =(0.0)
    reason_probability_map =(val_df.groupby("target_reason")["target_probability"].first().to_dict())
    consistency_values =[]
    for reason, probability in zip(predicted_reasons, predicted_probabilities,):
        if (reason in reason_probability_map and math.isfinite(probability)):
            expected = float(reason_probability_map[reason])
            consistency_values.append(abs(probability - expected) <= 0.011)
    if consistency_values:
        probability_reason_consistency =(float(np.mean(consistency_values)))
    else:
        probability_reason_consistency =(0.0)
    total_examples = len(val_df)
    valid_json_rate =(valid_json_count / total_examples)
    print("\n========================================")
    print("VALIDATION SUMMARY")
    print("========================================")
    print ("Reason accuracy:",f"{reason_accuracy :.4f}",)
    print ("Reason balanced accuracy:",f"{reason_balanced_accuracy :.4f}",)
    print ("Reason macro F1:",f"{reason_macro_f1 :.4f}",)
    print ("Probability MAE:",f"{probability_mae :.4f}",)
    print ("Probability correlation:",f"{probability_correlation :.4f}",)
    print ("Probability/reason consistency:",f"{probability_reason_consistency :.4f}",)
    print ("Valid JSON rate:",f"{valid_json_rate :.4f}",)
    print("Invalid outputs:", len(invalid_outputs),)
    return{"reason_accuracy": (reason_accuracy), "reason_balanced_accuracy": (reason_balanced_accuracy), "reason_macro_f1": (reason_macro_f1), "probability_mae": (probability_mae), "probability_correlation": (probability_correlation), "probability_reason_consistency": (probability_reason_consistency), "valid_json_rate": (valid_json_rate), "valid_json_count": (valid_json_count), "total_examples": (total_examples), "invalid_outputs": (invalid_outputs),}


def run_sft_fine_tuning(config: SFTConfig, train_df: pd.DataFrame, val_df: pd.DataFrame,):
    (base_model, tokenizer,) = load_model_and_tokenizer(config)
    train_dataset =(build_tokenized_dataset(train_df, tokenizer, config.max_length, "training",))
    val_dataset =(build_tokenized_dataset(val_df, tokenizer, config.max_length, "validation",))
    audit_labels(train_dataset, tokenizer, index = 0,)
    model = apply_lora(base_model, config,)
    model.config.use_cache = False
    audit_trainable_parameters(model)
    configure_greedy_generation(model)
    training_args =(create_training_arguments(config))
    data_collator =(create_data_collator(tokenizer, model,))
    # --------------------------------------------------------
    # processing_class replaces deprecated tokenizer=...
    # --------------------------------------------------------
    # Hugging Face will train one epoch, evaluate validation loss,
    # save the checkpoint, and then decide whether to continue.
    # Patience=3 means three consecutive epoch evaluations without
    # a meaningful (> threshold) improvement cause training to stop.
    early_stopping_callback = EarlyStoppingCallback(early_stopping_patience = config.early_stopping_patience, early_stopping_threshold = config.early_stopping_threshold,)
    trainer = Trainer(model = model, args = training_args, train_dataset =(train_dataset), eval_dataset =(val_dataset), data_collator =(data_collator), processing_class =(tokenizer), callbacks =[early_stopping_callback],)
    preflight = numerical_preflight(trainer)
    print("\n========================================")
    print("STARTING EPOCH-BY-EPOCH FINE-TUNING")
    print("========================================")
    print (f"Maximum epochs: {config .max_epochs }")
    print("Validation is evaluated after every epoch.")
    print("Early-stopping patience:", config.early_stopping_patience, "epochs",)
    print("Minimum eval-loss improvement:", config.early_stopping_threshold,)
    # This single Trainer call is still epoch-by-epoch training:
    # Trainer evaluates at the end of every epoch and the callback
    # can stop before max_epochs is reached.
    result = trainer.train()
    print("\nTraining finished.")
    print("Epoch actually reached:", trainer.state.epoch,)
    print("Best validation loss:", trainer.state.best_metric,)
    print("Best checkpoint:", trainer.state.best_model_checkpoint,)
    print("Final training loss:", result.training_loss,)
    trainer.save_model(config.output_model_path)
    tokenizer.save_pretrained(config.output_model_path)
    print("\nModel saved to:", config.output_model_path,)
    configure_greedy_generation(trainer.model)
    validation_results =(validate_model(model = trainer.model, tokenizer = tokenizer, val_df = val_df, config = config,))
    return (trainer, validation_results, preflight,)

def main():
    config = SFTConfig()
    configure_reproducibility(config.seed)
    cleanup_cuda()
    print("\n========================================")
    print("ENVIRONMENT")
    print("========================================")
    print("Transformers:", transformers.__version__,)
    print("PyTorch:", torch.__version__,)
    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0),)
        print("CUDA:", torch.version.cuda,)
        # LOAD DATA
    df = load_training_dataframe(config.sft_data_path)
    cluster_probability_map =(audit_cluster_probabilities(df))
    (train_df, val_df,) = split_dataframe(df = df, validation_fraction =(config.validation_fraction), seed = config.seed,)
    # SAMPLE TARGETS
    print("\n========================================")
    print("SAMPLE TRAINING EXAMPLES")
    print("========================================")
    for i in range(min(3, len(train_df),)):
        row = train_df.iloc[i]
        print (f"\nExample {i +1 }")
        print("\nInput:")
        print(row["input_text"][:700])
        print("\nTarget:")
        print(build_target_json(row["target_reason"], row["target_probability"],))
        # TRAIN
    (trainer, validation_results, preflight,) = run_sft_fine_tuning(config = config, train_df = train_df, val_df = val_df,)
    # FINAL RESULTS
    print("\n========================================")
    print("FINAL RESULTS")
    print("========================================")
    print("\nCompletion-only label masking:")
    print("Prompt ignored tokens:", preflight["ignored_tokens"],)
    print("JSON learnable tokens:", preflight["learned_tokens"],)
    print()
    for key, value in (validation_results.items()):
        if key ==("invalid_outputs"):
            continue
        print (f"{key } = {value }")

if __name__ == "__main__": main()