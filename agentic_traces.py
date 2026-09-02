# AGENTIC RL TRACES STARTER
from dataclasses import dataclass, field
from datasets import Dataset
from datetime import datetime
import glob
import json
import numpy as np
import os
import pandas as pd
from peft import LoraConfig, PeftModel
import random
import re
import time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from trl import SFTTrainer
from typing import Dict, Any, List, Optional, Any

from data_classes import get_true_outcome, TimeSlot, ConferenceSimulator, PersonDescriptor, safe_extract_json

from npcpy.npc_compiler import NPC
from npcpy.llm_funcs import get_llm_response

# Additional imports:

from transformers import StoppingCriteria, StoppingCriteriaList

QWEN_MODEL = "qwen3:1.7b"
TRACE_SCHEMA_VERSION = "success_prob_v2"
DESCRIPTOR_TEMPERATURE = 0.1

def calculate_reward(trace):
    final_rec = trace.get("final_recommendation_parsed")
    if not final_rec:
        return -1.0
    recommendation = str(final_rec.get("recommendation", "")).upper().strip()
    reasoning = str(final_rec.get("reasoning", "")).strip()
    if recommendation not in {"YES", "NO"}:
        return -1.0
    if not reasoning or not trace.get("completed_naturally", False):
        return -0.9
    correct = recommendation == get_true_outcome(float(trace["ground_truth"]))
    used_tools = bool(trace.get("tools_used"))
    if correct and used_tools:
        return 1.0
    if correct:
        return 0.5
    if used_tools:
        return -0.5
    return -0.9


#### Core data classes
from data_classes import TimeSlot, ConferenceSimulator, PersonDescriptor, safe_extract_json
system_prompt_configurations = [ { "name": "Pax",
        "primary_directive": """You are a balanced meeting advisor.
Evaluate compatibility and potential for productive discussion, weighing both positive and
negative signals equally to provide a well-rounded recommendation. Use tools to gather information,
then reason through your decision step by step."""
    }, { "name": "Vigil",
        "primary_directive": """You are a cautious meeting gatekeeper.
Your priority is to prevent unproductive meetings. Only recommend meetings if there is
extremely high confidence of success. Use tools to thoroughly analyze potential issues.
Avoid false positives at all costs."""
    }, { "name": "Aura",
        "primary_directive": """You are an optimistic connector.
Your goal is to foster connections. Use tools to find potential synergies and common ground.
Err on the side of recommending meetings, even with moderate compatibility, to encourage networking."""
    }, { "name": "Cortex",
        "primary_directive": """You are a data-driven analyst.
Use tools systematically to gather quantitative insights. Your recommendations must strictly
follow the analytical data. Process each tool output carefully before proceeding."""
    }, { "name": "Echo",
        "primary_directive": """You are an intuitive empath.
Use tools to understand the interpersonal dynamics, then rely heavily on your intuitive
assessment. Focus on reading between the lines of what the tools tell you."""
    }, { "name": "Maven",
        "primary_directive": """You are a strategic prioritizer.
Use tools to understand potential synergies and strategic value. Prioritize meetings that
offer the highest strategic value or growth potential for participants."""
    }, { "name": "Nexus",
        "primary_directive": """You are an efficiency expert.
Use tools to assess time efficiency and return on investment. Only recommend meetings
that represent highly efficient use of time for both parties."""
    } ]
# THE ABOVE ARE PROVIDED FOR SIMPLICITY. LEARNERS MAY TWEAK THESE AS THEY SEE FIT
# TO FURTHER IMPROVE THE QUALITY OF THE RESPONSES AND TRAINING DATA.



def extract_common_interests_and_topics(person_a_desc: str, person_b_desc: str) -> str:
   """
   Analyzes two person descriptions to identify common interests and conversation topics.

   Args:
       person_a_desc: Description of the first person
       person_b_desc: Description of the second person

   Returns:
       JSON string with topics, interests, compatibility score and explanation
   """
   prompt = f"""Analyze the two descriptions and identify common interests and potential conversation topics.

Person A: {person_a_desc}
Person B: {person_b_desc}

Return a single JSON object with:
- "topics": list of potential conversation topics
- "interests": list of common interests
- "compatibility_score": float from 0-1
- "explanation": string explaining the analysis

like so
""" + """

{
   'topics': ['topic1', 'topic2'],
   'interests': ['interest1', 'interest2'],
   'compatibility_score': 0.x,
   'explanation': '

}
"""
   response = get_llm_response(prompt, model=QWEN_MODEL, provider='ollama', format='json', temperature=0.1)
   json_out = response['response']

   return json_out

def assess_time_slot_fit_and_energy(person_a_desc: str, person_b_desc: str, time_slot: str) -> str:
    """
    Evaluates how well a time slot matches both people's energy levels and availability.

    Args:
        person_a_desc: Description of the first person
        person_b_desc: Description of the second person
        time_slot: The proposed meeting time slot

    Returns:
        JSON string with fit scores, energy levels, and potential issues
    """
    prompt = f"""Evaluate how well this time slot fits both people's energy levels and state.

    Person A: {person_a_desc}
    Person B: {person_b_desc}
    Time Slot: {time_slot}

    Return a JSON object with:
    - "fit_score": float from 0-1
    - "person_a_energy": float from 0-1 (e.g., 0.2 for low, 0.8 for high)
    - "person_b_energy": float from 0-1 (e.g., 0.2 for low, 0.8 for high)
    - "summary": string summary
    - "red_flags": list of potential issues
    like so """+ """

    {
    'fit_score': 0.x,
    'person_a_energy': 0.x,
    'person_b_energy': 0.x,
    'summary': 'string',
    'red_flags':['item1', 'item2']
    }
    """
    response = get_llm_response(prompt, model=QWEN_MODEL, provider='ollama', format='json', temperature=0.1)

    json_out = response['response']
    return json_out

def predict_follow_up_potential(person_a_desc: str, person_b_desc: str, time_slot: str) -> str:
    """
    Predicts the likelihood of productive follow-up interactions after a meeting.

    Args:
        person_a_desc: Description of the first person
        person_b_desc: Description of the second person
        time_slot: The meeting time slot

    Returns:
        JSON string with follow-up probability, relationship potential and suggestions
    """
    prompt = f"""Predict the follow-up potential for a meeting between these people.

    Person A: {person_a_desc}
    Person B: {person_b_desc}
    Time Slot: {time_slot}

    Return a JSON object with:
    - "follow_up_probability": float from 0-1
    - "relationship_potential": string (low/medium/high)
    - "business_potential": string (low/medium/high)
    - "rationale": string explanation
    - "suggested_next_steps": list of strings

    like: """ + """

    {
    'follow_up_probability': 0.x,
    'relationship_potential': 'low/medium/high',
    'business_potential': 'low/medium/high',
    'rationale': 'string
    'suggested_next_steps': ['item1', 'item2']


    }
    """
    response = get_llm_response(prompt, model=QWEN_MODEL, provider='ollama', format='json', temperature=0.1)
    json_out = response['response']

    return json_out


_SFT_MODEL = None
_SFT_TOKENIZER = None

BASE_MODEL_NAME = "google/gemma-3-270m-it"
SFT_ADAPTER_PATH = "models/sft_prediction_model_gemma_270m"


def build_sft_prompt(person_a_desc: str, person_b_desc: str, time_slot: str) -> str:
    input_text = f"""Person A: {person_a_desc}
Person B: {person_b_desc}
Time: {time_slot}

What is the likelihood of a successful meeting? Respond with JSON: {{"reason": "word", "probability": 0.XX}}"""
    return f"<start_of_turn>user\n{input_text}<end_of_turn>\n<start_of_turn>model\n"


class StopAfterJson(StoppingCriteria):
    def __init__(self, tokenizer, prompt_length: int):
        self.tokenizer = tokenizer
        self.prompt_length = prompt_length

    def __call__(self, input_ids, scores, **kwargs) -> bool:
        generated_ids = input_ids[0, self.prompt_length:]
        text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
        return "{" in text and "}" in text


def get_sft_model():
    global _SFT_MODEL, _SFT_TOKENIZER

    if _SFT_MODEL is not None and _SFT_TOKENIZER is not None:
        return _SFT_MODEL, _SFT_TOKENIZER

    if not os.path.exists(SFT_ADAPTER_PATH):
        raise FileNotFoundError(f"SFT adapter not found at {SFT_ADAPTER_PATH}")

    base_model = AutoModelForCausalLM.from_pretrained( BASE_MODEL_NAME, torch_dtype=torch.float32,
        attn_implementation="eager", trust_remote_code=True, )

    _SFT_MODEL = PeftModel.from_pretrained(base_model, SFT_ADAPTER_PATH)
    _SFT_MODEL.eval()

    _SFT_MODEL.generation_config.do_sample = False
    _SFT_MODEL.generation_config.temperature = None
    _SFT_MODEL.generation_config.top_p = None
    _SFT_MODEL.generation_config.top_k = None
    _SFT_MODEL.generation_config.typical_p = None
    _SFT_MODEL.generation_config.min_p = None
    _SFT_MODEL.generation_config.penalty_alpha = None

    _SFT_TOKENIZER = AutoTokenizer.from_pretrained( BASE_MODEL_NAME, trust_remote_code=True, )

    if _SFT_TOKENIZER.pad_token_id is None:
        _SFT_TOKENIZER.pad_token = _SFT_TOKENIZER.eos_token

    _SFT_TOKENIZER.padding_side = "right"
    return _SFT_MODEL, _SFT_TOKENIZER


def extract_prediction_json(text: str) -> dict:
    match = re.search(r"\{.*?\}", text.strip(), flags=re.DOTALL)

    if match is None:
        raise ValueError(f"No JSON object found in response: {text}")

    result = json.loads(match.group(0))

    if not isinstance(result, dict):
        raise ValueError("Prediction is not a JSON object.")

    return result


def predict_meeting_success_tool(person_a_desc, person_b_desc, time_slot):
    print("\n========== ACTUAL SFT TOOL CALL ==========", flush=True)
    print("Person A:", person_a_desc, flush=True)
    print("Person B:", person_b_desc, flush=True)
    print("Time:", time_slot, flush=True)
    try:
        model, tokenizer = get_sft_model()
        prompt = build_sft_prompt(person_a_desc, person_b_desc, time_slot)
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512, padding=False)
        device = next(model.parameters()).device
        input_ids = inputs["input_ids"].to(device)
        attention_mask = inputs["attention_mask"].to(device)
        prompt_length = input_ids.shape[1]
        stopping_criteria = StoppingCriteriaList([StopAfterJson(tokenizer, prompt_length)])
        with torch.no_grad():
            generated = model.generate(input_ids=input_ids, attention_mask=attention_mask, max_new_tokens=24, do_sample=False, use_cache=True, pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id, stopping_criteria=stopping_criteria)
        response_text = tokenizer.decode(generated[0, prompt_length:], skip_special_tokens=True).strip()
        result = extract_prediction_json(response_text)
        if "reason" not in result or "probability" not in result:
            raise ValueError("SFT prediction missing reason or probability.")
        reason = str(result["reason"]).strip()
        probability = float(result["probability"])
        if not 0.0 <= probability <= 1.0:
            raise ValueError(f"Probability outside [0,1]: {probability}")
        print("SFT RESULT:", result, flush=True)
        print("==========================================\n", flush=True)
        return json.dumps({"status": "success", "probability": probability, "reason": reason}, separators=(",", ":"))
    except Exception as error:
        print("SFT ERROR:", error, flush=True)
        return json.dumps({"status": "error", "message": str(error)}, separators=(",", ":"))


TOOLS = [ predict_meeting_success_tool, extract_common_interests_and_topics, assess_time_slot_fit_and_energy,
    predict_follow_up_potential, ]


class AgentTraceCollector:
    def __init__(self):
        self.traces = []

    def record_trace(self, **kwargs):
        self.traces.append(kwargs)

    def save_traces_to_file(self, filename: str):
        if not self.traces:
            print("No traces to save.")
            return
        df_data = []
        for trace in self.traces:
            final_rec = trace.get("final_recommendation_parsed")
            row = {"trace_schema_version": TRACE_SCHEMA_VERSION, "scenario_id": trace.get("scenario_id"), "system_prompt_name": trace.get("system_prompt_name"), "system_prompt": trace.get("system_prompt"), "initial_user_prompt": trace.get("initial_user_prompt"), "tools_used": ",".join(trace.get("tools_used", [])), "total_iterations": trace.get("total_iterations"), "ground_truth_prob": trace.get("ground_truth"), "true_outcome": get_true_outcome(trace.get("ground_truth", 0.0)), "agent_outcome": (final_rec or {}).get("recommendation"), "reward": trace.get("reward"), "final_recommendation_reasoning": (final_rec or {}).get("reasoning"), "completed_naturally": trace.get("completed_naturally", False), "final_recommendation_parsed": json.dumps(final_rec, separators=(",", ":")) if final_rec else None, "agent_transcript": trace.get("agent_transcript")}
            df_data.append(row)
        pd.DataFrame(df_data).to_csv(filename, index=False)
        print(f"Traces saved to {filename}")



class AgentToolLoop:
    # Here we use the npcpy tool use loop because it can auto call our tools for us and report back the results
    def __init__(self, agent: NPC, max_iterations: int = 8):
        # the number of max iterations should be sufficient to allow the model to get somewhere
        # but shouldnt be so high that it is spending 20 some odd turns deciding what to do..

        self.agent = agent
        self.max_iterations = max_iterations
    def _parse_final_json(self, text: str) -> Optional[Dict]:
        try:
            match = re.search(r'\{.*\}', text, re.DOTALL)
            if not match:
                return None
            data = json.loads(match.group(0))
            return data
        except json.JSONDecodeError:
            return None

    def run_tool_loop(self, initial_prompt: str) -> Dict[str, Any]:
        enhanced_directive = self.agent.primary_directive + """

Use tools to gather enough information to decide whether the meeting should happen. Treat every tool, including the SFT predictor, as fallible evidence rather than ground truth. Reconcile conflicting evidence and make the final decision from the overall scenario.

When finished, your FINAL response MUST be exactly one JSON object:

{
  "recommendation": "YES",
  "reasoning": "your explanation"
}

Rules:
- "recommendation" must be exactly "YES" or "NO".
- probability >= 0.50 normally supports YES
- probability < 0.50 normally supports NO
- The second key must be exactly "reasoning".
- Do not use "reason", "should_meet", "meeting_recommendation", "action", or any other key.
- Do not use phrases such as "Proceed with the meeting" or "Not recommended".
- Do not include text before or after the JSON object.
"""
        messages = [{"role": "system", "content": enhanced_directive}]
        raw_responses = []
        final_recommendation_data = None
        completed_naturally = False
        current_prompt = initial_prompt
        for i in range(self.max_iterations):
            print(f"\nIteration {i}: calling agent...", flush=True)
            start_time = time.time()

            response_obj = self.agent.get_llm_response( current_prompt, messages=messages, auto_process_tool_calls=True
            )

            print(f"Iteration {i}: agent returned after {time.time() - start_time:.1f}s", flush=True)

            print(response_obj['response'][-500:])

            raw_responses.append(response_obj)
            messages = response_obj.get('messages') or messages
            last_assistant_content = messages[-1].get('content', '')

            final_recommendation_data = self._parse_final_json(last_assistant_content)

            if final_recommendation_data and 'recommendation' in final_recommendation_data and 'reasoning' in final_recommendation_data:

                completed_naturally = True
                break

            current_prompt = "Are you finished? If so, provide your final JSON recommendation. If not, continue."
            if i ==1:
                messages[0]['content'] +='''Once you have sufficient information,
                    provide your final recommendation as JSON:
                {
                "recommendation": "YES" or "NO",
                "reasoning": "detailed explanation of your decision"
                }
                Do not include any other keys in your response. Your JSON must only be these two keys. It doesnt matter if you gathered infromation from tools and want to share it with the user, they will be able
                to see it through the inspection of the tool calling traces. It is importantly only your task to return this information very plainly once it is finished.
                '''

        return { "raw_responses": raw_responses, "final_recommendation": final_recommendation_data,
            "total_iterations": i + 1, "completed_naturally": completed_naturally }




def serialize_agent_transcript(loop_result: Dict[str, Any]) -> str:
    parts = []
    for index, response in enumerate(loop_result.get("raw_responses", []), start=1):
        text = str((response or {}).get("response", "")).strip()
        if text:
            parts.append(f"[assistant_turn_{index}]\n{text}")
    return "\n\n".join(parts)

def generate_scenario(scenario_id: int, descriptor: PersonDescriptor) -> Dict[str, Any]:
    simulator = ConferenceSimulator(num_attendees=50, seed=random.randint(0, 10000))
    p1_id, p2_id = random.sample(list(simulator.attendees.keys()), 2)
    person_a = simulator.attendees[p1_id]
    person_b = simulator.attendees[p2_id]
    time_slot = random.choice(list(TimeSlot))
    p1_desc = descriptor.generate_description(person_a, time_slot)
    p2_desc = descriptor.generate_description(person_b, time_slot)
    ts_str = (time_slot.value.replace("_", " ").title())
    _, gt_prob = (simulator._calculate_meeting_success(person_a, person_b, time_slot))
    initial_prompt = f"""Your task is to decide if two people should meet.
Use the available tools to gather information step-by-step.
When you have enough information, stop using tools and provide
your final answer as a single JSON object.

Person A: {p1_desc}
Person B: {p2_desc}
Time Slot: {ts_str}

Begin your analysis by calling a tool."""

    print("\n=== SCENARIO DEBUG ===")
    print("Person A raw:", person_a)
    print("Person B raw:", person_b)
    print("Time slot:", time_slot)
    print("Person A description:", p1_desc)
    print("Person B description:", p2_desc)
    print("Simulator probability:", gt_prob)
    print("SFT prediction:", predict_meeting_success_tool(p1_desc, p2_desc, ts_str))
    print("======================")

    return { "scenario_id": scenario_id, "p1_desc": p1_desc, "p2_desc": p2_desc, "ts_str": ts_str,
        "ground_truth": gt_prob, "initial_prompt": initial_prompt, }

def generate_agent_traces_for_training(num_scenarios: int) -> List[Dict[str, Any]]:
    traces_collector = AgentTraceCollector()
    descriptor = PersonDescriptor(temperature=DESCRIPTOR_TEMPERATURE)

    for scenario_id in range(num_scenarios):
        scenario = generate_scenario(scenario_id=scenario_id, descriptor=descriptor)

        for config in system_prompt_configurations:
            print(f"\nScenario {scenario_id + 1}/{num_scenarios} - Agent: {config['name']}")

            current_agent = NPC( name=config["name"].lower(), primary_directive=config["primary_directive"],
                tools=TOOLS, model=QWEN_MODEL, provider="ollama", )

            tool_loop = AgentToolLoop(current_agent, max_iterations=8)
            loop_result = tool_loop.run_tool_loop(scenario["initial_prompt"])

            all_tools_used = []
            for resp in loop_result["raw_responses"]:
                if resp and resp.get("tool_calls"):
                    for call in resp["tool_calls"]:
                        all_tools_used.append(call["function"]["name"])

            trace_data = { "scenario_id": scenario["scenario_id"], "system_prompt_name": config["name"],
                "system_prompt": config["primary_directive"],
                "initial_user_prompt": scenario["initial_prompt"],
                "final_recommendation_parsed": loop_result["final_recommendation"],
                "tools_used": list(set(all_tools_used)), "total_iterations": loop_result["total_iterations"],
                "ground_truth": scenario["ground_truth"], "completed_naturally": loop_result["completed_naturally"], }

            trace_data["reward"] = calculate_reward(trace_data)
            traces_collector.record_trace(**trace_data)

            true_outcome_str = get_true_outcome(scenario["ground_truth"])
            agent_decision = loop_result.get("final_recommendation")

            print( f"Scenario {scenario_id + 1}/{num_scenarios} ({config['name']}): "
                f"True Outcome={true_outcome_str} " f"(Prob={scenario['ground_truth']:.2f}), "
                f"Agent said='{agent_decision}', " f"Reward={trace_data['reward']:.2f}" )

    return traces_collector.traces



if __name__ == "__main__":
    traces_csv_file = None
    csv_pattern = "traces/agent_traces_*.csv"
    existing_csvs = sorted(glob.glob(csv_pattern), key=os.path.getmtime, reverse=True)
    num_traces_per_agent = 30
    generated_traces_list = generate_agent_traces_for_training(num_traces_per_agent)
    traces_csv_file = f"traces/agent_traces_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    temp_collector = AgentTraceCollector()
    temp_collector.traces = generated_traces_list
    temp_collector.save_traces_to_file(traces_csv_file)
    print(' With your traces saved, we can now start doing reinforcement learning with DPO! hop on over to starter_agentic_rlft.py to begin.')
