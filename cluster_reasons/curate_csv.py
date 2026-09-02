import pandas as pd
import random

import read_reason_clusters as rrc


def save_dataframe_to_csv(df, output_path: str) -> None:
    df.to_csv(output_path, index=False)


def curate_reason(cluster_map, reason):
    reason = reason.strip().lower()
    if reason[-1] == ".": reason = reason[:-1]
    reason = cluster_map[reason]
    return reason


def modify_input_text(input_text):
    substring = 'Respond with JSON: {"probability": 0.XX, "reason": "word"}'
    new_substring = 'Respond with JSON: {"reason": "word", "probability": 0.XX}'
    return input_text.replace(substring, new_substring)


def compute_mean_of_target_probabilities_for_each_cluster(df, clusters):
    for cluster_name in clusters:
        mask = df["target_reason"] == cluster_name
        average_probability = df.loc[mask, "target_probability"].mean()
        df.loc[mask, "target_probability"] = round(average_probability, 2)


def switch_person_a_b(text: str) -> str:
    """
    Switch the Person A and Person B sections in input_text.

    Assumes the text contains:

    Person A: ...
    Person B: ...

    followed by the rest of the prompt, such as Time:, instructions, etc.
    """

    marker_a = "Person A:"
    marker_b = "Person B:"

    pos_a = text.find(marker_a)
    pos_b = text.find(marker_b)

    if pos_a == -1 or pos_b == -1:
        raise ValueError(
            "Could not find both 'Person A:' and 'Person B:'"
        )

    if pos_a > pos_b:
        raise ValueError(
            "'Person A:' appears after 'Person B:'."
        )

    # Find where the section after Person B begins.
    # In your dataset this is typically "Time:".
    pos_after_b = text.find("\nTime:", pos_b)

    if pos_after_b == -1:
        raise ValueError(
            "Could not find '\\nTime:' after Person B."
        )

    before_a = text[:pos_a]

    person_a_text = text[
        pos_a + len(marker_a):pos_b
    ].strip()

    person_b_text = text[
        pos_b + len(marker_b):pos_after_b
    ].strip()

    rest = text[pos_after_b:]

    switched_text = (
        before_a
        + marker_a
        + " "
        + person_b_text
        + "\n"
        + marker_b
        + " "
        + person_a_text
        + rest
    )

    if random.random() < 0.01:
        print(f"text:\n{text}\n\nswitched_text:\n{switched_text}\n")

    return switched_text


def add_switched_persons(df: pd.DataFrame) -> pd.DataFrame:

    augmented_rows = []

    for index, row in df.iterrows():

        # Clone the row
        new_row = row.copy()

        # Modify only input_text
        new_row["input_text"] = switch_person_a_b(
            row["input_text"]
        )

        # target_reason and target_probability
        # automatically remain unchanged because
        # new_row is a copy of the original row.

        augmented_rows.append(new_row)

    # Convert cloned rows into a DataFrame
    augmented_df = pd.DataFrame(
        augmented_rows
    )

    # Add all augmented rows to the end
    result_df = pd.concat(
        [
            df,
            augmented_df,
        ],
        ignore_index=True,
    )

    return result_df


def main():
    csv_path = "../data/sft_training_data.csv"
    df = pd.read_csv(csv_path)
    
    clusters = rrc.load_clusters("curated_reason_clusters.txt")
    cluster_map = rrc.invert_cluster_mapping(clusters)

    for index, row in df.iterrows():
        df.at[index, "target_reason"] = curate_reason(cluster_map, row["target_reason"])
        df.at[index, "input_text"] = modify_input_text(row["input_text"])
    
    df = df.drop(columns=["ground_truth_prob"])

    compute_mean_of_target_probabilities_for_each_cluster(df, clusters)

    df = add_switched_persons(df)

    csv_path2 = "../data/sft_curated_training_data.csv"
    save_dataframe_to_csv(df, csv_path2)

    df = pd.read_csv(csv_path2)
    curated_target_reasons = set(df["target_reason"].astype(str))
    print("Curated Target Reasons:")
    for i, reason in enumerate(curated_target_reasons, 1):

        mask = df["target_reason"] == reason
        probs = df.loc[mask, "target_probability"].to_list()
        
        print(f"{i}: {reason} probs: {probs}")
    

if __name__ == "__main__":
    main()