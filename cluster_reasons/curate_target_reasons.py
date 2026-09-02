import pandas as pd

def save_reasons_to_txt(reasons: set[str], output_path: str) -> None:
    with open(output_path, "w", encoding="utf-8") as f:
        for reason in sorted(reasons):
            f.write(reason + "\n")

def main():
    csv_path = "../data/sft_training_data.csv"
    df = pd.read_csv(csv_path)
    target_reasons = set(df["target_reason"].astype(str))
    print(f"Number of unique target reasons: {len(target_reasons)}")
    print("\nUnique target reasons:")
    curated_reasons = set()
    for reason in sorted(target_reasons):
        reason = reason.strip()
        if reason[-1] == ".": reason = reason[:-1]
        curated_reasons.add(reason)
    print(f"Number of unique curated reasons: {len(curated_reasons)}")
    for reason in sorted(curated_reasons):
        print(reason)
    save_reasons_to_txt(curated_reasons, "curated_reasons.txt")

if __name__ == "__main__":
    main()