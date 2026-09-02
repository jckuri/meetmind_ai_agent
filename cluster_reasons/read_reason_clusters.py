def load_clusters(file_path: str) -> dict[str, list[str]]:
    clusters = {}
    current_cluster = None

    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()

            # Ignore empty lines
            if not line:
                continue

            # Cluster header, e.g. [engagement]
            if line.startswith("[") and line.endswith("]"):
                current_cluster = line[1:-1].strip()

                if current_cluster in clusters:
                    raise ValueError(
                        f"Duplicate cluster found: {current_cluster}"
                    )

                clusters[current_cluster] = []
                continue

            # Every other non-empty line is an element of the current cluster
            if current_cluster is None:
                raise ValueError(
                    f"Element found before any cluster header: {line}"
                )

            clusters[current_cluster].append(line)

    return clusters


def invert_cluster_mapping(
    cluster_to_reasons: dict[str, list[str]]
) -> dict[str, str]:
    reason_to_cluster = {}

    for cluster, reasons in cluster_to_reasons.items():
        for reason in reasons:
            if reason in reason_to_cluster:
                raise ValueError(
                    f"Reason '{reason}' appears in more than one cluster."
                )

            reason_to_cluster[reason] = cluster

    return reason_to_cluster


def main():
    clusters = load_clusters("curated_reason_clusters.txt")
    cluster_map = invert_cluster_mapping(clusters)
    n_reasons = 0
    n_clusters = len(clusters)
    for cluster in sorted(clusters):
        print(f'[{cluster}]')
        reasons = clusters[cluster]
        for reason in sorted(reasons):
            c = cluster_map[reason]
            print(f'{reason} ({c})')
            n_reasons += 1
        print()
    print(f'n_reasons={n_reasons}, n_clusters={n_clusters}')


if __name__ == "__main__":
    main()