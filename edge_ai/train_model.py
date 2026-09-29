import json
import random
from pathlib import Path

import numpy as np
from sklearn.tree import DecisionTreeClassifier


FEATURE_NAMES = [
    "active_robots",
    "nearby_robots",
    "minimum_distance",
    "predicted_conflicts",
    "junction_pressure",
    "head_on_pairs",
    "average_wait",
    "max_urgency",
    "low_battery_ratio",
    "jammed_cells",
]


def make_sample():
    active = random.randint(1, 12)
    nearby = random.randint(0, min(active - 1, 8))
    minimum_distance = random.randint(0, 8)
    predicted_conflicts = random.randint(0, 6)
    junction_pressure = random.random()
    head_on_pairs = random.randint(0, 3)
    average_wait = random.uniform(0, 15)
    max_urgency = random.randint(1, 5)
    low_battery_ratio = random.random()
    jammed_cells = random.randint(0, 10)

    score = 0

    if nearby >= 4:
        score += 1

    if minimum_distance <= 2:
        score += 2

    if predicted_conflicts >= 2:
        score += 3

    if junction_pressure > 0.65:
        score += 2

    if head_on_pairs >= 1:
        score += 2

    if average_wait > 7:
        score += 1

    if low_battery_ratio > 0.6:
        score += 1

    if jammed_cells >= 5:
        score += 2

    if active >= 8:
        score += 1

    if score >= 8:
        label = 2       # HIGH
    elif score >= 4:
        label = 1       # MEDIUM
    else:
        label = 0       # LOW

    features = [
        active,
        nearby,
        minimum_distance,
        predicted_conflicts,
        junction_pressure,
        head_on_pairs,
        average_wait,
        max_urgency,
        low_battery_ratio,
        jammed_cells,
    ]

    return features, label


def create_dataset(samples=6000):
    X = []
    y = []

    for _ in range(samples):
        features, label = make_sample()
        X.append(features)
        y.append(label)

    return np.array(X), np.array(y)


def export_tree(model, output_file):
    tree = model.tree_
    nodes = []

    for i in range(tree.node_count):

        left = int(tree.children_left[i])
        right = int(tree.children_right[i])

        # Leaf node
        if left == -1 and right == -1:

            probabilities = tree.value[i][0]

            total = probabilities.sum()

            if total > 0:
                probabilities = probabilities / total

            nodes.append({
                "leaf": True,
                "probabilities": probabilities.tolist()
            })

        # Decision node
        else:

            nodes.append({
                "leaf": False,
                "feature": int(tree.feature[i]),
                "threshold": float(tree.threshold[i]),
                "left": left,
                "right": right
            })

    data = {
        "model": "HiveLane Edge-AI Decision Tree",
        "version": 1,
        "features": FEATURE_NAMES,
        "classes": [
            "LOW",
            "MEDIUM",
            "HIGH"
        ],
        "nodes": nodes
    }

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

    print(f"Model exported to: {output_file}")


def main():

    print("=" * 55)
    print("HiveLane Edge-AI Model Training")
    print("=" * 55)

    print("\nGenerating warehouse training data...")

    X, y = create_dataset(6000)

    print(f"Training samples : {len(X)}")
    print(f"Features         : {X.shape[1]}")

    model = DecisionTreeClassifier(
        max_depth=6,
        min_samples_leaf=8,
        random_state=42
    )

    print("\nTraining Decision Tree...")

    model.fit(X, y)

    accuracy = model.score(X, y)

    print(f"Training accuracy: {accuracy * 100:.2f}%")

    output_file = Path(__file__).parent / "model.json"

    export_tree(model, output_file)

    print("\nModel ready.")
    print(f"Saved at: {output_file}")


if __name__ == "__main__":
    main()