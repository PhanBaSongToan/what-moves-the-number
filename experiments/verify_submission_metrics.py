"""Check reported accuracy and macro-F1 using aggregate confusion matrices."""

import json
import math
import sys
from pathlib import Path


def scores(matrix):
    size = len(matrix)
    total = sum(sum(row) for row in matrix)
    accuracy = sum(matrix[i][i] for i in range(size)) / total
    f1 = []
    for i in range(size):
        count = sum(matrix[i]) + sum(row[i] for row in matrix)
        f1.append(2 * matrix[i][i] / count if count else 0.0)
    return accuracy, sum(f1) / size


def main(path):
    results = json.loads(Path(path).read_text(encoding="utf-8"))
    for entry in results:
        accuracy, macro_f1 = scores(entry["confusion_matrix"])
        if not math.isclose(accuracy, entry["accuracy"], abs_tol=1e-10):
            raise ValueError(f"Accuracy mismatch: {entry['name']}")
        if not math.isclose(macro_f1, entry["macro_f1"], abs_tol=1e-10):
            raise ValueError(f"Macro-F1 mismatch: {entry['name']}")
    print(f"Verified {len(results)} runs.")


if __name__ == "__main__":
    main(sys.argv[1])
