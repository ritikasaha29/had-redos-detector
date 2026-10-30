from detection.had_detector import HADDetector
import time

# ---------- Normalization ----------
def normalize(s):
    return s.replace("²", "2").replace("³", "3")


# ---------- Load labeled dataset ----------
def load_dataset(file):
    data = []
    with open(file, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                pattern, label = line.strip().split(",")
                data.append((pattern, label))
    return data


# ---------- Load external dataset ----------
def load_external(file):
    patterns = []
    with open(file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                if line.startswith("/") and line.endswith("/"):
                    line = line[1:-1]
                patterns.append(line)
    return patterns


# ---------- MAIN ----------
detector = HADDetector()

# ===== PART 1: Labeled Dataset Evaluation =====
print("=== Labeled Dataset Evaluation ===")

test_data = load_dataset("regex_dataset.txt")

correct = 0
tp = fp = tn = fn = 0
latencies = []

for pattern, expected in test_data:
    start = time.time()
    report = detector.analyze(pattern)
    end = time.time()

    latencies.append(end - start)

    predicted = normalize(report.worst_complexity)
    expected = normalize(expected)

    print(f"{pattern:20} → predicted: {predicted}, expected: {expected}")

    # Accuracy
    if predicted == expected:
        correct += 1

    # SAFE = O(n)
    is_true_vuln = expected != "O(n)"
    is_pred_vuln = predicted != "O(n)"

    if is_pred_vuln and is_true_vuln:
        tp += 1
    elif is_pred_vuln and not is_true_vuln:
        fp += 1
    elif not is_pred_vuln and not is_true_vuln:
        tn += 1
    elif not is_pred_vuln and is_true_vuln:
        fn += 1


# ---------- Metrics ----------
accuracy = correct / len(test_data)

precision = tp / (tp + fp) if (tp + fp) else 0
recall = tp / (tp + fn) if (tp + fn) else 0
f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0
fpr = fp / (fp + tn) if (fp + tn) else 0

avg_latency = sum(latencies) / len(latencies)

print("\n=== Metrics ===")
print(f"Accuracy : {accuracy:.2f}")
print(f"Precision: {precision:.2f}")
print(f"Recall   : {recall:.2f}")
print(f"F1 Score : {f1:.2f}")
print(f"FPR      : {fpr:.2f}")
print(f"Latency  : {avg_latency*1000:.2f} ms")

print("\nConfusion Matrix:")
print(f"TP: {tp}, FP: {fp}, TN: {tn}, FN: {fn}")


# ===== PART 2: Snort Dataset Analysis =====
print("\n=== Snort Dataset Analysis ===")

external_data = load_external("snort_dataset.txt")

l0 = l1 = l2 = safe = 0

for pattern in external_data:
    report = detector.analyze(pattern)

    if report.layer2:
        l2 += 1
    elif report.layer1:
        l1 += 1
    elif report.layer0:
        l0 += 1
    else:
        safe += 1

total = len(external_data)

print(f"Total patterns: {total}")
print(f"L0   : {l0} ({l0/total:.2f})")
print(f"L1   : {l1} ({l1/total:.2f})")
print(f"L2   : {l2} ({l2/total:.2f})")
print(f"Safe : {safe} ({safe/total:.2f})")


# ===== PART 3: PyPI Dataset Analysis =====
print("\n=== PyPI Dataset Analysis ===")

pypi_data = load_external("pypi_dataset.txt")

l0 = l1 = l2 = safe = 0

for pattern in pypi_data:
    report = detector.analyze(pattern)

    if report.layer2:
        l2 += 1
    elif report.layer1:
        l1 += 1
    elif report.layer0:
        l0 += 1
    else:
        safe += 1

total = len(pypi_data)

print(f"Total patterns: {total}")
print(f"L0   : {l0} ({l0/total:.2f})")
print(f"L1   : {l1} ({l1/total:.2f})")
print(f"L2   : {l2} ({l2/total:.2f})")
print(f"Safe : {safe} ({safe/total:.2f})")