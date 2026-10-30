from detection.had_detector import HADDetector

def load_patterns(file_path):
    with open(file_path, "r") as f:
        return [line.strip() for line in f if line.strip()]

def run_dataset(file_path):
    detector = HADDetector()
    patterns = load_patterns(file_path)

    print("\nDataset Analysis:\n")

    for p in patterns:
        report = detector.analyze(p)
        print(f"{p:20} → {report.worst_complexity}")

if __name__ == "__main__":
    run_dataset("regex_dataset.txt")