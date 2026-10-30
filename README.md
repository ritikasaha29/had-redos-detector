# HAD — Hierarchical Ambiguity Detector for Regex ReDoS

HAD is a static analysis framework that detects **Regular Expression Denial of Service (ReDoS)** vulnerabilities, including patterns that use **backreferences**, which most existing detectors cannot handle. It classifies a regex by worst-case matching complexity, explains which detection layer fired, and suggests repairs.

## Features

- **Layered detection**
  - **Layer 0 (L0):** Structural ambiguity such as nested quantifiers and overlapping alternation, e.g. `(a+)+`, `(a|aa)+`. Complexity: `O(n²)`.
  - **Layer 1 (L1):** Backreference-driven ambiguity from variable-length captures, e.g. `(a+)\1`, `(\w+)\s+\1`. Complexity: `O(n²)`.
  - **Layer 2 (L2):** Compound ambiguity where a looping construct wraps a capture and its backreference, e.g. `((a+)\2)+`. Complexity: `O(n³)`.
- **Complexity classification** (`O(n)`, `O(n²)`, `O(n³)`)
- **Repair engine** that suggests safer rewrites for vulnerable patterns
- **Streamlit web UI** for interactive analysis
- **Evaluation tooling** with accuracy, precision, recall, F1, false-positive rate and latency, plus corpus analysis on Snort and PyPI regex datasets

## Project Structure

```
.
├── app.py               # Streamlit web interface
├── run_had.py           # Test suite runner and single-pattern demo
├── evaluate.py          # Metrics on labeled data + Snort/PyPI corpus analysis
├── load_dataset.py      # Simple batch analyzer for a pattern file
├── regex_dataset.txt    # Labeled dataset: pattern,expected_complexity
├── snort_dataset.txt    # Snort-style regexes (/pattern/)
├── pypi_dataset.txt     # Regexes sampled from PyPI packages
├── detection/           # HADDetector and HADReport
└── repair/              # RepairEngine and RepairResult
```

## Installation

```bash
git clone https://github.com/<your-username>/<your-repo>.git
cd <your-repo>
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
pip install streamlit
```

Requires Python 3.8+.

## Usage

### Web app

```bash
streamlit run app.py
```

Enter a regex and click **Analyze** to see whether it is vulnerable, its worst-case complexity, and which layers were triggered.

### Analyze a single pattern (with repair suggestions)

```bash
python run_had.py "(a+)+"
```

### Run the test suite

```bash
python run_had.py              # summary table
python run_had.py --verbose    # full report + repair suggestions per pattern
```

Test categories: `SAFE`, `L0`, `L1`, `L2`, `EDGE`, and `REAL` (real-world regexes).

### Run the evaluation

```bash
python evaluate.py
```

This reports:

1. Metrics on `regex_dataset.txt` (accuracy, precision, recall, F1, FPR, average latency, confusion matrix)
2. Layer distribution (L0 / L1 / L2 / Safe) over `snort_dataset.txt`
3. Layer distribution over `pypi_dataset.txt`

### Batch analysis

```bash
python load_dataset.py
```

## Dataset Formats

| File | Format | Example |
|------|--------|---------|
| `regex_dataset.txt` | `pattern,expected_complexity` | `(a+)+,O(n2)` |
| `snort_dataset.txt` | one regex per line, wrapped in `/.../` | `/(a+)+/` |
| `pypi_dataset.txt` | one raw regex per line | `(\w+)\s+\1` |

## Programmatic Use

```python
from detection.had_detector import HADDetector
from repair.repair_engine import RepairEngine

detector = HADDetector()
report = detector.analyze(r"(a+)\1")

print(report.is_vulnerable)       # True
print(report.worst_complexity)    # O(n²)
print(report.summary())

for fix in RepairEngine().repair(report):
    print(fix)
```

## Example Results

| Pattern | Layer | Complexity |
|---------|-------|------------|
| `a+b` | — | safe |
| `(a+)+` | L0 | O(n²) |
| `(a+)\1` | L1 | O(n²) |
| `((a+)\2)+` | L2 | O(n³) |

## Contributing

Issues and pull requests are welcome. Please run `python run_had.py` before submitting to confirm the test suite passes.

## License

Add a license of your choice (e.g. MIT) in a `LICENSE` file.
