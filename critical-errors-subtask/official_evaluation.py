"""
Evaluation script for the WMT22 Critical Error Detection Task.

This script computes the Mathew's Correlation Coefficient, Recall 
and Precision at a threshold determined by the true number of 
critical errors. The script can be executed for both the unconstrained 
and constrained settings.

In the unconstrained setting, the models can use the provided training 
data for the task and produce high scores for critical errors. In this 
case, the threshold is the Kth highest score, where K is the number of 
critical errors in and all scores above the computed threshold are 
considered positive. 

In the constrained setting, the models are purely trained with quality 
annotations, such as DA's, MQM and/or HTER. In this setting, it is 
expected that critical errors are attributed lower scores. Thus, the 
threshold is the Kth lowest score, where K is the number of critical 
errors and all scores below the computed threshold are consedered positive.
"""
import argparse
import typing

def parse_args():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "-s", "--scores", 
        required=True, 
        help="Path to file with predicted scores (one score per line).",
    )
    parser.add_argument(
        "-l", "--labels", 
        required=True, 
        help="Path to file with OK/BAD labels (one label per line).",
    )
    constrain_group = parser.add_mutually_exclusive_group(required=True)
    constrain_group.add_argument(
        "-u", "--unconstrained", 
        action="store_true", 
        help="Evaluate model in the unconstrained setting.",
    )
    constrain_group.add_argument(
        "-c", "--constrained", 
        action="store_true", 
        help="Evaluate model in the constrained setting.",
    )
    return parser.parse_args()

def read_file_lines(filename: str) -> typing.List[str]:
    with open(filename, "r") as f:
        return [l.rstrip() for l in f]

def read_scores(filename: str) -> typing.List[float]:
    text_scores = read_file_lines(filename)
    return [float(s) for s in text_scores]

def read_labels(filename: str) -> typing.List[int]:
    def parse_label(label: str) -> int:
        if label in ("OK", "0"):
            return 0
        if label in ("BAD", "1"):
            return 1
        raise ValueError(f"Unknown label: \"{label}\"")
    
    text_labels = read_file_lines(filename)
    return [parse_label(l) for l in text_labels]

def main():
    args = parse_args()

    scores = read_scores(args.scores)
    labels = read_labels(args.labels)

    num_errors = sum(1 for l in labels if l == 1)

    if args.unconstrained:
        # In the unconstrained setting, the models are trained on the CED train data to
        # predict critical errors and so higher scores indicate critical errors.
        sorted_scores = sorted(scores, reverse=True)
        # Threshold considering top K scores are correct
        threshold = sorted_scores[num_errors-1]
        preds = [1 if s >= threshold else 0 for s in scores]
    elif args.constrained:
        # In the constrained setting, the models are not trained on the CED train data and
        # the critical errors are the ones with the lowest scores
        sorted_scores = sorted(scores)
        # Threshold considering bottom K scores are correct
        threshold = sorted_scores[num_errors-1]
        preds = [1 if s <= threshold else 0 for s in scores]
    else:
        raise ValueError("constrained or unconstrained flags are not set.")

    print(f"Threshold: {'{:.4f}'.format(threshold)}")
    
    tp = sum(1 for p, l in zip(preds, labels) if p == l and l == 1)
    fp = sum(1 for p, l in zip(preds, labels) if p != l and l == 0)
    tn = sum(1 for p, l in zip(preds, labels) if p == l and l == 0)
    fn = sum(1 for p, l in zip(preds, labels) if p != l and l == 1)

    recall = tp / (tp + fn)
    precision = tp / (tp + fp) if tp + fp != 0 else 0
    mcc_numerator = (tp * tn) - (fp * fn)
    mcc_denominator = ((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)) ** 0.5
    mcc = mcc_numerator / mcc_denominator if mcc_denominator != 0 else 0
    
    at = num_errors/len(labels) * 100
    print(f"MCC@{'{:.2f}'.format(at)}: {'{:.4f}'.format(mcc)}")
    print(f"Recall@{'{:.2f}'.format(at)}: {'{:.4f}'.format(recall)}")
    print(f"Precision@{'{:.2f}'.format(at)}: {'{:.4f}'.format(precision)}")

if __name__ == "__main__":
    main()