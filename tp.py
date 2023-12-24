import numpy as np
from wordseg.evaluate import evaluate
from wordseg.prepare import prepare, gold
from wordseg.algos import tp
from wordseg.separator import Separator


def load_file(file_path):
    """Load in the file and put it into their weird formatting"""
    lines = []
    for line in open(file_path, "r").readlines():
        line = f"{line.strip()} ".replace(" ", ";eword").replace(".", " ").replace("|", ";syll")
        lines.append(line)
    return lines


def run_iter(num):
    """Run one iteration of the TP learning model"""
    separator = Separator(phone=' ', syllable=';syll', word=';eword')
    train = load_file(f"data/train_{num}.txt")
    test = load_file(f"data/test_{num}.txt")
    prepared_train = list(prepare(train, separator=separator))
    prepared_test = list(prepare(test, separator=separator))
    gold_test = list(gold(test, separator=separator))
    segmented_tp = tp.segment(
        text=prepared_test,
        train_text=prepared_train,
        threshold="relative"
    )
    eval_tp = evaluate(segmented_tp, gold_test)
    return eval_tp


def run_experiment():
    """Collect performance over all 10 random initializations"""
    overall_res = {}
    for i in range(10):
        print(f"Running {i}th iteration...")
        res = run_iter(i)
        for label, num in res.items():
            if label not in overall_res:
                overall_res[label] = []
            overall_res[label].append(num)
    for label in overall_res:
        values = np.asarray(overall_res[label])
        mean = np.mean(values)
        std = np.std(values)
        print(f"{label}: {mean :.3f} ({std :.3f})")


if __name__ == "__main__":
    run_experiment()

