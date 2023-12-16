from typing import List
import argparse


def read_file(file_path: str) -> List[str]:
    with open(file_path, "r") as f:
        lines = f.readlines()
        lines = [line.strip() for line in lines]
        return lines


def main(args):
    labels = read_file(args.label_file)
    results = read_file(args.result_file)

    correct = 0
    for label, result in zip(labels, results):
        if label == result:
            correct += 1

    print(f"accuracy: {round(correct / len(labels) * 100, 2)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--label_file", type=str, default="data/labels.txt")
    parser.add_argument("--result_file", type=str, default="data/results.txt")
    args = parser.parse_args()
    main(args)
