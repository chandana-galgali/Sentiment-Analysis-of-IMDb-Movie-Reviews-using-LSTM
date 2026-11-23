"""Command-line predictor for custom reviews."""
import argparse
from .model import load_artifacts, predict


def main():
    parser = argparse.ArgumentParser(description="Classify a movie review")
    parser.add_argument("review", help="Movie review text to classify")
    parser.add_argument("--model-dir", default="models")
    args = parser.parse_args()
    model, metadata = load_artifacts(args.model_dir)
    result = predict(args.review, model, metadata)
    print(f"{result['label']} | positive probability: {result['positive_probability']:.2%}")


if __name__ == "__main__":
    main()
