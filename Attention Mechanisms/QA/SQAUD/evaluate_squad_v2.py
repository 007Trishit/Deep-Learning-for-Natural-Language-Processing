import argparse
from datasets import load_dataset, load_metric
from transformers import pipeline, AutoModelForQuestionAnswering, AutoTokenizer
from tqdm import tqdm


def evaluate_squad_v2(model_path):
    # Load the fine-tuned model and tokenizer
    model = AutoModelForQuestionAnswering.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    # Create the question-answering pipeline
    qa_pipeline = pipeline("question-answering",
                           model=model, tokenizer=tokenizer, device="cuda:7")

    # Load the SQuAD v2.0 dataset
    dataset = load_dataset("squad_v2", split="validation")

    # Load the SQuAD metric
    squad_metric = load_metric("squad_v2")

    # Prepare lists to store predictions and references
    predictions = []
    references = []

    # Evaluate the model on the dataset
    for example in tqdm(dataset, desc="Evaluating"):
        context = example["context"]
        question = example["question"]

        # Get the model's prediction
        prediction = qa_pipeline(
            question=question, context=context, handle_impossible_answer=True)

        # Prepare the prediction and reference dictionaries
        pred_dict = {
            "id": example["id"],
            "prediction_text": prediction["answer"],
            "no_answer_probability": 1 - prediction["score"] if prediction["answer"] == "" else 0
        }

        ref_dict = {
            "id": example["id"],
            "answers": example["answers"],
            "no_answer_threshold": 0.5  # You can adjust this threshold
        }

        predictions.append(pred_dict)
        references.append(ref_dict)

    # Compute the metrics
    results = squad_metric.compute(
        predictions=predictions, references=references)

    # Print the results
    print("\nEvaluation results on SQuAD v2.0:")
    print(f"Exact Match: {results['exact']:.2f}")
    print(f"F1 Score: {results['f1']:.2f}")
    print(f"Total: {results['total']}")
    print(f"HasAns_exact: {results['HasAns_exact']:.2f}")
    print(f"HasAns_f1: {results['HasAns_f1']:.2f}")
    print(f"HasAns_total: {results['HasAns_total']}")
    print(f"NoAns_exact: {results['NoAns_exact']:.2f}")
    print(f"NoAns_f1: {results['NoAns_f1']:.2f}")
    print(f"NoAns_total: {results['NoAns_total']}")
    print(f"Best_exact: {results['best_exact']:.2f}")
    print(f"Best_exact_thresh: {results['best_exact_thresh']:.2f}")
    print(f"Best_f1: {results['best_f1']:.2f}")
    print(f"Best_f1_thresh: {results['best_f1_thresh']:.2f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate a fine-tuned model on SQuAD v2.0 dataset")
    parser.add_argument("model_path", type=str,
                        help="Path to the saved fine-tuned model")

    args = parser.parse_args()

    evaluate_squad_v2(args.model_path)
