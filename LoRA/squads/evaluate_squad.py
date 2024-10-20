import argparse
from datasets import load_dataset, load_metric
from transformers import pipeline, AutoModelForQuestionAnswering, AutoTokenizer
from tqdm import tqdm


def evaluate_squad_v1(model_path):
    # Load the fine-tuned model and tokenizer
    model = AutoModelForQuestionAnswering.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    # Create the question-answering pipeline
    qa_pipeline = pipeline("question-answering",
                           model=model, tokenizer=tokenizer, device="cuda:7")

    # Load the SQuAD v1.1 dataset
    dataset = load_dataset("squad", split="validation")

    # Load the SQuAD metric
    squad_metric = load_metric("squad")

    # Prepare lists to store predictions and references
    predictions = []
    references = []

    # Evaluate the model on the dataset
    for example in tqdm(dataset, desc="Evaluating"):
        context = example["context"]
        question = example["question"]

        # Get the model's prediction
        prediction = qa_pipeline(question=question, context=context)

        # Prepare the prediction and reference dictionaries
        pred_dict = {
            "id": example["id"],
            "prediction_text": prediction["answer"]
        }

        ref_dict = {
            "id": example["id"],
            "answers": example["answers"]
        }

        predictions.append(pred_dict)
        references.append(ref_dict)

    # Compute the metrics
    results = squad_metric.compute(
        predictions=predictions, references=references)

    # Print the results
    print("\nEvaluation results on SQuAD v1.1:")
    print(f"Exact Match: {results['exact_match']:.2f}")
    print(f"F1 Score: {results['f1']:.2f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate a fine-tuned model on SQuAD v1.1 dataset")
    parser.add_argument("model_path", type=str,
                        help="Path to the saved fine-tuned model")

    args = parser.parse_args()

    evaluate_squad_v1(args.model_path)
