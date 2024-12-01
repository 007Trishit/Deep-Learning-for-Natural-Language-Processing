from sentence_transformers import SentenceTransformer
from sklearn.model_selection import KFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, mean_absolute_error
import numpy as np
from typing import List, Tuple, Dict
import re
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch


class PromptTopicLLaMA:
    def __init__(self, n_folds: int = 10, n_topics: int = 10):
        self.n_folds = n_folds
        self.n_topics = n_topics
        self.sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
        self.rating_predictor = LogisticRegression(
            multi_class='multinomial', max_iter=1000)

        # Initialize LLaMA
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b")
        self.model = AutoModelForCausalLM.from_pretrained(
            "meta-llama/Llama-2-7b").to(self.device)

        # Example topics for restaurant domain
        self.example_topics = [
            "Food Quality: Discussion of taste, presentation, and freshness of dishes",
            "Service: Mentions of staff behavior, attentiveness, and overall service experience",
            "Ambiance: Description of atmosphere, decor, and overall feel of the establishment",
            "Value: Comments about pricing, portion sizes, and value for money",
            "Cleanliness: Discussion of hygiene standards and cleanliness"
        ]

    def preprocess_text(self, text: str) -> str:
        if not isinstance(text, str):
            return ""
        text = text.lower()
        text = re.sub(r'[^\w\s]', ' ', text)
        text = re.sub(r'\s+', ' ', text)
        return text.strip()

    def create_folds(self, texts: List[str], ratings: List[int]) -> List[Tuple]:
        kf = KFold(n_splits=self.n_folds, shuffle=True, random_state=42)
        splits = []

        print(f"\nCreating {self.n_folds}-fold cross validation splits")
        for fold_idx, (train_idx, val_idx) in enumerate(kf.split(texts), 1):
            X_train = [texts[i] for i in train_idx]
            X_val = [texts[i] for i in val_idx]
            y_train = [ratings[i] for i in train_idx]
            y_val = [ratings[i] for i in val_idx]

            print(f"Fold {fold_idx}: Train size = {
                  len(X_train)}, Val size = {len(X_val)}")
            splits.append((X_train, X_val, y_train, y_val))

        return splits

    def assign_topic_llama(self, text: str) -> str:
        """Use LLaMA to assign a topic to a given text"""
        prompt = f"""Please assign the most relevant topic to this restaurant review from the following options:
{self.example_topics}

Review: {text}

Most relevant topic:"""

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=50,
            temperature=0.1,
            top_p=0.95,
            num_return_sequences=1
        )
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return response.split("Most relevant topic:")[-1].strip()

    def get_topic_probabilities(self, texts: List[str], topics: List[str]) -> np.ndarray:
        """Convert topic assignments to probability distributions using embeddings"""
        text_embeddings = self.sentence_model.encode(texts)
        topic_embeddings = self.sentence_model.encode(topics)

        # Calculate similarities
        similarities = np.dot(text_embeddings, topic_embeddings.T)

        # Convert to probabilities using softmax
        probs = np.exp(similarities) / \
            np.exp(similarities).sum(axis=1, keepdims=True)
        return probs

    def analyze(self, reviews: List[str], ratings: List[int]) -> Dict:
        print(f"Initial data: {len(reviews)} reviews")

        processed_reviews = [self.preprocess_text(
            review) for review in reviews]
        processed_reviews = [r for r in processed_reviews if r]

        if len(processed_reviews) != len(ratings):
            ratings = ratings[:len(processed_reviews)]

        print(f"After preprocessing: {len(processed_reviews)} reviews")

        rating_dist = pd.Series(ratings).value_counts().sort_index()
        print("\nRatings distribution:")
        for rating, count in rating_dist.items():
            print(f"Rating {rating}: {count} reviews ({
                  count/len(ratings)*100:.1f}%)")

        splits = self.create_folds(processed_reviews, ratings)
        results = []

        for fold_idx, (X_train, X_val, y_train, y_val) in enumerate(splits, 1):
            print(f"\nProcessing fold {fold_idx}/{self.n_folds}")

            # Assign topics using LLaMA
            train_topics = [self.assign_topic_llama(text) for text in X_train]
            print("Generated topic assignments for training set")

            # Get topic probabilities
            train_probs = self.get_topic_probabilities(
                X_train, self.example_topics)
            print(f"Topic feature shape: {train_probs.shape}")

            # Get validation features
            val_probs = self.get_topic_probabilities(
                X_val, self.example_topics)
            print(f"Validation feature shape: {val_probs.shape}")

            # Train and evaluate rating predictor
            self.rating_predictor.fit(train_probs, y_train)
            y_pred = self.rating_predictor.predict(val_probs)

            metrics = {
                'accuracy': accuracy_score(y_val, y_pred),
                'mae': mean_absolute_error(y_val, y_pred)
            }
            results.append(metrics)
            print(f"Fold {fold_idx} results:", metrics)

            # Display topic distribution
            topic_counts = pd.Series(train_topics).value_counts()
            print("\nTopic Distribution:")
            for topic, count in topic_counts.items():
                print(f"{topic}: {count} documents ({
                      count/len(train_topics)*100:.1f}%)")

        avg_results = {
            metric: {
                'mean': np.mean([r[metric] for r in results]),
                'std': np.std([r[metric] for r in results])
            }
            for metric in results[0].keys()
        }

        return {
            'fold_results': results,
            'average_results': avg_results
        }


def main():
    # Load data
    try:
        df = pd.read_csv('../data/TripAdvisor/New_Delhi_reviews.csv')
    except:
        print("Error loading data file")
        return

    # Convert ratings and reviews to lists
    ratings = df['rating_review'].tolist()
    reviews = df['review_full'].tolist()

    # Initialize analyzer
    analyzer = PromptTopicLLaMA(
        n_folds=10, n_topics=5)  # Using 5 example topics

    try:
        # Run analysis
        results = analyzer.analyze(reviews, ratings)

        # Print final results
        print("\nFinal Results (averaged over {} folds):".format(analyzer.n_folds))
        for metric, values in results['average_results'].items():
            print(f"{metric}: {values['mean']:.4f} ± {values['std']:.4f}")

    except Exception as e:
        print(f"Error during analysis: {str(e)}")
        raise


if __name__ == "__main__":
    main()
