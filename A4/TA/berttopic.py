from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
from sklearn.model_selection import KFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, mean_absolute_error
import pandas as pd
import numpy as np
import re
from typing import List, Tuple, Dict


class RestaurantReviewAnalyzer:
    def __init__(self, n_folds: int = 10, n_topics: int = 10):
        self.n_folds = n_folds
        self.n_topics = n_topics
        self.sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
        self.topic_model = None
        self.rating_predictor = LogisticRegression(
            multi_class='multinomial', max_iter=1000)

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

    def extract_topics(self, texts: List[str]) -> Tuple[List[int], np.ndarray]:
        self.topic_model = BERTopic(
            embedding_model=self.sentence_model,
            nr_topics=self.n_topics,
            verbose=True
        )
        topics, probs = self.topic_model.fit_transform(texts)
        if probs.ndim == 1:
            probs = probs.reshape(-1, 1)
        return topics, probs

    def get_topic_features(self, texts: List[str]) -> np.ndarray:
        _, probs = self.topic_model.transform(texts)
        if probs.ndim == 1:
            probs = probs.reshape(-1, 1)
        return probs

    def display_topics(self) -> None:
        """Display topics from the fitted model"""
        try:
            # Get topic info
            topic_info = self.topic_model.get_topic_info()

            print("\nTopic Analysis:")
            print("-" * 80)

            # Display topics excluding outliers (-1)
            for _, row in topic_info.iterrows():
                topic_id = row['Topic']
                if topic_id != -1:  # Skip outlier topic
                    # Get words and scores for this topic
                    words = self.topic_model.get_topic(topic_id)

                    # Format output
                    word_str = ", ".join(
                        [f"{word} ({score:.3f})" for word, score in words[:10]])
                    print(f"\nTopic {topic_id} ({row['Count']} documents)")
                    print(f"Top words: {word_str}")

            print("\nTopic Distribution:")
            print(f"Total Topics: {
                  len(topic_info[topic_info['Topic'] != -1])}")
            print(f"Total Documents: {topic_info['Count'].sum()}")

        except Exception as e:
            print(f"Error displaying topics: {str(e)}")

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

            train_topics, train_probs = self.extract_topics(X_train)
            print(f"Topic feature shape: {train_probs.shape}")

            val_probs = self.get_topic_features(X_val)
            print(f"Validation feature shape: {val_probs.shape}")

            self.rating_predictor.fit(train_probs, y_train)
            y_pred = self.rating_predictor.predict(val_probs)

            metrics = {
                'accuracy': accuracy_score(y_val, y_pred),
                'mae': mean_absolute_error(y_val, y_pred)
            }
            results.append(metrics)
            print(f"Fold {fold_idx} results:", metrics)

            # Display topics for each fold
            print(f"\nTopics for fold {fold_idx}:")
            self.display_topics()

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
    df = pd.read_csv('../data/TripAdvisor/New_Delhi_reviews.csv')

    # Convert ratings and reviews to lists
    ratings = df['rating_review'].tolist()
    reviews = df['review_full'].tolist()

    # Initialize analyzer
    analyzer = RestaurantReviewAnalyzer(n_folds=10, n_topics=10)

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
