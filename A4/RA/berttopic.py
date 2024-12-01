import pandas as pd
import numpy as np
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
import gensim.corpora as corpora
from gensim.models.coherencemodel import CoherenceModel
from sklearn.feature_extraction.text import CountVectorizer
import re
import nltk
from nltk.corpus import stopwords
nltk.download('stopwords', quiet=True)
nltk.download('punkt', quiet=True)


def clean_text(text):
    """Clean and preprocess text."""
    if not isinstance(text, str):
        return ""

    # Remove special characters but keep useful punctuation
    text = re.sub(r'[^a-zA-Z\s\-]', ' ', text)

    # Convert to lowercase and strip
    text = text.lower().strip()

    # Remove extra whitespace
    text = ' '.join(text.split())

    return text


def preprocess_data(filepath):
    """Load and preprocess the data."""
    df = pd.read_csv(filepath)

    # Clean and combine title and abstract
    df['TITLE'] = df['TITLE'].apply(clean_text)
    df['ABSTRACT'] = df['ABSTRACT'].apply(clean_text)
    df['text'] = df['TITLE'] + ' ' + df['ABSTRACT']

    return df['text'].tolist()


def calculate_npmi_coherence(documents, topics_words):
    """Calculate NPMI coherence score for topics."""
    try:
        # Tokenize documents
        tokenized_docs = [doc.split() for doc in documents]

        # Create dictionary
        dictionary = corpora.Dictionary(tokenized_docs)

        # Filter topics to ensure they contain valid words
        filtered_topics = [
            [w for w in topic if w in dictionary.token2id]
            for topic in topics_words
        ]

        # Remove empty topics
        filtered_topics = [
            topic for topic in filtered_topics if len(topic) > 1]

        if not filtered_topics:
            return 0.0

        # Calculate coherence
        coherence_model = CoherenceModel(
            topics=filtered_topics,
            texts=tokenized_docs,
            dictionary=dictionary,
            coherence='c_npmi'
        )

        return coherence_model.get_coherence()

    except Exception as e:
        print(f"Error calculating coherence: {str(e)}")
        return 0.0


def calculate_topic_diversity(topics_words):
    """Calculate topic diversity as ratio of unique words."""
    if not topics_words:
        return 0.0

    unique_words = set(word for topic in topics_words for word in topic)
    total_words = sum(len(topic) for topic in topics_words)

    return len(unique_words) / total_words if total_words > 0 else 0.0


def main():
    try:
        print("Loading datasets...")
        train_texts = preprocess_data('train.csv')
        test_texts = preprocess_data('test.csv')

        print(f"Loaded {len(train_texts)} training and {
              len(test_texts)} test documents")

        # Custom stop words for scientific text
        stop_words = set(stopwords.words('english'))
        stop_words.update(['et', 'al', 'fig', 'figure', 'table'])

        # Initialize BERTopic with adjusted settings
        print("Initializing BERTopic model...")
        sentence_model = SentenceTransformer('all-MiniLM-L6-v2')

        vectorizer = CountVectorizer(
            stop_words=list(stop_words),
            min_df=2,        # At least 2 documents
            max_df=0.95,     # Present in at most 95% of documents
            ngram_range=(1, 2)  # Include bigrams
        )

        topic_model = BERTopic(
            embedding_model=sentence_model,
            vectorizer_model=vectorizer,
            min_topic_size=2,     # Minimum 2 documents per topic
            nr_topics=6,     # Let the model determine number of topics
            top_n_words=20,       # Get more words per topic for better evaluation
            verbose=True
        )

        # Fit model
        print("Fitting BERTopic model...")
        topics, probs = topic_model.fit_transform(train_texts)

        # Get topic info
        topic_info = topic_model.get_topic_info()
        print(f"\nDetected {len(topic_info)-1} topics")

        # Transform test set
        print("\nEvaluating on test set...")
        test_topics, test_probs = topic_model.transform(test_texts)

        # Get topic words for evaluation
        topics_words = []
        for topic in topic_model.get_topics():
            # Get top 10 words for each topic without their scores
            words = [word for word, _ in topic_model.get_topic(topic)[:10]]
            topics_words.append(words)

        # Calculate metrics
        coherence = calculate_npmi_coherence(test_texts, topics_words)
        diversity = calculate_topic_diversity(topics_words)

        print("\nMetrics:")
        print(f"Topic Coherence (NPMI): {coherence:.3f}")
        print(f"Topic Diversity: {diversity:.3f}")

        # Save topic information and metrics
        topic_info.to_csv('topic_info.csv', index=False)
        pd.DataFrame({
            'Metric': ['Topic Coherence', 'Topic Diversity'],
            'Value': [coherence, diversity]
        }).to_csv('topic_modeling_results.csv', index=False)

        # Print example topics
        print("\nTop 5 Topics with Words:")
        for idx, topic in enumerate(topic_model.get_topics()):
            if idx < 5 and idx != -1:  # Skip outlier topic (-1)
                words = [f"{word}" for word, score in topic_model.get_topic(idx)[
                    :5]]
                print(f"Topic {idx}: {', '.join(words)}")

    except Exception as e:
        print(f"An error occurred: {str(e)}")
        raise


if __name__ == "__main__":
    main()
