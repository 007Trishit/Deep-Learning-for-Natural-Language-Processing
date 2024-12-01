import pandas as pd
import numpy as np
from bertopic import BERTopic
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from sklearn.preprocessing import MultiLabelBinarizer
import plotly.express as px

# Load the data


def load_data(train_path, test_path):
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    # Combine title and abstract for better topic modeling
    train_df['text'] = train_df['TITLE'] + ' ' + train_df['ABSTRACT']
    test_df['text'] = test_df['TITLE'] + ' ' + test_df['ABSTRACT']

    return train_df, test_df

# Convert multi-label columns to binary format


def prepare_labels(df):
    label_columns = ['Computer Science', 'Physics', 'Mathematics',
                     'Statistics', 'Quantitative Biology', 'Quantitative Finance']
    return df[label_columns].values

# Train BERTopic model and get topics


def train_bertopic(docs, n_topics=10):
    # Initialize and train BERTopic
    topic_model = BERTopic(nr_topics=n_topics, verbose=True)
    topics, probs = topic_model.fit_transform(docs)

    return topic_model, topics, probs

# Evaluate topic quality


def evaluate_topics(true_labels, predicted_topics):
    # Convert predicted topics to binary format for comparison
    n_topics = len(np.unique(predicted_topics))
    topic_binary = np.zeros((len(predicted_topics), n_topics))
    for i, topic in enumerate(predicted_topics):
        topic_binary[i, topic] = 1

    # Calculate metrics
    nmi = normalized_mutual_info_score(
        true_labels.argmax(axis=1), predicted_topics)
    ari = adjusted_rand_score(true_labels.argmax(axis=1), predicted_topics)

    return {
        'NMI': nmi,
        'ARI': ari
    }

# Visualize topic distribution


def plot_topic_distribution(topic_model):
    # Get topic info
    topic_info = topic_model.get_topic_info()

    # Create bar plot
    fig = px.bar(topic_info.head(10),
                 x='Topic',
                 y='Count',
                 title='Top 10 Topics Distribution')
    return fig


def main():
    # Load data
    train_df, test_df = load_data(
        '../data/Research Articles/train.csv', '../data/Research Articles/test.csv')

    # Prepare documents
    train_docs = train_df['text'].tolist()
    test_docs = test_df['text'].tolist()

    # Prepare labels
    train_labels = prepare_labels(train_df)
    test_labels = prepare_labels(test_df)

    # Train model
    topic_model, topics, probs = train_bertopic(train_docs)

    # Get topics for test set
    test_topics, test_probs = topic_model.transform(test_docs)

    # Evaluate
    train_metrics = evaluate_topics(train_labels, topics)
    test_metrics = evaluate_topics(test_labels, test_topics)

    print("\nTraining Set Metrics:")
    for metric, value in train_metrics.items():
        print(f"{metric}: {value:.3f}")

    print("\nTest Set Metrics:")
    for metric, value in test_metrics.items():
        print(f"{metric}: {value:.3f}")

    # Print top topics
    print("\nTop Topics and their Keywords:")
    for topic in topic_model.get_topic_info().head(5).itertuples():
        print(f"\nTopic {topic.Topic}:")
        if topic.Topic != -1:  # Skip outlier topic
            words = topic_model.get_topic(topic.Topic)
            print(", ".join([word[0] for word in words[:5]]))


if __name__ == "__main__":
    main()
