import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch
from typing import List, Dict, Tuple
from tqdm import tqdm


class TopicGPT:
    def __init__(self):
        # Load GPT-2 model and tokenizer
        self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        self.model = GPT2LMHeadModel.from_pretrained('gpt2')
        if torch.cuda.is_available():
            self.model = self.model.cuda()
        self.model.eval()

        # Load sentence transformer for topic refinement
        self.sentence_model = SentenceTransformer('all-mpnet-base-v2')
        self.topics = []

    def generate_text(self, prompt: str, max_length: int = 100) -> str:
        """Generate text using GPT-2"""
        inputs = self.tokenizer.encode(prompt, return_tensors='pt')
        if torch.cuda.is_available():
            inputs = inputs.cuda()

        with torch.no_grad():
            outputs = self.model.generate(
                inputs,
                max_length=max_length,
                num_return_sequences=1,
                no_repeat_ngram_size=2,
                do_sample=True,
                top_k=50,
                top_p=0.95,
                temperature=0.7,
                pad_token_id=self.tokenizer.eos_token_id
            )

        generated_text = self.tokenizer.decode(
            outputs[0], skip_special_tokens=True)
        # Extract the generated part after the prompt
        return generated_text[len(prompt):].strip()

    def generate_topics(self, documents: List[str], example_topics: List[str]) -> List[str]:
        """Generate topics using GPT-2"""
        topics = []
        print("Generating topics...")

        # Process first 100 docs for efficiency
        for doc in tqdm(documents[:100]):
            prompt = f"""Given the document and example topics, identify a relevant topic.

Example topics:
{' '.join(example_topics)}

Document:
{doc[:500]}  # Truncate document to fit context window

Topic:"""

            try:
                generated_topic = self.generate_text(prompt, max_length=50)
                # Clean up the generated topic
                if ':' in generated_topic:
                    generated_topic = generated_topic.split(':')[0] + ':'
                topics.append(generated_topic)
            except Exception as e:
                print(f"Error generating topic: {e}")
                topics.append("Error")

        return self._refine_topics(topics)

    def _refine_topics(self, topics: List[str]) -> List[str]:
        """Refine topics by merging similar ones and removing infrequent ones"""
        print("Refining topics...")
        # Embed topics
        embeddings = self.sentence_model.encode(topics)

        # Calculate similarity matrix
        similarity_matrix = np.inner(embeddings, embeddings)

        # Merge similar topics (similarity > 0.8)
        merged_topics = []
        used_indices = set()

        for i in range(len(topics)):
            if i in used_indices:
                continue

            similar_indices = np.where(similarity_matrix[i] > 0.8)[0]
            used_indices.update(similar_indices)

            if len(similar_indices) > 1:
                # Take the most frequent topic from similar ones
                similar_topics = [topics[idx] for idx in similar_indices]
                merged_topics.append(
                    max(similar_topics, key=similar_topics.count))
            else:
                merged_topics.append(topics[i])

        # Remove duplicates while preserving order
        seen = set()
        refined_topics = [x for x in merged_topics if not (
            x in seen or seen.add(x))]

        # Remove infrequent topics (appearing less than 2 times)
        topic_counts = {topic: merged_topics.count(
            topic) for topic in refined_topics}
        final_topics = [
            topic for topic in refined_topics if topic_counts[topic] >= 2]

        return final_topics

    def assign_topics(self, documents: List[str]) -> List[str]:
        """Assign topics to documents using semantic similarity"""
        print("Assigning topics...")
        assignments = []

        # Encode all topics once
        topic_embeddings = self.sentence_model.encode(self.topics)

        # Process documents in batches
        batch_size = 32
        for i in tqdm(range(0, len(documents), batch_size)):
            batch_docs = documents[i:i + batch_size]

            # Encode documents
            doc_embeddings = self.sentence_model.encode(batch_docs)

            # Calculate similarity with all topics
            similarity_matrix = np.inner(doc_embeddings, topic_embeddings)

            # Assign most similar topic
            batch_assignments = [self.topics[idx]
                                 for idx in similarity_matrix.argmax(axis=1)]
            assignments.extend(batch_assignments)

        return assignments


def evaluate_topics(true_labels: np.ndarray, predicted_topics: List[str], unique_topics: List[str]) -> Dict[str, float]:
    """Evaluate topic assignments"""
    # Convert predicted topics to numeric labels
    topic_to_id = {topic: idx for idx, topic in enumerate(unique_topics)}
    predicted_labels = np.array([topic_to_id[topic]
                                for topic in predicted_topics])

    # Calculate metrics
    nmi = normalized_mutual_info_score(
        true_labels.argmax(axis=1), predicted_labels)
    ari = adjusted_rand_score(true_labels.argmax(axis=1), predicted_labels)

    return {
        'NMI': nmi,
        'ARI': ari
    }


def main():
    # Load data
    print("Loading data...")
    train_df = pd.read_csv('../data/Research Articles/train.csv')
    test_df = pd.read_csv('../data/Research Articles/test.csv')

    # Combine title and abstract
    train_df['text'] = train_df['TITLE'] + ' ' + train_df['ABSTRACT']
    test_df['text'] = test_df['TITLE'] + ' ' + test_df['ABSTRACT']

    # Example topics based on the dataset categories
    example_topics = [
        "Computer Science: Research on algorithms and computation",
        "Physics: Studies of physical phenomena",
        "Mathematics: Theoretical mathematical concepts",
        "Statistics: Statistical methods",
        "Quantitative Biology: Mathematical modeling in biology",
        "Quantitative Finance: Mathematical methods in finance"
    ]

    # Initialize TopicGPT
    topic_gpt = TopicGPT()

    # Generate topics
    topics = topic_gpt.generate_topics(
        train_df['text'].tolist(), example_topics)
    topic_gpt.topics = topics

    print("\nGenerated Topics:")
    for topic in topics:
        print(f"- {topic}")

    # Assign topics
    train_assignments = topic_gpt.assign_topics(train_df['text'].tolist())
    test_assignments = topic_gpt.assign_topics(test_df['text'].tolist())

    # Prepare true labels
    label_columns = ['Computer Science', 'Physics', 'Mathematics',
                     'Statistics', 'Quantitative Biology', 'Quantitative Finance']
    train_labels = train_df[label_columns].values
    test_labels = test_df[label_columns].values

    # Evaluate
    train_metrics = evaluate_topics(train_labels, train_assignments, topics)
    test_metrics = evaluate_topics(test_labels, test_assignments, topics)

    print("\nTraining Set Metrics:")
    for metric, value in train_metrics.items():
        print(f"{metric}: {value:.3f}")

    print("\nTest Set Metrics:")
    for metric, value in test_metrics.items():
        print(f"{metric}: {value:.3f}")

    # Print sample assignments
    print("\nSample Assignments:")
    for i in range(5):
        print(f"\nDocument: {train_df['TITLE'].iloc[i]}")
        print(f"Assigned Topic: {train_assignments[i]}")


if __name__ == "__main__":
    main()
