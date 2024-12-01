import pandas as pd
import numpy as np
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from collections import defaultdict
import torch
from typing import List, Dict, Tuple
from tqdm import tqdm


class PromptTopic:
    def __init__(self, K: int = 20, G: int = 40):
        """
        K: final number of topics
        G: intermediate number of topics (G > K)
        """
        # Load models
        self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        self.model = GPT2LMHeadModel.from_pretrained('gpt2')
        self.sentence_model = SentenceTransformer('all-mpnet-base-v2')

        if torch.cuda.is_available():
            self.model = self.model.cuda()
        self.model.eval()

        self.K = K
        self.G = G
        self.topics = []
        self.doc_topic_map = {}

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

        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)[len(prompt):].strip()

    def stage1_document_topic_generation(self, documents: List[str]) -> List[str]:
        """Stage 1: Generate initial topics for each document"""
        print("Stage 1: Generating document topics...")
        topics = []
        doc_topics = {}

        for idx, doc in enumerate(tqdm(documents)):
            prompt = f"""Extract the main topic from this research article text:

Text: {doc[:500]}

Topic (respond with a single topic phrase):"""

            topic = self.generate_text(prompt, max_length=50)
            topics.append(topic)
            doc_topics[idx] = topic

        self.doc_topic_map = doc_topics
        return topics

    def calculate_ctfidf(self, documents: List[str], topics: List[str]) -> Dict[str, List[str]]:
        """Calculate class-based TF-IDF for topics"""
        # Group documents by topic
        topic_docs = defaultdict(list)
        for doc_idx, doc in enumerate(documents):
            topic = self.doc_topic_map[doc_idx]
            topic_docs[topic].append(doc)

        # Calculate c-TF-IDF for each topic
        vectorizer = TfidfVectorizer(max_features=1000)
        topic_words = {}

        for topic in topics:
            if topic in topic_docs:
                topic_docs_text = " ".join(topic_docs[topic])
                # Get TF-IDF matrix for topic documents
                tfidf_matrix = vectorizer.fit_transform([topic_docs_text])
                # Get feature names and their scores
                feature_names = vectorizer.get_feature_names_out()
                scores = tfidf_matrix.toarray()[0]
                # Sort words by score
                sorted_idx = np.argsort(scores)[::-1]
                topic_words[topic] = [feature_names[i]
                                      for i in sorted_idx[:20]]

        return topic_words

    def stage2_pbm_collapse(self, topics: List[str]) -> List[str]:
        """Stage 2: Collapse topics using Prompt-Based Matching"""
        print("Stage 2: Collapsing topics using PBM...")
        collapsed_topics = set(topics)

        while len(collapsed_topics) > self.G:
            # Calculate topic similarities using sentence embeddings
            topic_embeddings = self.sentence_model.encode(
                list(collapsed_topics))
            similarity_matrix = np.inner(topic_embeddings, topic_embeddings)

            # Find most similar topic pairs
            np.fill_diagonal(similarity_matrix, -1)
            max_sim_idx = np.unravel_index(
                similarity_matrix.argmax(), similarity_matrix.shape)

            topic1 = list(collapsed_topics)[max_sim_idx[0]]
            topic2 = list(collapsed_topics)[max_sim_idx[1]]

            # Generate merged topic using prompt
            prompt = f"""Merge these two similar research topics into a single topic:

Topic 1: {topic1}
Topic 2: {topic2}

Merged topic (respond with a single topic phrase):"""

            merged_topic = self.generate_text(prompt, max_length=50)

            # Update topic set
            collapsed_topics.remove(topic1)
            collapsed_topics.remove(topic2)
            collapsed_topics.add(merged_topic)

            # Update doc-topic mapping
            for doc_id, topic in self.doc_topic_map.items():
                if topic in [topic1, topic2]:
                    self.doc_topic_map[doc_id] = merged_topic

        return list(collapsed_topics)

    def stage2_wsm_collapse(self, topics: List[str], topic_words: Dict[str, List[str]]) -> List[str]:
        """Stage 2: Collapse topics using Word Similarity Matching"""
        print("Stage 2: Collapsing topics using WSM...")
        collapsed_topics = set(topics)

        while len(collapsed_topics) > self.K:
            max_similarity = 0
            merge_pair = None

            # Find topic pair with highest word overlap
            topic_list = list(collapsed_topics)
            for i in range(len(topic_list)):
                for j in range(i + 1, len(topic_list)):
                    topic1 = topic_list[i]
                    topic2 = topic_list[j]

                    if topic1 in topic_words and topic2 in topic_words:
                        words1 = set(topic_words[topic1])
                        words2 = set(topic_words[topic2])
                        similarity = len(words1 & words2) / \
                            len(words1 | words2)

                        if similarity > max_similarity:
                            max_similarity = similarity
                            merge_pair = (topic1, topic2)

            if merge_pair and max_similarity > 0.3:  # Threshold for merging
                topic1, topic2 = merge_pair
                # Keep the topic with more documents as the representative
                topic1_docs = sum(
                    1 for t in self.doc_topic_map.values() if t == topic1)
                topic2_docs = sum(
                    1 for t in self.doc_topic_map.values() if t == topic2)

                merged_topic = topic1 if topic1_docs >= topic2_docs else topic2

                # Update topic set
                collapsed_topics.remove(topic1)
                collapsed_topics.remove(topic2)
                collapsed_topics.add(merged_topic)

                # Update doc-topic mapping
                for doc_id, topic in self.doc_topic_map.items():
                    if topic in [topic1, topic2]:
                        self.doc_topic_map[doc_id] = merged_topic
            else:
                break

        return list(collapsed_topics)

    def stage3_topic_word_generation(self, documents: List[str], topics: List[str]) -> Dict[str, List[str]]:
        """Stage 3: Generate representative words for each topic using c-TF-IDF"""
        print("Stage 3: Generating topic words...")
        return self.calculate_ctfidf(documents, topics)

    def fit_transform(self, documents: List[str], method: str = 'wsm') -> Tuple[List[str], Dict[str, List[str]]]:
        """Main method to run the complete pipeline"""
        # Stage 1: Generate initial topics
        initial_topics = self.stage1_document_topic_generation(documents)

        # Calculate c-TF-IDF for initial topics
        topic_words = self.calculate_ctfidf(documents, initial_topics)

        # Stage 2: Collapse topics
        if method == 'pbm':
            collapsed_topics = self.stage2_pbm_collapse(initial_topics)
        else:  # wsm
            collapsed_topics = self.stage2_wsm_collapse(
                initial_topics, topic_words)

        # Stage 3: Generate final topic words
        final_topic_words = self.stage3_topic_word_generation(
            documents, collapsed_topics)

        self.topics = collapsed_topics
        return collapsed_topics, final_topic_words


def main():
    # Load data
    print("Loading data...")
    train_df = pd.read_csv('../data/Research Articles/train.csv')
    test_df = pd.read_csv('../data/Research Articles/test.csv')

    # Combine title and abstract
    train_df['text'] = train_df['TITLE'] + ' ' + train_df['ABSTRACT']
    test_df['text'] = test_df['TITLE'] + ' ' + test_df['ABSTRACT']

    # Initialize PromptTopic
    prompt_topic = PromptTopic(K=20, G=40)

    # Fit and transform
    topics, topic_words = prompt_topic.fit_transform(train_df['text'].tolist())

    # Print results
    print("\nFinal Topics and their Top Words:")
    for topic in topics:
        if topic in topic_words:
            print(f"\nTopic: {topic}")
            print("Top words:", ", ".join(topic_words[topic][:10]))


if __name__ == "__main__":
    main()
