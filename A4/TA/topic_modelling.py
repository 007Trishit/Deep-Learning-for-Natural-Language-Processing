import pandas as pd
import numpy as np
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
import torch
import re
import logging
from typing import List, Dict, Tuple
import os


class TopicModelingComparison:
    def __init__(self, n_splits: int = 10, val_size: float = 0.2):
        """
        Initialize the topic modeling comparison class
        
        Args:
            n_splits: Number of train-validation splits to average over
            val_size: Size of validation set as a fraction of total data
        """
        self.n_splits = n_splits
        self.val_size = val_size

        # Initialize models
        self.bert_model = None
        self.sentence_transformer = SentenceTransformer('all-MiniLM-L6-v2')

        # Initialize GPT-2
        self.gpt2_tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        self.gpt2_model = GPT2LMHeadModel.from_pretrained('gpt2')
        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu')
        self.gpt2_model.to(self.device)

    def create_splits(self, texts: List[str]) -> List[Tuple[List[str], List[str]]]:
        """Create multiple train-validation splits"""
        splits = []
        for _ in range(self.n_splits):
            train_texts, val_texts = train_test_split(
                texts,
                test_size=self.val_size,
                random_state=np.random.randint(0, 10000)
            )
            splits.append((train_texts, val_texts))
        return splits

    def load_data(self) -> pd.DataFrame:
        """Load and preprocess the TripAdvisor dataset"""
        # Read the CSV using fs API
        file_content = window.fs.readFile('trip_adv.csv', {encoding: 'utf8'})
        df = pd.read_csv(pd.StringIO(file_content))

        # Basic preprocessing
        df['review_full'] = df['review_full'].fillna('')
        df['review_full'] = df['review_full'].astype(str)
        df['review_full'] = df['review_full'].apply(
            lambda x: re.sub(r'[^\w\s]', '', x))

        return df

    def run_bertopic(self, train_texts: List[str], val_texts: List[str], num_topics: int = 10) -> Tuple[BERTopic, List[int]]:
        """Run BERTopic model on the texts"""
        # Initialize and fit BERTopic on training data
        vectorizer = CountVectorizer(stop_words="english")
        model = BERTopic(
            vectorizer_model=vectorizer,
            nr_topics=num_topics,
            min_topic_size=5
        )
        model.fit(train_texts)

        # Transform validation data
        topics, probs = model.transform(val_texts)
        return model, topics

    def generate_topic_gpt2(self, text: str, max_length: int = 50) -> str:
        """Generate topic using GPT-2"""
        prompt = f"Main topics in this review: {text[:200]}\nTopics:"

        inputs = self.gpt2_tokenizer.encode(prompt, return_tensors='pt')
        inputs = inputs.to(self.device)

        with torch.no_grad():
            outputs = self.gpt2_model.generate(
                inputs,
                max_length=max_length,
                num_return_sequences=1,
                pad_token_id=self.gpt2_tokenizer.eos_token_id,
                temperature=0.7,
                do_sample=True,
                top_k=50,
                top_p=0.95
            )

        topic = self.gpt2_tokenizer.decode(
            outputs[0], skip_special_tokens=True)
        topic = topic.replace(prompt, '').strip()
        return topic

    def run_topicgpt(self, train_texts: List[str], val_texts: List[str], num_topics: int = 10) -> List[Dict]:
        """Run TopicGPT on validation texts after learning from training texts"""
        # First, learn topics from training data
        train_topics = []
        sample_size = min(100, len(train_texts))
        sampled_train = np.random.choice(
            train_texts, sample_size, replace=False)

        for text in sampled_train:
            try:
                topic = self.generate_topic_gpt2(text)
                train_topics.append({"text": text, "topic": topic})
            except Exception as e:
                logging.error(f"Error in TopicGPT training: {str(e)}")
                continue

        # Then generate topics for validation data
        val_topics = []
        for text in val_texts:
            try:
                topic = self.generate_topic_gpt2(text)
                val_topics.append({"text": text, "topic": topic})
            except Exception as e:
                logging.error(f"Error in TopicGPT validation: {str(e)}")
                continue

        return val_topics

    def run_prompttopic(self, train_texts: List[str], val_texts: List[str], num_topics: int = 10) -> List[Dict]:
        """Run PromptTopic on validation texts after learning from training texts"""
        prompt_template = """
        Review topics to identify:
        1. Service quality
        2. Food quality
        3. Ambiance
        4. Value for money
        
        Review: {text}
        
        Topics found:
        """

        # Learn from training data
        train_topics = []
        sample_size = min(100, len(train_texts))
        sampled_train = np.random.choice(
            train_texts, sample_size, replace=False)

        for text in sampled_train:
            try:
                formatted_prompt = prompt_template.format(text=text[:200])
                generated_topics = self.generate_topic_gpt2(formatted_prompt)
                topic_list = [t.strip()
                              for t in generated_topics.split('\n') if t.strip()]
                train_topics.append({"text": text, "topics": topic_list})
            except Exception as e:
                logging.error(f"Error in PromptTopic training: {str(e)}")
                continue

        # Generate topics for validation data
        val_topics = []
        for text in val_texts:
            try:
                formatted_prompt = prompt_template.format(text=text[:200])
                generated_topics = self.generate_topic_gpt2(formatted_prompt)
                topic_list = [t.strip()
                              for t in generated_topics.split('\n') if t.strip()]
                val_topics.append({"text": text, "topics": topic_list})
            except Exception as e:
                logging.error(f"Error in PromptTopic validation: {str(e)}")
                continue

        return val_topics

    def evaluate_coherence(self, model_topics: List[Dict]) -> float:
        """Calculate topic coherence score"""
        coherence_scores = []

        for topic in model_topics:
            if isinstance(topic, dict) and 'topics' in topic:
                # For PromptTopic/TopicGPT
                topic_words = ' '.join(topic['topics']).lower().split()
            else:
                # For BERTopic
                topic_words = str(topic).lower().split()

            # Calculate word co-occurrence
            word_pairs = [(topic_words[i], topic_words[j])
                          for i in range(len(topic_words))
                          for j in range(i+1, len(topic_words))]

            if word_pairs:
                coherence_scores.append(len(set(word_pairs)) / len(word_pairs))

        return np.mean(coherence_scores) if coherence_scores else 0.0

    def run_comparison(self, num_topics: int = 10):
        """Run and compare all three topic modeling approaches over multiple splits"""
        # Load data
        df = self.load_data()
        texts = df['review_full'].tolist()

        # Create splits
        splits = self.create_splits(texts)

        # Store results for each split
        bert_coherence_scores = []
        topicgpt_coherence_scores = []
        prompttopic_coherence_scores = []

        # Run models on each split
        for i, (train_texts, val_texts) in enumerate(splits, 1):
            print(f"\nProcessing split {i}/{self.n_splits}")

            # Run BERTopic
            print("Running BERTopic...")
            _, bert_topics = self.run_bertopic(
                train_texts, val_texts, num_topics)
            bert_coherence = self.evaluate_coherence(bert_topics)
            bert_coherence_scores.append(bert_coherence)

            # Run TopicGPT
            print("Running TopicGPT...")
            topicgpt_topics = self.run_topicgpt(
                train_texts, val_texts, num_topics)
            topicgpt_coherence = self.evaluate_coherence(topicgpt_topics)
            topicgpt_coherence_scores.append(topicgpt_coherence)

            # Run PromptTopic
            print("Running PromptTopic...")
            prompttopic_topics = self.run_prompttopic(
                train_texts, val_texts, num_topics)
            prompttopic_coherence = self.evaluate_coherence(prompttopic_topics)
            prompttopic_coherence_scores.append(prompttopic_coherence)

        # Calculate average scores and standard deviations
        results = {
            'bertopic': {
                'mean_coherence': np.mean(bert_coherence_scores),
                'std_coherence': np.std(bert_coherence_scores)
            },
            'topicgpt': {
                'mean_coherence': np.mean(topicgpt_coherence_scores),
                'std_coherence': np.std(topicgpt_coherence_scores)
            },
            'prompttopic': {
                'mean_coherence': np.mean(prompttopic_coherence_scores),
                'std_coherence': np.std(prompttopic_coherence_scores)
            }
        }

        # Print results
        print("\nFinal Results (averaged over {} splits):".format(self.n_splits))
        print(
            f"BERTopic Coherence: {results['bertopic']['mean_coherence']:.4f} ± {results['bertopic']['std_coherence']:.4f}")
        print(
            f"TopicGPT Coherence: {results['topicgpt']['mean_coherence']:.4f} ± {results['topicgpt']['std_coherence']:.4f}")
        print(
            f"PromptTopic Coherence: {results['prompttopic']['mean_coherence']:.4f} ± {results['prompttopic']['std_coherence']:.4f}")

        return results


# Usage example
if __name__ == "__main__":
    # Initialize the comparison class with 10 splits
    comparison = TopicModelingComparison(n_splits=10, val_size=0.2)

    # Run the comparison
    results = comparison.run_comparison(num_topics=10)
