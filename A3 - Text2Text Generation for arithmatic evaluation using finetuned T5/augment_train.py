import pandas as pd
import nltk
from nltk.corpus import wordnet
import random
import re

# Download required NLTK data files
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('omw-1.4')


def get_synonyms(word):
    """Get synonyms for a word using WordNet."""
    synonyms = set()
    for syn in wordnet.synsets(word):
        for lemma in syn.lemmas():
            # Exclude the word itself to avoid identity replacement
            if lemma.name().lower() != word.lower():
                synonyms.add(lemma.name().replace('_', ' '))
    return list(synonyms)


def replace_synonyms(sentence, n):
    """
    Replace up to n words in the sentence with their synonyms.
    Numbers in the sentence are preserved.
    """
    words = nltk.word_tokenize(sentence)
    new_sentence = words.copy()
    count = 0
    indices = list(range(len(words)))
    random.shuffle(indices)
    for i in indices:
        word = words[i]
        # Skip numbers and punctuation
        if re.match(r'^\d+$', word) or not word.isalpha():
            continue
        synonyms = get_synonyms(word)
        if synonyms:
            synonym = random.choice(synonyms)
            new_sentence[i] = synonym
            count += 1
        if count >= n:
            break
    return ' '.join(new_sentence)


def augment_text(text, num_augmentations=1, num_replacements=2):
    """
    Generate augmented versions of the input text.
    """
    augmented_texts = []
    for _ in range(num_augmentations):
        augmented = replace_synonyms(text, num_replacements)
        augmented_texts.append(augmented)
    return augmented_texts


# Load your existing Excel dataset
# Replace with your actual file path
input_file = 'ArithOps_Train.xlsx'
df = pd.read_excel(input_file)

# Augment the 'Description' and 'Question' columns
augmented_rows = []

for index, row in df.iterrows():
    description = row['Description']
    question = row['Question']
    # Keep the numbers the same using regex
    numbers_in_description = re.findall(r'\d+', description)
    numbers_in_question = re.findall(r'\d+', question)

    # Augment the text
    num_augmentations = 5  # Number of augmented versions per original sentence
    num_replacements = 3   # Number of words to replace in each sentence
    augmented_descriptions = augment_text(
        description, num_augmentations, num_replacements)
    augmented_questions = augment_text(
        question, num_augmentations, num_replacements)

    for desc_aug, ques_aug in zip(augmented_descriptions, augmented_questions):
        # Ensure numbers are the same in augmented text
        desc_aug_numbers = re.findall(r'\d+', desc_aug)
        for orig_num, aug_num in zip(numbers_in_description, desc_aug_numbers):
            desc_aug = desc_aug.replace(aug_num, orig_num, 1)
        ques_aug_numbers = re.findall(r'\d+', ques_aug)
        for orig_num, aug_num in zip(numbers_in_question, ques_aug_numbers):
            ques_aug = ques_aug.replace(aug_num, orig_num, 1)

        # Create a new row with augmented data
        new_row = row.copy()
        new_row['Description'] = desc_aug
        new_row['Question'] = ques_aug
        augmented_rows.append(new_row)

# Create a new DataFrame with augmented data
augmented_df = pd.DataFrame(augmented_rows, columns=df.columns)

# Combine the original and augmented data
combined_df = pd.concat([df, augmented_df], ignore_index=True)

# Save the augmented dataset to a new Excel file
# Replace with your desired output file path
output_file = 'augmented-ArithOps_Train.xlsx'
combined_df.to_excel(output_file, index=False)

print(f"Augmented dataset saved to {output_file}")
