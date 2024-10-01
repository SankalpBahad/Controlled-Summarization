import nltk
import re
import string
import numpy as np
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MinMaxScaler
from fuzzywuzzy import fuzz
from collections import defaultdict

# Download required nltk resources
nltk.download('punkt')
nltk.download('stopwords')

# Define pre-processing and feature extraction functions
def preprocess_text(text):
    """
    Preprocess the input text by performing case folding, punctuation removal,
    stopword removal, stemming, and tokenization.
    """
    # Case folding: Convert to lowercase
    text = text.lower()
    
    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    words = word_tokenize(text)
    words = [word for word in words if word not in stop_words]
    
    return ' '.join(words)

def extract_features(sentences, original_text):
    """
    Extract features for each sentence: title feature, length feature, term weight, 
    position feature, proper noun feature, and numerical data feature.
    """
    # Define the title as the first sentence of the document
    title = sentences[0] if sentences else ""
    
    features = defaultdict(list)
    
    # Feature 1: Title feature
    for sentence in sentences:
        title_matches = sum([1 for word in word_tokenize(sentence) if word in word_tokenize(title)])
        title_feature = title_matches / len(word_tokenize(title)) if len(word_tokenize(title)) > 0 else 0
        features['title_feature'].append(title_feature)
    
    # Feature 2: Sentence Length feature
    max_length = max([len(sentence) for sentence in sentences]) if sentences else 1
    for sentence in sentences:
        length_feature = len(sentence) / max_length
        features['length_feature'].append(length_feature)
    
    # Feature 3: Term Weight feature (TF-IDF)
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(sentences)
    tfidf_scores = np.array(tfidf_matrix.sum(axis=1)).flatten()
    max_tfidf = max(tfidf_scores) if len(tfidf_scores) > 0 else 1
    features['term_weight_feature'] = (tfidf_scores / max_tfidf).tolist()
    
    # Feature 4: Sentence Position feature
    for i, sentence in enumerate(sentences):
        position_feature = 1 - (i / len(sentences))
        features['position_feature'].append(position_feature)
    
    # Feature 5: Proper Noun feature
    for sentence in sentences:
        proper_nouns = sum(1 for word, pos in nltk.pos_tag(word_tokenize(sentence)) if pos == 'NNP')
        features['proper_noun_feature'].append(proper_nouns / len(word_tokenize(sentence)) if len(word_tokenize(sentence)) > 0 else 0)
    
    # Feature 6: Numerical Data feature
    for sentence in sentences:
        numerical_data = len(re.findall(r'\d+', sentence))
        features['numerical_data_feature'].append(numerical_data / len(word_tokenize(sentence)) if len(word_tokenize(sentence)) > 0 else 0)
    
    return features

# Define fuzzy logic-based scoring function
def fuzzy_logic_scoring(features):
    """
    Calculate the overall score for each sentence using a simple weighted sum of features.
    """
    # Normalizing each feature using MinMax Scaler
    scaler = MinMaxScaler()
    normalized_features = {key: scaler.fit_transform(np.array(values).reshape(-1, 1)).flatten() for key, values in features.items()}
    
    # Assign weights to each feature (can be adjusted based on preference)
    weights = {
        'title_feature': 0.2,
        'length_feature': 0.1,
        'term_weight_feature': 0.3,
        'position_feature': 0.2,
        'proper_noun_feature': 0.1,
        'numerical_data_feature': 0.1,
    }
    
    # Calculate the final score for each sentence
    scores = []
    for i in range(len(normalized_features['title_feature'])):
        score = sum([normalized_features[feature][i] * weight for feature, weight in weights.items()])
        scores.append(score)
    
    return scores

# Summarization function
def generate_summary(text, summary_length=5, style='concise', salience='term_weight'):
    """
    Generate a summary of the given text based on the user-defined parameters.
    """
    # Pre-process the text and split into sentences
    original_text = text
    sentences = sent_tokenize(text)
    preprocessed_sentences = [preprocess_text(sentence) for sentence in sentences]
    
    # Extract features for each sentence
    features = extract_features(preprocessed_sentences, original_text)
    
    # Calculate fuzzy logic-based scores for each sentence
    scores = fuzzy_logic_scoring(features)
    
    # Rank sentences based on the scores
    ranked_sentences = sorted(((scores[i], s) for i, s in enumerate(sentences)), reverse=True)
    
    # Select the top N sentences based on the summary length
    selected_sentences = [sentence for _, sentence in ranked_sentences[:summary_length]]
    
    # Adjust summary style based on user preference
    if style == 'detailed':
        summary = ' '.join(selected_sentences)
    else:  # concise style
        summary = ' '.join(selected_sentences[:summary_length // 2])
    
    return summary

# Main execution
if __name__ == "__main__":
    # Upload text file
    file_path = input("Enter the path of the text file: ")
    
    with open(file_path, 'r') as file:
        text_content = file.read()
    
    # Ask user for summary preferences
    summary_length = int(input("Enter the desired length of the summary (in number of sentences): "))
    style = input("Enter the style of the summary ('concise' or 'detailed'): ").lower()
    salience = input("Enter the most salient feature to consider (e.g., 'term_weight', 'position'): ").lower()
    
    # Generate and print the summary
    summary = generate_summary(text_content, summary_length, style, salience)
    print("\nGenerated Summary:\n", summary)
