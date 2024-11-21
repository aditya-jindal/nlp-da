from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

def extract_topics_from_questions(file_path, num_topics=5, top_n_keywords=5):
    with open(file_path, 'r') as f:
        questions = [line.strip() for line in f.readlines()]
    
    # SBERT
    model = SentenceTransformer('all-MiniLM-L6-v2')
    
    embeddings = model.encode(questions, convert_to_tensor=False)
    
    kmeans = KMeans(n_clusters=num_topics, random_state=42)
    kmeans.fit(embeddings)
    
    cluster_labels = kmeans.labels_
    
    clusters = {i: [] for i in range(num_topics)}
    for i, label in enumerate(cluster_labels):
        clusters[label].append(questions[i])
    
    topic_keywords = {}
    for cluster_id, cluster_questions in clusters.items():
        # Using TF-IDF to find important words in the cluster
        vectorizer = TfidfVectorizer(stop_words='english')
        tfidf_matrix = vectorizer.fit_transform(cluster_questions)
        tfidf_scores = np.array(tfidf_matrix.sum(axis=0)).flatten()
        keywords = [
            vectorizer.get_feature_names_out()[index]
            for index in tfidf_scores.argsort()[-top_n_keywords:][::-1]
        ]
        topic_keywords[cluster_id] = keywords
    
    for cluster_id, keywords in topic_keywords.items():
        print(f"Topic {cluster_id}: {', '.join(keywords)}")
        print("Sample Questions:")
        print("\n".join(clusters[cluster_id][:3])) 
        print()
    
    return topic_keywords

file_path = "crick.txt"
topics = extract_topics_from_questions(file_path)
print("Extracted Topics and Keywords:")
for cluster_id, keywords in topics.items():
    print(f"Topic {cluster_id}: {', '.join(keywords)}")
