import streamlit as st
from sentence_transformers import SentenceTransformer, util
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np



def extract_topics_from_questions(file_path, num_topics=5, top_n_keywords=5):
    with open(file_path, 'r') as f:
        questions = [line.strip() for line in f.readlines()]

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
        vectorizer = TfidfVectorizer(stop_words='english')
        tfidf_matrix = vectorizer.fit_transform(cluster_questions)
        tfidf_scores = np.array(tfidf_matrix.sum(axis=0)).flatten()
        keywords = [
            vectorizer.get_feature_names_out()[index]
            for index in tfidf_scores.argsort()[-top_n_keywords:][::-1]
        ]
        topic_keywords[cluster_id] = keywords

    return topic_keywords



def check_topic_discussion(transcribed_text, topics_file, similarity_threshold=0.5):
    with open(topics_file, 'r') as f:
        topics = [line.strip() for line in f.readlines()]

    model = SentenceTransformer('all-MiniLM-L6-v2')
    transcribed_sentences = [sentence.strip(
    ) for sentence in transcribed_text.split('. ') if sentence.strip()]

    if not transcribed_sentences:
        raise ValueError("Transcribed text contains no valid sentences.")
    if not topics:
        raise ValueError("Topics file is empty.")

    text_embeddings = model.encode(
        transcribed_sentences, convert_to_tensor=True)
    topic_embeddings = model.encode(topics, convert_to_tensor=True)

    cosine_similarities = util.pytorch_cos_sim(
        text_embeddings, topic_embeddings)

    flagged_sentences = []
    for i, sentence in enumerate(transcribed_sentences):
        for j, topic in enumerate(topics):
            if cosine_similarities[i][j] > similarity_threshold:
                flagged_sentences.append({
                    "sentence": sentence,
                    "topic": topic,
                    "similarity": float(cosine_similarities[i][j])
                })

    return flagged_sentences



def main():
    st.title("Exam Cheating Detection System")

    st.sidebar.title("Input Files")
    audio_file = st.sidebar.file_uploader(
        "Upload Audio File (MP3)", type=["mp3"])
    questions_file = st.sidebar.file_uploader(
        "Upload Questions File (TXT)", type=["txt"])

    if audio_file and questions_file:

        audio_path = f"uploaded_audio.mp3"
        with open(audio_path, "wb") as f:
            f.write(audio_file.getbuffer())

        questions_path = f"uploaded_questions.txt"
        with open(questions_path, "wb") as f:
            f.write(questions_file.getbuffer())

        st.header("1. Extracted Topics")
        topics = extract_topics_from_questions(
            questions_path, num_topics=5, top_n_keywords=5)
        for cluster_id, keywords in topics.items():
            st.write(f"**Topic {cluster_id}:** {', '.join(keywords)}")

        with open("transcription.txt", "r") as file:
            transcribed_text = file.read()

        st.header("2. Transcribed Audio")
        st.text_area("Transcribed Text", transcribed_text, height=200)

        st.header("3. Detected Topic Discussions")
        flagged_sentences = check_topic_discussion(
            transcribed_text, questions_path)
        if flagged_sentences:
            st.write("Flagged Sentences for Discussion:")
            for flagged in flagged_sentences:
                st.write(f"- **Sentence:** {flagged['sentence']}")
                st.write(f"  - **Topic:** {flagged['topic']}")
                st.write(
                    f"  - **Similarity Score:** {flagged['similarity']:.2f}")
        else:
            st.write("No suspicious topic discussions detected.")


if __name__ == "__main__":
    main()
