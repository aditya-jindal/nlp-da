from sentence_transformers import SentenceTransformer, util


def check_topic_discussion(transcribed_text, topics_file, similarity_threshold=0.75):
    with open(topics_file, 'r') as f:
        topics = [line.strip() for line in f.readlines()]

    #SBERT model
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



with open("transcription.txt", "r") as file:
    transcribed_text = file.read()

topics_file = "crick.txt"
try:
    flagged_sentences = check_topic_discussion(transcribed_text, topics_file, 0.5)
    if flagged_sentences:
        print("Flagged Sentences for Discussion:")
        for flagged in flagged_sentences:
            print(f"Exam Question: {flagged['topic']}")
            print(f"Similar Sentence: {flagged['sentence']}")
            print(f"Similarity: {flagged['similarity']:.2f}\n")
    else:
        print("No suspicious topic discussions detected.")
except Exception as e:
    print(f"An error occurred: {e}")
