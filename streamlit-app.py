import streamlit as st
import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain.document_loaders import AssemblyAIAudioTranscriptLoader
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain



load_dotenv()

langchain_llm = ChatGroq(model="llama3-8b-8192")


def load_and_transcribe(uploaded_file, save_to="transcription.txt"):
    temp_file_path = f"temp_{uploaded_file.name}"
    with open(temp_file_path, "wb") as f:
        f.write(uploaded_file.read())

    loader = AssemblyAIAudioTranscriptLoader(file_path=temp_file_path)
    docs = loader.load()
    transcribed_text = docs[0].page_content

    with open(save_to, "w") as f:
        f.write(transcribed_text)

    os.remove(temp_file_path)

    return transcribed_text


def extract_topics_from_questions(questions_text, save_to: str = "topics.txt"):
    topic_extraction_prompt = PromptTemplate(
        input_variables=["questions"],
        template=(
            "Given the following exam questions, extract the main topics or concepts they are about. Be concise, I only want the comma separated list of few of the main topics. Say absolutely nothing else."
            "Questions: {questions}"
        ),
    )
    topic_extraction_chain = LLMChain(
        llm=langchain_llm, prompt=topic_extraction_prompt)
    topics_response = topic_extraction_chain.run({"questions": questions_text})

    topics_list = [topic.strip() for topic in topics_response.split(",")]

    with open(save_to, "w") as f:
        f.write("\n".join(topics_list))

    return "\n".join(topics_list)  


topic_prompt = PromptTemplate(
    input_variables=["text", "topics"],
    template=(
        "Does the following text discuss any of the test topics provided? "
        "Topics: {topics}\nText: {text}"
    ),
)

cheating_behavior_prompt = PromptTemplate(
    input_variables=["text"],
    template=(
        "Analyze if this text contains out of context indications of someone spelling out mcq options "
        "(example: 'a, b, c, d') or giving numerical choices(example: '1, 2, 3, 4'). "
        "Provide your answer in the format: (Yes/No), followed by necessary justification. "
        "Text: {text}"
    ),
)

topic_chain = LLMChain(llm=langchain_llm, prompt=topic_prompt)
cheating_chain = LLMChain(llm=langchain_llm, prompt=cheating_behavior_prompt)


def analyze_transcription(transcribed_text, topics):
    results = {"topic_discussion": None, "cheating_behavior": None}

    topic_result = topic_chain.run(
        {"text": transcribed_text, "topics": topics})
    if "yes" in topic_result.lower():
        results["topic_discussion"] = topic_result

    cheating_result = cheating_chain.run({"text": transcribed_text})
    print("cheating result: ", cheating_result)
    if "yes" in cheating_result.lower():
        results["cheating_behavior"] = cheating_result

    return results


def main():
    st.set_page_config(layout="wide")
    st.title("Cheating Detection in Online Assessments")

    upload_col1, upload_col2 = st.columns(2)
    with upload_col1:
        questions_file = st.file_uploader(
            "Upload the questions.txt file", type="txt")
    with upload_col2:
        audio_file = st.file_uploader(
            "Upload the audio file (e.g., test.mp3)", type=["mp3", "wav"])

    if questions_file and audio_file:
        transcription_col, topics_col = st.columns(2)

        with transcription_col:
            st.subheader("Transcribed Audio")
            transcribed_text = load_and_transcribe(audio_file)
            st.text_area("Transcription", value=transcribed_text, height=300)

        with topics_col:
            st.subheader("Extracted Exam Topics")
            questions_text = questions_file.read().decode("utf-8")
            exam_topics = extract_topics_from_questions(questions_text)
            st.text_area("Topics", value=exam_topics, height=300)

        st.subheader("Analysis Results")

        analysis_results = analyze_transcription(transcribed_text, exam_topics)

        if analysis_results["topic_discussion"]:
            st.warning(
                "**Flagged:** The transcription discusses test-related topics.")
            st.write(
                f"**Suspicious Segment (Topic Discussion):** {analysis_results['topic_discussion']}")

        if analysis_results["cheating_behavior"]:
            st.warning("**Flagged:** Suspicious cheating behavior detected.")
            st.write(
                f"**Suspicious Segment (Cheating Behavior):** {analysis_results['cheating_behavior']}")

        if not analysis_results["topic_discussion"] and not analysis_results["cheating_behavior"]:
            st.success("No suspicious activity found.")
    else:
        st.info("Please upload both files to proceed.")


if __name__ == "__main__":
    main()
