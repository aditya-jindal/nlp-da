import streamlit as st
import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain.document_loaders import AssemblyAIAudioTranscriptLoader
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

# Load environment variables
load_dotenv()

# Initialize the LLM (Llama3 via ChatGroq)
langchain_llm = ChatGroq(model="llama3-8b-8192")

# Load and transcribe the audio file


def load_and_transcribe(uploaded_file, save_to="transcription.txt"):
    # Write the uploaded file to a temporary file
    temp_file_path = f"temp_{uploaded_file.name}"
    with open(temp_file_path, "wb") as f:
        f.write(uploaded_file.read())

    # Transcribe the audio
    loader = AssemblyAIAudioTranscriptLoader(file_path=temp_file_path)
    docs = loader.load()
    transcribed_text = docs[0].page_content

    # Save the transcription to a .txt file
    with open(save_to, "w") as f:
        f.write(transcribed_text)

    # Remove the temporary file
    os.remove(temp_file_path)

    return transcribed_text

# Extract topics from questions using the LLM


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
    # Split response into a list of topics
    with open(save_to, "w") as f:
        f.write("\n".join(topics_list))

    return topics_list


# Prompts for analyzing transcription
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
        "Analyze if this text suggests cheating behavior. Look for indications like someone spelling out mcq options "
        "(example: 'a, b, c, d') or giving numerical choices(example: '1, 2, 3, 4'). Text: {text}"
    ),
)

# Define chains
topic_chain = LLMChain(llm=langchain_llm, prompt=topic_prompt)
cheating_chain = LLMChain(llm=langchain_llm, prompt=cheating_behavior_prompt)

# Flagging mechanism


def analyze_transcription(transcribed_text, topics):
    """
    Analyze the transcription text for potential cheating.
    Returns a dictionary with analysis results.
    """
    results = {"topic_discussion": None, "cheating_behavior": None}

    # Check for topic discussion
    topic_result = topic_chain.run(
        {"text": transcribed_text, "topics": ", ".join(topics)})
    if "yes" in topic_result.lower():
        results["topic_discussion"] = topic_result

    # Check for suspicious cheating behavior
    cheating_result = cheating_chain.run({"text": transcribed_text})
    if "yes" in cheating_result.lower():
        results["cheating_behavior"] = cheating_result

    return results

# Streamlit UI


def main():
    st.title("Cheating Detection in Online Assessments")
    st.markdown(
        """
        Upload an audio file and a questions file to analyze whether the candidate's behavior suggests cheating.
        """
    )

    # File inputs
    questions_file = st.file_uploader(
        "Upload the questions.txt file", type="txt")
    audio_file = st.file_uploader(
        "Upload the audio file (e.g., test.mp3)", type=["mp3", "wav"])

    if questions_file and audio_file:
        st.write("**Processing your files...**")

        # Load and display questions
        questions_text = questions_file.read().decode("utf-8")
        exam_topics = extract_topics_from_questions(questions_text)
        st.subheader("Extracted Exam Topics")
        st.write(exam_topics)

        # Transcribe the audio
        transcribed_text = load_and_transcribe(audio_file)
        st.subheader("Transcribed Audio")
        st.text(transcribed_text)

        # Analyze the transcription
        analysis_results = analyze_transcription(transcribed_text, exam_topics)

        # Display results
        st.subheader("Analysis Results")
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
