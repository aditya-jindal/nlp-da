from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain.document_loaders import AssemblyAIAudioTranscriptLoader
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

load_dotenv()

# Initialize the LLM (Llama3 via ChatGroq)
langchain_llm = ChatGroq(model="llama3-8b-8192")

# Load and transcribe the audio file


def load_and_transcribe(audio_file: str, save_to: str = "transcription.txt") -> str:
    loader = AssemblyAIAudioTranscriptLoader(file_path=audio_file)
    docs = loader.load()
    transcribed_text = docs[0].page_content

    # Save the transcription to a .txt file
    with open(save_to, "w") as f:
        f.write(transcribed_text)
    return transcribed_text

# Extract topics from questions using the LLM


def extract_topics_from_questions(questions_file: str, save_to: str = "topics.txt") -> list:
    with open(questions_file, "r") as f:
        questions = f.read()  # Read all questions as a single string

    topic_extraction_prompt = PromptTemplate(
        input_variables=["questions"],
        template=(
            "Given the following exam questions, extract the main topics or concepts they are about. Be concise, I only want the comma separated list of few of the main topics."
            "Questions: {questions}"
        ),
    )
    topic_extraction_chain = LLMChain(
        llm=langchain_llm, prompt=topic_extraction_prompt)
    topics_response = topic_extraction_chain.run({"questions": questions})

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
        "Analyze if this text suggests cheating behavior. Look for indications like someone spelling out answers "
        "(e.g., 'a, b, c, d') or giving numerical choices. Text: {text}"
    ),
)

# Define chains
topic_chain = LLMChain(llm=langchain_llm, prompt=topic_prompt)
cheating_chain = LLMChain(llm=langchain_llm, prompt=cheating_behavior_prompt)

# Flagging mechanism


def analyze_transcription(transcribed_text: str, topics: list) -> dict:
    """
    Analyze the transcription text for potential cheating.
    Returns a dictionary with analysis results.
    """
    results = {"topic_discussion": None, "cheating_behavior": None}

    # Check for topic discussion
    topic_result = topic_chain.run(
        {"text": transcribed_text, "topics": ", ".join(topics)})
    if "yes" in topic_result.lower():
        # Extract the specific part of the text deemed relevant
        results["topic_discussion"] = topic_result

    # Check for suspicious cheating behavior
    cheating_result = cheating_chain.run({"text": transcribed_text})
    if "yes" in cheating_result.lower():
        # Extract the specific part of the text deemed suspicious
        results["cheating_behavior"] = cheating_result

    return results

# Full pipeline


def detect_cheating_in_audio(audio_file: str, questions_file: str):
    print("Extracting exam topics...")
    exam_topics = extract_topics_from_questions(questions_file)

    print("Transcribing audio...")
    transcribed_text = load_and_transcribe(audio_file)

    print("Analyzing transcription for cheating...")
    analysis_results = analyze_transcription(transcribed_text, exam_topics)

    # Print results
    if analysis_results["topic_discussion"]:
        print("Flagged: The transcription discusses test-related topics.")
        print(
            f"Suspicious segment (Topic Discussion): {analysis_results['topic_discussion']}")

    if analysis_results["cheating_behavior"]:
        print("Flagged: Suspicious cheating behavior detected.")
        print(
            f"Suspicious segment (Cheating Behavior): {analysis_results['cheating_behavior']}")

    if not analysis_results["topic_discussion"] and not analysis_results["cheating_behavior"]:
        print("No suspicious activity found.")

    return analysis_results


# Test the pipeline
if __name__ == "__main__":
    audio_path = "positive.mp3"         # Path to the audio file
    questions_path = "questions.txt"  # Path to the questions file
    detect_cheating_in_audio(audio_path, questions_path)
