# Import necessary libraries
import streamlit as st
import os
from langchain import PromptTemplate
from langchain.llms import OpenAI
from trubrics.integrations.streamlit import FeedbackCollector

# Check if 'response' is in the session state
if 'response' not in st.session_state:
    st.session_state.response = None

# Email improvement template
template = """
    Below is an email that may require enhancements in its presentation or translation to English. Your tasks are:
    - Translate the email to English if it's written in another language.
    - Improve provided text ensuring the email has a clear structure and format.
    - Modify the text to align with the chosen tone.
    - Tailor the content to fit the selected English dialect.

    For clarity, here are some examples:

    Tones:
    - Formal: "During our recent trip to Tokyo, we encountered several intriguing cultural nuances that we believe would be of interest to you."
    - Informal: "Guess what? We hit up Tokyo and saw some super cool stuff! Can't wait to spill the beans!"

    English Dialect Variants:
    - American:
    - Words: Sneakers, truck, fries, elevator, trash can, cookie, yard, pants, hood, faucet, vacation.
    - Sentence: "I grabbed some fries from the diner, then took an elevator to my apartment. Planning a vacation next month!"

    - British:
    - Words: Trainers, lorry, chips, lift, bin, biscuit, garden, trousers, bonnet, tap, holiday.
    - Sentence: "I fancied some chips from the cafe, then used the lift to reach my flat. Got a holiday lined up next month!"

    It's essential for the email to have a welcoming opening. If the original email doesn't have one, please incorporate a suitable introduction.

    Provided Details:
    TONE: {tone}
    VARIANT: {variant}
    EMAIL: {email}

    YOUR {variant} RESPONSE:
"""

prompt = PromptTemplate(
    input_variables=["tone", "variant", "email"],
    template=template,
)

def load_LLM(openai_api_key, temperature):
    """Logic for loading the chain"""
    llm = OpenAI(temperature=temperature, openai_api_key=openai_api_key)
    return llm

# Load API key from environment variable
openai_api_key = os.environ.get("OPENAI_API_KEY")
if not openai_api_key:
    st.sidebar.error("OpenAI API key not found in environment variables!")

st.set_page_config(
    page_title="Email Improvement with GPT Model",
    page_icon=":email:",
    layout="wide",
)

# Explanation at the beginning of the page
st.write("Welcome to the Email Improvement tool! This tool is part of a research project and will be running as long as users submit feedback. Your feedback is invaluable to us!")

# Create a sidebar for user inputs and settings
st.sidebar.header("Settings")

# Dropdown for selecting the tone of English: Formal or Informal
tone = st.sidebar.selectbox("Select Tone of English", ["Formal", "Informal"], index=0)

# Dropdown for selecting the variant of English: US or UK
variant = st.sidebar.selectbox("Select English Variant", ["US English", "UK English"], index=1)

# Slider to adjust the temperature of the model
temperature = st.sidebar.slider("Model Temperature", 0.0, 1.0, 0.5)

# Main content of the app
st.header("Input your email content (max 1000 characters)")
# Textarea for users to input their email content with a character limit
email_content = st.text_area("Email Content", "Type your email here...", max_chars=1000)

# Button to trigger the improvement process
if st.button("Improve Email"):
    # Load the LLM with the provided API key and temperature
    llm = load_LLM(openai_api_key, temperature)
    
    # Use the LLM to improve the email
    prompt_with_email = prompt.format(tone=tone, variant=variant, email=email_content)
    st.session_state.response = llm(prompt_with_email)

# Display the improved email content if it exists
if st.session_state.response:
    st.header("Improved Email Content")
    st.write(st.session_state.response)

# Optional: Add any additional information, instructions, or footers
st.info("Instructions: Input your email content in the provided text area (up to 1000 characters) and click 'Improve Email' to get a refined version of your email. Original email could be in English or another language.")

# Trubrics Feedback Collector
collector = FeedbackCollector(
    project="Email Improver",
    email=os.environ.get("TRUBRICS_EMAIL"),
    password=os.environ.get("TRUBRICS_PASSWORD"),
)

collector.st_feedback(
    component="default",
    feedback_type="thumbs",
    model="gpt-3.5-turbo",
    prompt_id=None,  # see prompts to log prompts and model generations
    open_feedback_label='[Optional] Provide additional feedback'
)

# Footer
st.write("---")
st.write("Made with :heart: using Streamlit and GPT")
