import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import google.generativeai as genai
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from youtube_transcript_api import YouTubeTranscriptApi

# Use environment variable for API Key
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY', 'AIzaSyA7aucBrnNlBQwdw8ghKEiuHPYk2iK72S0')

# Configure Gemini API with environment variable or fallback key
genai.configure(api_key=GEMINI_API_KEY)


# Configure Gemini API with the API key
genai.configure(api_key=GEMINI_API_KEY)

# Define the configuration for the Gemini model
generation_config = {
    "temperature": 1,
    "top_p": 0.95,
    "top_k": 40,
    "max_output_tokens": 8192,
    "response_mime_type": "text/plain",
}

# Create the Generative model instance
model = genai.GenerativeModel(
    model_name="gemini-1.5-pro",
    generation_config=generation_config,
)

# Initialize FastAPI app
app = FastAPI()

# Enable CORS for all origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allow all HTTP methods (GET, POST, etc.)
    allow_headers=["*"],  # Allow all headers
)

# Define request body structure for Article generation
class ArticleRequest(BaseModel):
    platform: str
    prompt: str

# Define request body structure for Code generation
class CodeRequest(BaseModel):
    description: str
    language: str  # Specify the programming language for code generation

# Define request body structure for Summarization
class SummarizeRequest(BaseModel):
    youtube_url: str  # The URL of the YouTube video to summarize

# Platform-specific pre-prompts for articles
platform_prompts = {
    "blog": "Write a well-structured, engaging, and informative blog post on the topic: ",
    "twitter": "Write a catchy and engaging Twitter/X thread on the topic: ",
    "linkedin": "Write a professional and insightful LinkedIn article on the topic: ",
    "facebook": "Write a friendly and engaging Facebook post on the topic: ",
    "instagram": "Write a creative and engaging Instagram caption on the topic: ",
}

# Define the /generate-article endpoint
@app.post("/generate-article/")
async def generate_article(request: ArticleRequest):
    try:
        # Check if the selected platform is valid
        if request.platform not in platform_prompts:
            raise HTTPException(status_code=400, detail="Invalid platform selected")

        # Get the appropriate pre-prompt based on the selected platform
        pre_prompt = platform_prompts[request.platform]

        # Prepare the input text for the model
        input_text = f"{pre_prompt} {request.prompt}"

        # Start a chat session with the model and send the message
        chat_session = model.start_chat(history=[])
        response = chat_session.send_message(input_text)
        
        # Return the generated text
        return JSONResponse(content={"generatedText": response.text})

    except Exception as e:
        raise HTTPException(status_code=500, detail="Error generating article: " + str(e))

# Define the /generate-code endpoint
@app.post("/generate-code/")
async def generate_code(request: CodeRequest):
    try:
        # Prepare the input text for the model
        input_text = f"Write a small, simple {request.language} program to solve the following problem: {request.description}. Include only the necessary parts of the code with clear comments explaining each section. Make the code as simple and short as possible."

        # Start a chat session with the model and send the message
        chat_session = model.start_chat(history=[])
        response = chat_session.send_message(input_text)
        
        # Extract only the code from the response, ensuring minimal output
        generated_code = response.text.strip()
        
        # Return the generated code as plain text (without backticks)
        return JSONResponse(content={"generatedCode": generated_code})

    except Exception as e:
        raise HTTPException(status_code=500, detail="Error generating code: " + str(e))

# Define the /summarize endpoint
@app.post("/summarize/")
async def summarize(request: SummarizeRequest):
    try:
        # Check if youtube_url is valid and extract the video ID
        if "v=" not in request.youtube_url:
            raise HTTPException(status_code=400, detail="Invalid YouTube URL")
        
        video_id = request.youtube_url.split("v=")[1]
        
        # Extract the transcript from YouTube video using youtube_transcript_api
        transcript = YouTubeTranscriptApi.get_transcript(video_id)

        # Combine all transcript segments into one text
        video_text = " ".join([segment['text'] for segment in transcript])

        # Define the summarization prompt
        prompt = f"""
            You are a YouTube Video Summarizer tasked with providing an in-depth analysis of a video's content. Your goal is to generate a comprehensive summary that captures the main points, key arguments, and supporting details within a 750-word limit. Please summarize the transcript text provided in a structured format with the following sections:

            - Introduction: A brief overview of the video's topic.
            - Key Points: List the key takeaways from the video in bullet points.
            - Conclusion: Summarize the overall message or conclusion of the video.

            Here is the transcript text provided: {video_text}
            """


        # Start a chat session with the model and send the message
        chat_session = model.start_chat(history=[])
        response = chat_session.send_message(prompt)

        # Return the generated summary
        return JSONResponse(content={"summary": response.text})

    except Exception as e:
        raise HTTPException(status_code=500, detail="Error summarizing video: " + str(e))

# Define request body structure for AI Chat
class ChatRequest(BaseModel):
    message: str  # User's message to the AI

@app.post("/api/chat/")
async def ai_chat(request: ChatRequest):
    try:
        # Start a chat session with the model and send the user's message
        chat_session = model.start_chat(history=[])
        response = chat_session.send_message(request.message)
        
        # Return the AI-generated reply
        return JSONResponse(content={"reply": response.text})
    except Exception as e:
        raise HTTPException(status_code=500, detail="Error generating chat response: " + str(e))


@app.get("/")
def read_root():
    return {"message": "Welcome to youContent Server.Now it is working..."}

# # Run the server using: uvicorn server:app --reload
# if __name__ == "__main__":
#     uvicorn.run(app, host="0.0.0.0", port=8000)  # Make sure to install uvicorn (`pip install uvicorn`)
