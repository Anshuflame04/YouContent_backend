import os
import re
import json  # <-- Import json module
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import google.generativeai as genai
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled, NoTranscriptFound
from urllib.parse import urlparse, parse_qs

# Use environment variable for API Key (avoid hardcoding in production)
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY', 'AIzaSyA7aucBrnNlBQwdw8ghKEiuHPYk2iK72S0')

# Configure Gemini API once with the API key
genai.configure(api_key=GEMINI_API_KEY)

# Define configuration for the Gemini model
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
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------- Other Endpoints (Article, Code, Summarize, Chat) ----------

class ArticleRequest(BaseModel):
    platform: str
    prompt: str

@app.post("/generate-article/")
async def generate_article(request: ArticleRequest):
    try:
        platform_prompts = {
            "blog": "Write a well-structured, engaging, and informative blog post on the topic: ",
            "twitter": "Write a catchy and engaging Twitter/X thread on the topic: ",
            "linkedin": "Write a professional and insightful LinkedIn article on the topic: ",
            "facebook": "Write a friendly and engaging Facebook post on the topic: ",
            "instagram": "Write a creative and engaging Instagram caption on the topic: ",
        }
        if request.platform not in platform_prompts:
            raise HTTPException(status_code=400, detail="Invalid platform selected")
        input_text = f"{platform_prompts[request.platform]} {request.prompt}"
        chat_session = model.start_chat(history=[])
        response = chat_session.send_message(input_text)
        return JSONResponse(content={"generatedText": response.text})
    except Exception as e:
        raise HTTPException(status_code=500, detail="Error generating article: " + str(e))


class CodeRequest(BaseModel):
    description: str
    language: str

@app.post("/generate-code/")
async def generate_code(request: CodeRequest):
    try:
        input_text = f"Write a small, simple {request.language} program to solve the following problem: {request.description}. Include only the necessary parts of the code with clear comments explaining each section. Make the code as simple and short as possible."
        chat_session = model.start_chat(history=[])
        response = chat_session.send_message(input_text)
        generated_code = response.text.strip()
        return JSONResponse(content={"generatedCode": generated_code})
    except Exception as e:
        raise HTTPException(status_code=500, detail="Error generating code: " + str(e))


class SummarizeRequest(BaseModel):
    youtube_url: str

@app.post("/summarize/")
async def summarize(request: SummarizeRequest):
    try:
        parsed_url = urlparse(request.youtube_url)
        if parsed_url.netloc in ['www.youtube.com', 'youtube.com']:
            query_params = parse_qs(parsed_url.query)
            if 'v' not in query_params:
                raise HTTPException(status_code=400, detail="Invalid YouTube URL: No video ID found")
            video_id = query_params['v'][0]
        elif parsed_url.netloc == 'youtu.be':
            video_id = parsed_url.path.lstrip('/')
            if not video_id:
                raise HTTPException(status_code=400, detail="Invalid YouTube URL: No video ID found")
        else:
            raise HTTPException(status_code=400, detail="Invalid YouTube URL: Unsupported domain")

        try:
            transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)
            transcript = None
            for t in transcript_list:
                if t.language_code.startswith('en'):
                    transcript = t.fetch()
                    break
            if not transcript:
                transcript = transcript_list.find_transcript([t.language_code for t in transcript_list]).fetch()
                original_language = transcript_list.find_transcript([t.language_code for t in transcript_list]).language
                if not original_language.startswith('en'):
                    video_text = " ".join([segment['text'] for segment in transcript])
                    translation_prompt = f"Translate the following text from {original_language} to English: {video_text}"
                    chat_session = model.start_chat(history=[])
                    translation_response = chat_session.send_message(translation_prompt)
                    video_text = translation_response.text
                else:
                    video_text = " ".join([segment['text'] for segment in transcript])
            else:
                video_text = " ".join([segment['text'] for segment in transcript])
        except NoTranscriptFound:
            raise HTTPException(status_code=400, detail="No transcript available for this video")
        except TranscriptsDisabled:
            raise HTTPException(status_code=400, detail="Transcripts are disabled for this video")

        prompt = f"""
            You are a YouTube Video Summarizer tasked with providing an in-depth analysis of a video's content. Your goal is to generate a comprehensive summary that captures the main points, key arguments, and supporting details within a 750-word limit. Please summarize the transcript text provided in a structured format with the following sections:
            - Introduction: A brief overview of the video's topic.
            - Key Points: List the key takeaways from the video in bullet points.
            - Conclusion: Summarize the overall message or conclusion of the video.
            Here is the transcript text provided: {video_text}
            """
        chat_session = model.start_chat(history=[])
        response = chat_session.send_message(prompt)
        return JSONResponse(content={"summary": response.text})
    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(status_code=500, detail="Error summarizing video: " + str(e))


class ChatRequest(BaseModel):
    message: str

@app.post("/api/chat/")
async def ai_chat(request: ChatRequest):
    try:
        chat_session = model.start_chat(history=[])
        response = chat_session.send_message(request.message)
        return JSONResponse(content={"reply": response.text})
    except Exception as e:
        raise HTTPException(status_code=500, detail="Error generating chat response: " + str(e))


# ---------- MCQ Generation and Answer Checking ----------

class MCQRequest(BaseModel):
    text: str


@app.post("/generate-mcq/")
async def generate_mcq(request: MCQRequest):
    try:
        prompt = f"""
        You are an AI that generates multiple-choice questions (MCQs) based on the given text.
        Each question should have four options, with one correct answer clearly marked.

        Text:
        {request.text}

        Return only valid JSON without any explanation. Strictly follow this format:

        {{
            "mcqs": [
                {{
                    "question": "What is the main topic?",
                    "options": ["A", "B", "C", "D"],
                    "correct_answer": "B"
                }},
                {{
                    "question": "What is true about the text?",
                    "options": ["True", "False", "Maybe", "None"],
                    "correct_answer": "True"
                }}
            ]
        }}
        """

        response = model.generate_content(prompt)
        mcq_text = response.text.strip()

        # 🔍 Step 1: Extract JSON using regex (in case AI adds extra text)
        json_match = re.search(r'\{.*\}', mcq_text, re.DOTALL)
        if json_match:
            mcq_text = json_match.group(0)  # Extract only the JSON part

        # 🔄 Step 2: Try parsing JSON safely
        try:
            mcqs = json.loads(mcq_text)
        except json.JSONDecodeError:
            raise HTTPException(status_code=500, detail="Failed to parse MCQs. AI response was not valid JSON.")

        return JSONResponse(content=mcqs)
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating MCQs: {str(e)}")


class CheckAnswersRequest(BaseModel):
    questions: list
    answers: list

@app.post("/check-answers/")
async def check_answers(request: CheckAnswersRequest):
    try:
        correct_count = 0
        correct_answers = []  # Store correct answers

        for idx, question in enumerate(request.questions):
            correct_answer = question["correct_answer"].strip().lower()  # Normalize correct answer
            user_answer = request.answers[idx].strip().lower()  # Normalize user answer

            if user_answer == correct_answer:
                correct_count += 1

            correct_answers.append(question["correct_answer"])  # Keep original correct answer

        total = len(request.questions)

        return JSONResponse(content={
            "score": correct_count,
            "total": total,
            "correct_answers": correct_answers  # Send correct answers to frontend
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error checking answers: {str(e)}")


@app.get("/")
def read_root():
    return {"message": "Welcome to youContent Server. Now it is working..."}


# if __name__ == "__main__":
#     uvicorn.run(app, host="0.0.0.0", port=8000)
