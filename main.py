import os
import asyncio
import io

from dotenv import load_dotenv
import speech_recognition as sr
import pygame
from google import genai
from google.genai import types
import edge_tts

from spotify_helper import build_spotify_tool

load_dotenv()

# Initialize Gemini Client
try:
    api_key = os.environ.get("GOOGLE_API_KEY")

    if not api_key:
        raise ValueError("GOOGLE_API_KEY is missing from .env")

    client = genai.Client(api_key=api_key)

except Exception as e:
    print(f"Error initializing Gemini client: {e}")
    exit(1)


# Google SDK keeps chat context in the session itself.
chat = client.chats.create(
    model=os.getenv("model", "gemini-2.5-flash"),
    config=types.GenerateContentConfig(
        system_instruction=(
            "You are a helpful, lightning-fast voice assistant. "
            "Always prefer the latest web information for anything that may change over time, such as news, prices, weather, stock prices, current events, sports scores, product availability, or recent developments. "
            "Use the built-in web search tool before answering current-information questions. "
            "When the search tool is available, use it as the primary source for up-to-date facts. "
            "Keep answers short, friendly, and under 3 sentences."
        ),
        tools=[
            types.Tool(google_search=types.GoogleSearch()),
            build_spotify_tool(),
        ],
    ),
)


# Initialize the English neural voice
VOICE = "en-US-GuyNeural"

try:
    pygame.mixer.init()
except Exception as e:
    print(f"Error initializing audio playback: {e}")
    exit(1)


async def generate_speech(text):
    """Generate speech using an English neural voice."""

    communicate = edge_tts.Communicate(
        text=text,
        voice=VOICE,
        rate="+35%",
        pitch="+0Hz"
    )

    audio_data = bytearray()

    async for chunk in communicate.stream():
        if chunk["type"] == "audio":
            audio_data.extend(chunk["data"])

    return bytes(audio_data)


def is_error_response(text):
    """Return True when the response is an error message or a failed-generation fallback."""
    if not text:
        return False

    normalized = text.strip()
    lowered = normalized.lower()

    return (
        lowered.startswith("error")
        or "error connecting to gemini" in lowered
        or "sorry, i could not generate a response" in lowered
    )


def speak(text):
    """Speaks the response without saving an audio file."""

    if not text or is_error_response(text):
        if text:
            print(f"Skipping spoken error response: {text}")
        return

    print(f"AI: {text}")

    try:
        audio_data = asyncio.run(generate_speech(text))

        # Play audio directly from memory.
        audio_file = io.BytesIO(audio_data)

        pygame.mixer.music.load(audio_file, "mp3")
        pygame.mixer.music.play()

        while pygame.mixer.music.get_busy():
            pygame.time.Clock().tick(30)

    except Exception as e:
        print(f"Speech error: {e}")


def listen_and_transcribe():
    """Listens to the microphone and converts speech to text."""

    recognizer = sr.Recognizer()

    with sr.Microphone() as source:
        print("\nListening...")

        recognizer.adjust_for_ambient_noise(
            source,
            duration=0.8
        )

        try:
            audio = recognizer.listen(
                source,
                timeout=5,
                phrase_time_limit=10
            )

            print("Processing audio...")

            text = recognizer.recognize_google(
                audio,
                language="en-US"
            )

            print(f"You said: {text}")
            return text

        except sr.UnknownValueError:
            print("Could not understand audio.")
            return None

        except sr.RequestError:
            print("Speech API unavailable.")
            return None

        except sr.WaitTimeoutError:
            print("Listening timed out.")
            return None

        except Exception as e:
            print(f"Microphone error: {e}")
            return None


def query_gemini(user_prompt):
    """Send the user prompt to Google’s chat session so the SDK keeps context automatically."""

    try:
        response = chat.send_message(user_prompt)
        return response.text or "Sorry, I could not generate a response."
    except Exception as e:
        try:
            response = client.models.generate_content(
                model=os.getenv("model", "gemini-2.5-flash"),
                contents=(
                    "You are a helpful, lightning-fast voice assistant. "
                    "For current information, use web search or recent data whenever available. "
                    "Do not rely only on outdated training knowledge. "
                    "Keep answers short, friendly, and under 3 sentences.\n\n"
                    f"User: {user_prompt}"
                ),
                config=types.GenerateContentConfig(
                    tools=[types.Tool(google_search=types.GoogleSearch())]
                ),
            )
            return response.text or "Sorry, I could not generate a response."
        except Exception as fallback_error:
            return f"Error connecting to Gemini: {fallback_error}"


# Main Assistant Loop
if __name__ == "__main__":

    speak("Gemini AI voice system active. Go ahead, I'm listening.")

    try:
        while True:

            user_input = listen_and_transcribe()

            if user_input:

                # Check for termination keywords
                if any(
                    word in user_input.lower()
                    for word in ["exit", "stop", "goodbye"]
                ):
                    speak("Goodbye!")
                    break

                # Get a response from Gemini
                ai_response = query_gemini(user_input)

                if is_error_response(ai_response):
                    print(f"Skipping error response: {ai_response}")
                    continue

                # Speak the response
                speak(ai_response)

    except KeyboardInterrupt:
        print("\nAssistant stopped.")

    finally:
        pygame.mixer.quit()
        client.close()
