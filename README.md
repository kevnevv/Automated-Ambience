# Automated-Ambience

Automated-Ambience is an AI-powered DJ that listens to your environment and automatically shifts the music to match the "vibe" of your surroundings.

Whether you are working (Typing/Silence), having a rowdy dinner (Laughter/Dishes), or just relaxing, the AI processes ambient audio in real-time to ensure the soundscape always matches the moment without you ever having to touch a remote.

# Quick Start

1. Prerequisites

Ensure you have Python 3.8+ installed. You will also need to install the following dependencies:

pip install flask numpy sounddevice tensorflow tensorflow-hub ytmusicapi yt-dlp


2. Run the Server

From your terminal, run the main script:

python main.py


3. Access the Interface

Once the AI finishes loading the YAMNet model (this may take a moment), you can access the player at:

Host URL: http://127.0.0.1:5000

Network Client URL: http://<your-local-ip>:5000 (Useful for using a phone as a remote/output)

#  How It Works

The Listening Loop

The project uses Google's YAMNet deep learning model. It captures 1-second chunks of audio and classifies them into one of 521 categories.

Weighted Logic & Thresholds

To prevent the music from "flickering" between genres, we use a Weighted Threshold System.

Evidence Gathering: Every sound (e.g., Laughter) adds "points" toward a proposed mood.

Variable Weights: Distinct sounds like "Bird chirps" are weighted higher ($1.5$) than common sounds like "Speech" ($0.7$).

The Boost: If a sound is detected that belongs to a different genre than what is currently playing, its score is multiplied by 1.5x to help the AI break through the current mood faster.

The Music Engine

Once a threshold is hit:

Selection: The backend picks a random track from a curated YouTube Music playlist.

Streaming: yt-dlp extracts the direct audio stream.

Cross-fade: The frontend performs a smooth 2-second cross-fade between the old and new tracks.

#  File Structure

main.py: The Python Flask backend and AI processing loop.

mapping.json: Defines which YAMNet sounds (e.g., "Siren") trigger which genres (e.g., "Rock").

genres.json: Stores the YouTube Music playlist IDs for each mood.

templates/index.html: The glassmorphic UI, audio buffers, and cross-fade logic.

#  Key Controls

Automatic: The AI handles everything.

Manual Overrides: The UI includes Skip, Previous, Volume, and a Seek Bar for manual control.

Visual Feedback: The background color shifts dynamically to match the current genre's "color identity."

#  Limitations & Notes

Music Suppression: The AI is tuned to ignore its own music playback to prevent a feedback loop.

Browser Permission: Browsers require a user gesture to play audio. Click the "Enable Audio" button on the landing page to start the stream.