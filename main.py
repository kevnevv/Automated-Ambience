import json
import csv
import numpy as np
import sounddevice as sd
import tensorflow as tf
import tensorflow_hub as hub
import os
import time
import sys
import random
import threading
import logging
from flask import Flask, jsonify, render_template

#This imports all the essential APIs and useful resources to process and gather videos from youtube  

SAMPLE_RATE = 16000 
CHUNK_DURATION = 1.0 
#Records audio in 1-second chunks
#Uses 16kHz sample rate (standard for speech/audio AI)


OUT_OF_GENRE_SCORE_BOOST = 1.5 
OUT_OF_GENRE_WEIGHT_MULTIPLIER = 1.3
#Boosts score based on if it is expected or not
#e.g chill music to loud drill boosts the chance it is a deliberate change


MOOD_THRESHOLDS = {
    "upbeat": 4,
    "rock": 4,
    "metal": 4,
    "dark": 5,
    "focus": 4,
    "chill_pop": 4,
    "synthwave": 4,
    "classical": 5,
    "country": 4,
    "chill_jazz": 6,
    "chill": 8,
    "chill_lofi": 5,
    "lofi": 5,
    "default": 6
}
#Each genre needs a different amount of evidence before switching.
#This helps prevent constant flickering between moods


SOUND_WEIGHTS = {
    "Speech": 0.7,           
    "Laughter": 1.5,         
    "Music": 0.0,            
    "Silence": 1.0,
    "Animal": 1.5,
    "Bird": 1.5,
    "Crying, sobbing": 2.0,
    "Vehicle": 1.5,
    "Drill": 2.0,
    "Heavy metal": 2.0,
    "Jackhammer": 2.0,
    "Construction": 2.0,
}
#Different sounds have different weights, Music is set to 0 to ensure that the program doesnt loop in on itself
#Drill is unlikley to be mistaken so has a much higher weight than speech which is easy to detect

#Decay Settings
#Decay, reduces progress if signals aren’t consistent
DECAY_RATE = 0.5 
PERSISTENCE_BONUS = 1.15 
SILENCE_HARD_LIMIT = 0.35      
SOUND_BREAKTHROUGH_MIN = 0.10  
MUSIC_SUPPRESSION_THRESHOLD = 0.15 
LOFI_SILENCE_TIMEOUT = 30 

try:
    from ytmusicapi import YTMusic
    import yt_dlp
except ImportError as e:
    print(f"[Error] Missing libraries. Please install ytmusicapi and yt_dlp")
    sys.exit(1)

#YTMusic fetches playlists
#yt_dlp extract playable audio stream URLs


app = Flask(__name__, template_folder='templates')
log = logging.getLogger('werkzeug')
log.setLevel(logging.ERROR)
#Helps connect frontend to backend


#This is the music engine which takes the input and plays the relevant music based on the observed genre if it is above the threshold
class MoodMusicPlayer:
    def __init__(self):
        self.current_playlist_id = None
        self.current_tracks = []
        self.history = []  
        
        self.state = {
            "url": None,
            "title": "Waiting for mood...",
            "artists": "AI DJ",
            "mood": "Initializing...",
            "ai_detected": "None",
            "ai_proposed_mood": None,
            "ai_progress": 0,
            "ai_threshold": MOOD_THRESHOLDS["default"]
        }
        
        self.ytmusic = YTMusic()
        self.genres_file = "genres.json"
        self.mood_playlists = self.load_genres()

    #Loads playlists from genres.json
    #Default_genres exists as a backup if necersary
    def load_genres(self):
        default_genres = {
            "chill": "PLOzDu-MXXLliO9fBNZOQTBDddoA3FzZUo",
            "upbeat": "PLDIoUOhQQPlXr63I_vwF9GD8sAKh77dWU",
            "focus": "PL363lLGC9MvjZQgO1eWLxb_4Q7nw5ErKU",
            "dark": "PLCaC1c0-Iw9eFrFlEbYt8jdcZARMGsLpx",
            "chill_pop": "PL4QNnZJr8sRPmuz_d87ygGR6YAYEF-fmw",
            "chill_lofi": "PLOzDu-MXXLlj7croDcwz33c-a5rpNEBNe",
            "chill_jazz": "PL8F6B0753B2CCA128",
            "jazz": "PL4kZATn_awL5Psxy7jrpcBMhDY_DaSkpp",
            "rock": "PL0GvsLQil0MmYC96KEs_7dTNsLm1PS6JX",
            "metal": "PLxJJcP5YX06cLFAQ5MbdtXRH-0AxUs4ld",
            "synthwave": "PLoNgUIB3urjvBKHDCmd3JP-HqvzQf2awD",
            "classical": "PL2788304DC59DBEB4",
            "country": "PL3oW2tjiIxvQW6c-4Iry8Bpp3QId40S5S"
        }
        if not os.path.exists(self.genres_file):
            with open(self.genres_file, "w") as f:
                json.dump(default_genres, f, indent=4)
            return default_genres
        try:
            with open(self.genres_file, "r") as f:
                return json.load(f)
        except Exception:
            return default_genres

    #Uses yt_dlp to extract audio stream from YouTube
    def get_direct_stream_url(self, video_id):
        youtube_url = f"https://www.youtube.com/watch?v={video_id}"
        ydl_opts = {'format': 'bestaudio/best', 'quiet': True, 'noplaylist': True, 'no_warnings': True}
        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(youtube_url, download=False)
                return info.get('url')
        except Exception as e:
            return None

    #Updates current song info
    #Sets stream URL
    def _play_track(self, track, mood_category):
        video_id = track.get('videoId')
        title = track.get('title')
        artists = ", ".join([a['name'] for a in track.get('artists', [])])
        stream_url = self.get_direct_stream_url(video_id)
        if stream_url:
            self.state["url"] = stream_url
            self.state["title"] = title
            self.state["artists"] = artists
            self.state["mood"] = mood_category

    #Loads playlist for the genre and picks a random track
    def play_mood(self, mood_category):
        mood_category = mood_category.lower().strip()
        playlist_id = self.mood_playlists.get(mood_category, self.mood_playlists.get("chill"))
        if playlist_id != self.current_playlist_id:
            try:
                playlist = self.ytmusic.get_playlist(playlist_id)
                self.current_tracks = playlist.get('tracks', [])
                self.current_playlist_id = playlist_id
            except Exception: return
        if not self.current_tracks: return
        selected_track = random.choice(self.current_tracks)
        if self.state["url"]:
            self.history.append(self.state.copy())
            if len(self.history) > 10: self.history.pop(0)
        self._play_track(selected_track, mood_category)

    def skip_track(self):
        if self.current_tracks:
            selected_track = random.choice(self.current_tracks)
            self._play_track(selected_track, self.state["mood"])

    def previous_track(self):
        if self.history:
            self.state = self.history.pop()

    #Both of these are track controls

device = MoodMusicPlayer()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/state')
def get_state():
    return jsonify(device.state)

@app.route('/api/skip', methods=['POST'])
def skip():
    device.skip_track()
    return jsonify({"status": "skipped"})

@app.route('/api/prev', methods=['POST'])
def prev():
    device.previous_track()
    return jsonify({"status": "went back"})

def load_config(filename):
    if not os.path.exists(filename): return {}
    with open(filename, "r") as f: return json.load(f)

#Loads AI model YAMNet which is trained to recognise 500+ sounds 
def ai_listen_loop():
    print("[AI] Loading Model...")
    model = hub.load('https://tfhub.dev/google/yamnet/1')
    class_map_path = model.class_map_path().numpy().decode('utf-8')
    class_names = []
    #Gathers class names e.g. Speech, Dog bark, Music, Drill
    
    with open(class_map_path) as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader: class_names.append(row['display_name'])

    sound_to_genre = load_config("mapping.json")
    
    current_playing_genre = None
    last_detected_sound = None
    silence_counter = 0
    
    consecutive_new_mood_count = 0.0
    last_proposed_mood = None

    try:
        music_idx = class_names.index("Music")
    except ValueError: music_idx = None

    print(f"\n🎤 AI Listener Active with Comprehensive Genre Support...")

    try:
        with sd.InputStream(samplerate=SAMPLE_RATE, channels=1, dtype='float32') as stream:
            while True:
                audio_chunk, _ = stream.read(int(SAMPLE_RATE * CHUNK_DURATION))
                waveform = np.squeeze(audio_chunk)
                scores, _, _ = model(waveform)
                mean_scores = np.mean(scores.numpy(), axis=0) if scores.numpy().ndim > 1 else scores.numpy()
                
                #Suppress Music
                if music_idx is not None and mean_scores[music_idx] > MUSIC_SUPPRESSION_THRESHOLD: 
                    mean_scores[music_idx] = 0.0
                
                #Artificially inflate the confidence of sounds that belong to a different genre
                if current_playing_genre:
                    for i, sound_class in enumerate(class_names):
                        mapped_genre = sound_to_genre.get(sound_class, "chill")
                        if mapped_genre != current_playing_genre:
                            mean_scores[i] *= OUT_OF_GENRE_SCORE_BOOST
                
                # Determine best sound
                top_idx = np.argsort(mean_scores)[::-1]
                best_sound, best_score = class_names[top_idx[0]], mean_scores[top_idx[0]]
                
                final_selection = best_sound
                if best_score < 0.12: final_selection = "Silence"

                last_detected_sound = final_selection
                
                if final_selection == "Silence":
                    silence_counter += 1
                    target_genre = "lofi" if silence_counter >= LOFI_SILENCE_TIMEOUT else current_playing_genre
                else:
                    silence_counter = 0 
                    target_genre = sound_to_genre.get(final_selection, "chill")
                
                #Get weight for the specific sound detected
                current_sound_weight = SOUND_WEIGHTS.get(final_selection, 1.0)
                
                #Get threshold for the specific genre target
                active_threshold = MOOD_THRESHOLDS.get(target_genre, MOOD_THRESHOLDS["default"])
                
                if target_genre != current_playing_genre:
                    
                    #Boost points because it's breaking away from the current genre
                    current_sound_weight *= OUT_OF_GENRE_WEIGHT_MULTIPLIER
                    
                    if target_genre == last_proposed_mood:
                        consecutive_new_mood_count += current_sound_weight
                    else:
                        consecutive_new_mood_count = max(0.0, consecutive_new_mood_count - DECAY_RATE)
                        if consecutive_new_mood_count == 0:
                            last_proposed_mood = target_genre
                            consecutive_new_mood_count = current_sound_weight
                else:
                    consecutive_new_mood_count = max(0.0, consecutive_new_mood_count - DECAY_RATE)
                    if consecutive_new_mood_count == 0:
                        last_proposed_mood = None

                #Update Frontend State
                device.state["ai_detected"] = final_selection
                device.state["ai_proposed_mood"] = last_proposed_mood
                device.state["ai_progress"] = round(consecutive_new_mood_count, 2)
                device.state["ai_threshold"] = active_threshold

                #Console Log
                if last_proposed_mood:
                    print(f"Detect: {final_selection[:10]} (w:{current_sound_weight:.2f}) -> {last_proposed_mood} | Progress: {consecutive_new_mood_count:.1f}/{active_threshold}")

                if last_proposed_mood and consecutive_new_mood_count >= active_threshold:
                    print(f"\n🎵 SWITCHING MOOD TO: {last_proposed_mood.upper()}\n")
                    current_playing_genre = last_proposed_mood
                    device.play_mood(current_playing_genre)
                    consecutive_new_mood_count = 0.0
                    last_proposed_mood = None

    except Exception as e:
        print(f"Listener error: {e}")

if __name__ == "__main__":
    threading.Thread(target=ai_listen_loop, daemon=True).start()
    app.run(host="0.0.0.0", port=5000, use_reloader=False)
