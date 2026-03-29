import json
import csv
import numpy as np
import sounddevice as sd
import tensorflow as tf
import tensorflow_hub as hub
from collections import deque, Counter
import os
import time
import sys
import json
import random


# --- Configuration ---
SAMPLE_RATE = 16000 
CHUNK_DURATION = 1.0 
# 10 seconds provides a good balance for testing
SMOOTHING_WINDOW = 10

#-----

# --- VLC PATH FIX ---
# If you get a FileNotFoundError for libvlc.dll, set the path to your VLC folder here.
VLC_PATH = r"C:\Program Files\VideoLAN\VLC" 
if os.name == 'nt':  # Windows specific fix
    if os.path.exists(VLC_PATH):
        os.add_dll_directory(VLC_PATH)

try:
    import vlc
    from ytmusicapi import YTMusic
    import yt_dlp
    
except ImportError as e:
    print(f"[Error] Missing libraries. Run: pip install python-vlc yt-dlp ytmusicapi")
    sys.exit(1)

#------


class MoodMusicPlayer:
    """
    A backend mockup that fetches a random song from a mood-matched playlist
    and streams it directly using VLC.
    """
    def __init__(self):
        self.current_playlist_id = None
        self.current_tracks = []
        print("[System] Initializing Mood Music Device (VLC Stream Mode)...")
        self.ytmusic = YTMusic()
        
        try:
            self.vlc_instance = vlc.Instance('--no-video')
            self.player = self.vlc_instance.media_player_new()
        except NameError:
            print("[Error] VLC module failed to initialize. Ensure VLC is installed.")
            sys.exit(1)
        
        # Load genres from external JSON file
        self.genres_file = "genres.json"
        self.mood_playlists = self.load_genres()
        
    def load_genres(self):
        """Loads mood-to-playlist mappings from an external JSON file."""
        default_genres = {
            "chill": "PLMIz2z2gI_PtoJndQ9H2Mib1i7p-qUUX1",
            "upbeat": "PLMIz2z2gI_Ptq4F3H242Wnt5D6zM2xY9b",
            "focus": "PLMIz2z2gI_PtH01B2wX8gR2wRERqX2y4a"
        }

        # If file doesn't exist, create it
        if not os.path.exists(self.genres_file):
            print(f"[Error] {self.genres_file} not found. Creating a default one.")
            with open(self.genres_file, "w") as f:
                json.dump(default_genres, f, indent=4)
            return default_genres
        
        # If it exists, try to load it. If it's empty/corrupted, catch the error and reset.
        try:
            with open(self.genres_file, "r") as f:
                return json.load(f)
        except json.JSONDecodeError:
            print(f"[Error] {self.genres_file} is empty or corrupted. Recreating default.")
            with open(self.genres_file, "w") as f:
                json.dump(default_genres, f, indent=4)
            return default_genres

    def get_song_from_category(self, mood_category):
        print(f"\n[System] LLM selected category: '{mood_category}'")
        mood_category = mood_category.lower().strip()
        
        if mood_category not in self.mood_playlists:
            print(f"[Warning] Unknown category '{mood_category}'. Defaulting to 'chill'.")
            mood_category = "chill"
            
        playlist_id = self.mood_playlists[mood_category]
        playlist_url = f"https://www.youtube.com/playlist?list={playlist_id}"
        
        print(f"[Search] Fetching playlist: {playlist_url}")

        try:
            ydl_opts = {
                'quiet': True,
                'extract_flat': True,  # fast, no full metadata
            }

            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(playlist_url, download=False)
                entries = info.get('entries', [])

            if not entries:
                print("[Search] Playlist is empty.")
                return None

            selected = random.choice(entries)

            video_id = selected.get('id')
            title = selected.get('title', 'Unknown Title')

            print(f"[Search] Selected: {title}")
            return video_id, title, "Unknown Artist"

        except Exception as e:
            print(f"[Search] Error fetching playlist: {e}")
            return None

    def get_direct_stream_url(self, video_id):
        """Uses yt-dlp to extract the direct audio stream URL from a YouTube Video ID."""
        print("[Extractor] Extracting direct audio stream URL...")
        youtube_url = f"https://www.youtube.com/watch?v={video_id}"
        
        ydl_opts = {
            'format': 'bestaudio/best',
            'quiet': True,
            'noplaylist': True,
        }
        
        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(youtube_url, download=False)
                return info.get('url')
        except Exception as e:
            print(f"[Extractor] Error extracting stream: {e}")
            return None

    def play_mood(self, mood_category):
        mood_category = mood_category.lower().strip()

        if mood_category not in self.mood_playlists:
            print(f"[Warning] Unknown category '{mood_category}', defaulting to 'chill'")
            mood_category = "chill"

        playlist_id = self.mood_playlists[mood_category]

        # ✅ ONLY fetch if playlist changed
        if playlist_id != self.current_playlist_id:
            print(f"[System] New playlist detected → fetching...")
            
            try:
                playlist = self.ytmusic.get_playlist(playlist_id)
                tracks = playlist.get('tracks', [])

                if not tracks:
                    print("[Search] Playlist empty.")
                    return

                self.current_tracks = tracks
                self.current_playlist_id = playlist_id

            except Exception as e:
                print(f"[Search] Error fetching playlist: {e}")
                return
        else:
            print("[System] Reusing cached playlist")

        # ✅ Just pick from cache
        selected_track = random.choice(self.current_tracks)

        video_id = selected_track.get('videoId')
        title = selected_track.get('title')
        artists = ", ".join([a['name'] for a in selected_track.get('artists', [])])

        print(f"[Player] Now Playing: '{title}' by {artists}")

        stream_url = self.get_direct_stream_url(video_id)
        if not stream_url:
            print("[Player] Stream extraction failed.")
            return

        self.player.stop()
        media = self.vlc_instance.media_new(stream_url)
        self.player.set_media(media)
        self.player.play()

    def stop(self):
        """Stops the VLC player safely."""
        if hasattr(self, 'player'):
            print("[Player] Stopping playback.")
            self.player.stop()






def load_config(filename):
    """Generic function to load JSON files."""
    if not os.path.exists(filename):
        print(f"Error: {filename} not found. Please create it.")
        return {}
    with open(filename, "r") as f:
        return json.load(f)

def main():
    current_mood = None
    device = MoodMusicPlayer()
    print("Loading YAMNet model...")
    model = hub.load('https://tfhub.dev/google/yamnet/1')
    
    class_map_path = model.class_map_path().numpy().decode('utf-8')
    class_names = []
    with open(class_map_path) as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            class_names.append(row['display_name'])

    genres_db = load_config("genres.json")
    sound_to_genre = load_config("mapping.json")
    
    if not genres_db or not sound_to_genre: 
        print("Missing required config files! Exiting.")
        return

    history = deque(maxlen=SMOOTHING_WINDOW)
    current_playing_genre = None
    last_detected_sound = None

    print(f"\n🎤 Ready! Listening to ambient sounds... (Smoothing: {SMOOTHING_WINDOW}s)")
    print("Press Ctrl+C to stop")
    print("-" * 70)

    try:
        with sd.InputStream(samplerate=SAMPLE_RATE, channels=1, dtype='float32') as stream:
            while True:
                audio_chunk, overflowed = stream.read(int(SAMPLE_RATE * CHUNK_DURATION))
                waveform = np.squeeze(audio_chunk)
                
                scores, embeddings, spectrogram = model(waveform)
                mean_scores = np.mean(scores.numpy(), axis=0) # Convert to numpy for indexing
                
                # --- PERSISTENCE BONUS ---
                # If we heard a sound last time, give it a 1.5x boost to prevent jitter
                if last_detected_sound in class_names:
                    idx = class_names.index(last_detected_sound)
                    mean_scores[idx] *= 1.5

                top_3_indices = np.argsort(mean_scores)[::-1][:3]
                
                best_sound = class_names[top_3_indices[0]]
                best_score = mean_scores[top_3_indices[0]]
                second_sound = class_names[top_3_indices[1]]
                second_score = mean_scores[top_3_indices[1]]
                
                # --- SILENCE & SENSITIVITY TUNING ---
                # 1. We make it HARDER to switch to Silence (Threshold 0.40)
                # 2. We make it EASIER for other sounds to break through Silence (Threshold 0.08)
                
                final_selection = best_sound
                
                # If the AI thinks it's silence, it must be VERY sure (0.40)
                if best_sound in ["Silence", "Inside, small room"] and best_score < 0.40:
                    # If it's not sure about silence, check the second best sound
                    if second_score > 0.08:
                        final_selection = second_sound
                    else:
                        final_selection = "Silence"
                
                # If it's a regular sound, it only needs 0.12 confidence to be counted
                elif best_score < 0.12:
                    final_selection = "Silence"

                last_detected_sound = final_selection
                
                # Map the best sound to our genres
                if final_selection not in sound_to_genre:
                    target_genre = "chill"
                    if final_selection != "Silence":
                        print(f"  [!] Missing from mapping.json: '{final_selection}'")
                else:
                    target_genre = sound_to_genre.get(final_selection, "chill")
                
                history.append(target_genre)
                
                # Get the most common genre AND how many seconds it was heard
                counts = Counter(history).most_common(1)
                most_common_genre, vote_count = counts[0]
                
                playlist_id = genres_db.get(most_common_genre, "Unknown Playlist")

                # Print the TOP 3 for debugging
                top_3_strings = [f"{class_names[i]} ({mean_scores[i]:.2f})" for i in top_3_indices]
                print(f"Top 3: {', '.join(top_3_strings)}")
                
                print(f"-> Selected: [{final_selection}] mapped to '{target_genre}' | Playing: '{most_common_genre}' ({vote_count}/{SMOOTHING_WINDOW}s)\n")

                # Switch if it has 6/10 votes
                if most_common_genre != current_playing_genre and vote_count >= 3:
                    print("="*70)
                    print(f"🎵 SWITCHING PLAYLIST: Now playing '{most_common_genre}'")
                    print(f"🔗 YouTube Link: https://www.youtube.com/playlist?list={playlist_id}")
                    print("="*70 + "\n")
                    current_playing_genre = most_common_genre
                    current_mood = current_playing_genre
                    device.play_mood(current_mood)

    except KeyboardInterrupt:
        print("\nStopping Ambient DJ. Goodbye!")

if __name__ == "__main__":
    main()


#--------------------------------------------------------------------------------------------------------------------------

    
