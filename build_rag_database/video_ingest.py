from moviepy import VideoFileClip
import os
import speech_recognition as sr

def extract_frames(video_path, frame_dir, fps=1):
    os.makedirs(frame_dir, exist_ok=True)
    clip = VideoFileClip(video_path)
    clip.write_images_sequence(f"{frame_dir}/frame_%06d.jpg", fps=fps)

def video_to_audio(video_path, output_audio_path):
    clip = VideoFileClip(video_path)
    audio = clip.audio
    audio.write_audiofile(output_audio_path)

def audio_to_text(audio_path):
    recognizer = sr.Recognizer()
    audio = sr.AudioFile(audio_path)
    with audio as source:
        # Record the audio data
        audio_data = recognizer.record(source)
        try:
            # Recognize the speech
            text = recognizer.recognize_whisper(audio_data)
        except sr.UnknownValueError:
            print("Speech recognition could not understand the audio.")
        except sr.RequestError as e:
            print(f"Could not request results from service; {e}")
    return text

# extract_frames("ZXoaMa6jlO4.mp4", "frames", fps=1)
# video_to_audio("ZXoaMa6jlO4.mp4", "output_audio.wav")
# text = audio_to_text("output_audio.wav")
# print(text)