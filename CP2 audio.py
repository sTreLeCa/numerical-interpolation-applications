import numpy as np
import matplotlib.pyplot as plt
from gtts import gTTS
from scipy.io import wavfile
import librosa
from scipy.interpolate import PchipInterpolator
import soundfile as sf
# Function to convert text to speech and save as WAV file
def text_to_wav(text, filename):
    tts = gTTS(text=text, lang='en')
    tts.save(filename)
    y, sr = librosa.load(filename)
    return y, sr

# Function for linear interpolation
def linear_interpolation(x, y, x_new):
    return np.interp(x_new, x, y)

# Function for Hermite interpolation
def hermite_interpolation(x, y, x_new):
    interp = PchipInterpolator(x, y)
    return interp(x_new)

# Function for cubic spline interpolation
def cubic_spline_interpolation(x, y, x_new):
    from scipy.interpolate import CubicSpline
    cs = CubicSpline(x, y)
    return cs(x_new)

# Save audio as WAV file
def save_audio_wav(y, sr, filename):
    wavfile.write(filename, sr, y)
    # sf.write(filename, y, sr)


# Generate input text
input_text = "This is sample text"

# User-defined parameters
num_interpolation_points =  100000# Number of interpolation points

# Convert text to WAV file
y_original, sr = text_to_wav(input_text, "original.wav")

# Time array for original audio
time_original = np.linspace(0, len(y_original) / sr, num_interpolation_points)
y_original = np.append(y_original, y_original[-1])
points = y_original[(time_original * sr).astype(int)]

# Linear interpolation
x_new = np.arange(0, len(y_original) / sr, 1 / 24000)
y_linear = linear_interpolation(time_original, points, x_new)
save_audio_wav(y_linear, sr, "linear_interpolated.wav")

# Hermite interpolation
y_hermite = hermite_interpolation(time_original, points, x_new)
save_audio_wav(y_hermite, sr, "hermite_interpolated.wav")

# Cubic spline interpolation
y_spline = cubic_spline_interpolation(time_original, points, x_new)
save_audio_wav(y_spline, sr, "cubic_spline_interpolated.wav")
sf.write("linear.wav",y_linear,sr)
sf.write("hermite.wav",y_hermite,sr)
sf.write("cubic.wav",y_spline,sr)
# Plot original and interpolated signals
plt.figure(figsize=(12, 8))

# Original signal
plt.subplot(4, 1, 1)
plt.plot(np.linspace(0, len(y_original) / sr, len(y_original)), y_original)
plt.title('Original Signal')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')

# Linear interpolation
plt.subplot(4, 1, 2)
plt.plot(x_new, y_linear)
plt.title('Linear Interpolation')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')

# Hermite interpolation
plt.subplot(4, 1, 3)
plt.plot(x_new, y_hermite)
plt.title('Hermite Interpolation')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')

# Cubic spline interpolation
plt.subplot(4, 1, 4)
plt.plot(x_new, y_spline)
plt.title('Cubic Spline Interpolation')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')

plt.tight_layout()
plt.show()