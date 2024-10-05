import cv2
import numpy as np
from mido import Message, MidiFile, MidiTrack, bpm2tempo, MetaMessage
import os

# Load the image
image_path = os.environ.get('IMAGE', 'palm_tree.png')  # Replace with your image path
image = cv2.imread(image_path)

# Resize image for easier processing (we'll adjust this based on average intensity)
# Convert to grayscale
gray_full = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Calculate the average intensity of the full image
avg_intensity_full = np.mean(gray_full)

# Determine scaling factor based on average intensity
# For simplicity, we'll map intensity range [0, 255] to scaling factor [30%, 100%]
scale_percent = int(30 + (avg_intensity_full / 255) * 70)  # Scale between 30% and 100%

# Ensure scale_percent is within reasonable bounds
scale_percent = max(30, min(scale_percent, 100))

# Resize image based on calculated scale_percent
width = int(image.shape[1] * scale_percent / 100)
height = int(image.shape[0] * scale_percent / 100)
dim = (width, height)
resized = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)

# Convert resized image to grayscale
gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)

# Use Canny edge detection
edges = cv2.Canny(gray, 50, 150)

# Find contours
contours, hierarchy = cv2.findContours(
    edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

# Check if any contours are found
if not contours:
    print("No contours found in the image.")
    exit()

# Assume the largest contour is the trunk
contour_areas = [cv2.contourArea(c) for c in contours]
max_index = np.argmax(contour_areas)
trunk_contour = contours[max_index]

# Create a mask for the trunk
mask = np.zeros_like(gray)
cv2.drawContours(mask, [trunk_contour], -1, 255, thickness=cv2.FILLED)

# Extract the trunk area
trunk = cv2.bitwise_and(gray, gray, mask=mask)

# Calculate the average intensity of the trunk area
avg_intensity_trunk = np.mean(trunk[trunk > 0])  # Consider only the trunk pixels

# Determine sampling rates based on trunk average intensity
# Map intensity [0, 255] to vertical step size [10, 2] (smaller step size means more samples)
vertical_step = int(10 - (avg_intensity_trunk / 255) * 8)  # Steps between 10 and 2
vertical_step = max(2, min(vertical_step, 10))

# Similarly for horizontal sampling, if desired
# For this example, we'll keep horizontal sampling as the full width

# Analyze patterns in the trunk
height, width = trunk.shape
pattern = []

for y in range(0, height, vertical_step):  # Sample every 'vertical_step' pixels vertically
    row = trunk[y, :]
    avg_row_intensity = np.mean(row[row > 0])  # Consider only non-zero pixels
    if np.isnan(avg_row_intensity):
        avg_row_intensity = 0
    pattern.append(avg_row_intensity)

# Constrain the pitch range to a specific octave (e.g., C3 to C5)
min_note = 48  # C3
max_note = 72  # C5

# Normalize the pattern to MIDI note numbers within the desired range
if pattern:
    min_intensity = min(pattern)
    max_intensity = max(pattern)
    if max_intensity - min_intensity == 0:
        normalized_pattern = [int((min_note + max_note) / 2)] * len(pattern)
    else:
        normalized_pattern = [
            int(min_note + (p - min_intensity) / (max_intensity - min_intensity) * (max_note - min_note))
            for p in pattern
        ]
else:
    print("No pattern data extracted from the trunk.")
    exit()

# Adjust velocity mapping based on intensity contrast
# Compute the difference between successive intensity values
intensity_differences = np.abs(np.diff(pattern))
intensity_differences = np.append(intensity_differences, 0)  # Append 0 to match pattern length

# Normalize velocity between 50 and 127 based on intensity differences
min_diff = min(intensity_differences)
max_diff = max(intensity_differences)
if max_diff - min_diff == 0:
    normalized_velocity = [80] * len(intensity_differences)  # Default velocity if no variation
else:
    normalized_velocity = [
        int(50 + (d - min_diff) / (max_diff - min_diff) * (127 - 50))
        for d in intensity_differences
    ]

# Create MIDI file
mid = MidiFile()
track = MidiTrack()
mid.tracks.append(track)

# Set tempo to 120 BPM using MetaMessage
track.append(MetaMessage('set_tempo', tempo=bpm2tempo(120), time=0))

# Quantize to 16th notes
ticks_per_beat = mid.ticks_per_beat
sixteenth_note = ticks_per_beat // 4

# Add notes to track
time = 0
for i, note in enumerate(normalized_pattern):
    velocity = normalized_velocity[i]

    # Add modulation based on average intensity
    # Map the average intensity to modulation wheel value (controller number 1)
    modulation = int((pattern[i] - min_intensity) / (max_intensity - min_intensity) * 127) if max_intensity - min_intensity != 0 else 0

    # Add Control Change message for modulation
    track.append(Message('control_change', control=1, value=modulation, time=time))
    time = 0  # Reset time for subsequent messages

    # Add Note On event
    track.append(Message('note_on', note=note, velocity=velocity, time=time))
    # Add Note Off event after a 16th note
    track.append(Message('note_off', note=note, velocity=0, time=sixteenth_note))
    time = 0  # Subsequent notes start immediately after the previous

# Save MIDI file
mid.save('palm_tree_music.mid')
print("MIDI file has been created as 'palm_tree_music.mid'")