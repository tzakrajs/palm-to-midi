import cv2
import numpy as np
from mido import Message, MidiFile, MidiTrack, bpm2tempo, MetaMessage
import os

# Load the image
image_path = os.environ.get('IMAGE', 'palm_tree.png')  # Replace with your image path
image = cv2.imread(image_path)

# Resize image for easier processing
scale_percent = 50  # Percent of original size
width = int(image.shape[1] * scale_percent / 100)
height = int(image.shape[0] * scale_percent / 100)
dim = (width, height)
resized = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)

# Convert to grayscale
gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)

# Use Canny edge detection
edges = cv2.Canny(gray, 50, 150)

# Find contours
contours, hierarchy = cv2.findContours(
    edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

# Assume the largest contour is the trunk
contour_areas = [cv2.contourArea(c) for c in contours]
if contour_areas:
    max_index = np.argmax(contour_areas)
    trunk_contour = contours[max_index]
else:
    print("No contours found.")
    exit()

# Create a mask for the trunk
mask = np.zeros_like(gray)
cv2.drawContours(mask, [trunk_contour], -1, 255, thickness=cv2.FILLED)

# Extract the trunk area
trunk = cv2.bitwise_and(gray, gray, mask=mask)

# Analyze patterns in the trunk
# We'll scan horizontally and sum pixel values to create a pattern
height, width = trunk.shape
pattern = []

for y in range(0, height, 5):  # Sample every 5 pixels vertically
    row = trunk[y, :]
    avg_intensity = np.mean(row)
    pattern.append(avg_intensity)

# Constrain the pitch range to a specific octave (e.g., C3 to C5)
min_note = 48  # C3
max_note = 72  # C5

# Normalize the pattern to MIDI note numbers within the desired range
min_intensity = min(pattern)
max_intensity = max(pattern)
normalized_pattern = [
    int(min_note + (p - min_intensity) / (max_intensity - min_intensity) * (max_note - min_note))
    for p in pattern
]

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
    modulation = int((pattern[i] - min_intensity) / (max_intensity - min_intensity) * 127)
    
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
