# Makefile

.PHONY: install run clean

PYTHON_VERSION=3.11  # Specify your desired Python version
IMAGE=your_image.jpg  # Replace with your image file
SCRIPT=palm_tree_to_midi.py  # Replace with your script name

install:
	pipenv --python $(PYTHON_VERSION)
	pipenv install opencv-python mido python-rtmidi

run: install
	pipenv run python $(SCRIPT)

clean:
	pipenv --rm
