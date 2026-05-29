Pedestrian Dwell Time Tracker
A real-time pedestrian dwell time extraction tool built on top of a public livestream, developed as part of the IoT & Big Data module at Erasmus Hogeschool Brussel (Postgraduate Applied AI, 2025–2026).
Based on the starter code by Maarten Dequanter: https://github.com/mdequanter/PostGraduatAI
What it does
Connects to a public YouTube livestream (Jackson Hole Town Square, Wyoming), allows the user to define a waiting zone by clicking two corners on the frame, and then detects and tracks pedestrians using YOLOv8. For each person who enters and exits the zone, it logs their entry timestamp, exit timestamp, dwell duration in seconds, and tracking ID to a CSV file.
Files
	•	trackDwellTime.py — main tracking script
	•	visualise_dwell.py — generates charts from the logged CSV data
	•	dwell_log.csv — sample output data
	•	dwell_visualisation.png — sample visualisation
How to run

Requirements

Python 3.10+
brew (for yt-dlp)

Installation
bash# Clone the repository
git clone https://github.com/JRoemeth/iot-dwell-time-tracker.git
cd iot-dwell-time-tracker

# Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate

# Install Python dependencies
pip install opencv-python ultralytics

# Install yt-dlp via Homebrew
brew install yt-dlp
How to run
bashsource venv/bin/activate
python trackDwellTime.py
Click two corners on the video frame to define the waiting zone. Press q to quit, r to reset the zone.