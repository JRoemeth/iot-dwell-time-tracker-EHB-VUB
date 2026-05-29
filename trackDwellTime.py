import cv2
import subprocess
import sys
import csv
import time
from datetime import datetime
from ultralytics import YOLO

YOUTUBE_URL = "https://www.youtube.com/watch?v=DoUOrTJbIu4"
MODEL_PATH = "yolov8n.pt"
CSV_FILE = "dwell_log.csv"

# Zone definition — set by clicking two corners on the frame
zone = []
zone_defined = False

# Tracks active persons in zone: {track_id: entry_timestamp}
active_in_zone = {}

# CSV setup
csv_file = open(CSV_FILE, "w", newline="")
csv_writer = csv.writer(csv_file)
csv_writer.writerow(["timestamp_entry", "timestamp_exit", "dwell_seconds", "track_id"])


def mouse_callback(event, x, y, flags, param):
    global zone, zone_defined
    if event == cv2.EVENT_LBUTTONDOWN:
        if len(zone) < 2:
            zone.append((x, y))
            print(f"Zone point {len(zone)} set: ({x}, {y})")
        if len(zone) == 2:
            zone_defined = True
            print(f"Zone defined: {zone[0]} -> {zone[1]}")


def in_zone(cx, cy):
    global zone_defined
    if not zone_defined:
        return False
    x1, y1 = min(zone[0][0], zone[1][0]), min(zone[0][1], zone[1][1])
    x2, y2 = max(zone[0][0], zone[1][0]), max(zone[0][1], zone[1][1])
    return x1 <= cx <= x2 and y1 <= cy <= y2


def get_stream_url(youtube_url: str) -> str:
    cmd = ["yt-dlp", "--cookies-from-browser", "chrome", "-g", youtube_url]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"yt-dlp error:\n{result.stderr}")
    for line in result.stdout.splitlines():
        line = line.strip()
        if line.startswith("http"):
            return line
    raise RuntimeError("No valid stream URL found.")


def main():
    global active_in_zone, zone_defined, zone

    try:
        print("Loading YOLO model...")
        model = YOLO(MODEL_PATH)

        print("Getting stream URL...")
        stream_url = get_stream_url(YOUTUBE_URL)
        print("Stream found.")

        cap = cv2.VideoCapture(stream_url)
        if not cap.isOpened():
            raise RuntimeError("Could not open stream.")

        window = "Pedestrian Dwell Time Tracker"
        cv2.namedWindow(window)
        cv2.setMouseCallback(window, mouse_callback)

        print("INSTRUCTIONS:")
        print("  1. Click TWO points on the frame to define the waiting zone")
        print("  2. The zone will turn green once defined")
        print("  3. Press 'r' to reset the zone")
        print("  4. Press 'q' to quit")
        print(f"  5. Data is being logged to: {CSV_FILE}")

        while True:
            ret, frame = cap.read()
            if not ret:
                print("No frame received. Stream may have ended.")
                break

            display = frame.copy()

            # Draw zone
            if len(zone) == 1:
                cv2.circle(display, zone[0], 5, (0, 255, 255), -1)
                cv2.putText(display, "Click second corner to define zone",
                            (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            elif zone_defined:
                x1 = min(zone[0][0], zone[1][0])
                y1 = min(zone[0][1], zone[1][1])
                x2 = max(zone[0][0], zone[1][0])
                y2 = max(zone[0][1], zone[1][1])
                cv2.rectangle(display, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(display, "WAITING ZONE", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            else:
                cv2.putText(display, "Click to define waiting zone (2 corners)",
                            (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

            # Run YOLO with tracking
            results = model.track(frame, persist=True, verbose=False, classes=[0])  # class 0 = person

            current_ids_in_zone = set()

            if (results[0].boxes is not None and
                    results[0].boxes.id is not None and
                    zone_defined):

                boxes = results[0].boxes.xyxy.cpu().numpy()
                track_ids = results[0].boxes.id.cpu().numpy().astype(int)

                for box, track_id in zip(boxes, track_ids):
                    x1, y1, x2, y2 = map(int, box)
                    cx = (x1 + x2) // 2
                    cy = (y1 + y2) // 2  # use bottom of box as ground point
                    cy_ground = y2

                    person_in_zone = in_zone(cx, cy_ground)

                    if person_in_zone:
                        current_ids_in_zone.add(track_id)

                        if track_id not in active_in_zone:
                            # Person just entered zone
                            active_in_zone[track_id] = time.time()
                            print(f"Person {track_id} entered zone at {datetime.now().strftime('%H:%M:%S')}")

                        # Calculate current dwell time
                        dwell = time.time() - active_in_zone[track_id]

                        # Draw person in zone
                        cv2.rectangle(display, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        cv2.putText(display, f"ID:{track_id} {dwell:.1f}s",
                                    (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX,
                                    0.6, (0, 255, 0), 2)
                    else:
                        # Draw person outside zone
                        cv2.rectangle(display, (x1, y1), (x2, y2), (255, 100, 0), 1)
                        cv2.putText(display, f"ID:{track_id}",
                                    (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX,
                                    0.5, (255, 100, 0), 1)

            # Check for people who left the zone
            exited_ids = set(active_in_zone.keys()) - current_ids_in_zone
            for track_id in exited_ids:
                entry_time = active_in_zone.pop(track_id)
                exit_time = time.time()
                dwell_seconds = round(exit_time - entry_time, 2)
                entry_str = datetime.fromtimestamp(entry_time).strftime('%Y-%m-%d %H:%M:%S')
                exit_str = datetime.fromtimestamp(exit_time).strftime('%Y-%m-%d %H:%M:%S')
                csv_writer.writerow([entry_str, exit_str, dwell_seconds, track_id])
                csv_file.flush()
                print(f"Person {track_id} left zone. Dwell time: {dwell_seconds}s")

            # Stats overlay
            cv2.putText(display,
                        f"In zone: {len(current_ids_in_zone)} | Logged: {sum(1 for _ in open(CSV_FILE)) - 1}",
                        (20, display.shape[0] - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            cv2.imshow(window, display)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key == ord("r"):
                zone.clear()
                zone_defined = False
                active_in_zone.clear()
                print("Zone reset.")

        cap.release()
        cv2.destroyAllWindows()
        csv_file.close()
        print(f"Data saved to {CSV_FILE}")

    except Exception as e:
        print(f"Error: {e}")
        csv_file.close()
        sys.exit(1)


if __name__ == "__main__":
    main()
