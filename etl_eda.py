import cv2
from detection import ObjectDetector
from tracking import ObjectTracker
from hazards import HazardDetector
from alerts import AlertSystem

def safety_pipeline(video_source=0):
    """Main vision-based safety pipeline."""
    detector = ObjectDetector()
    tracker = ObjectTracker()
    hazard_detector = HazardDetector()
    alerter = AlertSystem('sender@example.com', 'receiver@example.com')

    cap = cv2.VideoCapture(video_source)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        detections = detector.detect(frame)
        tracks = tracker.track(detections, frame)
        hazards = hazard_detector.detect_hazards(tracks)

        for hazard in hazards:
            alerter.send_alert(hazard)

        # Optional: Draw boxes, etc.
        cv2.imshow('Safety Monitoring', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# Usage: safety_pipeline('path/to/video.mp4')