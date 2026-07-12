"""
Hand Tracking with MediaPipe Tasks API (v0.10+)

Displays webcam feed with hand landmarks and connections.
Shows landmark coordinates for the first detected hand.
"""

import argparse
import cv2 as cv
import mediapipe as mp
import time
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# Landmark names mapping for reference
LANDMARK_NAMES = {
    0: "Wrist",
    1: "THUMB_CMC",
    2: "THUMB_MCP",
    3: "THUMB_IP",
    4: "THUMB_TIP",
    5: "INDEX_FINGER_MCP",
    6: "INDEX_FINGER_PIP",
    7: "INDEX_FINGER_DIP",
    8: "INDEX_FINGER_TIP",
    9: "MIDDLE_FINGER_MCP",
    10: "MIDDLE_FINGER_PIP",
    11: "MIDDLE_FINGER_DIP",
    12: "MIDDLE_FINGER_TIP",
    13: "RING_FINGER_MCP",
    14: "RING_FINGER_PIP",
    15: "RING_FINGER_DIP",
    16: "RING_FINGER_TIP",
    17: "PINKY_MCP",
    18: "PINKY_PIP",
    19: "PINKY_DIP",
    20: "PINKY_TIP",
}


def create_hand_landmarker(
    model_path: str, num_hands: int = 2
) -> vision.HandLandmarker:
    """
    Create and configure a HandLandmarker instance using the MediaPipe Tasks API.
    """
    base_options = python.BaseOptions(model_asset_path=model_path)
    options = vision.HandLandmarkerOptions(
        base_options=base_options,
        num_hands=num_hands,
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    return vision.HandLandmarker.create_from_options(options)


def draw_landmarks_with_names(image, hand_landmarks):
    """Draw hand landmarks with their names and coordinates."""
    image_height, image_width, _ = image.shape

    for idx, landmark in enumerate(hand_landmarks):
        # Convert normalized coordinates to pixel coordinates
        px = int(landmark.x * image_width)
        py = int(landmark.y * image_height)

        # Get landmark name
        landmark_name = LANDMARK_NAMES.get(idx, f"Landmark {idx}")

        # Draw circle at landmark position
        cv.circle(image, (px, py), 5, (0, 255, 0), -1)

        # Draw landmark name
        cv.putText(
            image,
            landmark_name,
            (px + 8, py - 8),
            cv.FONT_HERSHEY_SIMPLEX,
            0.4,
            (0, 255, 0),
            1,
        )

        # Draw coordinates
        coord_text = f"({landmark.x:.3f}, {landmark.y:.3f})"
        cv.putText(
            image,
            coord_text,
            (px + 8, py + 15),
            cv.FONT_HERSHEY_SIMPLEX,
            0.35,
            (255, 255, 0),
            1,
        )


def draw_landmark_connections(image, hand_landmarks):
    """Draw connections between hand landmarks."""
    # Define finger connections (MCP -> PIP -> DIP -> TIP)
    thumb_connections = [(1, 2), (2, 3), (3, 4)]
    index_connections = [(5, 6), (6, 7), (7, 8)]
    middle_connections = [(9, 10), (10, 11), (11, 12)]
    ring_connections = [(13, 14), (14, 15), (15, 16)]
    pinky_connections = [(17, 18), (18, 19), (19, 20)]

    # Wrist connections to MCPs
    wrist_connections = [(0, 1), (0, 5), (0, 9), (0, 13), (0, 17)]

    all_connections = [
        thumb_connections,
        index_connections,
        middle_connections,
        ring_connections,
        pinky_connections,
        wrist_connections,
    ]

    image_height, image_width, _ = image.shape

    # Draw connections
    for connections in all_connections:
        for start_idx, end_idx in connections:
            start_landmark = hand_landmarks[start_idx]
            end_landmark = hand_landmarks[end_idx]

            start_px = int(start_landmark.x * image_width)
            start_py = int(start_landmark.y * image_height)
            end_px = int(end_landmark.x * image_width)
            end_py = int(end_landmark.y * image_height)

            cv.line(image, (start_px, start_py), (end_px, end_py), (0, 255, 0), 2)


def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--model",
        help="Path to the hand landmarker model file.",
        required=False,
        default="hand_landmarker.task",
    )
    parser.add_argument(
        "--cameraId",
        help="Id of camera (default: 0 for built-in webcam).",
        required=False,
        type=int,
        default=0,
    )
    parser.add_argument(
        "--frameWidth",
        help="Width of frame to capture from camera.",
        required=False,
        type=int,
        default=1280,
    )
    parser.add_argument(
        "--frameHeight",
        help="Height of frame to capture from camera.",
        required=False,
        type=int,
        default=720,
    )
    args = parser.parse_args()

    print("Loading HandLandmarker...")
    # Initialize the hand landmarker model
    try:
        landmarker = create_hand_landmarker(args.model, num_hands=2)
        print(f"HandLandmarker loaded successfully from: {args.model}")
    except Exception as e:
        print(f"Error loading HandLandmarker: {e}")
        print(
            "Please ensure the hand_landmarker.task model file exists in the current directory."
        )
        print(
            "Download it from: https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
        )
        return

    print("Initializing webcam...")
    # Initialize webcam
    cap = cv.VideoCapture(args.cameraId)

    if not cap.isOpened():
        print(f"Error: Could not open webcam (ID: {args.cameraId})")
        landmarker.close()
        return

    # Set webcam resolution
    cap.set(cv.CAP_PROP_FRAME_WIDTH, args.frameWidth)
    cap.set(cv.CAP_PROP_FRAME_HEIGHT, args.frameHeight)

    print("\nHand Tracking (MediaPipe Tasks API)")
    print("=" * 50)
    print("Displaying:")
    print("  - Green circles: Hand landmarks")
    print("  - Green lines: Landmark connections")
    print("  - Text: Landmark name and coordinates")
    print("  - Top info: Hand count and FPS")
    print("\nControls:")
    print("  - Press 'q' or ESC to quit")
    print("  - Show your hand to the camera")
    print("=" * 50)

    # FPS calculation variables
    frame_counter = 0
    fps = 0
    start_time = time.time()
    fps_avg_frame_count = 10

    while True:
        # Capture frame-by-frame
        success, image = cap.read()

        if not success:
            print("Ignoring empty camera frame.")
            continue

        frame_counter += 1

        # Flip image horizontally for selfie-view
        image = cv.flip(image, 1)

        # Convert BGR to RGB as required by MediaPipe
        rgb_image = cv.cvtColor(image, cv.COLOR_BGR2RGB)

        # Convert to MediaPipe Image format
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_image)

        # Detect hand landmarks
        detection_result = landmarker.detect(mp_image)

        # Calculate FPS
        if frame_counter % fps_avg_frame_count == 0:
            end_time = time.time()
            fps = fps_avg_frame_count / (end_time - start_time)
            start_time = time.time()

        # Process detection results
        hand_count = len(detection_result.hand_landmarks)

        if hand_count > 0:
            # Draw landmarks for each detected hand
            for idx, hand_landmarks in enumerate(detection_result.hand_landmarks):
                # Draw connections first (so they appear behind landmarks)
                draw_landmark_connections(image, hand_landmarks)

                # Draw landmarks with names
                draw_landmarks_with_names(image, hand_landmarks)

                # Display handedness (left/right)
                if detection_result.handedness and idx < len(
                    detection_result.handedness
                ):
                    handedness = detection_result.handedness[idx][0]
                    cv.putText(
                        image,
                        f"Hand {idx+1}: {handedness.category_name} ({handedness.score:.2f})",
                        (10, 30 + idx * 30),
                        cv.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (255, 0, 0),
                        2,
                    )
        else:
            cv.putText(
                image,
                "No hands detected",
                (10, 30),
                cv.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 255),
                2,
            )

        # Display FPS
        fps_text = f"FPS: {fps:.1f}"
        cv.putText(
            image,
            fps_text,
            (image.shape[1] - 100, 30),
            cv.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
        )

        # Display instructions
        cv.putText(
            image,
            "Press 'q' or ESC to quit",
            (10, image.shape[0] - 20),
            cv.FONT_HERSHEY_SIMPLEX,
            0.5,
            (200, 200, 200),
            1,
        )

        # Display the resulting frame
        cv.imshow("Hand Tracking", image)

        # Check for quit command
        key = cv.waitKey(5) & 0xFF
        if key == 27 or key == ord("q"):
            break

    # Release resources
    landmarker.close()
    cap.release()
    cv.destroyAllWindows()
    print("\nHand Tracking stopped.")


if __name__ == "__main__":
    main()
