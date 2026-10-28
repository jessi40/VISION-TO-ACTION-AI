from ultralytics import YOLO
from src.coordinate_mapping import map_coordinates
from src.action import decide_action
from src.robot_action import move_robot

print("=" * 50)
print("          VISION TO ACTION AI")
print("=" * 50)

# Load YOLO model
model = YOLO("yolo11n.pt")

# Input image
image_path = "test.jpg"

# Robot working area
robot_width = 100
robot_height = 100

print("\nAnalyzing image...")
print()

# Detect objects
results = model(image_path)

object_count = 0

for result in results:

    frame_height, frame_width = result.orig_shape

    for box in result.boxes:

        object_count += 1

        # Bounding box
        x1, y1, x2, y2 = box.xyxy[0]

        # Find object center
        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2

        # Get object name
        class_id = int(box.cls[0])
        object_name = model.names[class_id]

        # Confidence
        confidence = float(box.conf[0]) * 100

        # Camera coordinates → Robot coordinates
        robot_x, robot_y = map_coordinates(
            float(center_x),
            float(center_y),
            frame_width,
            frame_height,
            robot_width,
            robot_height
        )

        # Decide action
        action = decide_action(
            object_name,
            robot_x,
            robot_y
        )

        print("-" * 50)

        print("OBJECT DETECTED")
        print("Object      :", object_name)
        print("Confidence  :", round(confidence, 2), "%")

        print("\nCAMERA POSITION")
        print("X           :", round(float(center_x), 2))
        print("Y           :", round(float(center_y), 2))

        print("\nROBOT POSITION")
        print("X           :", round(robot_x, 2))
        print("Y           :", round(robot_y, 2))

        print("\nACTION DECISION")
        print("Action      :", action)

        print("\nROBOT SIMULATION")

        status = move_robot(robot_x, robot_y)

        print("Status      :", status)

        print("-" * 50)

print()
print("=" * 50)
print("        VISION TO ACTION COMPLETED")
print("=" * 50)
print("Objects detected:", object_count)
print("System status: SUCCESS")
print("=" * 50)