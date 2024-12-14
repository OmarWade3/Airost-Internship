import asyncio
import cv2
import base64
import numpy as np
import httpx
from tkinter import Tk, Label, Button, StringVar, Frame, Listbox, Scrollbar, VERTICAL, RIGHT, Y, W, BOTH
from PIL import Image, ImageTk
import threading
from collections import Counter
import sqlite3

# Configuration for Roboflow API
ROBOFLOW_API_KEY = "YOUR_API_KEY"
ROBOFLOW_MODEL = "airost-internship-project/4"
ROBOFLOW_SIZE = 640

upload_url = f"https://detect.roboflow.com/{ROBOFLOW_MODEL}?api_key={ROBOFLOW_API_KEY}&format=json"
CONFIDENCE_THRESHOLD = 0.85
IOU_THRESHOLD = 0.5

# Global variables
tracked_objects = {}
object_id = 0
is_counting = False
temp_list = []
latest_predictions = []

# Video capture setup
video = cv2.VideoCapture(0)
video.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
video.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
video.set(cv2.CAP_PROP_FPS, 30)

# Database setup
DB_FILE = "inventory_management.db"

def setup_database():
    """Create inventory table if it doesn't exist."""
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS inventory (
            item_name TEXT PRIMARY KEY,
            quantity INTEGER
        )
    """)
    conn.commit()
    conn.close()

def load_inventory():
    """Load inventory data from the database."""
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute("SELECT item_name, quantity FROM inventory")
    rows = cursor.fetchall()
    conn.close()
    return {row[0]: row[1] for row in rows}

def save_inventory(item_name, quantity):
    """Save or update an inventory item in the database."""
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO inventory (item_name, quantity)
        VALUES (?, ?)
        ON CONFLICT(item_name) DO UPDATE SET quantity=excluded.quantity
    """, (item_name, quantity))
    conn.commit()
    conn.close()

def update_inventory(counted_items, action):
    """Update inventory in the database based on the action."""
    for item, count in counted_items.items():
        if action == "check-in":
            inventory[item] = inventory.get(item, 0) + count
        elif action == "check-out" and inventory.get(item, 0) >= count:
            inventory[item] -= count
        save_inventory(item, inventory[item])

# Initialize database and inventory
setup_database()
inventory = load_inventory()

def calculate_iou(box1, box2):
    """Calculate Intersection Over Union (IOU) for two bounding boxes."""
    x1, y1, x2, y2 = box1
    x1_p, y1_p, x2_p, y2_p = box2

    inter_x1 = max(x1, x1_p)
    inter_y1 = max(y1, y1_p)
    inter_x2 = min(x2, x2_p)
    inter_y2 = min(y2, y2_p)

    inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)
    box1_area = (x2 - x1) * (y2 - y1)
    box2_area = (x2_p - x1_p) * (y2_p - y1_p)

    union_area = box1_area + box2_area - inter_area
    return inter_area / union_area if union_area > 0 else 0

def track_objects(predictions):
    """Track objects across frames and avoid counting duplicates."""
    global tracked_objects, object_id
    updated_tracked_objects = {}

    for pred in predictions:
        label = pred["class"]
        confidence = pred["confidence"]
        x, y, w, h = pred["x"], pred["y"], pred["width"] / 2, pred["height"] / 2
        new_box = (x - w, y - h, x + w, y + h)

        matched = False
        for obj_id, (old_label, old_box) in tracked_objects.items():
            if label == old_label and calculate_iou(new_box, old_box) > IOU_THRESHOLD:
                updated_tracked_objects[obj_id] = (label, new_box)
                matched = True
                break

        if not matched:
            updated_tracked_objects[object_id] = (label, new_box)
            temp_list.append(label)  # Count this new object
            object_id += 1

    tracked_objects = updated_tracked_objects

async def infer():
    ret, img = video.read()
    if not ret:
        return None, []

    # Resize image
    height, width, _ = img.shape
    scale = ROBOFLOW_SIZE / max(height, width)
    img_resized = cv2.resize(img, (round(scale * width), round(scale * height)))

    # Encode frame as a base64 string
    retval, buffer = cv2.imencode('.jpg', img_resized, [cv2.IMWRITE_JPEG_QUALITY, 50])
    img_str = base64.b64encode(buffer)

    async with httpx.AsyncClient() as requests:
        try:
            response = await requests.post(upload_url, data=img_str, headers={
                "Content-Type": "application/x-www-form-urlencoded"
            }, timeout=30.0)
            predictions = response.json().get("predictions", [])
        except httpx.RequestError as exc:
            print(f"Request error: {exc}")
            predictions = []

    return img, predictions

def detection_thread():
    global latest_predictions
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    while True:
        if is_counting:
            img, predictions = loop.run_until_complete(infer())
            latest_predictions = predictions if predictions else []

            # Track and persist detected items
            track_objects(latest_predictions)


def draw_predictions(frame, predictions):
    """Draw bounding boxes for tracked predictions."""
    for pred in predictions:
        x, y, w, h = pred["x"], pred["y"], pred["width"] / 2, pred["height"] / 2
        confidence = pred["confidence"]
        if confidence >= CONFIDENCE_THRESHOLD:
            cv2.rectangle(frame, (int(x - w), int(y - h)), (int(x + w), int(y + h)), (0, 255, 0), 2)
            label = pred["class"]
            cv2.putText(frame, f"{label} ({confidence:.1f}%)", (int(x - w), int(y - h - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    return frame

# UI Design
root = Tk()
root.title("AI Inventory Management System")
root.geometry("800x600")
root.configure(bg="#2C2C2C")

# Frames
top_frame = Frame(root, bg="#3E3E3E", relief="raised", bd=2)
top_frame.pack(side="top", fill="x", pady=5)

video_frame = Frame(root, bg="#2C2C2C")
video_frame.pack(side="left", fill="both", expand=True, padx=10, pady=10)

side_frame = Frame(root, bg="#3E3E3E", relief="raised", bd=2)
side_frame.pack(side="right", fill="y", padx=10)

# Video Display
video_label = Label(video_frame, bg="#000000", width=640, height=480)
video_label.pack()

detected_items = StringVar()
detected_items.set("No item detected")
counted_items_label = Label(video_frame, textvariable=detected_items, font=("Arial", 12), bg="#2C2C2C", fg="#FFFFFF", justify="left")
counted_items_label.pack(pady=10)



# Buttons and Status
status_label = Label(top_frame, text="Status: No Detection Yet", font=("Arial", 14), bg="#3E3E3E", fg="#FFFFFF")
status_label.pack(pady=5)

button_frame = Frame(top_frame, bg="#3E3E3E")
button_frame.pack()

start_button = Button(button_frame, text="Start Detection", command=lambda: start_detection(), bg="#28A745", fg="#FFFFFF", font=("Arial", 12), width=12)
start_button.grid(row=0, column=0, padx=5, pady=10)

checkin_button = Button(button_frame, text="Check-In", command=lambda: stop_detection("check-in"), bg="#17A2B8", fg="#FFFFFF", font=("Arial", 12), width=12)
checkin_button.grid(row=0, column=1, padx=5, pady=10)

checkout_button = Button(button_frame, text="Check-Out", command=lambda: stop_detection("check-out"), bg="#FFC107", fg="#FFFFFF", font=("Arial", 12), width=12)
checkout_button.grid(row=0, column=2, padx=5, pady=10)

# Inventory List
inventory_label = Label(side_frame, text="Inventory List", font=("Arial", 14), bg="#3E3E3E", fg="#FFFFFF")
inventory_label.pack(pady=5)

scrollbar = Scrollbar(side_frame, orient=VERTICAL)
inventory_list = Listbox(side_frame, yscrollcommand=scrollbar.set, font=("Arial", 12), bg="#2C2C2C", fg="#FFFFFF", width=30, height=25)
scrollbar.config(command=inventory_list.yview)
scrollbar.pack(side=RIGHT, fill=Y)
inventory_list.pack(side="left", fill=BOTH, expand=True, padx=5, pady=5)

def update_inventory(counted_items, action):
    """Update inventory in the database based on the action."""
    for item, count in counted_items.items():
        if action == "check-in":
            inventory[item] = inventory.get(item, 0) + count
            save_inventory(item, inventory[item])
        elif action == "check-out":
            if inventory.get(item, 0) >= count:
                inventory[item] -= count
                save_inventory(item, inventory[item])
            else:
                print(f"Warning: Not enough {item} in inventory to check out.")

def update_inventory_display():
    """Update the inventory list display."""
    inventory_list.delete(0, 'end')  # Clear the current list
    for item, quantity in inventory.items():
        inventory_list.insert('end', f"{item}: {quantity}")  # Add updated inventory to the list

update_inventory_display()

# Video Stream and Logic
def video_stream():
    global latest_predictions

    ret, frame = video.read()
    if ret:
        # Draw predictions on the frame
        high_confidence_preds = [
            pred for pred in latest_predictions if pred["confidence"] >= CONFIDENCE_THRESHOLD
        ]
        frame = draw_predictions(frame, high_confidence_preds)

        counted_items = Counter(temp_list)
        detected_items.set(", ".join([f"{item}: {count}" for item, count in counted_items.items()]))

        # Convert frame to RGB and display in Tkinter
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = ImageTk.PhotoImage(image=Image.fromarray(frame))
        video_label.imgtk = img
        video_label.configure(image=img)

    video_label.after(25, video_stream)


def start_detection():
    global is_counting
    is_counting = True
    status_label.config(text="Status: Detection Started", fg="#28A745")

def stop_detection(action):
    global is_counting, temp_list

    is_counting = False
    status_label.config(text=f"Status: {action.capitalize()} Completed", fg="#FFC107")

    if temp_list:
        # Count detected items
        counted_items = Counter(temp_list)

        # Update inventory based on action
        update_inventory(counted_items, action)

        # Display the counted items in a label
        counted_items_text = "\n".join([f"{item}: {count}" for item, count in counted_items.items()])
        detected_items.set(f"Counted Items:\n{counted_items_text}")

        # Clear temp_list after processing
        temp_list.clear()

        # Update inventory display in UI
        update_inventory_display()
    else:
        status_label.config(text="No items detected to process.", fg="#FF0000")





# Start detection thread
thread = threading.Thread(target=detection_thread, daemon=True)
thread.start()

# Start video and app
video_stream()
root.mainloop()

# Release resources
video.release()
cv2.destroyAllWindows()
