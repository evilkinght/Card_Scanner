import re
import tkinter as tk
from tkinter import messagebox, filedialog
import cv2
from PIL import Image, ImageTk
from ultralytics import YOLO
import csv
import boto3

# Globals
scannedCards = {}
WHITELISTS = {
    'ID': set('ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'),
    'Name': set('ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz')
}

yolo_model = YOLO('yolo11z2.pt')
allow_prediction = False

CATEGORY_MAP = {
    'Common Filled': 'Common', 'Common Empty': 'Common', 'Rare Filled': 'Rare', 'Rare Empty': 'Rare',
    'Majestic Filled': 'Majestic', 'Majestic Empty': 'Majestic', 'Legendary Filled': 'Legendary',
    'Legendary Empty': 'Legendary', 'Fable Filled': 'Fable', 'Fable Empty': 'Fable', 'Super Filled': 'Super',
    'Super Empty': 'Super', 'Promo': 'Promo', 'Marvel': 'Marvel', 'Token Empty': 'Token', 'Token Filled': 'Token'
}

# AWS Configuration
AWS_REGION = 'us-east-1'  # Update with your region
textract = boto3.client('textract', region_name=AWS_REGION)

# Updating the camera frame in the UI
def update_frame():
    global allow_prediction
    ret, frame = cap.read()

    if ret:
        display_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = ImageTk.PhotoImage(Image.fromarray(display_frame))
        cameraLabel.imgtk = img
        cameraLabel.configure(image=img)

        if allow_prediction:
            process_card_detection(frame.copy())
            allow_prediction = False

    cameraLabel.after(10, update_frame)

# Enable manual entry of card quantity
def update_amount(unique_key, value):
    try:
        scannedCards[unique_key]["amount"] = int(value)
    except ValueError:
        messagebox.showerror("Invalid Input", "Amount must be a number!")

# Function to update the list in the GUI
def update_list():
    for widget in listFrame.winfo_children():
        widget.destroy()

    for unique_key, card in scannedCards.items():
        rowFrame = tk.Frame(listFrame)
        rowFrame.pack(fill="x", pady=5)

        # Name and ID with foil type
        text = f"{card['name']}\n{card['id']}"
        nameLabel = tk.Label(rowFrame, text=text, anchor="w", justify="left")
        nameLabel.pack(side="left", padx=5, fill="x", expand=True)

        # Foil type
        foilLabel = tk.Label(rowFrame, text=card['foil'], width=4)
        foilLabel.pack(side="left", padx=5)

        # Amount entry
        amountVar = tk.StringVar(value=str(card['amount']))
        amountEntry = tk.Entry(rowFrame, textvariable=amountVar, width=6)
        amountEntry.pack(side="right", padx=5)

        amountVar.trace_add("write", lambda *args, uk=unique_key, var=amountVar:
        update_amount(uk, var.get()))

# adds card to list
def add_card(name, cardId, amount, foil, detected_zones):
    unique_key = f"{cardId}-{foil}"

    if unique_key not in scannedCards:
        scannedCards[unique_key] = {
            "name": name,
            "id": cardId,
            "amount": amount,
            "foil": foil,
            "zones": detected_zones  # Store detection data
        }
    else:
        scannedCards[unique_key]["amount"] += amount
    update_list()

# YOLO-based detection
def detect_card_zones(card_image):
    results = yolo_model(card_image)
    category_dict = {}

    for result in results[0].boxes.data:
        x1, y1, x2, y2, confidence, cls = result.tolist()
        original_label = yolo_model.names[int(cls)]
        base_category = CATEGORY_MAP.get(original_label, original_label)

        if base_category not in category_dict or confidence > category_dict[base_category]["confidence"]:
            category_dict[base_category] = {
                "label": original_label,
                "confidence": confidence,
                "box": (int(x1), int(y1), int(x2), int(y2))
            }

    return list(category_dict.values())

def extract_text_with_textract(image):
    try:
        _, img_encoded = cv2.imencode('.png', image)
        response = textract.detect_document_text(
            Document={'Bytes': img_encoded.tobytes()}
        )
        return ' '.join(block['Text'] for block in response['Blocks'] if block['BlockType'] == 'LINE')
    except Exception as e:
        print(f"Textract error: {e}")
        return ""


def extract_text_from_zone(card_image, box, label):
    x1, y1, x2, y2 = box
    cropped = card_image[y1:y2, x1:x2]

    raw_text = extract_text_with_textract(cropped)
    filtered = ''.join([c for c in raw_text.strip() if c in WHITELISTS[label]])

    return validate_id(filtered) if label == "ID" else format_name(filtered)

def format_name(text):
    cleaned = re.sub(r'(?<!^)(?<! )([A-Z])', r' \1', text)
    return ' '.join(cleaned.split()).title()


def validate_id(text):
    if len(text) != 6 or not text[:3].isalpha():
        return "Invalid"

    substitutions = {'I': '1', 'O': '0', 'S': '5', 'Z': '2'}
    last_part = ''.join([substitutions.get(c, c) for c in text[3:]])
    return f"{text[:3]}{last_part}" if last_part.isdigit() else "Invalid"


def process_card_detection(frame):
    detected_zones = detect_card_zones(frame)
    card_data = {"Name": None, "ID": None}

    for zone in detected_zones:
        if zone["label"] == "Name":
            card_data["Name"] = extract_text_from_zone(frame, zone["box"], "Name")
        elif zone["label"] == "ID":
            card_data["ID"] = extract_text_from_zone(frame, zone["box"], "ID")

    if card_data["ID"] != "Invalid" and card_data["Name"]:
        add_card(card_data["Name"], card_data["ID"], 1, foil_var.get(), detected_zones)

def draw_boxes(frame, zones):
    for zone in zones:
        x1, y1, x2, y2 = zone["box"]
        label = zone["label"]
        confidence = zone["confidence"]

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 1)

        # Add label
        cv2.putText(frame, f"{label} {confidence:.2f}", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)


def export_to_binder_format():
    try:
        with open('Fab_IDs.csv', 'r') as fab_file:
            fab_reader = csv.DictReader(fab_file)
            fab_data = {row['SKU']: row['Binder Id'] for row in fab_reader}
    except FileNotFoundError:
        messagebox.showerror("Error", "Fab_IDs.csv file not found!")
        return

    output_data = []
    for unique_key, card in scannedCards.items():
        # Get stored detection data from when card was scanned
        detected_zones = card.get('zones', [])

        # Extract components
        card_id = card['id']
        foil = card['foil']
        rarity = next((z['label'] for z in detected_zones if "Filled" in z['label']), "")

        # Build SKU
        if any(exp in card_id for exp in ['WTR', 'ARC']) and "Filled" in rarity:
            foil_code = 'NO' if foil == 'NF' else foil
            sku = f"U-{card_id}-EN-U{foil_code}-1"
        elif any(exp in card_id for exp in ['WTR', 'ARC']) and "Empty" in rarity:
            foil_code = 'NO' if foil == 'NF' else foil
            sku = f"{card_id}-EN-1E{foil_code}-1"
        elif any(exp in card_id for exp in ['CRU','MON', 'ELE']) and "Empty" in rarity:
            foil_code = 'NO' if foil == 'NF' else foil
            sku = f"U-{card_id}-EN-U{foil_code}-1"
        elif any(exp in card_id for exp in ['CRU','MON', 'ELE']) and "Filled" in rarity:
            foil_code = 'NO' if foil == 'NF' else foil
            sku = f"{card_id}-EN-1E{foil_code}-1"
        else:
            foil_code = 'REG' if foil == 'NF' else foil
            sku = f"{card_id}-EN-{foil_code}-1"

        # Get Binder Id
        binder_id = fab_data.get(sku, 'NOT FOUND')

        # Build Variant Title
        base_title = "Near Mint"
        if any(exp in card_id for exp in ['WTR', 'ARC']) and "Filled" in rarity:
            base_title += " Unlimited"
        elif any(exp in card_id for exp in ['WTR', 'ARC']) and "Empty" in rarity:
            base_title += " 1st Edition"
        elif any(exp in card_id for exp in ['CRU', 'MON', 'ELE']) and "Filled" in rarity:
            base_title += " 1st Edition"
        elif any(exp in card_id for exp in ['CRU', 'MON', 'ELE']) and "Empty" in rarity:
            base_title += " Unlimited"

        foil_map = {
            'NF': '',
            'RF': ' Rainbow Foil',
            'CF': ' Cold Foil',
            'GF': ' Gold Foil'
        }
        variant_title = base_title + foil_map[foil]

        output_data.append({
            'Game Type': 'fleshAndBlood',
            'Binder Id': binder_id,
            'Variant Title': variant_title,
            'Quantity': card['amount']
        })

    # Save to new CSV
    file_path = filedialog.asksaveasfilename(
        defaultextension=".csv",
        filetypes=[("CSV files", "*.csv")],
        title="Save Binder File"
    )

    if not file_path:
        return

    try:
        with open(file_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=['Game Type', 'Binder Id', 'Variant Title', 'Quantity'])
            writer.writeheader()
            writer.writerows(output_data)
        messagebox.showinfo("Success", "Binder file created successfully!")
    except Exception as e:
        messagebox.showerror("Error", f"Failed to save file: {str(e)}")

# Function to enable predictions again when Enter key is pressed
def reset_prediction(event):
    global allow_prediction
    allow_prediction = True

# Initialize the main tkinter window
window = tk.Tk()
window.title("Card Scanning App")
window.geometry("3840x2160")
window.state("zoomed")

# Camera Frame
cameraFrame = tk.Frame(window, bg="black", width=window.winfo_screenwidth() // 3)
cameraFrame.pack(side="left", fill="both", expand=False)
cameraLabel = tk.Label(cameraFrame, bg="black")
cameraLabel.pack(expand=True)

# Foil List
foilFrame = tk.LabelFrame(cameraFrame, text="Foil Type", bg="black", fg="white")
foilFrame.pack(side="bottom", fill="x", padx=10, pady=10)

foil_var = tk.StringVar(value="NF")
foil_options = [
    ("Non-Foil", "NF"),
    ("Rainbow-Foil", "RF"),
    ("Cold-Foil", "CF"),
    ("Gold-Foil", "GF")
]

for text, value in foil_options:
    rb = tk.Radiobutton(foilFrame, text=text, variable=foil_var,
                       value=value, bg="black", fg="white",
                       selectcolor="black", activebackground="black",
                       activeforeground="white")
    rb.pack(side="left", padx=5, pady=2)

# List
listFrame = tk.Frame(window, bg="white")
listFrame.pack(side="right", fill="both", expand=True, padx =10)

def create_submit_button():
    submit_btn = tk.Button(
        window,
        text="Submit",
        command=export_to_binder_format,
        bg="#4CAF50",  # Green color
        fg="white",
        font=('Arial', 12, 'bold'),
        padx=20,
        pady=10
    )
    submit_btn.place(relx=1.0, rely=1.0, anchor='se')  # Bottom right corner

create_submit_button()
window.bind('<Return>', reset_prediction)

# Start video capture from the default camera (index 0)
cap = cv2.VideoCapture(0)

# Call the function to update the camera frame
update_frame()

# Run the tkinter main loop
window.mainloop()

# Release the camera when the window is closed
cap.release()
cv2.destroyAllWindows()
