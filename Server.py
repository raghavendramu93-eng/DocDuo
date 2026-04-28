from fastapi import FastAPI, UploadFile, File, Body, Form
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import numpy as np
import cv2
import onnxruntime as ort
import json
import os
import uuid
import datetime
import pytesseract
import re
from PIL import Image
import io
from datetime import datetime, timedelta
from google import genai
import time

# --- LOCAL OCR PATH ---
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

app = FastAPI(title="JNNCE Virtual Hospital API", version="9.0")

app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

client = genai.Client(api_key="AIzaSyBACYGEl7lTsmPD0il2zX8QTIC5VpoDjeo")

print("Loading Local Edge Models...")
try:
    skin_model = ort.InferenceSession("skincancer_efficientnet.onnx")
    brain_model = ort.InferenceSession("braintumour_efficientnet.onnx")
    pneumonia_model = ort.InferenceSession("pneumonia_resnet18.onnx")
    print("All Vision Models Loaded Successfully.")
except Exception as e: print(f"⚠️ Warning: Model missing or failed to load. {e}")

def preprocess_image(image_bytes: bytes) -> np.ndarray:
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (224, 224))
    img = img.astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    img = (img - mean) / std
    img = np.transpose(img, (2, 0, 1))
    img = np.expand_dims(img, axis=0)
    return img

def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=1, keepdims=True)

# ==========================================
# DECOUPLED DATABASE ARCHITECTURE
# ==========================================
DB_FILE = "hospital_database.json"
RECORDS_FILE = "medical_records.json" # NEW: Separate JSON for heavy storage
HOSPITAL_ADMIN_SECRET = "admin"

def load_db():
    if os.path.exists(DB_FILE):
        with open(DB_FILE, "r") as f: return json.load(f)
    return {"patients": {}, "doctors": {}, "appointments": [], "outbreaks": {}}

def save_db(data):
    with open(DB_FILE, "w") as f: json.dump(data, f, indent=4)

def load_records():
    if os.path.exists(RECORDS_FILE):
        with open(RECORDS_FILE, "r") as f: return json.load(f)
    return {}

def save_records(data):
    with open(RECORDS_FILE, "w") as f: json.dump(data, f, indent=4)

# ==========================================
# ENDPOINTS
# ==========================================
@app.post("/auth/register-patient")
async def register_patient(name: str, age: int, phone: str, location: str = "Unknown"):
    db = load_db()
    patient_id = f"PID-{str(uuid.uuid4())[:6].upper()}"
    db["patients"][patient_id] = { "name": name, "age": age, "phone": phone, "location": location.title(), "triage_status": "Routine" }
    save_db(db)
    
    # Initialize their empty medical file in the decoupled DB
    records_db = load_records()
    records_db[patient_id] = []
    save_records(records_db)
    
    return {"status": "Success", "patient_id": patient_id}

@app.post("/auth/register-doctor")
async def register_doctor(name: str, specialty: str, admin_password: str):
    if admin_password != HOSPITAL_ADMIN_SECRET: return {"status": "Denied"}
    db = load_db()
    doc_id = f"DOC-{str(uuid.uuid4())[:6].upper()}"
    db["doctors"][doc_id] = {"name": f"Dr. {name}", "specialty": specialty, "patients_queue": []}
    save_db(db)
    return {"status": "Success", "doctor_id": doc_id}

@app.post("/auth/login")
async def login(user_id: str):
    db = load_db() 
    
    # 1. ADMIN LOGIN & MAP EXPIRY LOGIC
    if user_id == "ADMIN-123": 
        active_outbreaks = {}
        seven_days_ago = datetime.now().date() - timedelta(days=7)
        
        for loc, dates in db.get("outbreaks", {}).items():
            if isinstance(dates, int): dates = [str(datetime.now().date())] * dates
            valid_dates = [d for d in dates if datetime.strptime(d, "%Y-%m-%d").date() >= seven_days_ago]
            db["outbreaks"][loc] = valid_dates 
            
            if valid_dates:
                active_outbreaks[loc] = len(valid_dates) 
                
        save_db(db)
        return {"status": "Success", "role": "admin", "data": active_outbreaks}
    
    # 2. PATIENT LOGIN
    elif user_id in db.get("patients", {}):
        return {"status": "Success", "role": "patient", "data": db["patients"][user_id]}
        
    # 3. DOCTOR LOGIN (FIXED: Hydrating the Queue)
    elif user_id in db.get("doctors", {}):
        doc_data = db["doctors"][user_id]
        
        # Enrich the queue with actual patient details instead of just raw IDs
        enriched_queue = []
        for pid in doc_data.get("patients_queue", []):
            if pid in db.get("patients", {}):
                patient_info = db["patients"][pid]
                enriched_queue.append({
                    "id": pid,
                    "name": patient_info.get("name", "Unknown Patient"),
                    "triage": patient_info.get("triage_status", "Routine")
                })
        
        # Create a copy so we send the rich data to the frontend without messing up our DB
        response_data = doc_data.copy()
        response_data["patients_queue"] = enriched_queue
        
        return {"status": "Success", "role": "doctor", "data": response_data}
        
    # 4. INVALID ID
    return {"status": "Error", "message": "ID not found."}

@app.post("/emergency/sos")
async def trigger_sos(patient_id: str, latitude: float, longitude: float):
    db = load_db()
    if patient_id in db["patients"]: db["patients"][patient_id]["triage_status"] = "Critical"; save_db(db)
    return {"status": "Dispatched", "message": f"Ambulance routed. Triage escalated."}

@app.get("/patient/{patient_id}")
async def get_patient_data(patient_id: str):
    db = load_db()
    records_db = load_records()
    if patient_id not in db["patients"]: return {"status": "Error"}
    
    patient_data = db["patients"][patient_id]
    active_appt = next(({"doctor": db["doctors"][a["doctor_id"]]["name"], "department": db["doctors"][a["doctor_id"]]["specialty"]} 
                        for a in db["appointments"] if a["patient_id"] == patient_id and a.get("status") == "Scheduled" and a["doctor_id"] in db["doctors"]), None)
    
    patient_history = records_db.get(patient_id, [])
    
    return {"status": "Success", "data": patient_data, "appointment": active_appt, "history": patient_history}

@app.get("/doctors/{department}")
async def get_doctors_by_dept(department: str):
    db = load_db()
    return {"status": "Success", "doctors": [{"id": d, "name": v["name"]} for d, v in db["doctors"].items() if v["specialty"].lower() == department.lower()]}

@app.post("/appointments/book")
async def book_appointment(patient_id: str, doctor_id: str):
    db = load_db()
    # FIXED: Changed datetime.datetime.now() to datetime.now()
    db["appointments"].append({"patient_id": patient_id, "doctor_id": doctor_id, "status": "Scheduled", "date": str(datetime.now().date())})
    if patient_id not in db["doctors"][doctor_id]["patients_queue"]: 
        db["doctors"][doctor_id]["patients_queue"].append(patient_id)
    save_db(db) 
    return {"status": "Success"}

# --- NEW: PATIENT CLOSES THE APPOINTMENT ---
@app.post("/patient/close-appointment")
async def close_appointment(patient_id: str):
    db = load_db()
    for appt in db["appointments"]:
        if appt["patient_id"] == patient_id and appt.get("status") == "Scheduled":
            appt["status"] = "Closed"
    
    if patient_id in db["patients"]: db["patients"][patient_id]["triage_status"] = "Routine"
    save_db(db)
    return {"status": "Success"}

@app.post("/doctor/check-ddi")
async def check_ddi(payload: dict = Body(...)):
    patient_id = payload.get("patient_id")
    new_meds = payload.get("new_meds")
    
    # 1. Properly pull the patient's PAST medications from the decoupled records DB
    records_db = load_records()
    history = records_db.get(patient_id, [])
    past_meds = "None" if not history else history[-1].get("medication", "None")
    
    print(f"--- DDI CHECK ---")
    print(f"Patient is taking: {past_meds}")
    print(f"Doctor prescribing: {new_meds}")
    
    # 2. Use the aggressive retry loop we built earlier!
    prompt = f"Patient is currently taking: {past_meds}. Doctor wants to prescribe: {new_meds}. Are there severe, life-threatening drug interactions? If safe, reply EXACTLY with the word 'SAFE'. If dangerous, reply with a 2-sentence severe warning."
    
    response_text = call_gemini_with_retry(prompt)
    
    # 3. Handle the response strictly
    if "SAFE" in response_text[:10].upper(): 
        return {"status": "SAFE"}
        
    return {"status": "DANGER", "alert": response_text}

@app.get("/doctor/trend/{patient_id}")
async def get_patient_trend(patient_id: str):
    try:
        records = load_records().get(patient_id, [])
        if len(records) < 2: 
            return {"trend": "Not enough historical data for trend analysis. Minimum 2 past visits required."}
        
        # Grab the last 3 visits
        history_text = "\n".join([f"Date: {l.get('date', 'Unknown')} - Data: {l.get('reports', 'No data')}" for l in records[-3:]])
        
        prompt = f"Analyze these consecutive patient reports:\n{history_text}\nProvide a 2-sentence predictive trend alert (e.g. 'WBC is rising consistently, monitor for infection'). Keep it strictly medical."
        
        response = client.models.generate_content(model='gemini-2.5-flash', contents=prompt)
        return {"trend": response.text}
    except Exception as e:
        return {"trend": f"⚠️ AI Trend Analysis temporarily unavailable: {str(e)}"}

# --- NEW: DECOUPLED JSON PAYLOAD SAVING (Allows Base64 Image Strings) ---
@app.post("/doctor/save-patient-record")
async def save_patient_record(payload: dict = Body(...)):
    patient_id = payload.get("patient_id")
    doctor_id = payload.get("doctor_id")
    
    db = load_db()
    records_db = load_records()
    
    doc_name = db["doctors"].get(doctor_id, {}).get("name", "Unknown Doctor")
    doc_dept = db["doctors"].get(doctor_id, {}).get("specialty", "General")
    
    new_record = {
        # FIXED: Changed datetime.datetime.now() to datetime.now()
        "date": str(datetime.now().date()),
        "doctor": doc_name,
        "department": doc_dept,
        "summary": payload.get("summary", ""),
        "medication": payload.get("medication", ""),
        "reports": payload.get("report_text", ""),
        "image": payload.get("image_base64", "")
    }
    
    if patient_id not in records_db: records_db[patient_id] = []
    records_db[patient_id].append(new_record)
    save_records(records_db)
    
    return {"status": "Success"}

@app.post("/predict/pneumonia")
async def predict_pneumonia(patient_id: str = Form("Unknown"), file: UploadFile = File(...)):
    CLASSES = ['Normal', 'Pneumonia']
    input_data = preprocess_image(await file.read())
    probs = softmax(pneumonia_model.run(None, {pneumonia_model.get_inputs()[0].name: input_data})[0])[0]
    diagnosis = CLASSES[np.argmax(probs)]
    if diagnosis == 'Pneumonia' and patient_id != "Unknown":
        db = load_db(); loc = db["patients"].get(patient_id, {}).get("location", "Unknown Area")
        # Replace the old outbreaks line with this:
        if loc != "Unknown Area": 
            if "outbreaks" not in db: db["outbreaks"] = {}
            if loc not in db["outbreaks"] or isinstance(db["outbreaks"][loc], int): db["outbreaks"][loc] = []
            db["outbreaks"][loc].append(str(datetime.now().date()))
            save_db(db)
    return {"diagnosis": diagnosis, "confidence": f"{float(np.max(probs)*100):.2f}%"}

@app.post("/predict/dermatology")
async def predict_skin(file: UploadFile = File(...)):
    input_data = preprocess_image(await file.read())
    probs = softmax(skin_model.run(None, {skin_model.get_inputs()[0].name: input_data})[0])[0]
    return {"diagnosis": ['Actinic keratosis', 'Basal cell carcinoma', 'Benign keratosis', 'Dermatofibroma', 'Melanoma', 'Nevus', 'Vascular lesion', 'Squamous cell carcinoma', 'Unknown'][np.argmax(probs)], "confidence": f"{float(np.max(probs)*100):.2f}%"}

@app.post("/predict/brain-mri")
async def predict_brain(file: UploadFile = File(...)):
    input_data = preprocess_image(await file.read())
    probs = softmax(brain_model.run(None, {brain_model.get_inputs()[0].name: input_data})[0])[0]
    return {"diagnosis": ['Glioma', 'Meningioma', 'No Tumor', 'Pituitary'][np.argmax(probs)], "confidence": f"{float(np.max(probs)*100):.2f}%"}

@app.post("/analyze/blood-report")
async def analyze_blood(file: UploadFile = File(...)):
    # 1. UPGRADED CLINICAL RANGES (Handles all 8 major parameters)
    CBC_REFERENCES = {
        "Hemoglobin": {"min": 12.0, "max": 17.2, "unit": "g/dL"},
        "RBC": {"min": 4.0, "max": 5.9, "unit": "mill/cumm"},
        "Hematocrit (PCV)": {"min": 36.0, "max": 50.0, "unit": "%"},
        "MCV": {"min": 81.0, "max": 101.0, "unit": "fL"},
        "MCH": {"min": 27.0, "max": 33.0, "unit": "pg"},
        "MCHC": {"min": 31.5, "max": 34.5, "unit": "g/dL"},
        "WBC": {"min": 4000.0, "max": 11000.0, "unit": "cells/cumm"},
        "Platelets": {"min": 150000.0, "max": 450000.0, "unit": "cells/cumm"}
    }

    try:
        image_bytes = await file.read()
        nparr = np.frombuffer(image_bytes, np.uint8)
        img_cv = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        # 2. ADVANCED OPENCV PIPELINE (For WhatsApp/Paper Photos)
        # Upscale the image by 2x to make tiny text readable
        img_cv = cv2.resize(img_cv, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
        gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
        
        # Adaptive Thresholding (Destroys shadows and isolates text perfectly)
        thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 21, 11)
        img_pil = Image.fromarray(thresh)

        # 3. TESSERACT WITH PSM 4 (Forces Tesseract to read in "Table Columns")
        raw_text = pytesseract.image_to_string(img_pil, config='--psm 4')

        print("\n--- LOCAL TESSERACT OCR OUTPUT ---")
        print(raw_text)
        print("----------------------------------\n")

        # 4. HIGHLY FORGIVING REGEX PATTERNS
        patterns = {
            "Hemoglobin": r"(?i)h[ae]moglobin[^\d\n]*(\d+\.\d+|\d+)",
            "RBC": r"(?i)(?:rbc count|rbc\b)[^\d\n]*(\d+\.\d+|\d+)",
            "Hematocrit (PCV)": r"(?i)(?:hematocrit|pcv)[^\d\n]*(\d+\.\d+|\d+)",
            "MCV": r"(?i)\bmcv\b[^\d\n]*(\d+\.\d+|\d+)",
            "MCH": r"(?i)\bmch\b[^\d\n]*(\d+\.\d+|\d+)",
            "MCHC": r"(?i)\bmchc\b[^\d\n]*(\d+\.\d+|\d+)",
            "WBC": r"(?i)(?:wbc|white blood cell|total leucocyte count)[^\d\n]*(\d+\.\d+|\d+)",
            "Platelets": r"(?i)(?:platelet count|platelets)[^\d\n]*(\d+\.\d+|\d+)"
        }

        analysis_results = {}
        for biomarker, pattern in patterns.items():
            match = re.search(pattern, raw_text)
            if match:
                val = float(match.group(1))
                
                # 5. UNIT NORMALIZATION 
                # (Some hospitals use "319" for platelets, others use "319000". This fixes it.)
                if biomarker == "Platelets" and val < 2000:
                    val = val * 1000  
                if biomarker == "WBC" and val < 100:
                    val = val * 1000  
                    
                analysis_results[biomarker] = val

        final_report = []
        for b, v in analysis_results.items():
            ref = CBC_REFERENCES.get(b)
            if not ref: continue
            status, flag = ("Low", "🔴") if v < ref["min"] else ("High", "🔴") if v > ref["max"] else ("Normal", "🟢")
            final_report.append({"Biomarker": b, "Value": f"{v} {ref['unit']}", "Status": status, "Flag": flag})

        if not final_report:
            return {"status": "Error", "message": "OCR failed. Check VS Code terminal to see what Tesseract read."}

        return {"status": "Success", "data": final_report}

    except Exception as e:
        return {"status": "Error", "message": f"Extraction crashed: {str(e)}"}


def call_gemini_with_retry(prompt: str, retries: int = 4):
    for attempt in range(retries):
        try:
            # I updated this to gemini-2.5-flash for you!
            response = client.models.generate_content(model='gemini-2.5-flash', contents=prompt)
            return response.text
        except Exception as e:
            # If we hit a 503 traffic jam, wait 2 seconds and hammer the API again
            if "503" in str(e) and attempt < retries - 1:
                print(f"Gemini API busy. Retrying in 2 seconds... (Attempt {attempt + 1})")
                time.sleep(2) 
                continue
            # If it fails 4 times in a row, then we finally show the error
            return f"⚠️ API Error: {str(e)}"

@app.post("/chat/medical-assistant")
async def medical_chat(user_query: str, role: str = "patient"):
    prompt = f"Clinical AI for {'Doctor' if role == 'doctor' else 'Patient'}. Query: {user_query}"
    return {"response": call_gemini_with_retry(prompt)}

@app.post("/doctor/synthesize")
async def synthesize_data(payload: dict = Body(...)):
    prompt = f"Synthesize:\nVISION: {payload.get('vision_data')}\nBLOOD: {payload.get('blood_data')}\nPROMPT: {payload.get('custom_prompt')}"
    return {"response": call_gemini_with_retry(prompt)}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
