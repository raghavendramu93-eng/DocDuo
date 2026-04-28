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
from google import genai

pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

app = FastAPI(title="Virtual Hospital API", version="1.0")

app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

client = genai.Client(api_key="AIzaSyBVjh-wj5RwKuSxnz5gTws5kjyWfYhCIoM")

print("Loading Local Edge Models...")
try:
    skin_model = ort.InferenceSession("skincancer_efficientnet.onnx")
    brain_model = ort.InferenceSession("braintumour_efficientnet.onnx")
    pneumonia_model = ort.InferenceSession("pneumonia_resnet18.onnx")
    print(" All Vision Models Loaded Successfully.")
except Exception as e: print(f"Warning: Model missing or failed to load. {e}")

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


@app.post("/analyze/blood-report")
async def analyze_blood_report(file: UploadFile = File(...)):
    try:
        # 1. Read Image into OpenCV
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        # 2. Advanced OpenCV Preprocessing (fixes shadows on paper)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Adaptive thresholding handles uneven lighting from phone cameras
        thresh = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
        )

        # 3. Tesseract OCR Extraction
        # psm 6 assumes a single uniform block of text (good for tables/reports)
        custom_config = r'--oem 3 --psm 6'
        extracted_text = pytesseract.image_to_string(thresh, config=custom_config)

        # 4. Structure the Data via Gemini
        prompt = f"""
        You are a medical data parser. Extract the biomarkers, their values, and their status (High/Low/Normal) from this raw OCR text of a blood report.
        
        Return ONLY a strict JSON object in this exact format, with no markdown formatting or extra text:
        {{
            "data": [
                {{"Biomarker": "RBC", "Value": "5.0 mill/cumm", "Status": "Normal"}},
                {{"Biomarker": "MCV", "Value": "80.0 fL", "Status": "Low"}}
            ]
        }}
        
        RAW OCR TEXT:
        {extracted_text}
        """
        
        # Use your aggressive retry loop!
        response_text = call_gemini_with_retry(prompt)
        
        # Clean the JSON output just in case Gemini wraps it in ```json ... ```
        cleaned_json = response_text.replace("```json", "").replace("```", "").strip()
        
        return json.loads(cleaned_json)

    except Exception as e:
        print(f"OCR Error: {str(e)}")
        return {"message": "OCR Extraction Failed. Please ensure the image is clear and legible."}

@app.post("/auth/login")
async def login(user_id: str):
    db = load_db() 
    if user_id == "ADMIN-123": return {"status": "Success", "role": "admin", "data": db["outbreaks"]}
    elif user_id.startswith("PID-") and user_id in db["patients"]: return {"status": "Success", "role": "patient", "data": db["patients"][user_id]}
    elif user_id.startswith("DOC-") and user_id in db["doctors"]: 
        doc_data = db["doctors"][user_id].copy()
        enriched_queue = []
        for pid in doc_data["patients_queue"]:
            p_info = db["patients"].get(pid, {})
            
            # Check if patient closed their appointment
            triage_val = p_info.get("triage_status", "Routine")
            appt_status = "Scheduled"
            for a in db["appointments"]:
                if a["patient_id"] == pid and a["doctor_id"] == user_id:
                    appt_status = a.get("status", "Scheduled")
            
            if appt_status == "Closed": triage_val = "Closed"
            
            enriched_queue.append({"id": pid, "name": p_info.get("name", "Unknown"), "triage": triage_val})
            
        doc_data["patients_queue"] = enriched_queue
        return {"status": "Success", "role": "doctor", "data": doc_data}
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
    db["appointments"].append({"patient_id": patient_id, "doctor_id": doctor_id, "status": "Scheduled", "date": str(datetime.datetime.now().date())})
    if patient_id not in db["doctors"][doctor_id]["patients_queue"]: db["doctors"][doctor_id]["patients_queue"].append(patient_id)
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
    records_db = load_records()
    history = records_db.get(payload.get("patient_id"), [])
    past_meds = "None" if not history else history[-1].get("medication", "None")
    
    prompt = f"Patient takes: {past_meds}. Doctor prescribes: {payload.get('new_meds')}. Are there severe interactions? Reply EXACTLY 'SAFE' or a 2-sentence warning."
    response = client.models.generate_content(model='gemini-2.5-flash', contents=prompt)
    if "SAFE" in response.text[:10].upper(): return {"status": "SAFE"}
    return {"status": "DANGER", "alert": response.text}

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
        return {"trend": f"AI Trend Analysis temporarily unavailable: {str(e)}"}

@app.post("/doctor/save-patient-record")
async def save_patient_record(payload: dict = Body(...)):
    patient_id = payload.get("patient_id")
    doctor_id = payload.get("doctor_id")
    
    db = load_db()
    records_db = load_records()
    
    doc_name = db["doctors"].get(doctor_id, {}).get("name", "Unknown Doctor")
    doc_dept = db["doctors"].get(doctor_id, {}).get("specialty", "General")
    
    new_record = {
        "date": str(datetime.datetime.now().date()),
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
        if loc != "Unknown Area": db["outbreaks"][loc] = db["outbreaks"].get(loc, 0) + 1; save_db(db)
    return {"diagnosis": diagnosis, "confidence": f"{float(np.max(probs)*100):.2f}%"}
  
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
