import json
import os
import torch
import torch.nn as nn
import spacy
from geopy.geocoders import Nominatim
from transformers import AutoTokenizer, AutoModelForCausalLM, DistilBertTokenizer, pipeline
from transformers import DistilBertModel

# ========== BASE DIRECTORY ==========
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ========== 0. Load Custom Crisis Classifier ==========
class CrisisClassifier(nn.Module):
    def __init__(self, num_types=5, num_urgencies=3):
        super().__init__()
        self.bert = DistilBertModel.from_pretrained("distilbert-base-uncased")
        self.dropout = nn.Dropout(0.3)
        self.type_head = nn.Linear(self.bert.config.hidden_size, num_types)
        self.urgency_head = nn.Linear(self.bert.config.hidden_size, num_urgencies)

    def forward(self, input_ids, attention_mask):
        output = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled = self.dropout(output.last_hidden_state[:, 0])
        return self.type_head(pooled), self.urgency_head(pooled)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
classifier_tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")

classifier_model_path = r"C:\Users\ASUS\Documents\CrisisCompass\CrisisCompass\CrisisCompassAPI\api\models\crisis_model.pt"
classifier_model = CrisisClassifier()
classifier_model.load_state_dict(torch.load(classifier_model_path, map_location=device))
classifier_model.to(device)
classifier_model.eval()

def classify_crisis(text):
    inputs = classifier_tokenizer(text, return_tensors="pt", truncation=True, padding=True).to(device)
    out_type, out_urgency = classifier_model(**inputs)
    type_idx = torch.argmax(out_type, dim=1).item()
    urgency_idx = torch.argmax(out_urgency, dim=1).item()
    type_labels = ["Medical", "Food", "Shelter", "Search & Rescue", "Infrastructure Damage"]
    urgency_labels = ["Low", "Medium", "High"]
    return type_labels[type_idx], urgency_labels[urgency_idx]

# ========== 1. Load Fine-tuned Instruction LLM ==========
LLM_PATH = r"C:\Users\ASUS\Documents\CrisisCompass\CrisisCompass\CrisisCompassAPI\api\models\crisis_llm_model"
llm_tokenizer = AutoTokenizer.from_pretrained(LLM_PATH, use_fast=False)
llm_model = AutoModelForCausalLM.from_pretrained(LLM_PATH)
llm_pipe = pipeline("text-generation", model=llm_model, tokenizer=llm_tokenizer, device=0 if torch.cuda.is_available() else -1)

# ========== 2. Named Entity Recognition ==========
nlp = spacy.load("en_core_web_sm")
geolocator = Nominatim(user_agent="crisiscompass")

def extract_location(text):
    doc = nlp(text)
    for ent in doc.ents:
        if ent.label_ in ["GPE", "LOC"]:
            return ent.text
    return "Unknown"

# ========== 3. Geolocation ==========
def geocode_location(location):
    try:
        loc = geolocator.geocode(location, addressdetails=True, country_codes='in')
        if loc:
            lat = loc.latitude
            lon = loc.longitude
            full_address = loc.address
            address_dict = loc.raw.get("address", {})
            state = address_dict.get("state") or address_dict.get("region") or "Unknown"
            return lat, lon, full_address, state
    except Exception as e:
        print("GeoError:", e)
    return None, None, "", "Unknown"

# ========== 4. Emergency Contact Mapping ==========
contact_path = os.path.join(BASE_DIR, "emergency_contacts.json")
with open(contact_path, "r") as f:
    contact_dict = json.load(f)

def get_contact(state):
    return contact_dict.get(state, "Not Available")

# ========== 5. Instruction Generation ==========
def generate_instruction(crisis_type, urgency, location, contact):
    prompt = f"<s>[INST] Crisis Type: {crisis_type}, Urgency: {urgency}, Location: {location}, Contact: {contact} [/INST]"
    result = llm_pipe(prompt, max_new_tokens=100, do_sample=True, temperature=0.7)
    return result[0]["generated_text"].split("[/INST]")[-1].strip()

# ========== 6. Dynamic Logic Validation (based on recent disasters) ==========
recent_disasters = [
    {"type": "Flood", "state": "Assam"},
    {"type": "Flood", "state": "Meghalaya"},
    {"type": "Flood", "state": "Mizoram"},
    {"type": "Landslide", "state": "Uttarakhand"},
    {"type": "Cloudburst", "state": "Uttarakhand"},
    {"type": "Flash Flood", "state": "Himachal Pradesh"},
    {"type": "Flood", "state": "Arunachal Pradesh"},
    {"type": "Flash Flood", "state": "Jammu & Kashmir"},
    {"type": "Avalanche", "state": "Uttarakhand"},
]

def dynamic_logic_warning(crisis_type, state):
    if crisis_type.lower() == "search & rescue":
        return None

    type_map = {
        "Flood": ["flood", "flash flood"],
        "Landslide": ["landslide", "cloudburst"],
        "Avalanche": ["avalanche"],
        "Search & Rescue": ["rescue", "landslide", "collapse", "flood", "disaster"],
    }
    c = crisis_type.lower()
    s = state.lower()

    for rec in recent_disasters:
        if s == rec["state"].lower():
            for match in type_map.get(crisis_type, [c]):
                if match.lower() in rec["type"].lower():
                    return None

    if crisis_type in type_map:
        return f"⚠️ {crisis_type} is uncommon in {state}. Please verify."
    return None

# ========== 7. Full Pipeline ==========
def crisis_pipeline(report_text):
    crisis_type, urgency = classify_crisis(report_text)
    location = extract_location(report_text)
    lat, lon, full_address, state = geocode_location(location)
    contact = get_contact(state)
    instruction = generate_instruction(crisis_type, urgency, location, contact)
    logic_warning = dynamic_logic_warning(crisis_type, state)

    result = {
        "crisis_type": crisis_type,
        "urgency": urgency,
        "location": location,
        "state": state,
        "latitude": lat,
        "longitude": lon,
        "contact": contact,
        "instruction": instruction,
        "logic_warning": logic_warning
    }

    return result
