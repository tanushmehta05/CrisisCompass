import os
from supabase import create_client
from dotenv import load_dotenv
from datetime import datetime

load_dotenv()
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

def insert_report_to_supabase(result, text):
    data = {
        "text": text,
        "crisis_type": result.get("crisis_type"),
        "urgency": result.get("urgency"),
        "location": result.get("location"),
        "state": result.get("state"),
        "latitude": result.get("latitude"),
        "longitude": result.get("longitude"),
        "contact": result.get("contact"),
        "instruction": result.get("instruction"),
        "logic_warning": result.get("logic_warning"),
        "timestamp": datetime.utcnow().isoformat()
    }
    response = supabase.table("reports").insert(data).execute()
    print("✅ Report inserted:", response)
