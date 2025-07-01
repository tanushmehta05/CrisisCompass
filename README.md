# 🧭 CrisisCompass – AI-Powered Emergency Response System

CrisisCompass is an intelligent, offline-first crisis response platform that leverages LLMs, machine learning, and geolocation to generate instant, context-aware emergency instructions — without relying on any external APIs.

---

## 🚀 Featuress

- 🤖 **Fine-Tuned Instruction LLM** (TinyLlama)
- 🧠 **Dual-Head Classifier** for crisis type + urgency
- 📍 **Location Detection** via NER + fallback dropdown
- 📞 **Emergency Contact Lookup** (state-wise)
- 📌 **LLM-Powered Instruction Generation** trained on synthetic + real scenarios
- ✅ Works fully offline (no OpenAI/Gemini APIs)
- 🧱 Modular pipeline ready for deployment

---

## 🧠 How It Works

1. **Crisis Report Ingested** → Text input from user or feed
2. **ML Classifier** → Predicts `Crisis Type` and `Urgency`
3. **NER Location Extractor** → Extracts geolocation or uses dropdown fallback
4. **Emergency Contact Lookup** → Maps state to known helpline
5. **LLM Generator** → Fine-tuned TinyLlama generates action instructions

---

## 🛠 Tech Stack

- 🤗 `transformers`, `datasets`, `peft`, `accelerate`
- 🦙 Fine-tuned `TinyLlama-1.1B-Chat`
- 🔍 `spaCy` / `HuggingFace NER` pipeline
- 📦 End-to-end runnable on **Kaggle or Colab** (no token required)

---

## 📂 Folder Structure

```bash
├── model/
│   └── classifier_model/      # Trained crisis classifier
│   └── crisis_llm/            # Fine-tuned instruction LLM
├── data/
│   ├── synthetic_reports.csv  # Augmented + real crisis data
│   └── instruction_dataset.jsonl
├── pipeline/
│   └── crisis_pipeline.py     # Full triage logic
├── notebooks/
│   └── training_notebooks.ipynb
└── README.md

🧪 Example
yaml
Copy
Edit
Input:
"12 people trapped in Siliguri after floods. Water rising."

🧠 → Crisis Type: Rescue
⚠️ → Urgency: High
📍 → Location: Siliguri (West Bengal)
📞 → Contact: +91-33-11223344

📝 Generated Instruction:
"Deploy search and rescue teams immediately to Siliguri. Coordinate with local response u

👨‍🔬 Future Work
🔊 Auto-call generation via Twilio or Android app

🌍 GPS-based routing to closest responders

📡 SMS gateway for rural/offline operation

🧠 Advanced instruction LLM trained with real world + crowdsourced data

👤 Author
Built with 💡 by Mehhta – always open to contributions or ideas!

📜 License
MIT – use freely, just don’t leave people behind. 🆘