# 🌍 Pakistan Multi-Hazard AI System

Machine learning-based preliminary risk assessment for **Heavy Rain, Heatwave,
Flood, and Landslide** across all provinces of Pakistan (Punjab, Sindh,
Balochistan, KP, Azad Jammu & Kashmir, Gilgit-Baltistan, ICT) — 154 districts.

⚠️ **Educational research prototype — not an official emergency warning
service.** Trained on a synthetic, terrain-calibrated proxy dataset
(`multi_hazard_pakistan.csv`), not raw historical disaster records.

## Features

1. **District Assessment** — live weather (Open-Meteo) + ML prediction + Groq AI explanation
2. **Compare Districts** — side-by-side multi-hazard comparison
3. **Multi-Agent Dashboard** — Climate Sentinel (data) → Disaster Agent (predict) → Field Agent (action)
4. **Voice Alert** — Urdu text + audio (gTTS) emergency alert generation
5. **Evacuation Route Finder** — Graph + Dijkstra shortest-path avoiding simulated flood zones
6. **AI Chatbot** — natural-language Q&A (English / Urdu / Roman Urdu)
7. **Damage Estimation** — heuristic exposure-based house/road risk estimate

## Tech stack

- **ML**: scikit-learn, XGBoost (Logistic Regression / Random Forest / XGBoost, best-model auto-selection per hazard)
- **Weather**: Open-Meteo (free, no API key)
- **AI explanations & chatbot**: Groq API (`openai/gpt-oss-120b`)
- **Voice**: gTTS (Google Text-to-Speech, Urdu)
- **Routing**: NetworkX (Dijkstra's algorithm on a simulated grid road network)
- **UI**: Streamlit

## Setup

```bash
git clone <your-repo-url>
cd <your-repo-name>
pip install -r requirements.txt
```

Get a free Groq API key at [console.groq.com](https://console.groq.com).

### Run locally

```bash
export GROQ_API_KEY="your_actual_key_here"      # Linux/Mac
setx GROQ_API_KEY "your_actual_key_here"         # Windows

streamlit run streamlit_app.py
```

### Deploy on Streamlit Community Cloud

1. Push this repo to GitHub (see below)
2. Go to [share.streamlit.io](https://share.streamlit.io) → **New app**
3. Select your repo, branch `main`, main file `streamlit_app.py`
4. **Settings → Secrets**, add:
   ```toml
   GROQ_API_KEY = "your_actual_key_here"
   ```
5. Deploy

## Repo structure

```
.
├── streamlit_app.py              # Main Streamlit app (7 tabs)
├── pakistan_districts.py         # District coordinates + terrain classification (154 districts)
├── best_model_heavy_rain.joblib  # Trained model
├── best_model_heatwave.joblib    # Trained model
├── best_model_flood.joblib       # Trained model
├── best_model_landslide.joblib   # Trained model
├── requirements.txt
├── README.md
└── .gitignore
```

## Limitations (be upfront about these)

- Training data is **synthetic**, terrain-calibrated by rule, not real historical
  weather or disaster records.
- Evacuation routes use a **simulated grid road network**, not real Pakistani
  road data (would require `osmnx` + OpenStreetMap integration).
- Damage estimation is a **heuristic formula**, not a trained ML model — no
  historical damage dataset exists to train one.
- Landslide predictions only apply to mountainous/hilly-terrain districts.
