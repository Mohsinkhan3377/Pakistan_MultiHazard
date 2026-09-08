"""
Pakistan Multi-Hazard AI System — Streamlit (full version, matches the Colab notebook).

Loads PRE-TRAINED models (best_model_*.joblib) — no retraining at startup.

Repo must contain (all in the same folder):
    streamlit_app.py
    pakistan_districts.py
    best_model_heavy_rain.joblib
    best_model_heatwave.joblib
    best_model_flood.joblib
    best_model_landslide.joblib
    requirements.txt

In Streamlit Cloud: Settings -> Secrets, add:
    GROQ_API_KEY = "your_actual_key_here"

⚠️ multi_hazard_pakistan.csv (training data) is a synthetic, terrain-calibrated
proxy dataset — say so explicitly in any report/demo, not raw historical records.
"""

import os
import random
from functools import lru_cache

import joblib
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import requests
import streamlit as st
from gtts import gTTS

from pakistan_districts import DISTRICTS

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

st.set_page_config(page_title="Pakistan Multi-Hazard AI System", page_icon="🌍", layout="wide")

GROQ_API_KEY = st.secrets.get("GROQ_API_KEY", os.environ.get("GROQ_API_KEY", ""))
OPEN_METEO_URL = "https://api.open-meteo.com/v1/forecast"
GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
GROQ_MODEL = "openai/gpt-oss-120b"
REQUEST_TIMEOUT = 20

FEATURES = [
    "rainfall_24h", "rainfall_3day", "rainfall_7day", "max_daily_rain",
    "forecast_rain", "max_temperature", "average_temperature",
    "average_humidity", "soil_moisture", "max_wind_speed",
]
TARGETS = ["heavy_rain", "heatwave", "flood", "landslide"]
LANDSLIDE_TERRAINS = ["mountainous", "hilly"]

MODEL_FILES = {
    "heavy_rain": "best_model_heavy_rain.joblib",
    "heatwave": "best_model_heatwave.joblib",
    "flood": "best_model_flood.joblib",
    "landslide": "best_model_landslide.joblib",
}

RISK_COLORS = {"Low": "#2ecc71", "Moderate": "#f1c40f", "High": "#e67e22", "Severe": "#e74c3c"}
RISK_EMOJI = {"Low": "🟢", "Moderate": "🟡", "High": "🟠", "Severe": "🔴"}
TERRAIN_POP_DENSITY = {"plains": 600, "arid": 150, "coastal": 2500, "hilly": 300, "mountainous": 80}

PROVINCES = ["All"] + sorted(set(v[2] for v in DISTRICTS.values()))
ALL_DISTRICTS = sorted(DISTRICTS.keys())


# ---------------------------------------------------------------------------
# Load pre-trained models
# ---------------------------------------------------------------------------

@st.cache_resource(show_spinner="Loading trained models...")
def load_models():
    models = {}
    for target, filename in MODEL_FILES.items():
        path = os.path.join(os.path.dirname(__file__), filename)
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing model file: {filename}")
        models[target] = joblib.load(path)
    return models


MODELS = load_models()


# ---------------------------------------------------------------------------
# Live weather + feature engineering
# ---------------------------------------------------------------------------

@lru_cache(maxsize=64)
def get_weather(lat, lon):
    params = {
        "latitude": lat, "longitude": lon,
        "hourly": "temperature_2m,relative_humidity_2m,precipitation,soil_moisture_0_to_1cm,wind_speed_10m",
        "daily": "temperature_2m_max,temperature_2m_min,precipitation_sum",
        "past_days": 7, "forecast_days": 2, "timezone": "auto",
    }
    response = requests.get(OPEN_METEO_URL, params=params, timeout=REQUEST_TIMEOUT)
    if response.status_code == 429:
        raise Exception("Weather API is temporarily rate-limited. Try again in a few minutes.")
    response.raise_for_status()
    return response.json()


def create_weather_features(data):
    hourly, daily = data["hourly"], data["daily"]
    temperature = pd.Series(hourly["temperature_2m"]).dropna()
    humidity = pd.Series(hourly["relative_humidity_2m"]).dropna()
    precipitation = pd.Series(hourly["precipitation"]).dropna()
    soil = pd.Series(hourly["soil_moisture_0_to_1cm"]).dropna()
    wind = pd.Series(hourly["wind_speed_10m"]).dropna()
    daily_rain = pd.Series(daily["precipitation_sum"]).dropna()
    daily_max_temp = pd.Series(daily["temperature_2m_max"]).dropna()

    return {
        "rainfall_24h": precipitation.tail(24).sum(),
        "rainfall_3day": precipitation.tail(72).sum(),
        "rainfall_7day": precipitation.sum(),
        "max_daily_rain": daily_rain.max(),
        "forecast_rain": daily_rain.tail(2).sum(),
        "max_temperature": daily_max_temp.max(),
        "average_temperature": temperature.mean(),
        "average_humidity": humidity.mean(),
        "soil_moisture": soil.mean(),
        "max_wind_speed": wind.max(),
    }


def risk_level(probability):
    if probability < 0.25:
        return "Low"
    elif probability < 0.50:
        return "Moderate"
    elif probability < 0.75:
        return "High"
    return "Severe"


def prepare_live_input(features):
    return pd.DataFrame([features])[FEATURES]


def predict_all_hazards(district, features):
    terrain = DISTRICTS[district][3]
    input_data = prepare_live_input(features)
    predictions = {}
    for target in TARGETS:
        if target == "landslide" and terrain not in LANDSLIDE_TERRAINS:
            predictions[target] = {"probability": 0.0, "risk": "Low",
                                    "note": "Not a landslide-prone terrain here"}
            continue
        proba = float(MODELS[target].predict_proba(input_data)[0][1])
        predictions[target] = {"probability": proba, "risk": risk_level(proba)}
    return predictions


def run_hazard_assessment(district):
    lat, lon, province, terrain = DISTRICTS[district]
    weather = get_weather(lat, lon)
    features = create_weather_features(weather)
    predictions = predict_all_hazards(district, features)
    return features, predictions


# ---------------------------------------------------------------------------
# Groq explanation layer
# ---------------------------------------------------------------------------

def explain_with_groq(district, features, predictions):
    if not GROQ_API_KEY:
        return "⚠️ Groq explanation unavailable — no GROQ_API_KEY configured for this app."

    hazard_lines = "\n".join(
        f"{h}: {p['probability']:.1%} risk ({p['risk']})" for h, p in predictions.items()
    )
    prompt = f"""You are an AI assistant for an educational multi-hazard environmental risk system in Pakistan.

District: {district}

Machine-learning predictions:
{hazard_lines}

Weather data:
7-day rainfall: {features['rainfall_7day']:.2f} mm
24-hour rainfall: {features['rainfall_24h']:.2f} mm
Forecast rainfall: {features['forecast_rain']:.2f} mm
Maximum temperature: {features['max_temperature']:.2f} C
Average humidity: {features['average_humidity']:.2f} %
Soil moisture: {features['soil_moisture']:.3f}
Wind: {features['max_wind_speed']:.2f} km/h

Explain the results in simple language. Mention which hazards have the
highest risk and which weather factors contributed. Do not claim a disaster
will definitely happen. Clearly state this is an ML-based preliminary
assessment, not an official emergency warning."""

    headers = {"Authorization": f"Bearer {GROQ_API_KEY}", "Content-Type": "application/json"}
    payload = {"model": GROQ_MODEL, "messages": [{"role": "user", "content": prompt}],
               "temperature": 0.2, "max_tokens": 300}
    try:
        response = requests.post(GROQ_URL, headers=headers, json=payload, timeout=30)
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"].strip()
    except Exception as e:
        return f"⚠️ Groq explanation failed: {e}"


# ---------------------------------------------------------------------------
# Feature 1 — Multi-Agent System
# ---------------------------------------------------------------------------

def climate_sentinel_agent(district):
    lat, lon, province, terrain = DISTRICTS[district]
    weather = get_weather(lat, lon)
    return create_weather_features(weather)


def disaster_agent(district, features):
    return predict_all_hazards(district, features)


def field_agent(predictions):
    actions = {}
    for hazard, result in predictions.items():
        risk = result["risk"]
        label = hazard.replace("_", " ").title()
        if risk == "Severe":
            actions[hazard] = f"🔴 URGENT: Evacuate at-risk areas for {label}. Alert PDMA immediately."
        elif risk == "High":
            actions[hazard] = f"🟠 Prepare: Pre-position relief supplies for {label}."
        elif risk == "Moderate":
            actions[hazard] = f"🟡 Monitor {label} conditions closely."
        else:
            actions[hazard] = f"🟢 No action needed for {label}."
    return actions


def run_multi_agent_dashboard(district):
    features = climate_sentinel_agent(district)
    predictions = disaster_agent(district, features)
    actions = field_agent(predictions)
    return features, predictions, actions


# ---------------------------------------------------------------------------
# Feature 2 — Urdu Voice Alert
# ---------------------------------------------------------------------------

def generate_voice_alert(district, predictions):
    top_hazard = max(predictions, key=lambda h: predictions[h]["probability"])
    risk = predictions[top_hazard]["risk"]
    label = top_hazard.replace("_", " ").title()

    prompt = f"""Generate a short emergency alert (1-2 sentences) in Urdu script.
District: {district}
Hazard: {label}
Risk level: {risk}
Keep it short, clear, urgent but not panic-inducing.
Respond ONLY with the Urdu sentence, nothing else."""

    headers = {"Authorization": f"Bearer {GROQ_API_KEY}", "Content-Type": "application/json"}
    payload = {"model": GROQ_MODEL, "messages": [{"role": "user", "content": prompt}],
               "max_tokens": 150, "temperature": 0.3}
    r = requests.post(GROQ_URL, headers=headers, json=payload)

    if r.status_code != 200:
        raise Exception(f"Groq API error {r.status_code}: {r.text}")
    result = r.json()
    if "choices" not in result:
        raise Exception(f"Unexpected Groq response: {result}")

    urdu_text = result["choices"][0]["message"]["content"].strip()
    if not urdu_text:
        urdu_text = f"خطرہ: {district} میں {label} کا خطرہ {risk} ہے۔"

    roman_urdu_text = f"Khatra: {district} mein {label} ka risk {risk} hai."

    audio_path = f"/tmp/alert_{district.replace(' ', '_')}.mp3"
    gTTS(text=urdu_text, lang="ur").save(audio_path)

    return urdu_text, roman_urdu_text, audio_path


# ---------------------------------------------------------------------------
# Feature 3 — Evacuation Route Finder (Graph + Dijkstra)
# ---------------------------------------------------------------------------

def build_district_road_graph(district, grid_size=8, seed=42):
    """Simulated local road network (grid graph) for demo purposes.
    For real roads, replace with osmnx-fetched OSM data."""
    random.seed(seed)
    G = nx.grid_2d_graph(grid_size, grid_size)
    for u, v in G.edges():
        G.edges[u, v]["weight"] = round(random.uniform(1, 5), 2)
    return G


def mark_flood_zone(G, predictions):
    flood_risk = predictions["flood"]["risk"]
    flooded_nodes = set()
    if flood_risk in ("High", "Severe"):
        nodes = list(G.nodes())
        center = nodes[len(nodes) // 2]
        for n in nodes:
            if abs(n[0] - center[0]) <= 1 and abs(n[1] - center[1]) <= 1:
                flooded_nodes.add(n)
    return flooded_nodes


def find_safe_route(district, predictions, start=(0, 0), end=(7, 7)):
    G = build_district_road_graph(district)
    flooded = mark_flood_zone(G, predictions)
    G_safe = G.copy()
    G_safe.remove_nodes_from(flooded)

    try:
        path = nx.dijkstra_path(G_safe, start, end, weight="weight")
        distance = nx.dijkstra_path_length(G_safe, start, end, weight="weight")
        status = "Safe route found, avoiding flood zone" if flooded else "No flood zone — direct route"
    except nx.NetworkXNoPath:
        path, distance, status = None, None, "⚠️ No safe route — all paths blocked by flood zone!"

    return path, distance, status, flooded, G


def draw_route_graph(G, path, flooded_nodes, title):
    pos = {n: n for n in G.nodes()}
    fig, ax = plt.subplots(figsize=(6, 6))
    colors = ["red" if n in flooded_nodes else "lightgray" for n in G.nodes()]
    nx.draw(G, pos, ax=ax, node_color=colors, node_size=200, edge_color="lightgray")
    if path:
        path_edges = list(zip(path, path[1:]))
        nx.draw_networkx_edges(G, pos, edgelist=path_edges, edge_color="blue", width=3, ax=ax)
        nx.draw_networkx_nodes(G, pos, nodelist=path, node_color="blue", node_size=100, ax=ax)
    ax.set_title(title)
    return fig


# ---------------------------------------------------------------------------
# Feature 4 — AI Chatbot Copilot
# ---------------------------------------------------------------------------

def ai_chatbot(question):
    mentioned_district = None
    for d in DISTRICTS:
        if d.lower() in question.lower():
            mentioned_district = d
            break

    context = ""
    if mentioned_district:
        features, predictions = run_hazard_assessment(mentioned_district)
        context = "\nCurrent data for " + mentioned_district + ": " + ", ".join(
            f"{h}={p['risk']}({p['probability']:.0%})" for h, p in predictions.items()
        )

    prompt = f"""You are a helpful disaster-preparedness assistant for Pakistan.
User question: {question}
{context}
Answer helpfully and concisely, in the same language style as the question
(English / Urdu / Roman Urdu). If asked about emergency kits, give practical
general advice. If no live data is available for the location, say so honestly."""

    headers = {"Authorization": f"Bearer {GROQ_API_KEY}", "Content-Type": "application/json"}
    payload = {"model": GROQ_MODEL, "messages": [{"role": "user", "content": prompt}],
               "max_tokens": 250, "temperature": 0.4}
    r = requests.post(GROQ_URL, headers=headers, json=payload)
    return r.json()["choices"][0]["message"]["content"].strip()


# ---------------------------------------------------------------------------
# Feature 5 — Damage Estimation (heuristic)
# ---------------------------------------------------------------------------

def estimate_damage(district, predictions):
    terrain = DISTRICTS[district][3]
    flood_prob = predictions["flood"]["probability"]
    density = TERRAIN_POP_DENSITY[terrain]

    affected_area_km2 = flood_prob * 5
    estimated_houses = int(affected_area_km2 * density / 6)
    estimated_roads = max(0, int(flood_prob * 4))

    return {
        "estimated_houses_at_risk": estimated_houses,
        "estimated_roads_at_risk": estimated_roads,
        "note": "Heuristic exposure-based estimate — not a trained ML model (no historical damage dataset available).",
    }


# ---------------------------------------------------------------------------
# Chart helper
# ---------------------------------------------------------------------------

def make_bar_chart(hazards, probabilities, colors, title):
    fig, ax = plt.subplots(figsize=(7, 4.2))
    bars = ax.bar(hazards, probabilities, color=colors)
    ax.set_ylim(0, 1)
    ax.set_ylabel("ML Probability")
    ax.set_title(title, fontweight="bold")
    for bar, prob in zip(bars, probabilities):
        ax.text(bar.get_x() + bar.get_width() / 2, prob + 0.02, f"{prob:.0%}",
                 ha="center", fontsize=9, fontweight="bold")
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Streamlit UI
# ---------------------------------------------------------------------------

st.markdown(
    """
    <div style="background: linear-gradient(135deg, #1e3a5f 0%, #2c5f8a 100%);
                padding: 28px 32px; border-radius: 12px; color: white; margin-bottom: 12px;">
        <h1 style="color:white; margin:0;">🌍 Pakistan Multi-Hazard AI System</h1>
        <p style="margin:8px 0 0 0;">Machine Learning · Live Weather · AI Explanations —
        Heavy Rain, Heatwave, Flood &amp; Landslide across every province of Pakistan.</p>
    </div>
    """,
    unsafe_allow_html=True,
)
st.caption("⚠️ Educational research system — not an official emergency warning service. "
           "Trained on a synthetic, terrain-calibrated proxy dataset.")

tabs = st.tabs([
    "📍 District Assessment", "📊 Compare Districts", "🤖 Multi-Agent Dashboard",
    "🔊 Voice Alert", "🗺️ Evacuation Route", "💬 Chatbot", "🏚️ Damage Estimate",
])

# ---- Tab 1: District Assessment ----
with tabs[0]:
    col1, col2 = st.columns([1, 2])
    with col1:
        province_filter = st.selectbox("Filter by Province", PROVINCES, key="p1")
        opts = [d for d, v in DISTRICTS.items() if province_filter == "All" or v[2] == province_filter]
        district = st.selectbox("Select District", sorted(opts), key="d1")
        run_btn = st.button("Run Assessment", type="primary", key="run1")

    if run_btn:
        try:
            with st.spinner(f"Fetching live weather for {district}..."):
                features, predictions = run_hazard_assessment(district)
                explanation = explain_with_groq(district, features, predictions)

            hazards = [h.replace("_", " ").title() for h in predictions]
            probs = [predictions[h]["probability"] for h in predictions]
            colors = [RISK_COLORS[predictions[h]["risk"]] for h in predictions]

            with col1:
                for h in predictions:
                    p = predictions[h]
                    note = f" — {p['note']}" if "note" in p else ""
                    st.markdown(f"**{RISK_EMOJI[p['risk']]} {h.replace('_',' ').title()}**: "
                                f"{p['risk']} ({p['probability']:.1%}){note}")
            with col2:
                st.pyplot(make_bar_chart(hazards, probs, colors, f"{district} — Multi-Hazard Risk"))

            with st.expander("🧠 AI Explanation", expanded=True):
                st.markdown(explanation)
        except Exception as e:
            st.error(f"⚠️ Error assessing **{district}**: {e}")

# ---- Tab 2: Compare Districts ----
with tabs[1]:
    compare_dd = st.multiselect("Select Districts (2+)", ALL_DISTRICTS,
                                 default=["Peshawar", "Swat", "Karachi Central", "Quetta"])
    compare_btn = st.button("Compare", type="primary", key="run2")

    if compare_btn:
        if len(compare_dd) < 2:
            st.warning("Select at least 2 districts.")
        else:
            rows = []
            with st.spinner("Fetching live weather..."):
                for d in compare_dd:
                    try:
                        features, predictions = run_hazard_assessment(d)
                        rows.append({
                            "District": d,
                            "Heavy Rain": predictions["heavy_rain"]["probability"],
                            "Heatwave": predictions["heatwave"]["probability"],
                            "Flood": predictions["flood"]["probability"],
                            "Landslide": predictions["landslide"]["probability"],
                        })
                    except Exception:
                        rows.append({"District": d, "Heavy Rain": 0, "Heatwave": 0, "Flood": 0, "Landslide": 0})

            comp_df = pd.DataFrame(rows)
            x = np.arange(len(comp_df))
            width = 0.2
            fig, ax = plt.subplots(figsize=(max(7, len(compare_dd) * 1.6), 4.5))
            ax.bar(x - 1.5*width, comp_df["Heavy Rain"], width, label="Heavy Rain")
            ax.bar(x - 0.5*width, comp_df["Heatwave"], width, label="Heatwave")
            ax.bar(x + 0.5*width, comp_df["Flood"], width, label="Flood")
            ax.bar(x + 1.5*width, comp_df["Landslide"], width, label="Landslide")
            ax.set_xticks(x)
            ax.set_xticklabels(comp_df["District"], rotation=20, ha="right")
            ax.set_ylim(0, 1)
            ax.legend()
            fig.tight_layout()

            riskiest = comp_df.set_index("District")[["Heavy Rain","Heatwave","Flood","Landslide"]].max(axis=1).idxmax()
            st.markdown(f"🔴 **Highest overall risk right now:** {riskiest}")
            st.pyplot(fig)
            st.dataframe(comp_df, use_container_width=True)

# ---- Tab 3: Multi-Agent Dashboard ----
with tabs[2]:
    agent_district = st.selectbox("Select District", ALL_DISTRICTS, index=ALL_DISTRICTS.index("Swat"), key="d3")
    agent_btn = st.button("Run Agents", type="primary", key="run3")

    if agent_btn:
        try:
            with st.spinner("Running agent pipeline..."):
                features, predictions, actions = run_multi_agent_dashboard(agent_district)

            st.markdown("### 🛰️ Climate Sentinel — Data Collected")
            st.markdown(f"Rain 7-day: {features['rainfall_7day']:.1f}mm | "
                        f"Temp: {features['max_temperature']:.1f}°C | "
                        f"Humidity: {features['average_humidity']:.0f}%")

            st.markdown("### 🎯 Disaster Agent — Predictions")
            for h, p in predictions.items():
                st.markdown(f"- **{h.replace('_',' ').title()}**: {p['risk']} ({p['probability']:.1%})")

            st.markdown("### 🚨 Field Agent — Recommended Actions")
            for h, a in actions.items():
                st.markdown(f"- {a}")
        except Exception as e:
            st.error(f"⚠️ Error: {e}")

# ---- Tab 4: Voice Alert ----
with tabs[3]:
    voice_district = st.selectbox("Select District", ALL_DISTRICTS, index=ALL_DISTRICTS.index("Swat"), key="d4")
    voice_btn = st.button("Generate Alert", type="primary", key="run4")

    if voice_btn:
        try:
            with st.spinner("Generating voice alert..."):
                features, predictions = run_hazard_assessment(voice_district)
                urdu_text, roman_text, audio_path = generate_voice_alert(voice_district, predictions)

            st.markdown(f"**Urdu:** {urdu_text}")
            st.markdown(f"**Roman Urdu:** {roman_text}")
            st.audio(audio_path)
        except Exception as e:
            st.error(f"⚠️ Error: {e}")

# ---- Tab 5: Evacuation Route ----
with tabs[4]:
    route_district = st.selectbox("Select District", ALL_DISTRICTS, index=ALL_DISTRICTS.index("Swat"), key="d5")
    route_btn = st.button("Find Safe Route", type="primary", key="run5")

    if route_btn:
        try:
            with st.spinner("Computing route..."):
                features, predictions = run_hazard_assessment(route_district)
                path, distance, status, flooded, G = find_safe_route(route_district, predictions)
                fig = draw_route_graph(G, path, flooded, f"{route_district} — Simulated Evacuation Route")

            st.markdown(f"**Status:** {status}")
            st.markdown(f"**Flood-affected nodes avoided:** {len(flooded)}")
            st.markdown(f"**Route length (simulated units):** {distance if distance else 'N/A'}")
            st.pyplot(fig)
            st.caption("⚠️ Simulated grid road network for demo — not real Pakistani road data.")
        except Exception as e:
            st.error(f"⚠️ Error: {e}")

# ---- Tab 6: Chatbot ----
with tabs[5]:
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    for q, a in st.session_state.chat_history:
        st.chat_message("user").write(q)
        st.chat_message("assistant").write(a)

    question = st.chat_input("Ask a question (English / Urdu / Roman Urdu)")
    if question:
        st.chat_message("user").write(question)
        try:
            answer = ai_chatbot(question)
        except Exception as e:
            answer = f"⚠️ Error: {e}"
        st.chat_message("assistant").write(answer)
        st.session_state.chat_history.append((question, answer))

# ---- Tab 7: Damage Estimate ----
with tabs[6]:
    damage_district = st.selectbox("Select District", ALL_DISTRICTS, index=ALL_DISTRICTS.index("Swat"), key="d7")
    damage_btn = st.button("Estimate Damage", type="primary", key="run7")

    if damage_btn:
        try:
            with st.spinner("Estimating..."):
                features, predictions = run_hazard_assessment(damage_district)
                damage = estimate_damage(damage_district, predictions)

            st.markdown(f"**Estimated houses at risk:** ~{damage['estimated_houses_at_risk']}")
            st.markdown(f"**Estimated roads at risk:** ~{damage['estimated_roads_at_risk']}")
            st.caption(damage["note"])
        except Exception as e:
            st.error(f"⚠️ Error: {e}")
