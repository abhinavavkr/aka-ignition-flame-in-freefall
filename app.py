"""AstroFlame SI Streamlit dashboard with a NumPy semantic RAG co-pilot."""

import json
import os
import pickle
import re
import time

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from google import genai
from google.genai import types


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "psi_data.json")
EMBEDDINGS_CACHE_PATH = os.path.join(BASE_DIR, "embeddings_cache.pkl")
EMBEDDING_MODELS = ("gemini-embedding-001", "text-embedding-004")


st.set_page_config(
    page_title="NASA PSI Fire Safety Intelligence",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    :root {
        --orbital-void: #0A0E17;
        --glass-panel: rgba(16, 24, 39, 0.70);
        --glass-border: rgba(255, 107, 53, 0.15);
        --ignition-orange: #FF6B35;
        --plasma-cyan: #00E5FF;
        --telemetry-white: #E5EEF7;
        --telemetry-muted: #9FB3C8;
    }

    .stApp,
    [data-testid="stAppViewContainer"],
    [data-testid="stHeader"] {
        background: var(--orbital-void);
        color: var(--telemetry-white);
    }

    .stApp {
        font-family: Inter, "Roboto Mono", ui-monospace, SFMono-Regular, Consolas, monospace;
    }

    [data-testid="stSidebar"] {
        background:
            linear-gradient(180deg, rgba(10, 14, 23, 0.97), rgba(16, 24, 39, 0.92)),
            radial-gradient(circle at 20% 0%, rgba(0, 229, 255, 0.12), transparent 35%);
        border-right: 1px solid rgba(0, 229, 255, 0.12);
    }

    [data-testid="stSidebar"] h1,
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3,
    h1, h2, h3 {
        color: var(--telemetry-white);
        letter-spacing: 0.03em;
        text-shadow: 0 0 18px rgba(0, 229, 255, 0.18);
    }

    [data-testid="stMetric"],
    .telemetry-card,
    div[data-testid="stExpander"],
    [data-testid="stChatMessage"] {
        background: var(--glass-panel);
        border: 1px solid var(--glass-border);
        border-radius: 12px;
        backdrop-filter: blur(12px);
        box-shadow: 0 12px 32px rgba(0, 0, 0, 0.18);
    }

    [data-testid="stMetric"] {
        padding: 16px;
    }

    [data-testid="stMetricValue"] {
        color: var(--ignition-orange);
        text-shadow: 0 0 10px rgba(255, 107, 53, 0.55);
        font-family: "Roboto Mono", Inter, ui-monospace, monospace;
    }

    [data-testid="stMetricLabel"],
    [data-testid="stCaptionContainer"] {
        color: var(--telemetry-muted);
    }

    div[data-testid="stExpander"] {
        overflow: hidden;
        transition: border-color 180ms ease, box-shadow 180ms ease;
    }

    div[data-testid="stExpander"]:hover {
        border-color: rgba(0, 229, 255, 0.55);
        box-shadow: 0 0 22px rgba(0, 229, 255, 0.13);
    }

    button[data-baseweb="tab"] {
        color: var(--telemetry-muted);
        font-family: Inter, "Roboto Mono", ui-monospace, monospace;
        transition: color 180ms ease, text-shadow 180ms ease;
    }

    button[data-baseweb="tab"]:hover,
    button[data-baseweb="tab"][aria-selected="true"] {
        color: var(--plasma-cyan);
        text-shadow: 0 0 12px rgba(0, 229, 255, 0.65);
    }

    div[data-baseweb="tab-highlight"] {
        background: var(--plasma-cyan);
        box-shadow: 0 0 12px rgba(0, 229, 255, 0.75);
    }

    [data-testid="stChatMessage"] {
        padding: 12px 16px;
        margin: 10px 0;
    }

    [data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) {
        background: rgba(31, 60, 92, 0.72);
        border-color: rgba(80, 148, 255, 0.28);
    }

    [data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-assistant"]) {
        border-left: 3px solid var(--ignition-orange);
        box-shadow: -4px 0 18px rgba(255, 107, 53, 0.12);
    }

    .pill {
        display: inline-block;
        padding: 3px 9px;
        margin: 2px 4px 2px 0;
        border-radius: 999px;
        background: rgba(0, 229, 255, 0.12);
        color: #8DEBFF;
        font-size: 0.78rem;
        border: 1px solid rgba(0, 229, 255, 0.18);
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def load_local_api_key():
    """Read GEMINI_API_KEY from local files, environment, or Streamlit secrets."""
    env_paths = [
        os.path.join(BASE_DIR, ".env.local"),
        os.path.join(BASE_DIR, ".env"),
    ]
    for env_path in env_paths:
        try:
            with open(env_path, "r", encoding="utf-8") as env_file:
                for line in env_file:
                    name, separator, value = line.partition("=")
                    if separator and name.strip() == "GEMINI_API_KEY":
                        return value.strip().strip("\"'")
        except FileNotFoundError:
            continue
    environment_key = os.environ.get("GEMINI_API_KEY", "")
    if environment_key:
        return environment_key
    try:
        return st.secrets.get("GEMINI_API_KEY", "")
    except Exception:
        return ""


def _first_value(record, keys, default=""):
    for key in keys:
        value = record.get(key)
        if value not in (None, ""):
            return value
    return default


def _safe_float(value, default):
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return number if np.isfinite(number) else default


def _stringify(value):
    if value in (None, ""):
        return ""
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    return str(value)


def _join_key(value):
    return re.sub(r"[^a-z0-9]+", " ", _stringify(value).lower()).strip()


def _related_to_investigation(related, investigation):
    investigation_keys = {
        _join_key(investigation.get(key))
        for key in ("id", "psi_id", "PSI ID", "PSI ID & Short Name", "title", "Title")
    }
    related_keys = {
        _join_key(related.get(key))
        for key in (
            "investigation_id",
            "investigationId",
            "domain",
            "id",
            "psi_id",
            "PSI ID",
            "title",
            "Title",
        )
    }
    investigation_keys.discard("")
    related_keys.discard("")
    for investigation_key in investigation_keys:
        for related_key in related_keys:
            if (
                investigation_key == related_key
                or investigation_key in related_key
                or related_key in investigation_key
            ):
                return True
            investigation_prefix = investigation_key.split(" ", 1)[0]
            related_prefix = related_key.split(" ", 1)[0]
            if (
                len(investigation_prefix) >= 3
                and investigation_prefix == related_prefix
            ):
                return True
    return False


def _normalize_record(record, materials=None, spinoffs=None):
    """Convert flat, spreadsheet, or relational records to one canonical shape."""
    materials = materials or []
    spinoffs = spinoffs or []
    related_materials = [item for item in materials if _related_to_investigation(item, record)]
    related_spinoffs = [item for item in spinoffs if _related_to_investigation(item, record)]

    existing_materials = _first_value(
        record,
        ("Tested Materials", "materials", "Materials"),
    )
    existing_earth_benefits = _first_value(
        record,
        (
            "Research Impacts / Earth Benefits",
            "Earth Benefits",
            "Research Impacts",
            "earth_benefits",
            "spinoffs",
        ),
    )
    all_materials = []
    if existing_materials not in (None, ""):
        all_materials.append(existing_materials)
    all_materials.extend(related_materials)
    all_earth_benefits = []
    if existing_earth_benefits not in (None, ""):
        all_earth_benefits.append(existing_earth_benefits)
    all_earth_benefits.extend(related_spinoffs)

    normalized = dict(record)
    normalized.update({
        "psi_id": _stringify(
            _first_value(record, ("psi_id", "PSI ID", "PSI ID & Short Name", "id"))
        ),
        "project_type": _stringify(
            _first_value(record, ("project_type", "Project Type", "projectType"))
        ),
        "title": _stringify(
            _first_value(record, ("title", "Title", "Proposal Title", "name"))
        ),
        "objectives": _stringify(
            _first_value(record, ("objectives", "Objectives"))
        ),
        "approach": _stringify(_first_value(record, ("approach", "Approach"))),
        "hypothesis": _stringify(
            _first_value(record, ("hypothesis", "Hypothesis"))
        ),
        "findings": _stringify(_first_value(record, ("findings", "Findings"))),
        "earth_benefits": _stringify(all_earth_benefits),
        "materials": _stringify(all_materials),
    })
    return normalized


@st.cache_data
def load_psi_data(file_mtime):
    """Load and normalize flat, spreadsheet, and relational PSI JSON shapes."""
    try:
        with open(DATA_PATH, "r", encoding="utf-8") as data_file:
            data = json.load(data_file)
    except FileNotFoundError:
        st.error("psi_data.json was not found. Add the PSI dataset beside app.py.")
        return []
    except json.JSONDecodeError as error:
        st.error(f"psi_data.json is not valid JSON: {error}")
        return []

    if isinstance(data, list):
        return [_normalize_record(record) for record in data if isinstance(record, dict)]
    if isinstance(data, dict):
        investigations = data.get("investigations")
        if not isinstance(investigations, list):
            investigations = data.get("records")
        if isinstance(investigations, list):
            materials = data.get("materials", [])
            spinoffs = data.get("spinoffs", [])
            if not isinstance(materials, list):
                materials = []
            if not isinstance(spinoffs, list):
                spinoffs = []
            normalized = [
                _normalize_record(record, materials, spinoffs)
                for record in investigations
                if isinstance(record, dict)
            ]
            for related_item in materials:
                if isinstance(related_item, dict) and not any(
                    _related_to_investigation(related_item, investigation)
                    for investigation in investigations
                    if isinstance(investigation, dict)
                ):
                    normalized.append(_normalize_record(related_item, [related_item]))
            for related_item in spinoffs:
                if isinstance(related_item, dict) and not any(
                    _related_to_investigation(related_item, investigation)
                    for investigation in investigations
                    if isinstance(investigation, dict)
                ):
                    normalized.append(_normalize_record(related_item, [], [related_item]))
            return normalized
    st.error("psi_data.json must contain a list, an investigations list, or a records list.")
    return []


def _record_text(record):
    """Serialize all non-empty canonical fields for search and grounding."""
    labels = (
        ("PSI ID", "psi_id"),
        ("Title", "title"),
        ("Project Type", "project_type"),
        ("Objectives", "objectives"),
        ("Approach", "approach"),
        ("Hypothesis", "hypothesis"),
        ("Findings", "findings"),
        ("Tested Materials", "materials"),
        ("Earth Benefits / Spinoffs", "earth_benefits"),
    )
    return "\n".join(
        f"{label}: {record[key]}"
        for label, key in labels
        if record.get(key) not in (None, "")
    )


def _embedding_vectors(response):
    """Read one or many embedding vectors from the Gemini response."""
    embeddings = getattr(response, "embeddings", None)
    if embeddings:
        return [np.asarray(embedding.values, dtype=float) for embedding in embeddings]
    embedding = getattr(response, "embedding", None)
    if embedding:
        return [np.asarray(embedding.values, dtype=float)]
    raise ValueError("Gemini returned no embedding values.")


def _unit_vector(vector):
    vector = np.asarray(vector, dtype=float)
    norm = np.linalg.norm(vector)
    return vector / norm if norm else vector


def _cosine_similarity(vec_a, vec_b):
    vector_a = _unit_vector(vec_a)
    vector_b = _unit_vector(vec_b)
    if vector_a.size == 0 or vector_b.size == 0 or vector_a.shape != vector_b.shape:
        return 0.0
    return float(np.dot(vector_a, vector_b))


def _embed_request(client, model, contents, task_type):
    """Try task-aware embedding first, then the SDK-compatible no-config form."""
    try:
        return client.models.embed_content(
            model=model,
            contents=contents,
            config=types.EmbedContentConfig(task_type=task_type),
        )
    except Exception as error:
        error_text = str(error).lower()
        if "quota" not in error_text and "resource_exhausted" not in error_text:
            return client.models.embed_content(model=model, contents=contents)
        raise


def _is_quota_error(error):
    error_text = str(error).lower()
    return "quota" in error_text or "resource_exhausted" in error_text or "429" in error_text


def _embedding_failure(error):
    if _is_quota_error(error):
        return RuntimeError(
            "Gemini embedding quota is exhausted (429 RESOURCE_EXHAUSTED). "
            "Wait for the quota window to reset or use a project/API key with embedding quota."
        )
    return RuntimeError(f"Gemini embedding request failed: {error}")


def _embed_batch(client, texts, task_type):
    """Batch embed all texts once, falling back to sequential calls if necessary."""
    last_error = None
    for model in EMBEDDING_MODELS:
        try:
            response = _embed_request(client, model, texts, task_type)
            vectors = _embedding_vectors(response)
            if len(vectors) != len(texts):
                raise ValueError("Gemini returned an unexpected number of embeddings.")
            return [_unit_vector(vector) for vector in vectors]
        except Exception as error:
            if _is_quota_error(error):
                raise _embedding_failure(error) from error
            last_error = error

    vectors = []
    for text in texts:
        single_error = None
        for model in EMBEDDING_MODELS:
            try:
                response = _embed_request(client, model, text, task_type)
                vectors.append(_unit_vector(_embedding_vectors(response)[0]))
                single_error = None
                break
            except Exception as error:
                if _is_quota_error(error):
                    raise _embedding_failure(error) from error
                single_error = error
        if single_error is not None:
            raise _embedding_failure(single_error) from single_error
    if not vectors:
        raise _embedding_failure(last_error) from last_error
    return vectors


def _record_identity(record):
    value = _first_value(
        record,
        ("psi_id", "id", "PSI ID", "PSI ID & Short Name", "title", "Title"),
    )
    return _join_key(value)


def _cache_matches_records(cached_records, records):
    if not isinstance(cached_records, list) or len(cached_records) != len(records):
        return False
    if any(not isinstance(item, dict) or "_embedding" not in item for item in cached_records):
        return False
    return [
        _record_identity(item) for item in cached_records
    ] == [_record_identity(item) for item in records]


@st.cache_data(show_spinner=False)
def generate_document_embeddings(records, api_key):
    """Generate normalized document vectors with strict free-tier throttling."""
    if os.path.exists(EMBEDDINGS_CACHE_PATH):
        try:
            with open(EMBEDDINGS_CACHE_PATH, "rb") as cache_file:
                cached_records = pickle.load(cache_file)
            if _cache_matches_records(cached_records, records):
                return cached_records
        except Exception as error:
            st.warning(f"Embedding cache could not be loaded; rebuilding it: {error}")

    client = genai.Client(api_key=api_key, http_options={'api_version': 'v1beta'})
    embedded_records = []
    for record_index, source_record in enumerate(records, start=1):
        text_to_embed = _record_text(source_record)[:8000]
        vector = None
        last_error = None

        for attempt in range(5):
            try:
                response = None
                for model in EMBEDDING_MODELS:
                    try:
                        response = _embed_request(
                            client,
                            model,
                            text_to_embed,
                            "RETRIEVAL_DOCUMENT",
                        )
                        break
                    except Exception as error:
                        last_error = error
                if response is None:
                    raise last_error

                vectors = _embedding_vectors(response)
                if not vectors:
                    raise ValueError("Gemini returned no embedding for the record.")
                vector = _unit_vector(vectors[0])
                break
            except Exception as error:
                last_error = error
                st.warning(
                    f"Embedding record {record_index} failed "
                    f"(attempt {attempt + 1}/5): {error}"
                )
                if attempt == 4:
                    raise RuntimeError(
                        "Unable to generate PSI embeddings after 5 retries."
                    ) from last_error
                time.sleep(30)

        embedded_record = dict(source_record)
        embedded_record["_embedding"] = vector
        embedded_record["_search_text"] = _record_text(embedded_record)
        embedded_records.append(embedded_record)
        time.sleep(2)

    try:
        with open(EMBEDDINGS_CACHE_PATH, "wb") as cache_file:
            pickle.dump(embedded_records, cache_file)
    except Exception as error:
        st.warning(f"Embedding cache could not be saved: {error}")
    return embedded_records


def _record_value(record, *keys, default="Unknown"):
    for key in keys:
        value = record.get(key)
        if value not in (None, ""):
            return value
    return default


def _json_items(value):
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError:
            return []
    if isinstance(value, dict):
        return [value]
    return value if isinstance(value, list) else []


def _record_materials(records):
    materials = []
    for record in records:
        for material in _json_items(record.get("materials", "")):
            if isinstance(material, dict):
                item = dict(material)
                item["_parent_psi_id"] = record.get("psi_id", "")
                item["_parent_title"] = record.get("title", "")
                materials.append(item)
            elif material:
                materials.append(
                    {
                        "name": str(material),
                        "_parent_psi_id": record.get("psi_id", ""),
                        "_parent_title": record.get("title", ""),
                    }
                )
    return materials


def _material_metrics(material):
    fallback = {
        "PMMA": (17.0, 1.8, 14.2),
        "SIBAL Fabric": (18.0, 1.1, 16.0),
        "Nomex": (26.0, 0.4, 29.5),
        "Delrin": (18.5, 1.2, 18.0),
        "Polycarbonate": (21.5, 0.7, 24.7),
        "PU Foam": (16.0, 2.4, 11.8),
        "Cotton/Fiberglass": (20.0, 0.9, 22.1),
        "Teflon": (25.0, 0.2, 31.0),
        "Kapton": (22.0, 0.5, 27.0),
        "n-Dodecane (C12)": (15.0, 3.0, 10.0),
        "n-Heptane (C7)": (14.0, 3.4, 9.0),
        "n-Decane (C10)": (14.5, 3.2, 9.5),
    }
    name = str(
        material.get("name")
        or material.get("short_name")
        or material.get("material")
        or "Unknown Material"
    )
    benchmark = fallback.get(name, (20.0, 1.0, 20.0))
    return {
        "name": name,
        "loc": _safe_float(
            material.get("loc_percent", material.get("LOC", benchmark[0])),
            benchmark[0],
        ),
        "spread": _safe_float(
            material.get("flame_spread_mm_s", material.get("flame_spread", benchmark[1])),
            benchmark[1],
        ),
        "heat_flux": _safe_float(
            material.get(
                "critical_heat_flux_kw_m2",
                material.get("critical_heat_flux", benchmark[2]),
            ),
            benchmark[2],
        ),
        "psi_id": material.get("_parent_psi_id", "Benchmark"),
        "parent_title": material.get("_parent_title", "NASA benchmark fallback"),
        "fallback": not any(
            key in material
            for key in ("loc_percent", "LOC", "flame_spread_mm_s", "flame_spread")
        ),
    }


def _mission_risk(oxygen, gravity, pressure, airflow):
    oxygen_component = min(45.0, oxygen * 0.55)
    gravity_component = (1.0 - gravity) * 20.0
    airflow_peak = max(0.0, 1.0 - abs(airflow - 15.0) / 20.0) * (15.0 if gravity <= 0.05 else 7.0)
    pressure_component = max(0.0, min(15.0, (101.3 - pressure) / 8.0))
    return max(0.0, min(100.0, oxygen_component + gravity_component + airflow_peak + pressure_component))


def _flame_regime(gravity, airflow):
    if gravity <= 0.05 and airflow < 5:
        return "🔵 Spherical Diffusion / Cool-Flame (0g)"
    if gravity <= 0.05:
        return "🟠 Microgravity Forced-Flow Spread"
    if gravity <= 0.40:
        return "🟡 Fractional-g Mixed Convection"
    return "🔴 Buoyancy-Dominated Plume (1g)"


def _suppression_protocol(oxygen, gravity, airflow):
    if gravity <= 0.05 and airflow < 5:
        return "Zero-Convection SAME-R + Fine Water Mist"
    if oxygen >= 50 or airflow >= 25:
        return "Ventilation Cutoff + N₂/CO₂ Deluge"
    return "FLEX Low-Flow Suppression + Optical Detection"


def _category_for_record(record):
    text = _record_text(record).lower()
    if any(word in text for word in ("smoke", "aerosol", "particle")):
        return "Smoke & Aerosols"
    if any(word in text for word in ("droplet", "cool flame", "dodecane")):
        return "Droplet & Cool Flames"
    return "Solid Material Spread"


def _embed_question(question, api_key):
    client = genai.Client(api_key=api_key, http_options={'api_version': 'v1beta'})
    return _embed_batch(client, [question], "RETRIEVAL_QUERY")[0]


STOPWORDS = {
    "and", "are", "for", "from", "how", "into", "its", "the", "this",
    "what", "when", "where", "which", "with", "does", "about", "can",
}


def _lexical_score(question, record):
    query_tokens = {
        token
        for token in re.findall(r"[a-z0-9]+", question.lower())
        if len(token) >= 3 and token not in STOPWORDS
    }
    if not query_tokens:
        return 0.0
    record_text = str(record.get("_search_text", _record_text(record))).lower()
    matched = sum(token in record_text for token in query_tokens)
    score = matched / len(query_tokens)
    psi_id = record.get("psi_id", "").lower()
    if psi_id and psi_id in question.lower():
        score = min(1.0, score + 0.15)
    return score


def _retrieve_top_records(question, records, api_key):
    if not question or not question.strip() or not records:
        return []
    question_vector = _embed_question(question, api_key)
    scored_records = []
    for record in records:
        semantic_score = _cosine_similarity(question_vector, record.get("_embedding", []))
        lexical_score = _lexical_score(question, record)
        hybrid_score = (0.85 * semantic_score) + (0.15 * lexical_score)
        scored_records.append(
            (hybrid_score, semantic_score, lexical_score, record)
        )
    scored_records.sort(key=lambda item: item[0], reverse=True)
    return scored_records[:3]


def _answer_question(question, matches, api_key):
    context_parts = []
    for hybrid_score, semantic_score, lexical_score, record in matches:
        context_parts.append(
            f"Hybrid Match: {hybrid_score * 100:.1f}%\n"
            f"Semantic Match: {semantic_score * 100:.1f}%\n"
            f"Keyword Match: {lexical_score * 100:.1f}%\n"
            f"{_record_text(record)}"
        )
    context = "\n\n---\n\n".join(context_parts)
    prompt = (
        "Use ONLY the three PSI records below to answer the user's question. "
        "Do not invent facts or use outside knowledge. If the records do not support "
        "an answer, say so clearly. Cite the PSI ID for every factual claim.\n\n"
        f"RETRIEVED PSI RECORDS:\n{context}\n\n"
        f"USER QUESTION:\n{question}"
    )
    client = genai.Client(api_key=api_key, http_options={'api_version': 'v1beta'})
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt,
        config=types.GenerateContentConfig(
            system_instruction=(
                "Answer using ONLY the provided 3 records. Treat the records as the complete "
                "ground truth for this response. Cite the PSI ID for every claim."
            ),
            temperature=0.1,
        ),
    )
    return response.text or "The Gemini model returned an empty response."


def main():
    file_mtime = os.path.getmtime(DATA_PATH) if os.path.exists(DATA_PATH) else 0
    records = load_psi_data(file_mtime)
    if not records:
        st.stop()

    with st.sidebar:
        st.markdown("## 🛰️ NASA // MISSION CONTROL")
        st.caption("SPACE & FLAME TELEMETRY")

        presets = {
            "ISS Standard Cabin": (21.0, 0.00, 101.3, 10.0),
            "Lunar Habitat (Artemis)": (34.0, 0.16, 56.5, 15.0),
            "Martian Surface Habitat": (26.0, 0.38, 70.0, 12.0),
            "Exploration EVA Suit": (85.0, 0.00, 29.6, 20.0),
            "SAFFIRE-1 Flow Duct": (21.6, 0.00, 101.3, 20.0),
        }
        if "mission_preset" not in st.session_state:
            st.session_state.mission_preset = "ISS Standard Cabin"
        if "oxygen" not in st.session_state:
            st.session_state.oxygen = 21.0
            st.session_state.gravity = 0.00
            st.session_state.pressure = 101.3
            st.session_state.airflow = 10.0

        def apply_preset():
            values = presets[st.session_state.mission_preset]
            st.session_state.oxygen, st.session_state.gravity = values[:2]
            st.session_state.pressure, st.session_state.airflow = values[2:]

        mission_preset = st.selectbox(
            "Mission profile preset",
            list(presets),
            key="mission_preset",
            on_change=apply_preset,
        )
        oxygen = st.slider("O₂ Concentration (%)", 10.0, 100.0, step=0.5, key="oxygen")
        gravity = st.slider("Gravity (g)", 0.00, 1.00, step=0.01, key="gravity")
        pressure = st.slider("Cabin Pressure (kPa)", 20.0, 120.0, step=1.0, key="pressure")
        airflow = st.slider("Ventilation Airflow (cm/s)", 0.0, 50.0, step=1.0, key="airflow")

    api_key = load_local_api_key()
    if not api_key:
        st.error("🚨 CRITICAL: GEMINI_API_KEY not found in .env.local")
        st.stop()

    ranked_records = []
    for record in records:
        relevance = _mission_risk(oxygen, gravity, pressure, airflow)
        category = _category_for_record(record)
        relevance = max(0.0, min(100.0, relevance + (8 if category == "Droplet & Cool Flames" and gravity < 0.1 else 0)))
        ranked_records.append((relevance, record, category))
    ranked_records.sort(key=lambda item: item[0], reverse=True)
    top_score, top_record, _ = ranked_records[0]
    risk_score = _mission_risk(oxygen, gravity, pressure, airflow)
    risk_index = "CRITICAL" if risk_score >= 75 else "SEVERE" if risk_score >= 55 else "MODERATE" if risk_score >= 35 else "NOMINAL"
    flame_regime = _flame_regime(gravity, airflow)
    suppression = _suppression_protocol(oxygen, gravity, airflow)

    st.title("🔥 AstroFlame SI · Space & Flame Telemetry")
    st.caption("NASA PSI mission-control view for flame-in-freefall environments.")
    metric_columns = st.columns(4)
    metric_columns[0].metric("Flammability Risk Index", f"{risk_index} · {risk_score:.0f}/100", f"{oxygen:.1f}% O₂")
    metric_columns[1].metric("Active Flame Regime", flame_regime)
    metric_columns[2].metric(
        "Top Investigation Match",
        f"{top_record.get('psi_id', 'PSI')} · {top_record.get('title', 'Unknown')}",
        f"{top_score:.0f}/100 relevance",
    )
    metric_columns[3].metric("Suppression & Detection", suppression)
    st.caption(
        f"Profile: {mission_preset} · Pressure {pressure:.1f} kPa · "
        f"Airflow {airflow:.0f} cm/s · Dataset revision {file_mtime:.0f}"
    )

    tab_one, tab_two, tab_three, tab_four = st.tabs(
        [
            "1. Environment Ranker",
            "2. Scenario Explorer",
            "3. Fire Controls & Spinoffs",
            "4. Semantic RAG Co-Pilot",
        ]
    )

    with tab_one:
        st.subheader("1. Mission Relevance Ranker")
        filter_columns = st.columns([2, 1])
        with filter_columns[0]:
            search_term = st.text_input("Search investigations", placeholder="flame, smoke, material...")
        with filter_columns[1]:
            category_filter = st.selectbox(
                "Category",
                ["All", "Droplet & Cool Flames", "Solid Material Spread", "Smoke & Aerosols"],
            )
        visible_records = [
            item
            for item in ranked_records
            if (not search_term or search_term.lower() in _record_text(item[1]).lower())
            and (category_filter == "All" or item[2] == category_filter)
        ]
        for rank, (score, record, category) in enumerate(visible_records, start=1):
            title = record.get("title", "Untitled PSI investigation")
            with st.expander(f"#{rank}  🛰️ {record.get('psi_id', 'PSI')} · {title} · {score:.0f}/100"):
                st.markdown(
                    f'<span class="pill">PSI ID · {record.get("psi_id", "Unknown")}</span>'
                    f'<span class="pill">Category · {category}</span>'
                    f'<span class="pill">Flame Regime · {flame_regime}</span>',
                    unsafe_allow_html=True,
                )
                st.progress(score / 100)
                detail_columns = st.columns(2)
                with detail_columns[0]:
                    st.markdown("**Objectives & Hypothesis**")
                    st.write(record.get("objectives") or "Not specified")
                    st.write(record.get("hypothesis") or "Not specified")
                with detail_columns[1]:
                    st.markdown("**Experimental Approach & Findings**")
                    st.write(record.get("approach") or "Not specified")
                    st.write(record.get("findings") or "Not specified")
                st.markdown("**Earth & spacecraft benefits**")
                st.write(record.get("earth_benefits") or "Not specified")
    with tab_two:
        st.subheader("2. Flame Physics & Scenario Explorer")
        materials = _record_materials(records)
        material_names = sorted(
            {item["name"] for item in materials if item.get("name")}
            | {"PMMA", "SIBAL Fabric", "Nomex", "Delrin", "Polycarbonate", "PU Foam", "Cotton/Fiberglass", "Teflon", "Kapton", "n-Dodecane (C12)", "n-Heptane (C7)", "n-Decane (C10)"}
        )
        scenario_left, scenario_right = st.columns([1, 2])
        with scenario_left:
            selected_material_name = st.selectbox("Material / fuel selector", material_names)
            selected_raw = next(
                (item for item in materials if item.get("name") == selected_material_name),
                {"name": selected_material_name},
            )
            selected_material = _material_metrics(selected_raw)
            st.markdown('<div class="telemetry-card">', unsafe_allow_html=True)
            st.metric("LOC", f"{selected_material['loc']:.1f}%")
            st.metric("Flame Spread", f"{selected_material['spread']:.2f} mm/s")
            st.metric("Critical Heat Flux", f"{selected_material['heat_flux']:.1f} kW/m²")
            st.write(f"Parent PSI: {selected_material['psi_id']}")
            st.write("🔥 IGNITABLE" if oxygen >= selected_material["loc"] else "❄️ SELF-EXTINGUISHING")
            if selected_material["fallback"]:
                st.caption("Numeric values use labeled NASA benchmark fallback data.")
            st.markdown("</div>", unsafe_allow_html=True)
        with scenario_right:
            morphology_columns = st.columns(3)
            morphology_columns[0].info("🔥 Earth 1g\n\nTeardrop buoyant flame · 1800–2200 K")
            morphology_columns[1].info("🔵 Microgravity 0g\n\nSpherical diffusion flame · radiative extinction")
            morphology_columns[2].info("❄️ Cool flame\n\n600–800 K low-temperature chemistry")
            airflow_values = list(range(0, 51, 5))
            scenario_rows = []
            for regime, factor in (
                ("0.0g ISS", 0.72),
                ("0.16g Lunar", 0.86),
                ("0.38g Mars", 0.93),
                ("1.0g Earth", 1.0),
            ):
                for airflow_value in airflow_values:
                    scenario_rows.append(
                        {
                            "Material": selected_material["name"],
                            "Gravity": regime,
                            "Airflow (cm/s)": airflow_value,
                            "Flame Spread Rate (mm/s)": round(
                                selected_material["spread"] * factor * (1 + airflow_value / 60),
                                3,
                            ),
                        }
                    )
            scenario_data = pd.DataFrame(scenario_rows)
            chart_columns = st.columns(2)
            with chart_columns[0]:
                spread_chart = px.line(
                    scenario_data,
                    x="Airflow (cm/s)",
                    y="Flame Spread Rate (mm/s)",
                    color="Gravity",
                    markers=True,
                    title="Flame spread vs. ventilation",
                    template="plotly_dark",
                )
                spread_chart.add_vline(x=airflow, line_dash="dash", line_color="#FF6B35")
                st.plotly_chart(spread_chart, use_container_width=True)
            matrix_rows = []
            for item in materials or [{"name": name} for name in material_names]:
                metric = _material_metrics(item)
                matrix_rows.append(
                    {
                        "Material": metric["name"],
                        "LOC (%)": metric["loc"],
                        "Critical Heat Flux (kW/m²)": metric["heat_flux"],
                        "Flame Spread (mm/s)": metric["spread"],
                        "Status": "Flammable" if oxygen >= metric["loc"] else "Safe",
                    }
                )
            with chart_columns[1]:
                matrix_data = pd.DataFrame(matrix_rows).drop_duplicates("Material")
                matrix_chart = px.scatter(
                    matrix_data,
                    x="LOC (%)",
                    y="Critical Heat Flux (kW/m²)",
                    size="Flame Spread (mm/s)",
                    color="Status",
                    hover_name="Material",
                    color_discrete_map={"Safe": "#00E5FF", "Flammable": "#FF6B35"},
                    title="Material flammability matrix",
                    template="plotly_dark",
                )
                matrix_chart.add_vline(x=oxygen, line_dash="dash", line_color="#FF9F1C")
                st.plotly_chart(matrix_chart, use_container_width=True)
            st.download_button(
                "📥 Export Flammability & Airflow Telemetry (CSV)",
                scenario_data.to_csv(index=False).encode("utf-8"),
                "flame_modeling_data.csv",
                "text/csv",
            )
    with tab_three:
        st.subheader("3. Fire Controls & Earth Spinoffs Bridge")
        control_columns = st.columns(3)
        with control_columns[0]:
            st.info("**Stage 1 · SAME-R / PSI-102**\n\n🌪️ Early aerosol and pyrolysis detection. Calibrate particle size and obscuration for Teflon, Kapton, Silicone Rubber, and DBP.")
        with control_columns[1]:
            st.info("**Stage 2 · FLEX / PSI-117 + CFI-G / PSI-159**\n\n💧 Droplet radiative/diffusive extinction, fine water mist, and cool-flame re-ignition control.")
        with control_columns[2]:
            st.info("**Stage 3 · SAFFIRE-1 / PSI-98 + BASS-II**\n\n⚡ Structural qualification, LOC screening, and concurrent vs. opposed flame propagation.")

        spinoff_rows = []
        for record in records:
            for spinoff in _json_items(record.get("earth_benefits", "")):
                if isinstance(spinoff, dict):
                    spinoff_rows.append({
                        "NASA Investigation & Domain": f"{record.get('psi_id', '')} · {spinoff.get('domain', '')}",
                        "Microgravity Innovation": spinoff.get("innovation", ""),
                        "Earth / Commercial Application": spinoff.get("application", spinoff.get("earth_application", "")),
                        "Quantifiable Safety Impact": spinoff.get("impact", ""),
                    })
        st.subheader("Earth Spinoff Technology Transfer")
        st.dataframe(
            pd.DataFrame(spinoff_rows, columns=[
                "NASA Investigation & Domain",
                "Microgravity Innovation",
                "Earth / Commercial Application",
                "Quantifiable Safety Impact",
            ]),
            use_container_width=True,
            hide_index=True,
        )

    with tab_four:
        st.subheader("Semantic RAG Co-Pilot")
        try:
            with st.spinner("Generating PSI document embeddings..."):
                embedded_records = generate_document_embeddings(records, api_key)
        except Exception as error:
            st.error(f"Unable to generate PSI embeddings: {error}")
            embedded_records = []

        if embedded_records:
            st.session_state.setdefault("rag_messages", [])
            quick_columns = st.columns(3)
            quick_prompts = [
                ("🧪 PMMA & Nomex", "Compare PMMA and Nomex flame behavior, SAME-R smoke detection, and Earth spinoffs."),
                ("🛰️ SAFFIRE-1 & SAME", "Explain SIBAL concurrent versus opposed burns and SAME telemetry materials."),
                ("❄️ Cool-Flame Physics", "Explain hot versus cool flame extinction diameters and burning-rate behavior."),
            ]
            for column, (label, prompt_text) in zip(quick_columns, quick_prompts):
                if column.button(label, use_container_width=True):
                    st.session_state.quick_rag_prompt = prompt_text
            for message in st.session_state.rag_messages:
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])

            question = st.chat_input("Ask a question grounded in the PSI investigations")
            if not question:
                question = st.session_state.pop("quick_rag_prompt", None)
            if question:
                st.session_state.rag_messages.append({"role": "user", "content": question})
                with st.chat_message("user"):
                    st.markdown(question)
                try:
                    matches = _retrieve_top_records(
                        question,
                        embedded_records,
                        api_key,
                    )
                    answer = _answer_question(question, matches, api_key)
                except Exception as error:
                    st.error(f"Semantic RAG request failed: {error}")
                else:
                    st.session_state.rag_last_matches = matches
                    st.session_state.rag_messages.append(
                        {"role": "assistant", "content": answer}
                    )
                    with st.chat_message("assistant"):
                        st.markdown(answer)
            latest_matches = st.session_state.get("rag_last_matches", [])
            if latest_matches:
                with st.expander("Diagnostics: View Semantic Retrieval Math"):
                    for hybrid_score, semantic_score, lexical_score, record in latest_matches:
                        st.write(
                            f"**{_record_value(record, 'Title', 'title', 'name')}** · "
                            f"PSI ID: {_record_value(record, 'PSI ID', 'psi_id', 'id')} · "
                            f"Hybrid Score: {hybrid_score * 100:.8f}% · "
                            f"Semantic Cosine: {semantic_score * 100:.8f}% · "
                            f"Keyword Match: {lexical_score * 100:.8f}%"
                        )


if __name__ == "__main__":
    main()
