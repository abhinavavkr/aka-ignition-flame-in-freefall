# **Architectural Specification: NASA PSI Fire Safety Intelligence Dashboard**

This document serves as the master specification and prompt guide for AI coding assistants to build the **NASA Physical Sciences Informatics (PSI) Fire Safety Intelligence Dashboard** in Python and Streamlit.

---

# **1\. Project Overview & Scope Guardrails**

## **1.1 Mission Statement**

The NASA PSI Fire Safety Intelligence Dashboard provides astronautical engineers and life support system designers with interactive risk modeling, microgravity flammability analytics, and LLM-powered synthesis of spaceflight fire safety data derived from NASA's Physical Sciences Informatics system.

## **1.2 Strict Technical Guardrails**

* **Zero Local ML Training/Inference Heavyweight Libraries:** Do not use `torch`, `tensorflow`, `scikit-learn`, or heavy statistical engines. All analytical models must use deterministic physics-based scoring algorithms or pre-computed lookup tables.  
* **No Mock or Random Data:** All visualizations, tables, and scoring models must draw directly from the static dataset (`psi_data.json`).  
* **Single External AI Integration:** All generative capabilities, contextual Q\&A, and report drafting must exclusively use the `google-genai` SDK using the Gemini API model (`gemini-2.5-flash`).  
* **RAG Context Constraint:** The Gemini API must receive the structured contents of `psi_data.json` alongside user-configured dashboard states to form ground-truth context for all queries.

---

# **2\. File Architecture**

The repository must strictly adhere to the following file layout:

```
nasa-psi-dashboard/
├── .venv/                      # Isolated Python Virtual Environment
├── .env                        # Environment variables (GEMINI_API_KEY)
├── requirements.txt            # Explicitly pinned dependencies
├── psi_data.json               # Canonical NASA PSI Ground-Truth Dataset
├── development.md              # AI Coding Assistant log and implementation guide
└── app.py                      # Main Streamlit Application entrypoint
```

## **2.1 File Content Requirements**

### **`requirements.txt`**

```
streamlit>=1.31.0
numpy>=1.24.0
pandas>=2.1.0
plotly>=5.18.0
google-genai>=0.1.0
```

### **`psi_data.json` Schema Structure**

Ensure `psi_data.json` contains structured objects for:

* **`investigations`**: Metadata for core experiments (e.g., FLEX, SAME, BASS, Burning Rate Emulator).  
* **`materials`**: Specific NASA test sample data (e.g., PMMA, Delrin, Nomex, Cotton/Fiberglass blends) containing LOC (Limiting Oxygen Concentration), flame spread velocities (\$mm/s\$), and critical radiant heat flux.  
* **`spinoffs`**: Terrestrial engineering applications derived from microgravity fire research.

---

# **3\. Detailed Technical Specifications (`app.py`)**

## **3.1 Global UI Configuration & Session State**

```py
import streamlit as st
import pandas as pd
import plotly.express as px
import json
import os
from google import genai
from google.genai import types

# Page Config
st.set_page_config(
    page_title="NASA PSI Fire Safety Intelligence",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)
```

### **Session State Initialization**

The app must initialize the following keys if not present:

* `chat_history`: List of dicts `[{"role": "user"|"assistant", "content": str}]`.  
* `generated_report`: String tracking the current generated markdown report.  
* `filter_state`: Container storing current sidebar slider configurations.

---

## **3.2 Left Sidebar: Mission Control Panel**

The sidebar configures environmental parameters for spacecraft modules or extraterrestrial habitats.

### **UI Controls**

1. **Target Environment Preset (Dropdown):**  
   * *Options:* "ISS Standard (1 atm, 21% O2)", "Lunar Habitat (0.56 atm, 32% O2)", "Martian Habitat (0.38 atm, 26% O2)", "Exploration EVA Suit (0.3, 100% O2)", "Custom Spacecraft Profile".  
2. **Gravity Level (\$g\$) (Slider):** Range: `0.00` to `1.00`, Default: `0.00` (\$10^{-6}\\text{ g}\$ Microgravity), Step: `0.01`.  
3. **Oxygen Concentration (\$%\\text{ O}\_2\$) (Slider):** Range: `10.0` to `100.0`, Default: `21.0`, Step: `0.5`.  
4. **Total Environmental Pressure (\$\\text{kPa}\$) (Slider):** Range: `20.0` to `101.3`, Default: `101.3`, Step: `1.0`.  
5. **Forced Ventilation Airflow (\$\\text{m/s}\$) (Slider):** Range: `0.00` to `1.00`, Default: `0.15`, Step: `0.01`.

### **Preset Auto-Fill Logic**

Selecting a preset must automatically update the sliders for Gravity, Oxygen, and Pressure via callback functions.

---

## **3.3 Top Metric Summary Strip**

Positioned directly below the header banner, displaying 4 primary metrics responsive to the sidebar parameters:

| Metric Card | Computation / Logic | Display Format |
| :---- | :---- | :---- |
| **Active Target Regime** | Selected Preset Name | String Label |
| **Calculated \$\\text{ppO}\_2\$** | \$\\text{Pressure (kPa)} \\times (\\text{O}\_2% / 100)\$ | `XX.X kPa` |
| **Flammability Risk Index** | Base score calculated from \$\\text{ppO}\_2\$ & Forced Airflow | `Low` / `Moderate` / `Severe` / `Critical` |
| **PSI Corpus Reference Count** | Total records loaded from `psi_data.json` | `N Investigations` |

---

## **3.4 Tab 1: Habitable Crew Environment Relevance Ranker**

### **Scoring Algorithm**

Each material and experiment in `psi_data.json` must be evaluated against active sidebar parameters using a deterministic Indexing Formula:

\$\$\\text{Risk Score} \= \\left( \\frac{%\\text{ O}*2}{\\text{LOC}*{\\text{material}}} \\right) \\times \\left( 1 \+ 0.5 \\times \\text{Airflow} \\right) \\times \\left( \\frac{101.3}{\\text{Pressure}} \\right)^{0.3}\$\$

Where \$\\text{LOC}\_{\\text{material}}\$ is the Limiting Oxygen Concentration of the given material.

### **UI Components**

* **Dynamic Material Ranking Table:** Formatted displaying Material Name, NASA ID, Base LOC, Calculated Risk Score, and Flammability Status (Safe / Conditional / High Hazard).  
* **Expandable Details Cards (`st.expander`):**  
  * Display detailed investigation metadata: Experiment Title, Principal Investigator, Mission Flight (e.g., Space Shuttle STS-75, ISS Expedition 18-60), and Key Technical Findings.

---

## **3.5 Tab 2: Predictive Flammability & Scenario Explorer**

### **UI Layout & Functionality**

1. **Material Selector:** Radio buttons or drop-down to choose a specific test material (e.g., "Polymethyl Methacrylate (PMMA)", "Flexible Polyurethane Foam").  
2. **Side-by-Side Comparative Analysis Columns (`st.columns(2)`):**  
   * **Column A (Earth 1g Baseline):** Explains buoyant thermal plume dynamics, high convective cooling, upward propagation characteristics.  
   * **Column B (Selected Microgravity Environment):** Explains absence of natural convection, spherical flame front propagation, radiatively driven flame extinction limits (EXTINCTION/FLEX findings).  
3. **Interactive Plotly Chart:**  
   * **Type:** Scatter/Line chart mapping **Flame Spread Rate (\$\\text{mm/s}\$)** vs. **Forced Airflow (\$\\text{m/s}\$)** across three different Gravity levels (\$0.0\\text{ g}\$, \$0.38\\text{ g}\$, \$1.0\\text{ g}\$).  
   * Render using custom NASA color palette: Deep Blue (`#0B3D91`), Flame Amber (`#FC3D21`), and Cool Grey (`#E1E6EB`).  
4. **Data Export:** Button to download the generated multi-scenario comparison table as a `.csv` file.

---

## **3.6 Tab 3: Next-Gen Fire Controls & Earth Spinoff Bridge**

### **UI Layout & Structure**

* **Sub-Section A: Flight-Certified Fire Safety Framework (Grid Layout):**  
  * **Detection Systems:** Multi-wavelength optical flame sensors vs. smoke particulate counters in zero-convection airflow.  
  * **Suppression Technologies:** Transition from Halon 1301 to Water Mist / Nitrogen deluge systems on ISS.  
  * **Electrical Arc Control:** Circuit topology design to prevent microgravity thermal runaway.  
* **Sub-Section B: Earth Spinoff Technology Transfer Table:**

| NASA Research Domain | Spaceflight Innovation | Terrestrial Earth Spinoff Application | Commercial Impact Metric |
| :---- | :---- | :---- | :---- |
| **FLEX / Flame Extinction** | Low-flow gaseous fire suppression | Commercial Aircraft Cargo Fire Suppression | 40% reduction in agent toxicity |
| **BASS Material Testing** | Microgravity flammability testing | Mining Safety & Subsurface Enclosure Standards | Adopted by ISO/NFPA standards |
| **Particle Micro-sensors** | Zero-convection smoke detection | High-tech Cleanroom & Data Center Monitoring | Early detection enhancement (+35%) |

---

## **3.7 Tab 4: PSI Co-Pilot & 1-Click Open Science Report Generator**

### **Gemini API Integration (`google-genai` SDK)**

```py
def query_gemini_copilot(prompt: str, json_context: dict, env_state: dict) -> str:
    """Queries the Gemini API using the new google-genai SDK with grounding context."""
    client = genai.Client() # Automatically picks up GEMINI_API_KEY from environment
    
    system_instruction = (
        "You are the NASA PSI Fire Safety AI Specialist. You provide precise, highly technical answers "
        "grounded EXCLUSIVELY in the provided NASA Physical Sciences Informatics dataset and current "
        "environmental parameters. Do not hallucinate data outside the provided context."
    )
    
    context_str = f"""
    --- NASA PSI GROUND TRUTH DATA ---
    {json.dumps(json_context, indent=2)}
    
    --- CURRENT HABITAT ENVIRONMENTAL PARAMETERS ---
    {json.dumps(env_state, indent=2)}
    """
    
    full_prompt = f"{context_str}\n\nUSER QUESTION: {prompt}"
    
    response = client.models.generate_content(
        model='gemini-2.5-flash',
        contents=full_prompt,
        config=types.GenerateContentConfig(
            system_instruction=system_instruction,
            temperature=0.2,
        )
    )
    return response.text
```

### **Dual-Mode Interface**

#### **Mode A: Conversational Co-Pilot Chat**

* Interactive `st.chat_input` sending prompts to the Gemini model.  
* Message history stored in `st.session_state.chat_history` and rendered using `st.chat_message`.

#### **Mode B: 1-Click Open Science Report Generator**

* **Trigger:** Button labeled **"Generate Flight Safety Hazard Assessment Report"**.  
* **Action:** Compiles current sidebar settings, top material risk scores from Tab 1, and comparative limits from Tab 2 into a structured prompt sent to Gemini.  
* **Output:** Generates a complete, formal NASA Technical Memorandum format markdown document.  
* **Export:** `st.download_button` allowing users to save the output as `NASA_Fire_Safety_Assessment.md`.

---

# **4\. Execution Protocol for AI Coding Assistant**

When executing this build:

1. Verify `requirements.txt` installs correctly without dependency conflicts.  
2. Ensure `app.py` runs cleanly via `streamlit run app.py`.  
3. Validate that changing sidebar sliders immediately updates the **Top Metric Summary Strip**, **Tab 1 Risk Scores**, and **Tab 2 Plotly Graphs**.  
4. Confirm API calls use the `google-genai` library syntax (`from google import genai`) and handle missing API keys gracefully with clear `st.warning` messages.

