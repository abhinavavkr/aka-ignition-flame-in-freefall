# 🔥 AstroFlame SI (Team AKA Ignition)
**NASA Space Apps Challenge 2026 | Challenge: Flame in Freefall**

![AstroFlame Banner](https://img.shields.io/badge/NASA-Space_Apps_2026-0B3D91?style=for-the-badge&logo=nasa)
![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
![Gemini AI](https://img.shields.io/badge/Google_Gemini_API-8E75B2?style=for-the-badge)

## 🚀 Project Overview
**AstroFlame SI (Scientific Intelligence)** is a zero-cost, interactive web dashboard designed for astronautical engineers and life-support system designers. It summarizes, ranks, and interprets complex microgravity combustion data from NASA's **Physical Sciences Informatics (PSI)** database to deliver actionable fire safety insights for habitable crew environments.

Instead of requiring engineers to manually parse terabytes of raw sensor data, AstroFlame uses deterministic physics-scoring and a lightweight LLM RAG (Retrieval-Augmented Generation) pipeline to map historical space fires to predictive flammability models.

## 🛠️ Core Features

* **🛰️ Mission Environment Ranker:** Select a spacecraft profile (e.g., ISS 21% O2, Artemis Lunar Habitat 34% O2) to instantly re-score and rank NASA PSI investigations (like BASS-II, SAME-R, and FLEX) based on mission relevance.
* **📊 Predictive Flammability Explorer:** Compare how materials like PMMA (Acrylic) and Cotton burn on Earth (1g) vs. Microgravity (0g). Features interactive Plotly charts mapping flame spread rates and soot concentration.
* **🌍 Space-to-Earth Spinoffs:** A dedicated matrix mapping microgravity fire technology (like fine water-mist suppression and ion-wind control) to terrestrial firefighter gear improvements.
* **🤖 SI Co-Pilot & Open Science Exporter:** Ask direct questions to the PSI dataset using our integrated Gemini 2.5 Flash AI assistant, and generate a 1-click downloadable Markdown technical report for engineering reviews.

## 💻 Tech Stack
* **Frontend/Backend:** Python, Streamlit
* **Data Visualization:** Plotly Express, Pandas
* **AI/SI Engine:** Google Gemini 2.5 Flash API (google-genai SDK)
* **Dataset:** Curated metadata JSON derived from 24 NASA PSI Combustion Science investigations.

## ⚙️ How to Run Locally

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/abhinavavkr/aka-ignition-flame-in-freefall.git](https://github.com/abhinavavkr/aka-ignition-flame-in-freefall.git)
   cd aka-ignition-flame-in-freefall
   
