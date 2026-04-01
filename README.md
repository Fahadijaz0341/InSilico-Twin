<div align="center">

<img src="https://img.shields.io/badge/InSilico%20Twin-v2.0-38bdf8?style=for-the-badge&logo=flask&logoColor=white" alt="InSilico Twin v2.0"/>

# 🧬 InSilico Twin
### Multi-Scale Virtual Clinical Trial · Quantitative Systems Pharmacology Dashboard

> **A 3-layer biosimulation engine for in-silico drug trials — powered by ODE-based mechanistic models, synthetic patient generation, and real-time interactive analytics.**

[![Python](https://img.shields.io/badge/Python-3.9%2B-3776AB?style=flat-square&logo=python&logoColor=white)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.32-FF4B4B?style=flat-square&logo=streamlit&logoColor=white)](https://streamlit.io)
[![Tellurium](https://img.shields.io/badge/Tellurium-2.2-00C48C?style=flat-square)](https://tellurium.readthedocs.io)
[![SDV](https://img.shields.io/badge/SDV-GaussianCopula-c084fc?style=flat-square)](https://sdv.dev)
[![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)](LICENSE)

<img src="https://img.shields.io/badge/Author-Fahad%20Ijaz-818cf8?style=for-the-badge" alt="Author: Fahad Ijaz"/>

---

</div>
TRY > https://insilico-twin.streamlit.app/
## 📋 Table of Contents

- [Overview](#overview)
- [Architecture — 3-Layer Pipeline](#architecture--3-layer-pipeline)
- [Features](#features)
- [Tech Stack](#tech-stack)
- [Getting Started](#getting-started)
- [How to Run a Trial](#how-to-run-a-trial)
- [Model References](#model-references)
- [Project Structure](#project-structure)
- [Author](#author)

---

## 🔬 Overview

**InSilico Twin** is a Quantitative Systems Pharmacology (QSP) platform that simulates virtual clinical trials for diabetes drug candidates — entirely in silico, without a single real patient.

It addresses a fundamental bottleneck in drug development: **the cost and time of clinical trials**. By combining mechanistic ODE models with synthetic data generation and statistical analysis, InSilico Twin lets researchers rapidly explore how a drug candidate might perform across a heterogeneous patient population.

### What it does end-to-end:
1. **Generates** a synthetic cohort of hundreds of virtual diabetic patients using Gaussian Copula synthesis trained on real Pima Indian Diabetes data
2. **Simulates** each patient through a 3-layer mechanistic cascade — from molecular drug effects, to daily glucose dynamics, to long-term organ survival
3. **Analyses** the trial results with publication-grade statistics and interactive visualizations

---

## 🏗️ Architecture — 3-Layer Pipeline

```
 ┌─────────────────────────────────────────────────────────────────┐
 │                   InSilico Twin — 3-Layer QSP Pipeline           │
 └─────────────────────────────────────────────────────────────────┘

   INPUT: Virtual Patient (Age, BMI, Glucose, Group)
      │
      ▼
 ┌─────────────────────────────────────────────┐
 │  LAYER 1 — Molecular  (Insulin Signaling)   │
 │  ODE: IRS1/Akt phosphorylation cascade      │
 │  Ref: BIOMD0000000356                       │
 │  Output: Sensitivity Multiplier             │
 └──────────────────────┬──────────────────────┘
                        │  sensitivity_multiplier
                        ▼
 ┌─────────────────────────────────────────────┐
 │  LAYER 2 — Physiological  (Glucose-Insulin) │
 │  ODE: Dalla Man glucose-insulin dynamics    │
 │  Ref: BIOMD0000000379                       │
 │  Output: 24-hr avg glucose, hypo events     │
 └──────────────────────┬──────────────────────┘
                        │  avg_glucose
                        ▼
 ┌─────────────────────────────────────────────┐
 │  LAYER 3 — Prognostic  (Beta-Cell Survival) │
 │  ODE: Beta-cell mass dynamics               │
 │  Ref: BIOMD0000000341                       │
 │  Output: 5-yr beta-cell mass, failure time  │
 └─────────────────────────────────────────────┘
```

Each layer's output feeds directly as a parameter into the next — creating a **mechanistic handshake** that propagates drug effects across biological scales.

---

## ✨ Features

| Feature | Description |
|---|---|
| 🏥 **Virtual Trial Runner** | Simulate 100–1,000 patients in Control vs. Treatment arms |
| 🧬 **3-Layer ODE Engine** | Mechanistic models from the BioMD repository via Tellurium/libRoadRunner |
| 🤖 **Synthetic Cohort** | SDV `GaussianCopulaSynthesizer` trained on Pima Diabetes dataset |
| 📊 **Interactive Analytics** | Plotly scatter, histogram, and bubble charts with hover tooltips |
| 📈 **Statistical Summary** | Auto-computed t-tests, Cohen's d, and group mean comparisons |
| 🌑 **Dark Mode UI** | Professional "Midnight Pro" dark theme with glassmorphism-style cards |
| 📸 **Static Export** | Matplotlib publication-grade dashboard exported as PNG |
| ⚡ **Live Simulation** | Real-time single-patient biosimulation as sidebar sliders change |
| 💊 **Drug Potency Slider** | Adjust treatment-arm sensitivity multiplier and re-run instantly |

---

## 🛠️ Tech Stack

| Component | Technology |
|---|---|
| **UI Framework** | Streamlit 1.32 |
| **ODE Simulation** | Tellurium 2.2 + libRoadRunner |
| **Synthetic Data** | SDV GaussianCopulaSynthesizer |
| **Interactive Plots** | Plotly 5.19 |
| **Static Plots** | Matplotlib 3.8 |
| **Numerics** | NumPy 1.26, SciPy 1.11, Pandas 2.2 |
| **Model Language** | Antimony (SBML-compatible) |

---

## 🚀 Getting Started

### Prerequisites

- Python 3.9 or higher
- `pip` package manager

### 1. Clone the repository

```bash
git clone https://github.com/FahadIjaz/InSilico-Twin.git
cd InSilico-Twin
```

### 2. Create a virtual environment (recommended)

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate
```

### 3. Install dependencies

> ⚠️ **Important:** NumPy must be pinned below 2.0 for compatibility with Tellurium/libRoadRunner.

```bash
pip install -r requirements.txt
```

### 4. Launch the app

```bash
streamlit run app.py
```

The dashboard will open in your browser at `http://localhost:8501`.

---

## 🧪 How to Run a Trial

1. **Upload Data** — In the sidebar, upload a `diabetes.csv` file (Pima Indian Diabetes dataset format: `Glucose`, `BMI`, `Age`, and other standard columns).
2. **Set Parameters:**
   - **Cohort Size** — Choose how many virtual patients to simulate (100–1,000)
   - **Drug Potency** — Set the sensitivity multiplier for the treatment arm (1×–10×)
3. **Live Simulation** — Adjust BMI and drug toggle in the sidebar to see real-time single-patient traces update instantly across all 3 layers.
4. **Run Full Trial** — Click **🚀 Run Full Trial** to launch the complete multi-patient pipeline.
5. **Explore Results:**
   - Trial summary metrics (mean glucose, beta-cell mass, failure time)
   - Distribution histograms (Control vs. Treatment overlay)
   - BMI vs. Survival scatter with trend lines
   - Interactive bubble chart (bubble size = beta-cell mass)
   - Statistical table (t-statistic, p-value, Cohen's d)
   - Downloadable results CSV
   - Exportable static Matplotlib dashboard PNG

---

## 📚 Model References

| Layer | BioMD ID | Description |
|---|---|---|
| Molecular | `BIOMD0000000356` | IRS1/Akt insulin signaling cascade |
| Physiological | `BIOMD0000000379` | Dalla Man glucose-insulin dynamics model |
| Prognostic | `BIOMD0000000341` | Beta-cell mass dynamics and glucotoxicity |

All models are encoded in **Antimony** notation and simulated via the **libRoadRunner** SBML solver.

---

## 📁 Project Structure

```
InSilico-Twin/
├── app.py              # Main Streamlit application (all simulation + UI logic)
├── requirements.txt    # Python dependencies (pinned versions)
└── README.md           # This file
```

> The entire platform is self-contained in `app.py`. The architecture is modular inside the file: model builders → simulation functions → SDV synthesizer → figure builders → Streamlit UI.

---

## 👤 Author

<div align="center">

**Fahad Ijaz**

[![GitHub](https://img.shields.io/badge/GitHub-FahadIjaz-181717?style=for-the-badge&logo=github)](https://github.com/FahadIjaz)

*Computational Biology · Drug Discovery · Systems Pharmacology*

---

*Built with ❤️ using Python, Streamlit & Tellurium*

</div>
