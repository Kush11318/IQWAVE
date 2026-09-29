<div align="center">

# 📡 DAWC
### Digital Automated Waveform Classifier & Autonomous Protocol Recovery (SIH26147)

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100%2B-009688.svg?style=for-the-badge&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![Pytest Suite](https://img.shields.io/badge/Tests-258%2F258%20Passing-success.svg?style=for-the-badge&logo=pytest&logoColor=white)](https://pytest.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=for-the-badge)](https://opensource.org/licenses/MIT)
[![Interface: Light/Dark](https://img.shields.io/badge/UI-Laboratory%20Instrument-0284c7.svg?style=for-the-badge)](#interactive-laboratory-interface)

<p align="center">
  <b>An automated, zero-prior-knowledge RF signal intelligence and physical-layer protocol recovery framework.</b><br>
  Ingests raw In-Phase / Quadrature ($I/Q$) streams, estimates carrier and spectral parameters, classifies modulations, recovers symbol synchronization, computes exact bit log-likelihood ratios (LLR), blind-identifies FEC codes, and decodes underlying frame and CRC structures.
</p>

[Key Features](#-key-features) •
[Architecture](#-system-architecture) •
[DSP Pipeline](#-sequential-dsp-pipeline) •
[Getting Started](#-getting-started) •
[API Reference](#-rest-api-reference) •
[Testing](#-verification--tests)

---

</div>

## 🌟 Key Features

- **Zero-Prior-Knowledge Blind Recovery**: Operates directly on unlabelled $I/Q$ recordings without transmission metadata.
- **Laboratory Instrument Interface**:
  - High-contrast, sober academic aesthetic with 1-click Light / Dark mode switching.
  - Real-time digital storage oscilloscope (DSO) with animated phosphor beam sweep.
  - Real FFT Power Spectral Density (PSD) with **Blackman-Harris 4-term windowing**, dynamic Occupied Bandwidth (OBW) shading, and floating peak-tracking HUD capsules.
  - Interactive constellation visualizers illustrating Gardner timing and Costas Loop carrier phase convergence.
- **Scientifically Rigorous DSP**:
  - Cyclic autocorrelation & Ciblat estimators for non-data-aided symbol rate detection.
  - 4th-power Costas Phase-Locked Loop (PLL) & Gardner Timing Error Detector (TED).
  - Split-moment $M_2M_4$ SNR estimator cross-validated against Error Vector Magnitude (EVM).
  - Exact Gaussian soft-decision bit Log-Likelihood Ratio (LLR) mapping.
  - Linear algebraic GF(2) parity matrix solver for frame periodicity and blind FEC syndrome recovery.
- **Multi-Format Audit Reporting**: One-click generation of authoritative engineering audit reports in **HTML**, **Markdown**, and **JSON**.

---

## 🏛️ System Architecture

```
                    ┌──────────────────────────────────────────────┐
                    │            Raw RF I/Q Stream Ingestion       │
                    │       (.json, .iq, .wav, .bin, .dat)         │
                    └──────────────────────┬───────────────────────┘
                                           │
 ┌─────────────────────────────────────────┴─────────────────────────────────────────┐
 │                                 SEQUENTIAL DSP CORE                               │
 │                                                                                   │
 │   [01] Signal Validation    ──▶   [02] Higher-Order Moments & Symmetry Gating      │
 │              │                                      │                             │
 │              ▼                                      ▼                             │
 │   [03] Modulation AMC       ──▶   [04] Ciblat Baud Rate, CFO & Bandwidth Estim.   │
 │   (RF Forest + CNN)                                 │                             │
 │              │                                      ▼                             │
 │              ▼                    [05] Gardner & Costas Carrier/Timing Lock       │
 │   [06] M2M4 & EVM SNR       ◀──                     │                             │
 │              │                                      ▼                             │
 │              ▼                    [07] Exact Gaussian Soft-Decision LLRs          │
 │   [08] Blind FEC Decoder    ◀──                     │                             │
 │   (Hamming, BCH, Conv)                              ▼                             │
 │              │                    [09] GF(2) Parity Periodicity & Frame Search    │
 │              ▼                                      │                             │
 │   [10] CRC-16/32 Division & Matrix Interleaver De-interleaving                    │
 └─────────────────────────────────────────┬─────────────────────────────────────────┘
                                           │
                    ┌──────────────────────┴───────────────────────┐
                    │         Authoritative Engineering Report     │
                    │      (Interactive Web UI / HTML / JSON)      │
                    └──────────────────────────────────────────────┘
```

---

## 🔬 Sequential DSP Pipeline

The core analysis engine executes ten decoupled, mathematically rigorous processing stages:

| Stage | Name | Theoretical Foundation | Output Metrics |
| :---: | :--- | :--- | :--- |
| **01** | **Canonical I/Q Ingestion** | Power normalization & DC-offset rejection ($P_{\text{avg}} = 1.0$) | Unit-variance normalized $I/Q$ vector, sample validation |
| **02** | **Signal Observation** | 24 higher-order cumulants ($C_{20}, C_{21}, C_{40}, C_{42}$), spectral symmetry | Kurtosis, phase variance, circularity coefficient |
| **03** | **Modulation AMC** | Hybrid Random Forest feature classification + Deep Convolutional Network | Predicted class (QPSK, BPSK, 8-PSK, 16-QAM, 64-QAM, GFSK) |
| **04** | **Parameter Estimation** | Cyclic autocorrelation (Ciblat algorithm), FFT spectral centroiding | Symbol rate ($R_s$), samples/symbol (SPS), CFO (Hz), OBW |
| **05** | **Carrier & Timing Sync** | Dual-stage Gardner Timing Error Detector + 4th-power Costas PLL | Compensated CFO, phase trajectory, synchronized symbols |
| **06** | **SNR & Quality** | Split-moment $M_2M_4$ amplitude ratio cross-validated against EVM | Calibrated SNR (dB), EVM (%), SINAD estimate |
| **07** | **Soft-Decision Demapping** | Exact closed-form Gaussian Log-Likelihood Ratio ($\text{LLR}$) calculation | Bit-level LLR stream, hard-decision agreement percentage |
| **08** | **Blind FEC Recognition** | Parity-check syndrome weight minimization over candidate codebooks | Top FEC code (Hamming, BCH, Convolutional), rate estimate |
| **09** | **Frame Structure** | Linear algebraic GF(2) null-space rank search & auto-correlation | Frame period $P$ (bits), boundary offset $\phi^*$, exact relations |
| **10** | **CRC & Interleaving** | GF(2) polynomial division over CRC-16/32 standards, rank-profile de-interleaver | CRC polynomial candidate, acceptance rate, interleaver depth |

---

## 💻 Interactive Laboratory Interface

The user interface is engineered with high visual standards suitable for scientific demonstration and RF lab work:

```
┌──────────────────────────────────────────────────────────────────────────────────┐
│  📡 IQWAVE  [✓ DSP ENGINE: OPERATIONAL]  [✓ 258/258 Tests Passed]     [🌙 Dark Mode]│
├──────────────────────────────────────────────────────────────────────────────────┤
│  [1] PRESET SELECTOR: Tactical VHF QPSK Carrier (142.850 MHz, SNR 24 dB, 1200 Sa) │
│  [ Fs: 1.0 MSa/s ]  [ Fc: 142.85 MHz ]  [ Mod: AUTO ]   [ ⚡ RUN COMPLETE ANALYSIS ]│
├──────────────────────────────────────────────────────────────────────────────────┤
│  ┌───────────────────────────────┐      ┌───────────────────────────────┐        │
│  │ 📈 Time-Domain Waveform       │      │ 📊 Power Spectral Density     │        │
│  │ In-Phase & Quadrature trace   │      │ Blackman-Harris, OBW, HUD     │        │
│  └───────────────────────────────┘      └───────────────────────────────┘        │
│  ┌───────────────────────────────┐      ┌───────────────────────────────┐        │
│  │ 🎯 Raw Signal Constellation   │      │ ✨ Synchronized Constellation │        │
│  │ Dispersed unsynchronized I/Q  │      │ Costas & Gardner locked grid  │        │
│  └───────────────────────────────┘      └───────────────────────────────┘        │
├──────────────────────────────────────────────────────────────────────────────────┤
│  01 INGESTION  02 FEATURES  03 AMC  04 PARAMETERS  05 SYNC  06 SNR  07 LLR ...   │
├──────────────────────────────────────────────────────────────────────────────────┤
│  [📋 Download HTML Report]    [Download Markdown Report]    [Download Raw JSON]  │
└──────────────────────────────────────────────────────────────────────────────────┘
```

- **Phosphor Beam Oscilloscope**: Dynamic left-to-right sweep with realistic beam luminescence.
- **Spectrum Equalizer Rise**: Responsive spectral power trace rising smoothly from the $-80\text{ dBFS}$ floor.
- **Costas Lock Convergence**: Animated constellation cluster convergence illustrating loop acquisition.

---

## 🚀 Getting Started

### Prerequisites
- **Python 3.10+** (Tested on Python 3.10, 3.11, 3.12)
- Modern web browser (Chrome, Edge, Firefox, Safari)

### Installation

```bash
# 1. Clone the repository
git clone https://github.com/Kush11318/IQWAVE.git
cd IQWAVE

# 2. (Optional) Create and activate a virtual environment
python -m venv .venv
# On Windows:
.venv\Scripts\activate
# On Linux/macOS:
source .venv/bin/activate

# 3. Install dependencies
pip install fastapi uvicorn numpy scipy scikit-learn pytest
```

### Running the Application

Launch the platform with the Uvicorn ASGI server:
```bash
python -m uvicorn backend.app.main:app --host 127.0.0.1 --port 8000
```
Navigate to your local browser:
**`http://127.0.0.1:8000/`**

---

## 📡 REST API Reference

The backend exposes clean, OpenAPI-compliant endpoints:

| Endpoint | Method | Description |
| :--- | :---: | :--- |
| `/api/pipeline/run` | `POST` | Execute end-to-end signal analysis from raw $I/Q$ arrays |
| `/api/pipeline/upload` | `POST` | Ingest external files (`.json`, `.iq`, `.wav`, `.bin`) and run pipeline |
| `/api/pipeline/fixture` | `GET` | Fetch calibrated high-fidelity signal presets (`qpsk`, `16qam`, `8psk`, `bpsk`, `gfsk`) |
| `/api/pipeline/report` | `POST` | Generate formatted engineering audit reports (`html`, `markdown`, `json`) |
| `/api/pipeline/status` | `GET` | Query pipeline architecture and engine availability |

---

## 🧪 Verification & Tests

The DSP algorithms and integration pipelines are validated by an extensive test suite:

```bash
python -m pytest backend/tests/ -v
```

```
============================= test session starts =============================
collected 258 items

backend/tests/test_api.py .............................                  [ 11%]
backend/tests/test_integration_pipeline.py ..............                [ 16%]
backend/tests/test_module1.py .............                              [ 21%]
backend/tests/test_module10.py ....................                      [ 29%]
backend/tests/test_module2.py ..............                             [ 34%]
backend/tests/test_module3.py .........                                  [ 38%]
backend/tests/test_module4.py .................................          [ 51%]
backend/tests/test_module5.py ........................................   [ 66%]
backend/tests/test_module6.py .......................                    [ 75%]
backend/tests/test_module7.py .......................                    [ 84%]
backend/tests/test_module8.py ......................                     [ 93%]
backend/tests/test_module9.py ..................                         [100%]

======================= 258 passed, 0 failed in 12.62s ========================
```

---

## 📄 License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.
