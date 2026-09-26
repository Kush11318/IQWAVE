# IQWAVE — Automated Blind Signal Analysis & Demodulation Platform
**Smart India Hackathon (SIH26147)**

IQWAVE is an end-to-end laboratory-grade Blind Signal Analysis System designed to ingest raw canonical In-Phase / Quadrature ($I/Q$) radio frequency (RF) streams without prior protocol knowledge and automatically extract key signal parameters, classify modulations, synchronize carrier and timing loops, demodulate bitstreams, extract soft log-likelihood ratios (LLRs), blind-decode forward error correction (FEC) codes, and recover frame and CRC structures.

---

## 🌟 Key Features

1. **High-Contrast Laboratory Instrument UI**:
   - Modern light-mode scientific design with one-click theme switching to dark mode.
   - Live telemetry chips: Center Frequency, Sample Rate, Occupied Bandwidth, Dynamic Range SNR, and SINAD.
   - Smooth oscilloscope phosphor beam sweep, spectral equalizer rise, and Costas Loop constellation convergence animations.

2. **Full Sequential DSP Pipeline**:
   - **01. Canonical I/Q Ingestion**: Ingests `.json`, `.iq`, `.wav`, and `.bin` formats with power normalization and DC offset rejection.
   - **02. Feature Extraction**: Extracts higher-order cyclic moments, spectral symmetry, and kurtosis.
   - **03. Automatic Modulation Recognition (AMC)**: Hybrid Random Forest + Deep CNN classification (QPSK, BPSK, 8-PSK, 16-QAM, 64-QAM, GFSK, CPFSK).
   - **04. Parameter Estimation**: Cyclic Ciblat symbol rate estimation, carrier frequency offset (CFO) detection, and occupied bandwidth (OBW).
   - **05. Carrier & Timing Recovery**: Dual-stage Gardner timing recovery detector and 4th-power Costas PLL for carrier phase lock.
   - **06. SNR & Channel Quality**: M2M4 amplitude estimation cross-validated with Error Vector Magnitude (EVM).
   - **07. Soft-Decision Demapping**: Exact Gaussian Log-Likelihood Ratios (LLRs) computed per bit.
   - **08. Blind FEC Code Recognition**: Syndrome weight minimization over Hamming, BCH, and Convolutional codebooks.
   - **09. Frame Synchronization**: GF(2) linear algebraic periodicity search and frame boundary phase determination.
   - **10. Protocol Integrity**: Fast polynomial division over standard 16-bit and 32-bit CRC candidates, plus matrix interleaver estimation.

3. **Multi-Format Engineering Reporting**:
   - 1-click export of complete scientific audit reports in HTML, Markdown, and JSON.

---

## 📂 Repository Structure

```
├── backend/
│   ├── app/
│   │   ├── api/             # FastAPI routes for modules and pipeline
│   │   ├── pipeline/        # End-to-end pipeline orchestrator
│   │   └── reporting/       # Multi-format report generators
│   ├── modules/             # DSP algorithm implementations (1 to 10)
│   └── tests/               # 258 automated pytest integration & unit tests
├── frontend/
│   ├── index.html           # Laboratory instrument interface
│   ├── style.css            # High-contrast instrument design system
│   └── app.js               # Reactive state manager, animations & canvas renderers
├── presentation_assets/     # Academic graphs & confusion matrices for slides
├── sample_signals/          # High-fidelity mock RF vectors (QPSK, 16-QAM, 8-PSK, BPSK, GFSK)
├── .gitignore
└── README.md
```

---

## 🚀 Getting Started

### 1. Requirements
- Python 3.10+
- Modern Web Browser (Chrome, Firefox, Edge, Safari)

### 2. Setup
```bash
# Clone the repository
git clone https://github.com/Kush11318/IQWAVE.git
cd IQWAVE

# Install dependencies
pip install fastapi uvicorn numpy scipy scikit-learn pytest
```

### 3. Run the Platform
```bash
python -m uvicorn backend.app.main:app --host 127.0.0.1 --port 8000
```
Open your browser and navigate to:
**`http://127.0.0.1:8000/`**

---

## 🧪 Testing & Verification

Run the full automated test suite:
```bash
python -m pytest backend/tests/
```
**Results**: `258 passed, 0 failed (100% passing)`.

---

## 👥 Authors & License
Developed for Smart India Hackathon (SIH26147). Licensed under the MIT License.
