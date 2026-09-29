# IQWAVE — Research Report & Improvement Plan
## Based on 27 Research Papers (PAPERS/ folder)

---

## CURRENT STATE OF MODULE 3 (AMC)

| Metric          | Engine A (Random Forest) | Engine B (1D CNN)   | Fusion            |
|-----------------|--------------------------|---------------------|-------------------|
| Accuracy        | ~69%                     | ~69%                | NONE              |
| Macro-F1        | ~69.06%                  | ~68.63%             | NOT_YET_VALIDATED |
| Features        | 24 handcrafted           | Raw IQ (128x2)      | —                 |
| Weights         | Must be trained          | Must be trained     | —                 |

### Root Causes of Low Accuracy
1. No trained weights exist — every run falls back to statistical heuristics
2. Fusion is completely absent — two engines never combine their evidence
3. Feature gaps — C60/C63 cumulants and cyclostationary features are missing
4. Single CNN with no ensemble — overconfident point predictions
5. Training data is too clean — no multipath fading, no IQ imbalance

---

## LITERATURE FINDINGS (Paper by Paper)

### 1. 2503.04142v2 — Uncertainty Quantification for AMC (UC San Diego, 2025)
What they did: Built a deep ensemble of multiple CNNs trained on RadioML2016.10a and
RadioML2018.01a (same signal types as IQWAVE).

Key findings:
- Equal-weighted ensemble of 3-5 CNNs consistently beats single CNN, BNNs,
  and SNR-aware weighted ensembles
- Ensemble produces wider confidence intervals on WRONG predictions —
  it knows when it doesn't know
- At low SNR (-10 dB): single CNN is overconfident on mistakes;
  ensemble expresses honest uncertainty
- They use the same 4-layer Conv architecture confirming our baseline is valid

Direct lessons for IQWAVE:
- Replace single CNN with 3-model ensemble -> +3-5% accuracy + honest confidence
- Add entropy-based uncertainty: H = -sum(p * log(p)) as confidence metric

---

### 2. Automatic_Modulation_Classification_A_Deep_Architecture_Survey.pdf
   (IEEE Access 2021)
What they surveyed: FFNN, RNN/LSTM, CNN architectures for AMC comprehensively.

Key findings by architecture type:
- FFNN (RF engine equivalent): works with 24-28 HOC + spectral + circular features
  Best reported: 86% at 5 dB SNR on 5 classes with 28 features
- LSTM: captures temporal dependencies better than CNN for low-SNR
- CNN: best overall — especially with residual connections (ResNet-style)
- GAP (Global Average Pooling): our current architecture uses this — CORRECT
- CL-DNN (CNN + LSTM combined): consistently best overall architecture

Direct lessons for IQWAVE:
- Add C60 and C63 cumulants (top discriminators for QAM16 vs QAM64)
- Add circular skewness + kurtosis (top discriminators for FSK sub-types)
- CNN needs residual connections — current 3-block stack suffers at deeper training

---

### 3. ilide.info DL Survey — Signal Representation & Preprocessing
   (IEEE TNNLS 2022)
What they surveyed: 4 signal representation strategies.

Results table:
  Representation               | Best Accuracy | Notes
  HOC features -> DNN          | 86% (5 class) | 28 features, Rayleigh fading
  Constellation image -> CNN   | 91% (5 class) | AlexNet/GoogLeNet
  IQ sequence -> CNN           | 88% (11 class)| RadioML2016.10a
  Combined (features+sequence) | 93%           | BEST overall

Feature importance from survey:
- C40 = most discriminative single cumulant
- C42 = second most important
- C60, C63 = add 3-5% improvement when included
- Circular features (variance, skewness, kurtosis) = critical for FSK vs PSK

Cyclostationary features (SCF): powerful for FSK/PSK but computationally expensive.
Currently NOT in our system.

---

### 4. s41598-026-35558-7.pdf — DBFCNN: Dual-Branch Feature Fusion CNN
   (Nature Scientific Reports, 2026)
What they did: Dual-branch CNN — Branch 1 processes raw sequences,
Branch 2 processes handcrafted features — then fuses them late.

Key findings:
- Multi-scale kernels (k=3, 5, 7) + dilation rates (d=1, 2, 5): +5% over single-scale
- GELU activation outperformed ReLU for ALL sequence tasks
- Late fusion (concat both branch outputs before classifier): +5% over either branch alone
- At BER < 10^-3: achieves 98% accuracy on 7 coding types

Direct lessons for IQWAVE:
- CNN should use multi-scale kernels with dilation
- RF features + CNN features must be fused by a TRAINED layer, not heuristic averaging
- GELU > ReLU for our task

---

### 5. s13638-020-01730-4.pdf — NDA SNR Estimation via ML (EURASIP 2020)
Relevant to: Module 6 (SNR estimation)

Key finding: Their ML estimator outperforms M2M4 method (which we use) by 2-4 dB
at low SNR. Algorithm uses iterative bisection on the log-likelihood function —
computable without any pilot symbols.

---

### 6. 2601.15903v2.pdf — Blind Code Identification: Subspace Approach (2026)
Relevant to: Module 8 (FEC blind identification)

Key finding: Subspace-distance decoder works for ANY code family. Works with
N=5-20 received codewords. Provides THEORETICAL guarantees on error probability.
Currently not in our Module 8 — we only use heuristic FEC detection.

---

### 7. Blind_Interleaver_Recognition_Using_Deep_Learning_Techniques.pdf
Relevant to: Module 9 (interleaver detection)

Key finding: DL-based interleaver recognition achieves higher accuracy than
algebraic rank-deficiency methods at BER > 1%. Our Module 9 uses rank-deficiency
analysis — DL replacement would improve performance under noise.

---

### 8. s44459-025-00013-y.pdf — OneWeb Signal Parameter Estimation
   (npj Wireless Technology, 2025)
Relevant to: Module 4 (symbol rate) and Module 5 (demodulation)

Key finding: Cyclostationary analysis (CAF) is gold standard for blind symbol rate
estimation of real-world SC-QPSK signals. Our Module 4 already uses this — CORRECT.
Also confirms roll-off factor estimation via template matching for Module 5.

---

## THE IMPROVEMENT PLAN

### PHASE 1 — Train Existing Engines Properly
Est. accuracy gain: +9-13%
Problem: No weights exist. Training data is too narrow.

What to change in training:
- Samples per class:  450 -> 2000 (14,000 total signals)
- SNR range:          5-30 dB -> -5 to 30 dB (low-SNR is where models fail)
- Add multipath Rayleigh fading channel model (currently only AWGN)
- Add IQ imbalance simulation (amplitude/phase mismatch between I and Q arms)
- Phase offset randomization: already done
- CFO randomization: already done

Expected result: RF engine ~78%, CNN engine ~75%

---

### PHASE 2 — Expand Feature Set (24 -> 32 features)
Est. accuracy gain: +3-5%
Problem: Missing 8 top-performing features from literature.

New Feature          | Paper Source          | Why It Matters
c60_norm             | IEEE TNNLS 2022       | 6th-order cumulant: QAM64 vs QAM16
c63_norm             | IEEE TNNLS 2022 Eq.3  | Cross-cumulant: PSK vs QAM boundary
circular_skewness    | DL Survey Sec III.C   | FSK vs PSK: GFSK vs 8PSK
circular_kurtosis    | DL Survey Sec III.C   | GFSK vs CPFSK sub-classification
peak_to_avg_power    | Standard              | PAPR: QAM > PSK > FSK
spectral_flatness    | Standard              | FSK is flat; QAM is not
cyclic_freq_peak     | OneWeb paper          | Confidence of symbol rate peak
norm_phase_variance  | DL Survey             | CFO-robust phase spread

Expected result: RF engine ~82%

---

### PHASE 3 — Upgrade CNN Architecture (Multi-Scale + Residual + GELU)
Est. accuracy gain: +5-8%
Problem: Current 3-block vanilla CNN has no residual connections, single kernel size,
uses ReLU only.

Target architecture (based on DBFCNN + Architecture Survey):
  Input (128, 2)
  -> Multi-Scale Block: parallel k=3, k=5, k=7 convolutions -> concat (192 ch)
  -> Dilated Block: parallel d=1, d=2, d=4 convolutions + residual -> (288 ch)
  -> Second Conv Block (256 ch) -> MaxPool
  -> Global Average Pool -> 256-dim vector
  -> Dense(128, GELU) -> Dropout(0.3) -> Dense(7, Softmax)

Key changes from current:
- Residual connections (no vanishing gradient)
- GELU instead of ReLU (better for sequences)
- Multi-scale kernels (captures patterns at different scales)
- Dilated convolutions (wider receptive field without losing resolution)

Expected result: CNN ~83%

---

### PHASE 4 — Deep Ensemble of 3 CNNs
Est. accuracy gain: +3-5%
Problem: Single CNN is overconfident, no uncertainty quantification.

What to build (from 2503.04142v2):
- Train 3 identical CNN architectures with DIFFERENT random seeds
- At inference: average their softmax outputs (equal weight)
- Compute inter-model variance as uncertainty signal
- Confidence levels:
  - HIGH:      entropy < 15%
  - MEDIUM:    entropy 15-40%
  - LOW:       entropy 40-65%
  - UNCERTAIN: entropy > 65%

Expected result: Ensemble ~87%

---

### PHASE 5 — Trained Dual-Branch Fusion (The Goldmine)
Est. accuracy gain: +5-10%
Problem: EvidenceLayer fusion is NOT_YET_VALIDATED. Two engines never truly combine.

What to build (from DBFCNN paper):
  RF features vector (32-dim)       -> Dense(64, GELU) -> rf_branch (64-dim)
  CNN ensemble output (256-dim GAP) -> cnn_branch (256-dim)
  -> Concatenate -> (320-dim)
  -> Dense(128, GELU) -> Dropout(0.3) -> Dense(7, Softmax)

Train this fusion head end-to-end.
Replaces heuristic "0.55*rf + 0.45*cnn" average with a LEARNED combination.

Expected result: Fused accuracy ~91-93%

---

### PHASE 6 — Uncertainty Quantification Output
Est. accuracy gain: 0% (practical value only)

Replace current binary "PREDICTION_SUCCESSFUL / NOT_YET_VALIDATED" with:
  predicted_class: "QPSK"
  confidence: 0.89
  confidence_level: "HIGH"
  uncertainty:
    normalized_entropy: 0.11
    inter_model_variance_max: 0.003

---

## EXPECTED ACCURACY TRAJECTORY

Phase                | Engine A (RF) | Engine B (CNN) | Fused   | Key Change
Current (no weights) |     ~69%      |     ~69%       |  NONE   | Stat fallback
P1: Train properly   |     ~78%      |     ~75%       |   —     | 2000/class, fading
P2: +8 features      |     ~82%      |     ~75%       |   —     | C60, C63, circular HOC
P3: Multi-scale CNN  |     ~82%      |     ~83%       |   —     | Residual+dilated+GELU
P4: CNN Ensemble x3  |     ~82%      |     ~87%       |   —     | Equal-weight averaging
P5: Trained fusion   |     ~82%      |     ~87%       |  ~91%   | Dual-branch fusion head
P6: UQ output        |     ~82%      |     ~87%       |  ~91%   | Calibrated confidence

---

## PER-CLASS DISCRIMINATION GUIDE (From Papers)

Hardest Pairs       | Key Discriminating Features                  | Why Hard
GFSK <-> CPFSK      | circular_variance, phase_entropy, circ_kurt  | Both constant envelope
QAM16 <-> QAM64     | c42_norm, radial_entropy, amp_kurtosis       | Same family, diff M-ary
QPSK <-> 8PSK       | phase_m4_concentration, phase_entropy        | Phase density differs
BPSK <-> QPSK       | phase_m2_concentration, iq_eigen_ratio       | Easy @high SNR, hard @0dB

---

## SUCCESS CRITERIA (When This Is Done)

Metric                   | Target
RF engine accuracy       | >= 80%
CNN ensemble accuracy    | >= 85%
Fused accuracy           | >= 90%
Macro-F1                 | >= 88%
GFSK recall              | >= 75% (currently worst class)
Confidence output        | Calibrated, not hardcoded
Fusion status            | TRAINED_VALIDATED (not NOT_YET_VALIDATED)
