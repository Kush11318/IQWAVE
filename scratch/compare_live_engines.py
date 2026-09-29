import sys
import os
sys.path.insert(0, os.path.abspath("."))
import json
import glob
import numpy as np
from backend.modules.module3.service import default_amc_service

print(f"{'Signal File':24s} | {'Ground Truth':12s} | {'Engine A (RF)':14s} | {'Engine B (CNN)':14s}")
print("-" * 72)
for p in sorted(glob.glob("sample_signals/*.json")):
    name = os.path.basename(p)
    d = json.load(open(p))
    iq = np.array(d['i']) + 1j * np.array(d['q'])
    res = default_amc_service.classify_signal(iq)
    rf_pred = res['engines']['engine_a_rf']['predicted_class']
    cnn_pred = res['engines']['engine_b_cnn']['predicted_class']
    gt = d['metadata'].get('modulation', 'N/A')
    print(f"{name:24s} | {gt:12s} | {rf_pred:14s} | {cnn_pred:14s}")
