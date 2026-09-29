import sys
import os
sys.path.insert(0, os.path.abspath("."))
import json
import glob
import numpy as np
from backend.modules.module3.feature_extractor import extract_24_features

for p in sorted(glob.glob("sample_signals/*.json")):
    name = os.path.basename(p)
    d = json.load(open(p))
    iq = np.array(d['i']) + 1j * np.array(d['q'])
    f = extract_24_features(iq)
    print(f"{name:24s} | amp_cv={f['amp_cv']:.3f} | radial_iqr={f['radial_iqr']:.3f} | c42={f['c42_norm']:.3f} | c40={f['c40_norm']:.3f} | m2={f['phase_m2_concentration']:.3f} | m4={f['phase_m4_concentration']:.3f}")
