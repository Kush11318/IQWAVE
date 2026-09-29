import sys
import os
sys.path.insert(0, os.path.abspath("."))
import json
import glob
from backend.app.pipeline.pipeline_service import default_pipeline_orchestrator

signals = glob.glob("sample_signals/*.json")
print(f"Found {len(signals)} sample signals.")

results = {}
for sig_path in sorted(signals):
    name = os.path.basename(sig_path)
    with open(sig_path, "r") as f:
        data = json.load(f)
    meta = data.get("metadata", {})
    i_ch = data.get("i", [])
    q_ch = data.get("q", [])
    
    print(f"\n==========================================")
    print(f"Testing Signal: {name}")
    print(f"Ground Truth Metadata: {meta}")
    
    # Run pipeline purely blind (no override_modulation)
    res = default_pipeline_orchestrator.run_pipeline(
        i_channel=i_ch,
        q_channel=q_ch,
        sample_rate=meta.get("sample_rate"),
        center_frequency=meta.get("center_frequency"),
        metadata=meta
    )
    
    m_statuses = res.get("module_statuses", {})
    m3 = res.get("module3_amc", {})
    m4 = res.get("module4_parameters", {})
    m5 = res.get("module5_recovery", {})
    m6 = res.get("module6_snr", {})
    m8 = res.get("module8_fec", {})
    m9 = res.get("module9_structure", {})
    m10 = res.get("module10_fec_crc", {})
    
    print(f"Pipeline Success: {res.get('pipeline_success')}")
    print(f"Module Statuses: {m_statuses}")
    print(f"Active Modulation: {res.get('active_modulation')}")
    print(f"RF Engine status: {m3.get('engines', {}).get('engine_a_rf', {}).get('status')}")
    print(f"CNN Engine status: {m3.get('engines', {}).get('engine_b_cnn', {}).get('status')}")
    print(f"Estimated SPS: {m4.get('samples_per_symbol')}, Symbol Rate: {m4.get('symbol_rate')}, CFO: {m4.get('cfo')}")
    print(f"Module 5 Sync status: {m5.get('status')}, Recovered Symbols: {len(m5.get('symbols', []))}, Bits: {len(m5.get('bits', []))}")
    print(f"Module 6 Recommended SNR: {m6.get('recommended_snr_db')} dB (M2M4: {m6.get('m2m4_snr_db')}, EVM SNR: {m6.get('evm_snr_db')})")
    print(f"Module 8 Blind FEC: {m8.get('top_candidate') or m8.get('best_candidate')}")
    print(f"Module 9 Frame Period: {m9.get('frame_period')}, Preamble: {m9.get('preamble_detected')}")
    print(f"Module 10 CRC: {m10.get('crc', {}).get('candidate')}, Interleaver: {m10.get('interleaver', {}).get('detected_width')}")
    
    results[name] = {
        "ground_truth_mod": meta.get("modulation"),
        "active_mod": res.get("active_modulation"),
        "ground_truth_snr": meta.get("snr_db"),
        "estimated_snr": m6.get("recommended_snr_db"),
        "ground_truth_sps": meta.get("samples_per_symbol"),
        "estimated_sps": m4.get("samples_per_symbol"),
        "cfo_hz": m4.get("cfo"),
        "recovered_bits": len(m5.get("bits", []))
    }

print("\n\nSUMMARY RESULTS:")
print(json.dumps(results, indent=2))
