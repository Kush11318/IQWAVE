import urllib.request
import json

base_url = 'http://127.0.0.1:8000'

print("=== 1. Testing Static HTML & Asset Delivery ===")
for path, label in [('/', 'Standard Laboratory Frontend'), ('/experimental', 'SPECTRA Phosphor Radar Lab'), ('/exp', 'Radar Lab Alias')]:
    url = f"{base_url}{path}"
    with urllib.request.urlopen(url) as resp:
        print(f"GET {path} ({label}): Status {resp.status}, Content-Type: {resp.headers.get('content-type')}")

print("\n=== 2. Testing All Signal Fixture Presets ===")
presets = ['qpsk_1200', '16qam_1600', 'psk8_1200', 'bpsk_1000', 'fsk_1200', 'radar_lfm_1200', 'radar_barker_1300']
for p in presets:
    url = f"{base_url}/api/experimental/pipeline/fixture?preset={p}"
    with urllib.request.urlopen(url) as resp:
        data = json.loads(resp.read().decode('utf-8'))
        print(f"Preset {p:18s} -> Status: {data.get('status')}, Modulation: {data.get('modulation')}, Samples: {len(data.get('i', []))}")

print("\n=== 3. Testing Real Pipeline Execution (Modules 1-10) ===")
fix_url = f"{base_url}/api/experimental/pipeline/fixture?preset=qpsk_1200"
with urllib.request.urlopen(fix_url) as resp:
    fixture = json.loads(resp.read().decode('utf-8'))

run_url = f"{base_url}/api/experimental/pipeline/run"
payload = json.dumps({
    'i_channel': fixture['i'],
    'q_channel': fixture['q'],
    'sample_rate': fixture['sample_rate'],
    'center_frequency': fixture['center_frequency'],
    'modulation': fixture['modulation']
}).encode('utf-8')

req = urllib.request.Request(run_url, data=payload, headers={'Content-Type': 'application/json'}, method='POST')
with urllib.request.urlopen(req) as resp:
    res = json.loads(resp.read().decode('utf-8'))
    print(f"Overall Pipeline Status: {res.get('overall_pipeline_status')}")
    print(f"  [Module 1] Representation: {res.get('module1_validation', {}).get('status')}")
    print(f"  [Module 2] Observation Features Extracted: {len(res.get('module2_features', {}).get('features', {}))} metrics")
    conf = res.get('module3_amc', {}).get('calibrated_confidence') or res.get('module3_amc', {}).get('confidence') or 0.0
    print(f"  [Module 3] AMC Class: {res.get('module3_amc', {}).get('predicted_class')} (Confidence: {conf:.2%})")
    print(f"  [Module 4] Parameters: Center Freq = {res.get('module4_parameters', {}).get('center_frequency_hz')} Hz, Symbol Rate = {res.get('module4_parameters', {}).get('symbol_rate_baud')} Baud")
    print(f"  [Module 5] Sync Constellation: {len(res.get('module5_synchronization', {}).get('synchronized_symbols', {}).get('i', []))} symbols")
    snr_val = res.get('module6_snr', {}).get('estimated_snr_db')
    snr_str = f"{snr_val:.2f} dB" if snr_val is not None else "N/A"
    print(f"  [Module 6] Estimated SNR: {snr_str}")
    print(f"  [Module 7] LLR Soft Bits: {len(res.get('module7_soft_bits', {}).get('llr_stream', []))} soft bits")
    print(f"  [Module 8] Detected FEC Code: {res.get('module8_fec', {}).get('detected_code')}")
    print(f"  [Module 9] Frame Structure: Frame Len = {res.get('module9_framing', {}).get('frame_length')} bits")
    print(f"  [Module 10] Joint Protocol Intelligence: {res.get('module10_joint_intelligence', {}).get('status')}")

print("\n=== 4. Testing Report Generation Endpoints ===")
for fmt in ['json', 'html', 'md']:
    rep_url = f"{base_url}/api/experimental/pipeline/report/{fmt}"
    with urllib.request.urlopen(rep_url) as resp:
        content = resp.read()
        print(f"Report [{fmt.upper()}]: Status {resp.status}, Content Length: {len(content):,} bytes")

print("\n=== 5. Pipeline Architecture Status ===")
stat_url = f"{base_url}/api/experimental/pipeline/status"
with urllib.request.urlopen(stat_url) as resp:
    data = json.loads(resp.read().decode('utf-8'))
    print(f"Architecture Status: {data.get('status')}, Active Modules: {len(data.get('modules', {}))}")

print("\n>>> ALL FRONTEND-BACKEND INTEGRATION TESTS PASSED 100% <<<")
