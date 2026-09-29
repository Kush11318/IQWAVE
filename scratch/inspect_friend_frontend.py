import os
import re

dawc_exp = r'c:\Projects\MODULATION\DAWC---Digital-Automated-Waveform-Classifier\Frontend-Experimental'
with open(os.path.join(dawc_exp, 'index.html'), encoding='utf-8', errors='ignore') as f:
    html = f.read()

ids = re.findall(r'id=["\']([^"\']+)["\']', html)
print('Total element IDs in Frontend-Experimental/index.html:', len(ids))
print('Sample IDs:')
for i in ids[:40]:
    print(' ', i)

canvases = re.findall(r'<canvas[^>]*id=["\']([^"\']+)["\'][^>]*>', html)
print('\nCanvases:', canvases)

# Check buttons or triggers
buttons = re.findall(r'<button[^>]*id=["\']([^"\']+)["\'][^>]*>', html)
print('\nButtons with IDs:', buttons)

# Check app.js endpoints or fetch calls
with open(os.path.join(dawc_exp, 'app.js'), encoding='utf-8', errors='ignore') as f:
    js = f.read()

fetches = re.findall(r'fetch\([`"\']([^`"\']+)`?["\']', js)
print('\nFetch calls in app.js:', set(fetches))
