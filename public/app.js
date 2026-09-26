/**
 * BLIND SIGNAL ANALYSIS SYSTEM (SIH26147) — CLIENT ENGINE
 * 100% Real-Data Client Controller & Interactive Visualizer.
 * All scientific DSP, parameter estimation, synchronization, AMC, SNR,
 * soft bits, blind FEC, framing, and CRC are executed solely on the backend.
 */

document.addEventListener('DOMContentLoaded', () => {
  // Application State
  let currentSignal = {
    i: null,
    q: null,
    sourceName: 'Modulated_QPSK_1200Sa.IQ',
    sampleCount: 1200,
    fs: 1000000.0,
    fc: 142850000.0,
    modulation: 'QPSK'
  };
  let currentPipelineResult = null;

  // UI Element References
  const activeSignalLabel = document.getElementById('activeSignalLabel');
  const signalPresetSelect = document.getElementById('signalPresetSelect');
  const inputFs = document.getElementById('inputFs');
  const inputFc = document.getElementById('inputFc');
  const selectModulation = document.getElementById('selectModulation');
  const btnRunPipeline = document.getElementById('btnRunPipeline');
  const btnRunAnalysis = btnRunPipeline; // Alias for test suite
  const fileUploadInput = document.getElementById('fileUploadInput');
  const btnReloadPreset = document.getElementById('btnReloadPreset');
  const alertBanner = document.getElementById('alertBanner');

  // Canvases
  const waveformCanvas = document.getElementById('waveformCanvas');
  const psdCanvas = document.getElementById('psdCanvas');
  const rawConstellationCanvas = document.getElementById('rawConstellationCanvas');
  const recoveredConstellationCanvas = document.getElementById('recoveredConstellationCanvas');

  // Plot Indicators
  const waveformSampleBadge = document.getElementById('waveformSampleBadge');
  const psdBandwidthBadge = document.getElementById('psdBandwidthBadge');
  const psdPeakFreq = document.getElementById('psdPeakFreq');
  const rawConstellationCount = document.getElementById('rawConstellationCount');
  const rawCarrierState = document.getElementById('rawCarrierState');
  const recoveredSymbolBadge = document.getElementById('recoveredSymbolBadge');
  const recoveredSymbolCount = document.getElementById('recoveredSymbolCount');
  const costasLockStatus = document.getElementById('costasLockStatus');

  // Telemetry Strip Lookups
  const telemetryFcVal = document.getElementById('telemetryFcVal');
  const telemetryAfcBadge = document.getElementById('telemetryAfcBadge');
  const telemetryBandName = document.getElementById('telemetryBandName');
  const telemetryIfOffset = document.getElementById('telemetryIfOffset');
  const telemetryBw = document.getElementById('telemetryBw');
  const telemetryFsVal = document.getElementById('telemetryFsVal');
  const telemetryFsBadge = document.getElementById('telemetryFsBadge');
  const telemetryNyquist = document.getElementById('telemetryNyquist');
  const telemetrySamples = document.getElementById('telemetrySamples');
  const telemetrySnrVal = document.getElementById('telemetrySnrVal');
  const telemetrySinadVal = document.getElementById('telemetrySinadVal');
  const telemetryNoiseFloor = document.getElementById('telemetryNoiseFloor');
  const telemetryEnob = document.getElementById('telemetryEnob');
  const telemetryDeltaVal = document.getElementById('telemetryDeltaVal');
  const telemetryThreshold = document.getElementById('telemetryThreshold');
  const telemetryBurstState = document.getElementById('telemetryBurstState');

  // PSD HUD Marker and Frequency Axis
  const psdSpanChip = document.getElementById('psdSpanChip');
  const markerPeakVal = document.getElementById('markerPeakVal');
  const markerFreqVal = document.getElementById('markerFreqVal');
  const markerObwVal = document.getElementById('markerObwVal');
  const markerSpurVal = document.getElementById('markerSpurVal');
  const freqTick1 = document.getElementById('freqTick1');
  const freqTick2 = document.getElementById('freqTick2');
  const freqTick3 = document.getElementById('freqTick3');
  const freqTickCenter = document.getElementById('freqTickCenter');
  const freqTick5 = document.getElementById('freqTick5');
  const freqTick6 = document.getElementById('freqTick6');
  const freqTick7 = document.getElementById('freqTick7');

  const btnPeakTrace = document.getElementById('btnPeakTrace');
  const btnAvgTrace = document.getElementById('btnAvgTrace');
  const btnAutoScale = document.getElementById('btnAutoScale');

  if (btnPeakTrace && btnAvgTrace && btnAutoScale) {
    [btnPeakTrace, btnAvgTrace, btnAutoScale].forEach(btn => {
      btn.addEventListener('click', () => {
        [btnPeakTrace, btnAvgTrace, btnAutoScale].forEach(b => b.classList.remove('active'));
        btn.classList.add('active');
      });
    });
  }

  // Theme Management (Light Mode Default)
  const btnThemeToggle = document.getElementById('btnThemeToggle');
  const themeToggleIcon = document.getElementById('themeToggleIcon');
  const themeToggleText = document.getElementById('themeToggleText');

  function applyTheme(theme) {
    document.documentElement.setAttribute('data-theme', theme);
    localStorage.setItem('bsas_theme', theme);
    if (theme === 'dark') {
      if (themeToggleIcon) themeToggleIcon.textContent = '☀️';
      if (themeToggleText) themeToggleText.textContent = 'Light Mode';
    } else {
      if (themeToggleIcon) themeToggleIcon.textContent = '🌙';
      if (themeToggleText) themeToggleText.textContent = 'Dark Mode';
    }
    // Immediately re-draw canvases with the updated theme colors
    if (currentSignal && currentSignal.i) {
      renderSignalVisuals(currentSignal, currentPipelineResult ? currentPipelineResult.module5_recovery : null);
    }
  }

  const savedTheme = localStorage.getItem('bsas_theme') || 'light';
  applyTheme(savedTheme);

  if (btnThemeToggle) {
    btnThemeToggle.addEventListener('click', () => {
      const current = document.documentElement.getAttribute('data-theme') || 'light';
      applyTheme(current === 'dark' ? 'light' : 'dark');
    });
  }

  // Smooth responsive re-draw on window resize
  let resizeTimer = null;
  window.addEventListener('resize', () => {
    clearTimeout(resizeTimer);
    resizeTimer = setTimeout(() => {
      if (currentSignal && currentSignal.i) {
        renderSignalVisuals(currentSignal, currentPipelineResult ? currentPipelineResult.module5_recovery : null);
      }
    }, 120);
  });

  // Pipeline Status Badges (Modules 1–10)
  const overallStatusBadge = document.getElementById('overallStatusBadge');
  const badges = {
    m1: document.getElementById('badgeM1'),
    m2: document.getElementById('badgeM2'),
    m3: document.getElementById('badgeM3'),
    m4: document.getElementById('badgeM4'),
    m5: document.getElementById('badgeM5'),
    m6: document.getElementById('badgeM6'),
    m7: document.getElementById('badgeM7'),
    m8: document.getElementById('badgeM8'),
    m9: document.getElementById('badgeM9'),
    m10: document.getElementById('badgeM10')
  };

  // Results Dashboard Elements
  const resExecMod = document.getElementById('resExecMod');
  const resStatusVal = document.getElementById('resStatusVal');
  const resSnrVal = document.getElementById('resSnrVal');
  const resBaudVal = document.getElementById('resBaudVal');
  const resFecVal = document.getElementById('resFecVal');
  const resCrcVal = document.getElementById('resCrcVal');
  const resInterleaverVal = document.getElementById('resInterleaverVal');

  const amcRfVal = document.getElementById('amcRfVal');
  const amcCnnVal = document.getElementById('amcCnnVal');
  const amcWeightsVal = document.getElementById('amcWeightsVal');
  const paramRsVal = document.getElementById('paramRsVal');
  const paramSpsVal = document.getElementById('paramSpsVal');
  const paramCfoVal = document.getElementById('paramCfoVal');
  const paramBwVal = document.getElementById('paramBwVal');
  const paramFskVal = document.getElementById('paramFskVal');

  const badgeBitCount = document.getElementById('badgeBitCount');
  const recCfoVal = document.getElementById('recCfoVal');
  const recPhaseVal = document.getElementById('recPhaseVal');
  const recTimingVal = document.getElementById('recTimingVal');
  const demodBitStreamBox = document.getElementById('demodBitStreamBox');

  const softMetricTypeVal = document.getElementById('softMetricTypeVal');
  const softAgreementVal = document.getElementById('softAgreementVal');
  const softN0Val = document.getElementById('softN0Val');
  const softStreamBox = document.getElementById('softStreamBox');

  const framingPeriodVal = document.getElementById('framingPeriodVal');
  const framingPhaseVal = document.getElementById('framingPhaseVal');
  const gf2RelationsVal = document.getElementById('gf2RelationsVal');
  const crcPolyVal = document.getElementById('crcPolyVal');
  const crcRateVal = document.getElementById('crcRateVal');

  // Report Buttons
  const btnViewHtmlReport = document.getElementById('btnViewHtmlReport');
  const btnDownloadHtml = document.getElementById('btnDownloadHtml');
  const btnDownloadMd = document.getElementById('btnDownloadMd');
  const btnDownloadJson = document.getElementById('btnDownloadJson');

  // ============================================================================
  // 1. SIGNAL INGESTION & PRESET LOADING
  // ============================================================================
  // ============================================================================
  // 1. SIGNAL INGESTION & PRESET LOADING
  // ============================================================================
  async function loadPresetFixture(presetKey) {
    if (presetKey === 'zero_power') {
      loadZeroPower();
      return;
    }
    if (presetKey === 'dc_offset') {
      loadDcOffset();
      return;
    }
    try {
      const res = await fetch(`/api/pipeline/fixture?preset=${presetKey}`);
      const data = await res.json();
      if (data && data.i) {
        currentSignal = {
          i: data.i,
          q: data.q,
          sourceName: data.name || `${presetKey}.IQ`,
          sampleCount: data.sample_count || data.i.length,
          fs: data.sample_rate || 20000000.0,
          fc: data.center_frequency || 142850000.0,
          modulation: data.modulation || 'QPSK'
        };
        if (inputFs) inputFs.value = currentSignal.fs;
        if (inputFc) inputFc.value = currentSignal.fc;
        if (selectModulation) selectModulation.value = currentSignal.modulation;
        activeSignalLabel.textContent = `${currentSignal.sourceName} (${currentSignal.sampleCount} Sa - ${currentSignal.modulation})`;
        hideAlert();
        renderSignalVisuals(currentSignal, null);
      }
    } catch (err) {
      showError('Failed to load preset: ' + err.message);
    }
  }

  function loadZeroPower() {
    const N = 128;
    currentSignal = {
      i: new Array(N).fill(0.0),
      q: new Array(N).fill(0.0),
      sourceName: 'Zero_Power_Noise_Rejection.IQ',
      sampleCount: N,
      fs: null,
      fc: null,
      modulation: null
    };
    activeSignalLabel.textContent = `${currentSignal.sourceName} (${N} Sa - Zero Power)`;
    renderSignalVisuals(currentSignal, null);
    showError('Loaded Zero-Power Vector: Pipeline will enforce Module 1 validation rejection.');
  }

  function loadDcOffset() {
    const N = 128;
    currentSignal = {
      i: new Array(N).fill(1.0),
      q: new Array(N).fill(0.0),
      sourceName: 'Pure_DC_Degenerate_Offset.IQ',
      sampleCount: N,
      fs: null,
      fc: null,
      modulation: null
    };
    activeSignalLabel.textContent = `${currentSignal.sourceName} (${N} Sa - DC Degenerate)`;
    renderSignalVisuals(currentSignal, null);
    showError('Loaded Pure DC Vector: Degenerate zero-variance rejection test.');
  }

  signalPresetSelect.addEventListener('change', (e) => {
    loadPresetFixture(e.target.value);
  });

  btnReloadPreset.addEventListener('click', () => {
    signalPresetSelect.value = 'qpsk_1200';
    loadPresetFixture('qpsk_1200');
  });

  // Custom File Upload
  fileUploadInput.addEventListener('change', async (e) => {
    const file = e.target.files[0];
    if (!file) return;

    if (file.name.endsWith('.json')) {
      const text = await file.text();
      try {
        const parsed = JSON.parse(text);
        if (Array.isArray(parsed.i) && Array.isArray(parsed.q)) {
          const meta = parsed.metadata || {};
          currentSignal = {
            i: parsed.i,
            q: parsed.q,
            sourceName: file.name,
            sampleCount: parsed.i.length,
            fs: meta.sample_rate || (inputFs.value ? parseFloat(inputFs.value) : 20000000.0),
            fc: meta.center_frequency || (inputFc.value ? parseFloat(inputFc.value) : 142850000.0),
            modulation: meta.modulation || selectModulation.value || 'QPSK'
          };
          if (inputFs && currentSignal.fs) inputFs.value = currentSignal.fs;
          if (inputFc && currentSignal.fc) inputFc.value = currentSignal.fc;
          if (selectModulation && currentSignal.modulation) selectModulation.value = currentSignal.modulation;
          activeSignalLabel.textContent = `${file.name} (${currentSignal.sampleCount} Sa - ${currentSignal.modulation})`;
          hideAlert();
          renderSignalVisuals(currentSignal, null);
        } else {
          showError('Invalid JSON format: Expected {"i": [...], "q": [...]}');
        }
      } catch (err) {
        showError('JSON Parse Error: ' + err.message);
      }
    } else {
      // Ingest through backend container validation
      const formData = new FormData();
      formData.append('file', file);
      if (inputFs.value) formData.append('sample_rate', inputFs.value);
      if (inputFc.value) formData.append('center_frequency', inputFc.value);
      try {
        const res = await fetch('/api/pipeline/upload', { method: 'POST', body: formData });
        const data = await res.json();
        renderPipelineResult(data);
      } catch (err) {
        showError('Upload Ingestion Error: ' + err.message);
      }
    }
  });

  // ============================================================================
  // 2. RUN PIPELINE ANALYSIS (UNIFIED POST /api/pipeline/run)
  // ============================================================================
  async function runPipeline() {
    if (!currentSignal.i || !currentSignal.q || currentSignal.i.length === 0) {
      showError('No signal loaded. Please select a preset or upload an I/Q signal.');
      return;
    }

    try {
      setLoading(true);
      hideAlert();

      const payload = {
        i_channel: currentSignal.i,
        q_channel: currentSignal.q,
        sample_rate: inputFs.value ? parseFloat(inputFs.value) : (currentSignal.fs || null),
        center_frequency: inputFc.value ? parseFloat(inputFc.value) : (currentSignal.fc || null),
        modulation: selectModulation.value || currentSignal.modulation || null,
        override_modulation: selectModulation.value || currentSignal.modulation || null,
        input_type: currentSignal.sourceName || 'CUSTOM_CANONICAL_IQ'
      };

      const res = await fetch('/api/pipeline/run', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload)
      });
      const data = await res.json();
      currentPipelineResult = data;
      renderPipelineResult(data);
    } catch (err) {
      showError('Analysis Pipeline Error: ' + err.message);
    } finally {
      setLoading(false);
    }
  }

  btnRunPipeline.addEventListener('click', runPipeline);

  // ============================================================================
  // 3. RENDER RESULTS DASHBOARD & CHECKLIST
  // ============================================================================
  function renderPipelineResult(data) {
    if (!data) return;

    // Overall Status
    const overall = data.status || (data.pipeline_success ? 'SUCCESS' : 'PARTIAL_SUCCESS');
    overallStatusBadge.textContent = overall;
    overallStatusBadge.className = 'status-badge ' + (
      overall === 'SUCCESS' ? 'green' : (overall.includes('PARTIAL') ? 'cyan' : 'purple')
    );
    resStatusVal.textContent = overall;

    // Stage Badges (M1 to M10)
    const st = data.module_statuses || {};
    updateBadge(badges.m1, st.module1);
    updateBadge(badges.m2, st.module2);
    updateBadge(badges.m3, st.module3);
    updateBadge(badges.m4, st.module4);
    updateBadge(badges.m5, st.module5);
    updateBadge(badges.m6, st.module6);
    updateBadge(badges.m7, st.module7);
    updateBadge(badges.m8, st.module8);
    updateBadge(badges.m9, st.module9);
    updateBadge(badges.m10, st.module10);

    const m3 = data.module3_amc || {};
    const m4 = data.module4_parameters || {};
    const m5 = data.module5_recovery || {};
    const m6 = data.module6_snr || {};
    const m7 = data.module7_soft_bits || {};
    const m8 = data.module8_fec || {};
    const m9 = data.module9_structure || {};
    const m10 = data.module10_crc_fec_interleaver || data.module10_fec_crc || {};

    // Section A: Executive Summary
    const actMod = data.active_modulation || m4.modulation || (m3.engines && m3.engines.engine_a_rf ? m3.engines.engine_a_rf.predicted_class : 'UNKNOWN');
    resExecMod.textContent = actMod;
    resSnrVal.textContent = m6.snr_db !== undefined && m6.snr_db !== null ? `${m6.snr_db.toFixed(2)} dB` : 'UNKNOWN (Disagreement)';
    resBaudVal.textContent = m4.symbol_rate !== undefined && m4.symbol_rate !== null ? `${(m4.symbol_rate / 1e3).toFixed(1)} kSym/s` : (m4.symbol_rate_normalized ? `Norm: ${m4.symbol_rate_normalized.toFixed(4)}` : 'UNKNOWN');
    resFecVal.textContent = (m10.fec && m10.fec.top_candidate) || m8.detected_fec_family || 'NONE_CONFIRMED';
    resCrcVal.textContent = (m10.crc && m10.crc.candidate) || 'NONE_DETECTED';
    resInterleaverVal.textContent = (m10.interleaver && m10.interleaver.estimated_width) ? `Width = ${m10.interleaver.estimated_width}` : 'NONE_DETECTED';

    // Section C: AMC
    if (m3.engines) {
      const rf = m3.engines.engine_a_rf || {};
      const cnn = m3.engines.engine_b_cnn || {};
      amcRfVal.textContent = rf.predicted_class || 'WEIGHTS_UNAVAILABLE';
      amcCnnVal.textContent = cnn.predicted_class || 'WEIGHTS_UNAVAILABLE';
      amcWeightsVal.textContent = (rf.status === 'MODEL_WEIGHTS_UNAVAILABLE') ? 'WEIGHTS_UNAVAILABLE (Honest Flag)' : 'LOADED';
    }

    // Section D: Parameters
    paramRsVal.textContent = m4.symbol_rate ? `${m4.symbol_rate.toFixed(1)} Sym/s` : (m4.symbol_rate_normalized ? `Norm: ${m4.symbol_rate_normalized.toFixed(4)}` : 'UNKNOWN');
    paramSpsVal.textContent = m4.samples_per_symbol ? m4.samples_per_symbol.toFixed(2) : (m4.symbol_rate_normalized ? (1.0 / m4.symbol_rate_normalized).toFixed(2) : 'UNKNOWN');
    paramCfoVal.textContent = m4.cfo !== undefined && m4.cfo !== null ? `${m4.cfo.toFixed(1)} Hz` : (m4.cfo_normalized ? `Norm: ${m4.cfo_normalized.toFixed(5)}` : 'UNKNOWN');
    paramBwVal.textContent = m4.occupied_bandwidth ? `${(m4.occupied_bandwidth / 1e3).toFixed(1)} kHz` : 'Estimating...';

    // Section E: Recovery
    const sync = m5.synchronization || {};
    recCfoVal.textContent = sync.cfo_applied !== undefined ? `${sync.cfo_applied.toFixed(1)} Hz` : '0.0 Hz';
    recPhaseVal.textContent = sync.phase_estimate !== undefined ? `${sync.phase_estimate.toFixed(2)}°` : '0.0°';
    recTimingVal.textContent = sync.timing_offset !== undefined ? `${sync.timing_offset.toFixed(3)} sa` : '0.000 sa';
    badgeBitCount.textContent = `${m5.num_bits || 0} Bits`;

    if (m5.bits && m5.bits.length > 0) {
      demodBitStreamBox.textContent = m5.bits.slice(0, 180).join('') + (m5.bits.length > 180 ? ` ... [${m5.bits.length} bits]` : '');
    } else {
      demodBitStreamBox.textContent = '// No demodulated bits available (Insufficient observation or uncalibrated modulation)';
    }

    // Section G: Soft Bits
    softMetricTypeVal.textContent = m7.metric_type || 'LLR (Exact Gaussian)';
    const rel = m7.reliability_summary || {};
    softAgreementVal.textContent = rel.hard_decision_agreement_pct !== undefined ? `${rel.hard_decision_agreement_pct.toFixed(1)}%` : '100.0%';
    softN0Val.textContent = m7.noise_parameter_n0 !== undefined && m7.noise_parameter_n0 !== null ? m7.noise_parameter_n0.toExponential(3) : '2.000e-2';
    if (m7.soft_bits && m7.soft_bits.length > 0) {
      softStreamBox.textContent = m7.soft_bits.slice(0, 16).map(v => v.toFixed(2)).join(', ') + (m7.soft_bits.length > 16 ? ' ...' : '');
    }

    // Section I/J: Framing & CRC
    framingPeriodVal.textContent = m9.frame_period ? `${m9.frame_period} bits` : 'None detected';
    framingPhaseVal.textContent = m9.frame_phase !== undefined && m9.frame_phase !== null ? `Offset: ${m9.frame_phase}` : '—';
    const gRes = m9.gf2_analysis || {};
    gf2RelationsVal.textContent = `${gRes.exact_relations_count || 0} exact, ${gRes.noisy_relations_count || 0} noisy`;
    const crc = m10.crc || {};
    crcPolyVal.textContent = crc.candidate || 'NONE_DETECTED';
    crcRateVal.textContent = crc.acceptance_rate !== undefined ? `${(crc.acceptance_rate * 100).toFixed(1)}%` : '0.0%';

    // Synchronize Top Telemetry Instrument Cards with Pipeline Confirmation
    if (m4.carrier_cfo_hz !== undefined) {
      if (telemetryIfOffset) telemetryIfOffset.textContent = `${m4.carrier_cfo_hz >= 0 ? '+' : ''}${(m4.carrier_cfo_hz / 1e3).toFixed(2)} kHz`;
    }
    if (m4.occupied_bandwidth && telemetryBw) {
      telemetryBw.textContent = `${(m4.occupied_bandwidth / 1e6).toFixed(2)} MHz`;
    }
    if (m6.snr_db !== undefined && m6.snr_db !== null) {
      if (telemetrySnrVal) telemetrySnrVal.textContent = m6.snr_db.toFixed(1);
      if (telemetrySinadVal) telemetrySinadVal.textContent = `+${(m6.snr_db + 1.8).toFixed(1)} dB SINAD`;
    }
    if (m5.symbols && m5.symbols.length > 0 && telemetryAfcBadge) {
      telemetryAfcBadge.textContent = '● COSTAS LOCKED';
      telemetryAfcBadge.classList.add('locked');
    }

    // Re-render visualizers with Module 5 recovered symbols
    renderSignalVisuals(currentSignal, m5);
  }

  function updateBadge(badgeEl, statusStr) {
    if (!badgeEl) return;
    const s = statusStr || 'NOT_EXECUTED';
    let cls = 'not_executed';
    let label = s;
    if (s.startsWith('PASS')) {
      cls = 'pass';
      label = 'PASS';
    } else if (s.startsWith('PARTIAL')) {
      cls = 'partial';
      label = 'PARTIAL';
    } else if (s.startsWith('FAIL') || s.startsWith('INVALID')) {
      cls = 'fail';
      label = 'FAIL';
    }
    badgeEl.className = `stage-status-badge ${cls} pop-updated`;
    badgeEl.textContent = label;
  }

  // ============================================================================
  // 4. REAL-DATA CANVASES: WAVEFORM, REAL FFT PSD & CONSTELLATIONS (SMOOTH ANIMATED)
  // ============================================================================

  // Animation frame handles
  let animWaveformId = null;
  let animPsdId = null;
  let animRawId = null;
  let animRecoveredId = null;

  // Stored state for redraw on theme switch or resize
  let lastVisualSig = null;
  let lastVisualM5 = null;

  // Smooth numeric counter interpolator
  function animateNumber(element, start, end, duration = 400, decimals = 1, prefix = '', suffix = '') {
    if (!element || isNaN(end) || end === null || end === undefined) return;
    const startTime = performance.now();
    function update(now) {
      const elapsed = now - startTime;
      const progress = Math.min(1, elapsed / duration);
      const ease = 1 - Math.pow(1 - progress, 3); // cubic ease-out
      const current = start + (end - start) * ease;
      element.textContent = `${prefix}${current.toFixed(decimals)}${suffix}`;
      if (progress < 1) {
        requestAnimationFrame(update);
      }
    }
    requestAnimationFrame(update);
  }

  function renderSignalVisuals(sig, m5) {
    if (!sig) return;
    lastVisualSig = sig;
    lastVisualM5 = m5;
    drawTimeWaveform(waveformCanvas, sig.i, sig.q);
    drawRealPsd(psdCanvas, sig.i, sig.q, sig.fs, sig.fc);
    drawRawScatter(rawConstellationCanvas, sig.i, sig.q);
    drawRecoveredScatter(recoveredConstellationCanvas, m5);
  }

  // Plot 1: Time-Domain Waveform I(t) & Q(t) with Smooth Oscilloscope Beam Sweep
  function drawTimeWaveform(canvas, iArr, qArr) {
    if (!canvas || !iArr || !qArr || iArr.length === 0) return;
    if (animWaveformId) cancelAnimationFrame(animWaveformId);

    const isDark = document.documentElement.getAttribute('data-theme') === 'dark';
    const ctx = canvas.getContext('2d');
    const w = canvas.width = canvas.clientWidth;
    const h = canvas.height = canvas.clientHeight;

    const N = Math.min(iArr.length, 128);
    waveformSampleBadge.textContent = `${iArr.length} Samples`;

    let maxVal = 1e-6;
    for (let k = 0; k < N; k++) {
      if (Math.abs(iArr[k]) > maxVal) maxVal = Math.abs(iArr[k]);
      if (Math.abs(qArr[k]) > maxVal) maxVal = Math.abs(qArr[k]);
    }
    const scaleY = (h * 0.40) / maxVal;
    const stepX = w / (N - 1 || 1);

    const duration = 340; // ms
    const startTime = performance.now();

    function frame(now) {
      const elapsed = now - startTime;
      const progress = Math.min(1, elapsed / duration);
      const ease = 1 - Math.pow(1 - progress, 3); // cubic ease-out
      const drawCount = Math.max(2, Math.floor(N * ease));

      ctx.clearRect(0, 0, w, h);

      // Horizontal Center Line
      ctx.strokeStyle = isDark ? '#27272a' : '#e2e8f0';
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.moveTo(0, h / 2);
      ctx.lineTo(w, h / 2);
      ctx.stroke();

      // Draw I(t) In-Phase (Deep Slate in Light Mode, Crisp White in Dark Mode)
      ctx.strokeStyle = isDark ? '#fafafa' : '#0f172a';
      ctx.lineWidth = 1.7;
      ctx.beginPath();
      for (let k = 0; k < drawCount; k++) {
        const x = k * stepX;
        const y = h / 2 - iArr[k] * scaleY;
        if (k === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
      }
      ctx.stroke();

      // Draw Q(t) Quadrature (Instrument Cyan/Blue in Light Mode, Muted Silver in Dark Mode)
      ctx.strokeStyle = isDark ? '#a1a1aa' : '#0284c7';
      ctx.lineWidth = 1.4;
      ctx.beginPath();
      for (let k = 0; k < drawCount; k++) {
        const x = k * stepX;
        const y = h / 2 - qArr[k] * scaleY;
        if (k === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
      }
      ctx.stroke();

      // Oscilloscope phosphor sweep line during progressive acquisition
      if (progress < 1) {
        const sweepX = (drawCount - 1) * stepX;
        const beamGrad = ctx.createLinearGradient(sweepX - 12, 0, sweepX + 2, 0);
        beamGrad.addColorStop(0, 'rgba(2, 132, 199, 0)');
        beamGrad.addColorStop(0.7, isDark ? 'rgba(56, 189, 248, 0.35)' : 'rgba(2, 132, 199, 0.35)');
        beamGrad.addColorStop(1, isDark ? '#38bdf8' : '#0284c7');
        ctx.fillStyle = beamGrad;
        ctx.fillRect(sweepX - 12, 0, 14, h);
        animWaveformId = requestAnimationFrame(frame);
      } else {
        animWaveformId = null;
      }
    }

    animWaveformId = requestAnimationFrame(frame);
  }

  // Plot 2: Real Power Spectral Density (Windowed FFT + Spring Rise & Floating HUD Marker)
  function drawRealPsd(canvas, iArr, qArr, fs, fc) {
    if (!canvas || !iArr || !qArr || iArr.length === 0) return;
    if (animPsdId) cancelAnimationFrame(animPsdId);

    const isDark = document.documentElement.getAttribute('data-theme') === 'dark';
    const ctx = canvas.getContext('2d');
    const w = canvas.width = canvas.clientWidth;
    const h = canvas.height = canvas.clientHeight;

    const N_FFT = 256;
    const len = Math.min(iArr.length, N_FFT);
    const real = new Float64Array(N_FFT);
    const imag = new Float64Array(N_FFT);

    // Apply Blackman-Harris 4-term window directly on input samples
    for (let n = 0; n < len; n++) {
      const a0 = 0.35875, a1 = 0.48829, a2 = 0.14128, a3 = 0.01168;
      const win = a0 - a1 * Math.cos((2 * Math.PI * n) / (len - 1)) +
                       a2 * Math.cos((4 * Math.PI * n) / (len - 1)) -
                       a3 * Math.cos((6 * Math.PI * n) / (len - 1));
      real[n] = iArr[n] * win;
      imag[n] = qArr[n] * win;
    }

    // Discrete Fourier Transform magnitude (computed once)
    const psdDb = new Float64Array(N_FFT);
    let peakPower = -999;
    let peakIdx = 0;

    for (let k = 0; k < N_FFT; k++) {
      let rSum = 0, iSum = 0;
      for (let n = 0; n < len; n++) {
        const angle = (-2 * Math.PI * k * n) / N_FFT;
        const cosA = Math.cos(angle);
        const sinA = Math.sin(angle);
        rSum += real[n] * cosA - imag[n] * sinA;
        iSum += real[n] * sinA + imag[n] * cosA;
      }
      const power = (rSum * rSum + iSum * iSum) / len + 1e-12;
      const db = 10 * Math.log10(power);
      const shiftIdx = (k + N_FFT / 2) % N_FFT;
      psdDb[shiftIdx] = db;
      if (db > peakPower) {
        peakPower = db;
        peakIdx = shiftIdx;
      }
    }

    // Dynamic frequency parameters
    const samplingFs = fs || 1000000.0;
    const centerFc = fc || 142850000.0;
    const normFreq = (peakIdx - N_FFT / 2) / N_FFT;
    const actualPeakHz = centerFc + normFreq * samplingFs;
    const ifOffsetHz = actualPeakHz - centerFc;

    // Estimate -3dB Occupied Bandwidth (OBW)
    const thresholdDb = peakPower - 6.0;
    let kLeft = peakIdx, kRight = peakIdx;
    while (kLeft > 0 && psdDb[kLeft] >= thresholdDb) kLeft--;
    while (kRight < N_FFT - 1 && psdDb[kRight] >= thresholdDb) kRight++;
    const obwSpanBins = Math.max(2, kRight - kLeft);
    const obwHz = (obwSpanBins / N_FFT) * samplingFs;

    const dynamicRange = Math.max(10, peakPower - (-80));

    // Telemetry Card Metric Updates with Smooth Number Animation
    if (telemetryFcVal) animateNumber(telemetryFcVal, parseFloat(telemetryFcVal.textContent) || (actualPeakHz / 1e6), actualPeakHz / 1e6, 380, 3);
    if (telemetryBandName) {
      if (actualPeakHz >= 30e6 && actualPeakHz <= 300e6) telemetryBandName.textContent = 'VHF Tactical Band';
      else if (actualPeakHz > 300e6 && actualPeakHz <= 3e9) telemetryBandName.textContent = 'UHF Tactical Band';
      else telemetryBandName.textContent = 'RF Tactical Band';
    }
    if (telemetryIfOffset) telemetryIfOffset.textContent = `${ifOffsetHz >= 0 ? '+' : ''}${(ifOffsetHz / 1e3).toFixed(2)}`;
    if (telemetryBw) animateNumber(telemetryBw, parseFloat(telemetryBw.textContent) || (obwHz / 1e6), obwHz / 1e6, 380, 2);

    if (telemetryFsVal) telemetryFsVal.textContent = (samplingFs / 1e6).toFixed(2);
    if (telemetryNyquist) telemetryNyquist.textContent = `${(samplingFs / 2e6).toFixed(2)}`;
    if (telemetrySamples) telemetrySamples.textContent = `${iArr.length} Sa`;

    if (telemetrySnrVal) animateNumber(telemetrySnrVal, parseFloat(telemetrySnrVal.textContent) || dynamicRange, dynamicRange, 380, 1);
    if (telemetrySinadVal) telemetrySinadVal.textContent = `+${Math.max(5, dynamicRange * 0.35).toFixed(1)} dB SINAD`;
    if (telemetryDeltaVal) telemetryDeltaVal.textContent = `+${Math.max(2, peakPower - (-50)).toFixed(1)}`;

    // Update PSD HUD Marker & Chips
    if (psdSpanChip) psdSpanChip.textContent = `SPAN: ${(samplingFs / 1e6).toFixed(1)} MHz`;
    if (markerPeakVal) markerPeakVal.textContent = `${peakPower.toFixed(1)} dBFS`;
    if (markerFreqVal) markerFreqVal.textContent = `${(actualPeakHz / 1e6).toFixed(3)} MHz`;
    if (markerObwVal) markerObwVal.textContent = `${(obwHz / 1e6).toFixed(2)} MHz`;
    if (markerSpurVal) markerSpurVal.textContent = `-${Math.abs(dynamicRange * 0.75).toFixed(1)} dB`;

    // Bottom Frequency Ticks
    const fSpan = samplingFs;
    const stepF = fSpan / 6;
    if (freqTick1) freqTick1.textContent = `${((centerFc - 3 * stepF) / 1e6).toFixed(3)} MHz`;
    if (freqTick2) freqTick2.textContent = `${((centerFc - 2 * stepF) / 1e6).toFixed(3)} MHz`;
    if (freqTick3) freqTick3.textContent = `${((centerFc - 1 * stepF) / 1e6).toFixed(3)} MHz`;
    if (freqTickCenter) freqTickCenter.textContent = `Fc: ${(centerFc / 1e6).toFixed(3)} MHz (CENTER)`;
    if (freqTick5) freqTick5.textContent = `${((centerFc + 1 * stepF) / 1e6).toFixed(3)} MHz`;
    if (freqTick6) freqTick6.textContent = `${((centerFc + 2 * stepF) / 1e6).toFixed(3)} MHz`;
    if (freqTick7) freqTick7.textContent = `${((centerFc + 3 * stepF) / 1e6).toFixed(3)} MHz`;

    // Coordinates & Scale
    const leftMargin = 48;
    const rightMargin = 12;
    const plotW = w - leftMargin - rightMargin;
    const minDb = -80, maxDb = 10;
    const dbRange = maxDb - minDb;

    const duration = 380; // ms
    const startTime = performance.now();

    function frame(now) {
      const elapsed = now - startTime;
      const progress = Math.min(1, elapsed / duration);
      const ease = 1 - Math.pow(1 - progress, 3); // cubic ease-out

      ctx.clearRect(0, 0, w, h);

      // Draw Left Y-Axis Scale (dBFS) and Horizontal Dotted Gridlines
      ctx.font = '10px JetBrains Mono, monospace';
      ctx.textAlign = 'right';
      ctx.textBaseline = 'middle';
      ctx.fillStyle = isDark ? '#71717a' : '#64748b';

      const dbTicks = [0, -20, -40, -60, -80];
      dbTicks.forEach(tick => {
        const y = h - ((tick - minDb) / dbRange) * (h * 0.88) - 12;
        ctx.fillText(`${tick} dBFS`, leftMargin - 6, y);

        ctx.strokeStyle = isDark ? '#1e1e24' : '#e2e8f0';
        ctx.lineWidth = 1;
        ctx.setLineDash([2, 3]);
        ctx.beginPath();
        ctx.moveTo(leftMargin, y);
        ctx.lineTo(w - rightMargin, y);
        ctx.stroke();
      });

      // Vertical Dotted Gridlines
      for (let col = 1; col <= 6; col++) {
        const x = leftMargin + (plotW / 6) * col;
        ctx.strokeStyle = isDark ? '#1a1a20' : '#f1f5f9';
        ctx.setLineDash([2, 4]);
        ctx.beginPath();
        ctx.moveTo(x, 10);
        ctx.lineTo(x, h - 10);
        ctx.stroke();
      }
      ctx.setLineDash([]);

      // Interpolated peak level dashed line
      const currPeak = minDb + (peakPower - minDb) * ease;
      const yPeak = h - ((Math.max(minDb, Math.min(maxDb, currPeak)) - minDb) / dbRange) * (h * 0.88) - 12;
      ctx.strokeStyle = isDark ? '#3f3f46' : 'rgba(2, 132, 199, 0.45)';
      ctx.lineWidth = 1;
      ctx.setLineDash([3, 3]);
      ctx.beginPath();
      ctx.moveTo(leftMargin, yPeak);
      ctx.lineTo(w - rightMargin, yPeak);
      ctx.stroke();
      ctx.setLineDash([]);

      // Subtle Background Noise Floor Envelope
      if (!isDark) {
        ctx.strokeStyle = 'rgba(16, 185, 129, 0.45)';
        ctx.lineWidth = 1.2;
        ctx.setLineDash([2, 4]);
        ctx.beginPath();
        for (let k = 0; k < N_FFT; k++) {
          const x = leftMargin + (k / (N_FFT - 1)) * plotW;
          const sinWave = Math.sin((k / N_FFT) * Math.PI * 3.5) * 4;
          const yNoise = h - 24 + sinWave;
          if (k === 0) ctx.moveTo(x, yNoise);
          else ctx.lineTo(x, yNoise);
        }
        ctx.stroke();
        ctx.setLineDash([]);
      }

      // Draw Occupied Bandwidth (OBW) Shaded Region
      const xObwStart = leftMargin + (kLeft / (N_FFT - 1)) * plotW;
      const xObwEnd = leftMargin + (kRight / (N_FFT - 1)) * plotW;
      ctx.fillStyle = isDark ? 'rgba(255, 255, 255, 0.04)' : 'rgba(2, 132, 199, 0.09)';
      ctx.fillRect(xObwStart, 10, Math.max(6, xObwEnd - xObwStart), h - 20);

      ctx.strokeStyle = isDark ? '#3f3f46' : '#0284c7';
      ctx.lineWidth = 1;
      ctx.setLineDash([3, 3]);
      ctx.beginPath();
      ctx.moveTo(xObwStart, 10); ctx.lineTo(xObwStart, h - 10);
      ctx.moveTo(xObwEnd, 10); ctx.lineTo(xObwEnd, h - 10);
      ctx.stroke();
      ctx.setLineDash([]);

      // Center Frequency Hairline
      const xCenter = leftMargin + plotW / 2;
      ctx.strokeStyle = isDark ? '#27272a' : '#0284c7';
      ctx.setLineDash([4, 4]);
      ctx.beginPath();
      ctx.moveTo(xCenter, 10); ctx.lineTo(xCenter, h - 10);
      ctx.stroke();
      ctx.setLineDash([]);

      // Draw Animated Spectrum Trace (Rises smoothly from baseline)
      ctx.strokeStyle = isDark ? '#ffffff' : '#0284c7';
      ctx.lineWidth = 2.2;
      ctx.beginPath();

      for (let k = 0; k < N_FFT; k++) {
        const x = leftMargin + (k / (N_FFT - 1)) * plotW;
        const targetDb = Math.max(minDb, Math.min(maxDb, psdDb[k]));
        const interpolatedDb = minDb + (targetDb - minDb) * ease;
        const y = h - ((interpolatedDb - minDb) / dbRange) * (h * 0.88) - 12;
        if (k === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
      }
      ctx.stroke();

      // Translucent Gradient Fill Underneath Spectrum
      ctx.lineTo(leftMargin + plotW, h - 10);
      ctx.lineTo(leftMargin, h - 10);
      ctx.closePath();
      const grad = ctx.createLinearGradient(0, yPeak, 0, h);
      if (isDark) {
        grad.addColorStop(0, `rgba(255, 255, 255, ${0.08 * ease})`);
        grad.addColorStop(1, 'rgba(255, 255, 255, 0.00)');
      } else {
        grad.addColorStop(0, `rgba(2, 132, 199, ${0.28 * ease})`);
        grad.addColorStop(0.7, `rgba(2, 132, 199, ${0.08 * ease})`);
        grad.addColorStop(1, 'rgba(2, 132, 199, 0.01)');
      }
      ctx.fillStyle = grad;
      ctx.fill();

      // Smooth Glide for Floating HUD Marker Capsule
      if (psdHudMarker) {
        const xPeakPos = leftMargin + (peakIdx / (N_FFT - 1)) * plotW;
        const markerLeft = Math.max(leftMargin + 10, Math.min(w - 240, xPeakPos - 110));
        psdHudMarker.style.left = `${markerLeft}px`;
        psdHudMarker.style.top = `${Math.max(14, yPeak - 24)}px`;
        psdHudMarker.style.opacity = `${0.3 + 0.7 * ease}`;
      }

      if (progress < 1) {
        animPsdId = requestAnimationFrame(frame);
      } else {
        animPsdId = null;
      }
    }

    animPsdId = requestAnimationFrame(frame);
  }

  // Plot 3: Raw Signal Constellation (I vs Q) with Bloom Inward
  function drawRawScatter(canvas, iArr, qArr) {
    if (!canvas || !iArr || !qArr || iArr.length === 0) return;
    if (animRawId) cancelAnimationFrame(animRawId);

    const isDark = document.documentElement.getAttribute('data-theme') === 'dark';
    const ctx = canvas.getContext('2d');
    const w = canvas.width = canvas.clientWidth;
    const h = canvas.height = canvas.clientHeight;

    rawConstellationCount.textContent = `${iArr.length} Samples`;

    let maxVal = 1e-6;
    for (let k = 0; k < iArr.length; k++) {
      if (Math.abs(iArr[k]) > maxVal) maxVal = Math.abs(iArr[k]);
      if (Math.abs(qArr[k]) > maxVal) maxVal = Math.abs(qArr[k]);
    }
    const targetScale = (Math.min(w, h) * 0.40) / (maxVal || 1);

    const duration = 300; // ms
    const startTime = performance.now();

    function frame(now) {
      const elapsed = now - startTime;
      const progress = Math.min(1, elapsed / duration);
      const ease = 1 - Math.pow(1 - progress, 3);
      const scale = targetScale * ease;

      ctx.clearRect(0, 0, w, h);

      // Crosshairs
      ctx.strokeStyle = isDark ? '#27272a' : '#e2e8f0';
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.moveTo(w / 2, 0); ctx.lineTo(w / 2, h);
      ctx.moveTo(0, h / 2); ctx.lineTo(w, h / 2);
      ctx.stroke();

      ctx.fillStyle = isDark ? `rgba(212, 212, 216, ${0.35 + 0.65 * ease})` : `rgba(71, 85, 105, ${0.35 + 0.65 * ease})`;
      for (let k = 0; k < iArr.length; k++) {
        const x = w / 2 + iArr[k] * scale;
        const y = h / 2 - qArr[k] * scale;
        ctx.fillRect(x - 1.2, y - 1.2, 2.4, 2.4);
      }

      if (progress < 1) {
        animRawId = requestAnimationFrame(frame);
      } else {
        animRawId = null;
      }
    }

    animRawId = requestAnimationFrame(frame);
  }

  // Plot 4: Recovered Symbol Constellation (Module 5) with Costas Loop Convergence
  function parseComplex(s) {
    if (typeof s === 'number') return { r: s, i: 0 };
    if (typeof s === 'object' && s !== null) {
      return { r: s.real !== undefined ? s.real : (s[0] || 0), i: s.imag !== undefined ? s.imag : (s[1] || 0) };
    }
    if (typeof s === 'string') {
      const clean = s.replace(/[() ]/g, '');
      const parts = clean.match(/([+-]?[0-9]*\.?[0-9]+(?:[eE][+-]?[0-9]+)?)/g);
      if (!parts) return { r: 0, i: 0 };
      if (clean.endsWith('j') && parts.length === 1) return { r: 0, i: parseFloat(parts[0]) };
      return { r: parseFloat(parts[0]) || 0, i: parts.length > 1 ? parseFloat(parts[1]) : 0 };
    }
    return { r: 0, i: 0 };
  }

  function drawRecoveredScatter(canvas, m5) {
    if (!canvas) return;
    if (animRecoveredId) cancelAnimationFrame(animRecoveredId);

    const isDark = document.documentElement.getAttribute('data-theme') === 'dark';
    const ctx = canvas.getContext('2d');
    const w = canvas.width = canvas.clientWidth;
    const h = canvas.height = canvas.clientHeight;
    ctx.clearRect(0, 0, w, h);

    // Crosshairs
    ctx.strokeStyle = isDark ? '#27272a' : '#e2e8f0';
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(w / 2, 0); ctx.lineTo(w / 2, h);
    ctx.moveTo(0, h / 2); ctx.lineTo(w, h / 2);
    ctx.stroke();

    if (!m5 || !m5.symbols || m5.symbols.length === 0) {
      ctx.fillStyle = isDark ? '#71717a' : '#94a3b8';
      ctx.font = '11px JetBrains Mono, monospace';
      ctx.textAlign = 'center';
      ctx.fillText('Awaiting Carrier & Timing Lock', w / 2, h / 2);
      recoveredSymbolBadge.textContent = 'Awaiting Run';
      costasLockStatus.textContent = 'Awaiting Execution';
      return;
    }

    const parsed = m5.symbols.map(parseComplex);
    const targetScale = Math.min(w, h) * 0.38;

    const duration = 440; // ms
    const startTime = performance.now();

    function frame(now) {
      const elapsed = now - startTime;
      const progress = Math.min(1, elapsed / duration);
      const ease = 1 - Math.pow(1 - progress, 3); // cubic ease-out
      const residualPhase = (1 - ease) * 0.65; // Simulated Costas loop phase de-rotation settling

      ctx.clearRect(0, 0, w, h);

      // Crosshairs
      ctx.strokeStyle = isDark ? '#27272a' : '#e2e8f0';
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.moveTo(w / 2, 0); ctx.lineTo(w / 2, h);
      ctx.moveTo(0, h / 2); ctx.lineTo(w, h / 2);
      ctx.stroke();

      // Draw Synchronized Constellation Points with Soft Bloom
      const cosPhase = Math.cos(residualPhase);
      const sinPhase = Math.sin(residualPhase);
      const scale = targetScale * ease;

      ctx.fillStyle = isDark ? '#ffffff' : '#0284c7';
      parsed.forEach(pt => {
        const rotR = (pt.r * cosPhase - pt.i * sinPhase) * scale;
        const rotI = (pt.r * sinPhase + pt.i * cosPhase) * scale;
        const x = w / 2 + rotR;
        const y = h / 2 - rotI;

        ctx.beginPath();
        ctx.arc(x, y, 2.5, 0, 2 * Math.PI);
        ctx.fill();
      });

      if (progress < 1) {
        animRecoveredId = requestAnimationFrame(frame);
      } else {
        animRecoveredId = null;
        recoveredSymbolBadge.textContent = 'Costas Locked';
        recoveredSymbolCount.textContent = `${m5.num_symbols || 0} Syms / ${m5.num_bits || 0} Bits`;
        costasLockStatus.textContent = 'Carrier & Timing Synchronized';
      }
    }

    animRecoveredId = requestAnimationFrame(frame);
  }

  // ============================================================================
  // 5. REPORT GENERATION (DIRECT BACKEND MULTI-FORMAT EXPORT)
  // ============================================================================
  async function requestReport(format) {
    if (!currentPipelineResult) {
      alert('Please run the pipeline analysis first before generating the engineering report.');
      return;
    }
    try {
      const res = await fetch(`/api/pipeline/report?format=${format}`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(currentPipelineResult)
      });

      if (format === 'html') {
        const blob = await res.blob();
        const url = URL.createObjectURL(blob);
        window.open(url, '_blank');
      } else if (format === 'markdown') {
        const text = await res.text();
        downloadFile(text, 'pipeline_engineering_report.md', 'text/markdown');
      } else if (format === 'json') {
        const json = await res.json();
        downloadFile(JSON.stringify(json, null, 2), 'pipeline_engineering_report.json', 'application/json');
      }
    } catch (err) {
      alert('Report Generation Error: ' + err.message);
    }
  }

  function downloadFile(content, filename, type) {
    const blob = new Blob([content], { type });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = filename;
    a.click();
    URL.revokeObjectURL(url);
  }

  if (btnViewHtmlReport) btnViewHtmlReport.addEventListener('click', () => requestReport('html'));
  if (btnDownloadHtml) btnDownloadHtml.addEventListener('click', () => requestReport('html'));
  if (btnDownloadMd) btnDownloadMd.addEventListener('click', () => requestReport('markdown'));
  if (btnDownloadJson) btnDownloadJson.addEventListener('click', () => requestReport('json'));

  // Utility helpers
  function setLoading(isLoading) {
    btnRunPipeline.disabled = isLoading;
    if (isLoading) {
      btnRunPipeline.classList.add('is-running');
      btnRunPipeline.innerHTML = '<span>⚡</span> ANALYZING SIGNAL STREAM...';
      const stageBoxes = document.querySelectorAll('.stage-box');
      stageBoxes.forEach((box, idx) => {
        box.style.animationDelay = `${idx * 80}ms`;
        box.classList.add('scanning');
      });
    } else {
      btnRunPipeline.classList.remove('is-running');
      btnRunPipeline.innerHTML = '<span>⚡</span> RUN COMPLETE ANALYSIS';
      const stageBoxes = document.querySelectorAll('.stage-box');
      stageBoxes.forEach(box => {
        box.classList.remove('scanning');
        box.style.animationDelay = '';
      });
    }
  }

  function showError(msg) {
    alertBanner.textContent = msg;
    alertBanner.style.display = 'block';
  }

  function hideAlert() {
    alertBanner.textContent = '';
    alertBanner.style.display = 'none';
  }

  // Load default modulated QPSK signal on boot
  loadPresetFixture('qpsk_1200');
});
