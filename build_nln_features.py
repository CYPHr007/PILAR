#!/usr/bin/env python3
"""
build_nln_features.py — Stream-process the full NLN-EMP 20GB archive
=====================================================================
Extracts features from EVERY fault class × load level × channel
by piping 7-Zip stdout directly into pandas — no temp files needed.

Output: nln_emp_features.csv  (~100 KB, all features pre-computed)
Usage:  python build_nln_features.py
"""
import subprocess, sys, io, json
import numpy as np
import pandas as pd
from pathlib import Path

SEVENZIP = r"C:\Program Files\7-Zip\7z.exe"
ARCHIVE  = (
    r"C:\Users\info\Downloads\extraction"
    r"\Motor Current and Vibration Monitoring Dataset for various Faults in an E-motor-driven Centrifugal Pump"
    r"\Dataset\Dataset.7z"
)
OUT_CSV   = "nln_emp_features.csv"
MAX_ROWS  = 4_000    # 0.2 s at 20 kHz — enough for RMS, kurtosis, and 50 Hz FFT
G_TO_MMS  = 9810.0 / (2.0 * np.pi * 50.0)   # ~31.2 mm/s per g-rms

FEATURE_MEDIANS = {
    'vibration': 0.61, 'temp_palier': 44.837, 'debit': 0.395,
    'pression_entree': 1.987, 'pression_sortie': 107.73,
    'courant_moteur': 4.579, 'temp_moteur': 58.043,
    'heure_fonctionnement': 1103.0,
}
COLONNES = list(FEATURE_MEDIANS.keys())

# ── Zone label mapping ────────────────────────────────────────────────────────
# Maps lowercase fault-class name keywords → (y_fault, zone)
# y_fault=0 means healthy/normal; zone=None means healthy
ZONE_MAP = [
    (['broken rotor bar', 'stator short'],          'MOT'),
    (['bearing bpfi', 'bearing bpfo', 'bearing bsf',
      'bearing contaminated', 'bearing pump'],       'ROL'),
    (['impeller'],                                   'IMP'),
    (['healthy', 'new motor'],                       None),   # normal
    (['loose foot', 'soft foot'],                    'MECH'), # structural
]

def _zone_of(fault_name: str):
    """Return zone string or None (normal). 'MECH' = unclassified mechanical."""
    fn = fault_name.lower()
    for keywords, zone in ZONE_MAP:
        if any(k in fn for k in keywords):
            return zone
    return 'OTHER'


def _rms(a):
    return float(np.sqrt(np.mean(a ** 2)))

def _kurtosis(a):
    if len(a) < 4: return 0.0
    mu, std = a.mean(), a.std()
    if std == 0: return 0.0
    return float(np.mean(((a - mu) / std) ** 4) - 3.0)

def _neg_seq_ratio(Ia, Ib, Ic):
    """
    Negative sequence current ratio (IEC/NEMA motor asymmetry indicator).
    Uses symmetrical components: I_neg = (Ia + a^2*Ib + a*Ic) / 3
    where a = exp(j*2*pi/3).
    Returns |I_neg| / |I_pos| as a ratio (0 = perfectly balanced).
    Motor faults: rotor bar break / stator short -> ratio 0.05-0.20+
    Healthy motor: ratio < 0.02
    """
    a  = np.exp(1j * 2 * np.pi / 3)
    a2 = np.exp(1j * 4 * np.pi / 3)
    # Build complex phasors from time-domain signals (use FFT at fundamental)
    N = len(Ia)
    freq_bins = np.fft.rfftfreq(N, d=1.0/20000.0)
    fa = np.fft.rfft(Ia) / N
    fb = np.fft.rfft(Ib) / N if Ib is not None else fa * 0
    fc = np.fft.rfft(Ic) / N if Ic is not None else fa * 0
    # Find fundamental frequency bin (closest to 50 Hz)
    fund_idx = int(np.argmin(np.abs(freq_bins - 50.0)))
    Ia_f = fa[fund_idx]
    Ib_f = fb[fund_idx]
    Ic_f = fc[fund_idx]
    I_pos = (Ia_f + a  * Ib_f + a2 * Ic_f) / 3.0
    I_neg = (Ia_f + a2 * Ib_f + a  * Ic_f) / 3.0
    mag_pos = abs(I_pos)
    if mag_pos < 1e-6:
        return 0.0
    return float(abs(I_neg) / mag_pos)

def _thd(a, fs=20000.0, fund_hz=50.0, n_harmonics=10):
    """
    Total Harmonic Distortion of a current signal.
    THD = sqrt(sum of harmonic^2) / fundamental amplitude.
    Returns THD as a ratio (0 = pure sine).
    Motor faults increase THD: stator short typically 0.05-0.20+
    """
    N = len(a)
    freqs = np.fft.rfftfreq(N, d=1.0/fs)
    spectrum = np.abs(np.fft.rfft(a)) / N
    fund_idx = int(np.argmin(np.abs(freqs - fund_hz)))
    fund_amp = spectrum[fund_idx]
    if fund_amp < 1e-6:
        return 0.0
    harmonic_power = 0.0
    for h in range(2, n_harmonics + 1):
        hidx = int(np.argmin(np.abs(freqs - h * fund_hz)))
        harmonic_power += spectrum[hidx] ** 2
    return float(np.sqrt(harmonic_power) / fund_amp)


def _stream_csv(archive_path: str, inner_path: str, max_rows: int) -> np.ndarray | None:
    """
    Stream one file from the 7z archive via stdout.
    Returns (n_rows × n_run_cols) numeric array, or None on failure.
    """
    cmd = [SEVENZIP, 'e', archive_path, inner_path, '-so', '-y']
    try:
        proc = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL
        )
        # Read raw bytes then wrap in StringIO for pandas
        raw = proc.stdout.read(max_rows * 400)   # ~400 bytes per row estimate
        proc.terminate()
        proc.wait()
        if not raw:
            return None
        # Decode — NLN-EMP CSVs are ASCII
        text = raw.decode('ascii', errors='replace')
        df = pd.read_csv(io.StringIO(text), nrows=max_rows, low_memory=False,
                         on_bad_lines='skip')
        if len(df) < 200:
            return None
        run_cols = [c for c in df.columns if str(c).strip().lstrip('-').isdigit()]
        if not run_cols:
            return None
        return df[run_cols].apply(pd.to_numeric, errors='coerce').dropna().values
    except Exception as e:
        return None


def extract_features(load: str, fault: str) -> list[dict]:
    """
    For one (load, fault) pair, stream Electric ch1/ch2/ch3 + Vibration ch1
    and return one feature dict per run column.
    """
    def _inner(subfolder, ch):
        fname = f"{subfolder}_Motor-2_{load}_time-{fault}-ch{ch}.csv"
        return f"{subfolder}\\Motor-2\\{load}\\{fault}\\{fname}"

    A = _stream_csv(ARCHIVE, _inner('Electric', 1), MAX_ROWS)
    if A is None:
        return []

    B  = _stream_csv(ARCHIVE, _inner('Electric', 2), MAX_ROWS)
    C  = _stream_csv(ARCHIVE, _inner('Electric', 3), MAX_ROWS)
    V1 = _stream_csv(ARCHIVE, _inner('Vibration', 1), MAX_ROWS)

    n_runs = A.shape[1]
    rows = []

    for i in range(n_runs):
        row = dict(FEATURE_MEDIANS)

        Ia = A[:, i]
        Ib = B[:, i] if (B is not None and i < B.shape[1]) else None
        Ic = C[:, i] if (C is not None and i < C.shape[1]) else None

        # 3-phase RMS
        phase_rms = [_rms(Ia)]
        if Ib is not None: phase_rms.append(_rms(Ib))
        if Ic is not None: phase_rms.append(_rms(Ic))
        phase_rms = np.array(phase_rms)
        row['courant_moteur'] = float(np.sqrt(np.mean(phase_rms ** 2)))

        # Negative sequence current ratio → motor electrical fault indicator
        # Healthy: ~0.01-0.02 | Rotor bar / stator short: 0.05-0.20+
        # Scale to temp_moteur range: baseline + neg_seq_ratio * 200
        # (at 20% neg_seq → temp_moteur = 58 + 40 = 98°C, physically plausible)
        if Ib is not None and Ic is not None:
            neg_seq = _neg_seq_ratio(Ia, Ib, Ic)
            row['temp_moteur'] = FEATURE_MEDIANS['temp_moteur'] + neg_seq * 200.0
        elif len(phase_rms) >= 2:
            # Fallback: simple imbalance proxy
            row['temp_moteur'] = FEATURE_MEDIANS['temp_moteur'] + float(phase_rms.std()) * 10.0

        # THD of phase A → harmonic distortion (motor winding faults)
        # Healthy: THD ~0.01-0.05 | Stator short: 0.05-0.25
        # Map to vibration as a secondary motor fault indicator:
        # normal vibration (0.61 mm/s baseline) + THD * 5
        thd_a = _thd(Ia)
        # We'll store THD as extra column (used in zone retrain)
        row['_thd'] = thd_a

        # Kurtosis of phase A → bearing proxy
        kurt_I = _kurtosis(Ia)
        row['temp_palier'] = FEATURE_MEDIANS['temp_palier'] + max(0.0, (kurt_I - 3.0) * 1.5)

        # Vibration (accelerometer in g → mm/s)
        if V1 is not None and i < V1.shape[1]:
            row['vibration'] = _rms(V1[:, i]) * G_TO_MMS
            kurt_v = _kurtosis(V1[:, i])
            row['temp_palier'] = max(row['temp_palier'],
                FEATURE_MEDIANS['temp_palier'] + max(0.0, (kurt_v - 3.0) * 1.5))

        rows.append(row)
    return rows


# ── Main: iterate every fault × load ─────────────────────────────────────────
FAULTS = [
    'bearing bpfi 1', 'bearing bpfi 2', 'bearing bpfi 3',
    'bearing bpfo 1', 'bearing bpfo 2', 'bearing bpfo 3',
    'bearing bsf', 'bearing contaminated',
    'bearing pump 1', 'bearing pump 2', 'bearing pump 3',
    'broken rotor bar',
    'healthy 1', 'healthy 2', 'healthy 3', 'healthy noise',
    'impeller 1', 'impeller 2', 'impeller 3',
    'loose foot motor', 'loose foot pump',
    'new motor',
    'soft foot 1', 'soft foot 2',
    'stator short 1', 'stator short 2',
]
LOADS = ['100', '75', '50']

all_rows = []
total = len(FAULTS) * len(LOADS)
done = 0

print(f"Processing {len(FAULTS)} faults × {len(LOADS)} loads = {total} combinations")
print("Streaming from 7z archive (no temp files)...\n")

for load in LOADS:
    for fault in FAULTS:
        done += 1
        zone = _zone_of(fault)
        rows = extract_features(load, fault)
        n = len(rows)
        status = f"[{done:3d}/{total}] {load}% {fault:30s} zone={zone or 'NORMAL':6s} -> {n} samples"
        print(status)
        sys.stdout.flush()
        if rows:
            for r in rows:
                r['fault']  = fault
                r['load']   = int(load)
                r['zone']   = zone if zone else 'NORMAL'
                r['y_fault'] = 0 if zone is None else 1
                all_rows.append(r)

if not all_rows:
    print("ERROR: no rows extracted.")
    sys.exit(1)

df = pd.DataFrame(all_rows)
df.to_csv(OUT_CSV, index=False)
print(f"\nSaved {len(df)} rows -> {OUT_CSV}")
print(f"Zone distribution:\n{df['zone'].value_counts().to_string()}")
print(f"Fault distribution:\n{df['fault'].value_counts().to_string()}")
