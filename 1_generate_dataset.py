"""
PhytoSense — Synthetic Bioelectric Signal Dataset Generator
=============================================================
Generates realistic plant bioelectric potential time-series for 5 states,
modeled on patterns reported in plant electrophysiology literature
(resting membrane potential drift, action potentials (APs), variation
potentials (VPs) / slow wave potentials, and stress-specific oscillatory
behavior).

This is SYNTHETIC data meant for demonstrating the ML pipeline before real
electrode data is collected from the PhytoSense hardware (AD620 + ADS1115 + ESP32).
Once real data is logged from the device, replace this generator's output
with real CSVs in the same schema and retrain.

Classes:
    0 = Healthy          - stable resting potential, small stochastic noise
    1 = Drought Stress    - slow gradual depolarization (downward drift), damped oscillation
    2 = Salt Stress        - rapid, higher-frequency oscillatory spiking
    3 = Mechanical/Wound  - sharp fast action-potential spike + slow variation-potential recovery
    4 = Heat Stress          - increased spike frequency + amplitude, faster irregular oscillation

Signal: 500 timesteps per sample (~simulates ADS1115 @ ~5 Hz over 100s window),
values in millivolts (mV), single channel (as recorded from one electrode pair).
"""

import numpy as np
import pandas as pd
import json

RNG = np.random.default_rng(42)
SEQ_LEN = 500
CLASSES = ["Healthy", "Drought Stress", "Salt Stress", "Mechanical/Wound", "Heat Stress"]
N_PER_CLASS = 260  # -> 1300 total samples

t = np.linspace(0, 100, SEQ_LEN)  # 100 seconds window


def resting_baseline(rng, level=-165.0, drift_scale=3.0):
    """Slow random-walk baseline resembling resting membrane potential drift."""
    drift = np.cumsum(rng.normal(0, drift_scale / np.sqrt(SEQ_LEN), SEQ_LEN))
    return level + drift


def add_noise(sig, rng, sigma=0.6):
    return sig + rng.normal(0, sigma, len(sig))


def gen_healthy(rng):
    sig = resting_baseline(rng, level=-165.0, drift_scale=2.0)
    # tiny circadian-like slow oscillation
    sig += 1.5 * np.sin(2 * np.pi * t / 90 + rng.uniform(0, 6.28))
    return add_noise(sig, rng, sigma=0.5)


def gen_drought(rng):
    # gradual depolarization: potential drifts upward (less negative) over time
    sig = resting_baseline(rng, level=-165.0, drift_scale=2.0)
    depol_start = rng.uniform(20, 40)
    depol = np.clip((t - depol_start) / 60, 0, 1) * rng.uniform(18, 32)
    sig += depol
    # oscillation amplitude damps as stress progresses
    damp = np.exp(-t / 70)
    sig += 2.0 * damp * np.sin(2 * np.pi * t / 40)
    return add_noise(sig, rng, sigma=0.7)


def gen_salt(rng):
    sig = resting_baseline(rng, level=-160.0, drift_scale=2.5)
    freq = rng.uniform(0.35, 0.55)  # higher frequency oscillation (Hz-ish in this timebase)
    amp = rng.uniform(6, 11)
    sig += amp * np.sin(2 * np.pi * freq * t + rng.uniform(0, 6.28))
    sig += amp * 0.4 * np.sin(2 * np.pi * (freq * 2.3) * t)
    return add_noise(sig, rng, sigma=1.0)


def gen_mechanical(rng):
    sig = resting_baseline(rng, level=-165.0, drift_scale=1.5)
    spike_t = rng.uniform(15, 30)
    idx = np.argmin(np.abs(t - spike_t))
    # fast action potential: sharp depolarization + fast repolarization
    ap = 45 * np.exp(-((t - spike_t) ** 2) / (2 * 0.6 ** 2))
    sig += ap
    # slow variation potential: long, broad recovery wave following the AP
    vp_center = spike_t + rng.uniform(15, 25)
    vp = 20 * np.exp(-((t - vp_center) ** 2) / (2 * 12 ** 2))
    sig += vp
    return add_noise(sig, rng, sigma=0.6)


def gen_heat(rng):
    sig = resting_baseline(rng, level=-158.0, drift_scale=3.0)
    n_spikes = rng.integers(4, 8)
    for _ in range(n_spikes):
        st = rng.uniform(5, 95)
        amp = rng.uniform(8, 18)
        width = rng.uniform(1.0, 2.5)
        sig += amp * np.exp(-((t - st) ** 2) / (2 * width ** 2))
    sig += 3.0 * np.sin(2 * np.pi * 0.25 * t)
    return add_noise(sig, rng, sigma=1.1)


GENERATORS = [gen_healthy, gen_drought, gen_salt, gen_mechanical, gen_heat]

rows = []
signal_matrix = []
labels = []
for class_idx, gen_fn in enumerate(GENERATORS):
    for _ in range(N_PER_CLASS):
        seed_rng = np.random.default_rng(RNG.integers(0, 2**31 - 1))
        sig = gen_fn(seed_rng)
        signal_matrix.append(sig)
        labels.append(class_idx)

signal_matrix = np.array(signal_matrix)  # (N, 500)
labels = np.array(labels)

# Shuffle
perm = RNG.permutation(len(labels))
signal_matrix = signal_matrix[perm]
labels = labels[perm]

# Save as CSV: sample_id, label, label_name, t0..t499
df = pd.DataFrame(signal_matrix, columns=[f"t{i}" for i in range(SEQ_LEN)])
df.insert(0, "label_name", [CLASSES[l] for l in labels])
df.insert(0, "label", labels)
df.insert(0, "sample_id", np.arange(len(labels)))

df.to_csv("D:\COLLEGE\project\Phytosense\signals\phytosense_dataset.csv", index=False)

# Save a small JSON sample set (a few per class) for embedding in the dashboard demo
demo_samples = []
for class_idx in range(len(CLASSES)):
    class_rows = df[df["label"] == class_idx].head(6)
    for _, row in class_rows.iterrows():
        demo_samples.append({
            "id": int(row["sample_id"]),
            "label": int(row["label"]),
            "label_name": row["label_name"],
            "signal": [round(float(row[f"t{i}"]), 3) for i in range(SEQ_LEN)]
        })

with open("D:\COLLEGE\project\Phytosense\signals\demo_samples.json", "w") as f:
    json.dump({"classes": CLASSES, "samples": demo_samples}, f)

print("Dataset shape:", signal_matrix.shape)
print("Class distribution:", {CLASSES[i]: int((labels == i).sum()) for i in range(len(CLASSES))})
print("Saved: phytosense_dataset.csv, demo_samples.json")
