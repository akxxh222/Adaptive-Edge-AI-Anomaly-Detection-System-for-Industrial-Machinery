# Adaptive Edge AI Anomaly Detection System for Industrial Machinery

> Machine-adaptive anomaly detection on Raspberry Pi using Edge AI — trains on any industrial machine, detects faults in real time, fully offline.

---

## Overview

This system detects mechanical faults in industrial machinery by learning the machine's own healthy vibration and temperature signature, then alerting when behaviour deviates from that baseline. It runs entirely on a Raspberry Pi with no cloud dependency, no pre-labelled fault data, and no internet connection required.

Unlike conventional predictive maintenance systems that are hardcoded for specific machines, this system is **machine-agnostic** — it adapts to any rotating equipment (motors, pumps, compressors, fans, generators) by training on that machine's unique operating profile.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     RASPBERRY PI                            │
│                                                             │
│  ┌──────────────┐     ┌──────────────┐                      │
│  │   ADXL345    │     │   DS18B20    │                      │
│  │  (Vibration) │     │ (Temperature)│                      │
│  │   I2C @ 0x53 │     │   1-Wire     │                      │
│  └──────┬───────┘     └──────┬───────┘                      │
│         │ 100 Hz             │ 1 Hz                         │
│         ▼                    ▼                              │
│  ┌─────────────────────────────────────┐                    │
│  │           SENSOR THREADS            │                    │
│  │  sensor_loop()    temperature_loop()│                    │
│  │  • High-pass filter (0.5 Hz RC)     │                    │
│  │  • 3-axis RMS computation           │                    │
│  │  • Shared Queue (maxsize=5000)      │                    │
│  └─────────────────┬───────────────────┘                    │
│                    │                                        │
│         ┌──────────▼──────────┐                             │
│         │  SIGNAL PROCESSING  │                             │
│         │  • Bandpass filter  │                             │
│         │    (Butterworth 4th │                             │
│         │     order 0.5-45Hz) │                             │
│         │  • Hanning window   │                             │
│         │  • 128-point FFT    │                             │
│         │  • Spectral energy  │                             │
│         └──────────┬──────────┘                             │
│                    │                                        │
│          ┌─────────▼──────────┐                             │
│          │    TRAIN MODE      │   (run once per machine)    │
│          │  • 15 min sampling │                             │
│          │  • Compute μ, σ    │                             │
│          │    per feature     │                             │
│          │  • Save model.json │                             │
│          └─────────┬──────────┘                             │
│                    │                                        │
│          ┌─────────▼──────────┐                             │
│          │   MONITOR MODE     │   (runs continuously)       │
│          │  • Z-score scoring │                             │
│          │    per feature     │                             │
│          │  • Composite score │                             │
│          │  • EMA baseline    │                             │
│          │    drift correction│                             │
│          │  • State detection │                             │
│          └─────────┬──────────┘                             │
│                    │                                        │
│     ┌──────────────▼──────────────────┐                     │
│     │           OUTPUTS               │                     │
│     │  LED (GPIO17)  monitor.csv  log │                     │
│     └─────────────────────────────────┘                     │
└─────────────────────────────────────────────────────────────┘

ANOMALY STATES
──────────────
IDLE    → RMS < 0.08          Machine off or not moving
NORMAL  → score ≤ 3.0         Healthy operation
WARNING → 3.0 < score ≤ 6.0  Deviation detected — LED blinks
ANOMALY → score > 6.0         Fault detected  — LED solid ON

ANOMALY SCORE = |z_rms| + |z_fft| + |z_temp|
where z = (x − μ) / σ  (Z-score per feature)
```

---

## Hardware

| Component | Role | Interface |
|---|---|---|
| Raspberry Pi (any model with 40-pin GPIO) | Main compute | — |
| ADXL345 accelerometer | Vibration sensing (X, Y, Z axes) | I2C (bus 1, addr 0x53) |
| DS18B20 temperature sensor | Thermal monitoring | 1-Wire |



---

## How It Works

### Detection Method — Adaptive Z-Score Statistical Baseline Modeling

This system uses **one-class statistical modeling** — it trains exclusively on healthy machine data, then flags anything that deviates beyond a learned threshold. No labelled fault examples are needed.

**Training phase** computes, for each feature:
- Mean (μ) — the machine's normal operating centre
- Standard deviation (σ) — the machine's normal operating spread

**Monitoring phase** computes a Z-score for each feature:
```
z = (current_value − μ) / σ
```

A composite anomaly score is formed by summing Z-scores across all three features (RMS vibration, spectral energy, temperature). Scores above 3.0 trigger WARNING; above 6.0 trigger ANOMALY.

**EMA drift correction** — during NORMAL state, the model's baseline slowly adapts to long-term ambient changes (e.g. temperature rise over the day) using an exponential moving average (α = 0.90). This prevents false alarms from gradual environmental drift without masking real faults.

### Signal Processing Pipeline

Each accelerometer sample goes through:
1. **High-pass filter (0.5 Hz)** — removes DC gravity component
2. **3-axis RMS** — combines X/Y/Z into single vibration magnitude
3. **Bandpass filter (0.5–45 Hz, Butterworth 4th order)** — isolates mechanical frequency range
4. **Hanning window** — reduces spectral leakage before FFT
5. **128-point FFT** — extracts spectral energy from frequency domain

---

## Installation

```bash
# Enable I2C and 1-Wire on your Pi
sudo raspi-config
# Interface Options → I2C → Enable
# Interface Options → 1-Wire → Enable

# Set I2C to 400kHz for reliable 100Hz polling
# Add to /boot/firmware/config.txt:
# dtparam=i2c_arm=on
# dtparam=i2c_arm_baudrate=400000

sudo reboot

# Install dependencies
pip install -r requirements.txt

# Verify ADXL345 is detected
sudo i2cdetect -y 1
# Should show 53 at address 0x53
```

---

## How to Run

### Step 1 — Train (run once per machine)

Attach the device to the machine while it is running in **healthy, normal operating condition**. Then:

```bash
python3 edge.py --train
```

- Runs for 15 minutes (configurable via `train_duration` in `config.ini`)
- Samples vibration and temperature at 100 Hz
- Computes statistical baseline (mean and standard deviation per feature)
- Saves learned model to `model.json`
- Saves raw training data to `train.csv`

> The machine must be healthy during training. The system learns what "normal" looks like — it has no concept of faults during this phase.

### Step 2 — Monitor (runs continuously)

```bash
python3 edge.py
```

- Loads `model.json`
- Begins real-time anomaly scoring at 100 Hz
- Writes state to `monitor.csv` (columns: `time, rms, fft, temp, state`)
- Logs all readings to `monitor.log`
- Controls LED based on detected state

### Output Files

| File | Created by | Contents |
|---|---|---|
| `model.json` | `--train` | Learned μ and σ per feature |
| `train.csv` | `--train` | Raw training samples (time, rms, fft, temp) |
| `monitor.csv` | monitor mode | Real-time readings with state label |
| `monitor.log` | both modes | Rotating log (1MB cap, 3 backups) |

### Config

All tunable parameters live in `config.ini`. Key settings:

```ini
train_duration = 900      ; training time in seconds
fault_threshold = 6.0     ; anomaly score to trigger ANOMALY state
alarm_threshold = 3.0     ; anomaly score to trigger WARNING state
alpha = 0.90              ; EMA smoothing (higher = slower adaptation)
```

---

## Auto-Start on Boot

The Pi is configured to start monitoring automatically after every boot using systemd. The service loads after the OS is fully ready, waits 10 seconds for hardware to stabilise, then starts `edge.py` in monitor mode.

**Service file location:** `/etc/systemd/system/edge-monitor.service`

```ini
[Unit]
Description=Edge Vibration Monitor
After=multi-user.target
After=sys-bus-w1.device

[Service]
Type=simple
User=teem5821edge
WorkingDirectory=/home/teem5821edge/Desktop
ExecStart=/usr/bin/python3 /home/teem5821edge/Desktop/edge.py
Restart=on-failure
RestartSec=10
StandardOutput=append:/home/teem5821edge/Desktop/monitor.log
StandardError=append:/home/teem5821edge/Desktop/monitor.log
ExecStartPre=/bin/sleep 10

[Install]
WantedBy=multi-user.target
```

To manage the service manually:
```bash
sudo systemctl status edge-monitor    # check status
sudo systemctl stop edge-monitor      # stop monitoring
sudo systemctl start edge-monitor     # start monitoring
sudo systemctl disable edge-monitor   # disable auto-start
```

> Training must be done manually with `python3 edge.py --train` before enabling the service. The service only runs monitor mode.

---


## Project Structure

```
adaptive-edge-ai-anomaly-detection/
├── edge.py           # Main system — sensor reading, training, monitoring
├── config.ini        # All tunable parameters
├── requirements.txt  # Python dependencies
└── README.md         # This file
```

---

## Dependencies

| Library | Purpose |
|---|---|
| `numpy` | Numerical arrays and FFT magnitude computation |
| `scipy` | Butterworth bandpass filter and FFT |
| `smbus2` | I2C communication with ADXL345 |
| `gpiozero` | LED and button GPIO control |
| `w1thermsensor` | DS18B20 1-Wire temperature reading |

---

## Limitations

- Training requires the machine to be in healthy operating condition — there is no automated health check during training
- The system detects anomalies relative to the trained baseline; it does not classify fault type (bearing fault, imbalance, overheating, etc.)
- One model per machine — if the device is moved to a different machine, re-training is required (`python3 edge.py --train`)

---

## Legal

© 2026 Jai Akash LJ & Jeevan Pranav SR. All rights reserved.

This project is publicly visible for portfolio and evaluation purposes only. No permission is granted to use, copy, modify, or distribute this code without explicit written consent from the author.

> ⚠️ This project is a research prototype. Not intended for safety-critical deployments without further validation.
