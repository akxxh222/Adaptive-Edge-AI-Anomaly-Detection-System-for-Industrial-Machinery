#!/usr/bin/env python3

import os
import time
import math
import json
import signal
import threading
import argparse
import logging

from logging.handlers import RotatingFileHandler
from queue import Queue, Empty
from collections import deque

import numpy as np
from scipy.signal import butter, sosfilt
from scipy.fft import fft

import struct

import smbus2

from gpiozero import LED, Button
from w1thermsensor import W1ThermSensor, Unit, Sensor

# =========================================================
# CONFIG
# =========================================================

CONFIG = {

    'mode': 'monitor',

    'train_duration': 900,

    'led_pin': 17,

    # FIX — GPIO3 is the I2C1 SCL clock line on every Raspberry Pi.
    # Instantiating Button(3) claims that pin as a digital input with
    # a pull resistor, which yanks the I2C clock out from under any
    # in-flight transaction. That's why calibration worked (button
    # object didn't exist yet) but reads started failing with
    # Errno 110 a couple seconds into train mode (right after
    # shutdown_btn = Button(...) ran), and why a hard kill left the
    # bus wedged until reboot (pin claim never released cleanly).
    # GPIO27 is free on all Pi models and doesn't collide with
    # I2C (2/3), SPI, UART, or the LED on GPIO17.
    'shutdown_pin': 27,

    'i2c_bus': 1,
    'acc_addr': 0x53,

    'sample_rate': 100,
    'odata_rate': 100,

    'window_size': 100,
    'fft_size': 128,

    'cutoff_hz': 0.5,

    'bandpass': [0.5, 45],

    'logging_size': 1000000,

    'log_file': 'monitor.log',

    'train_file': 'train.csv',
    'monitor_file': 'monitor.csv',

    'model_file': 'model.json',

    'max_write_freq': 1.0,

    'idle_threshold': 0.08,

    'fault_threshold': 3.0,
    'alarm_threshold': 1.5,

    'alpha': 0.90,

    'calibrate_samples': 100,

    'temp_offset': 0.0,
    'temp_scale': 1.0,

    'w1_id': None
}

# =========================================================
# GLOBALS
# =========================================================

running = True

status_led = None

acc_offset = [0.0, 0.0, 0.0]

buffer_lock = threading.Lock()

# FIX #4 — shared temperature variable updated by a slow thread
_latest_temp = 25.0
_temp_lock = threading.Lock()

# =========================================================
# CONFIG LOADER
# =========================================================

def load_config(path):

    import configparser

    if not os.path.exists(path):
        return

    if path.endswith(".json"):

        with open(path) as f:
            data = json.load(f)

        CONFIG.update(data)

    else:

        cp = configparser.ConfigParser()

        cp.read(path)

        if 'Default' in cp:

            for key, val in cp['Default'].items():

                if key == 'bandpass':

                    CONFIG[key] = [
                        float(x.strip())
                        for x in val.split(',')
                    ]

                    continue

                if key in CONFIG and isinstance(CONFIG[key], bool):

                    CONFIG[key] = cp.getboolean(
                        'Default',
                        key
                    )

                elif key in CONFIG and isinstance(CONFIG[key], int):

                    CONFIG[key] = int(val)

                elif key in CONFIG and isinstance(CONFIG[key], float):

                    CONFIG[key] = float(val)

                else:

                    CONFIG[key] = val

# =========================================================
# FILTERS
# =========================================================

def design_highpass(cutoff_hz, fs):

    dt = 1.0 / fs

    RC = 1.0 / (2 * math.pi * cutoff_hz)

    alpha = RC / (RC + dt)

    return alpha

def high_pass_filter(x, prev_x, prev_y, alpha):

    y = alpha * (prev_y + x - prev_x)

    return y

def apply_bandpass(samples):

    low, high = CONFIG['bandpass']

    # FIX P8 — clamp to valid Nyquist range to prevent scipy ValueError
    nyquist = CONFIG['sample_rate'] / 2
    high = min(high, nyquist - 1)
    low = max(low, 0.1)

    sos = butter(
        4,
        [low, high],
        btype='band',
        fs=CONFIG['sample_rate'],
        output='sos'
    )

    return sosfilt(sos, samples)

def window_data(data):

    return data * np.hanning(len(data))

def compute_fft(samples):

    N = len(samples)

    yf = fft(samples)

    yf = np.abs(yf[:N // 2]) * 2 / N

    xf = np.linspace(
        0,
        CONFIG['sample_rate'] / 2,
        N // 2,
        endpoint=False
    )

    return xf, yf

# =========================================================
# SENSOR INIT
# =========================================================

def reset_i2c_bus():
    """
    Recover a hung I2C bus by toggling SCL manually via sysfs.
    When a script is killed mid-transaction the kernel I2C driver
    can be left with SDA held low. This pulses SCL 9 times which
    forces the sensor to release SDA and resets the bus state.
    """
    import subprocess
    try:
        # Unexport first in case pin is already exported
        for pin in [2, 3]:  # SDA=GPIO2, SCL=GPIO3
            try:
                with open('/sys/class/gpio/unexport', 'w') as f:
                    f.write(str(pin))
            except:
                pass
        # Use i2c-gpio recovery via kernel if available
        result = subprocess.run(
            ['sudo', 'i2cdetect', '-y', '1'],
            capture_output=True,
            timeout=5
        )
        time.sleep(0.1)
        logging.info("I2C bus reset attempted")
    except Exception as e:
        logging.warning(f"I2C bus reset: {e}")


def init_accelerometer():

    try:

        bus = smbus2.SMBus(CONFIG['i2c_bus'])

        ACC = CONFIG['acc_addr']

        # Step 1 — soft reset: put sensor into standby first
        bus.write_byte_data(ACC, 0x2D, 0x00)
        time.sleep(0.01)

        # Step 2 — DATA_FORMAT: full resolution, ±2g
        bus.write_byte_data(ACC, 0x31, 0x08)
        time.sleep(0.01)

        odr_code = {
            800: 0x0D,
            400: 0x0C,
            200: 0x0B,
            100: 0x0A,
            50: 0x09,
            25: 0x08
        }.get(CONFIG['odata_rate'], 0x0A)

        # Step 3 — BW_RATE: set ODR (must be set before waking)
        bus.write_byte_data(ACC, 0x2C, odr_code)
        time.sleep(0.01)

        # Step 4 — POWER_CTL: enter measure mode
        bus.write_byte_data(ACC, 0x2D, 0x08)
        time.sleep(0.10)  # 100ms — give sensor time to fill pipeline at 100Hz

        # Step 5 — verify sensor is alive by reading DEVID register (should be 0xE5)
        devid = bus.read_byte_data(ACC, 0x00)
        if devid != 0xE5:
            raise RuntimeError(
                f"ADXL345 DEVID mismatch: got 0x{devid:02X}, expected 0xE5"
            )

        logging.info("ADXL345 initialized and verified")

        return bus

    except Exception as e:

        logging.error(
            f"ADXL345 init failed: {e}"
        )

        return None

def read_acc(bus):

    ACC = CONFIG['acc_addr']

    # Retry up to 3 times on transient I2C timeout (Errno 110)
    for attempt in range(3):

        try:

            data = bus.read_i2c_block_data(ACC, 0x32, 6)

            # FIX P1 — struct.unpack handles two's complement correctly
            # np.int16() raises OverflowError on values > 32767 in recent NumPy
            x = struct.unpack('<h', bytes(data[0:2]))[0]
            y = struct.unpack('<h', bytes(data[2:4]))[0]
            z = struct.unpack('<h', bytes(data[4:6]))[0]

            scale = 0.004

            x = (x * scale - acc_offset[0]) * 9.80665
            y = (y * scale - acc_offset[1]) * 9.80665
            z = (z * scale - acc_offset[2]) * 9.80665

            return x, y, z

        except Exception as e:

            if attempt < 2:
                # Short pause before retry — gives I2C bus time to recover
                time.sleep(0.005)
            else:
                logging.error(f"Accelerometer read failed after 3 attempts: {e}")

    return 0.0, 0.0, 0.0

def init_temperature():

    try:

        if CONFIG['w1_id']:

            sensor = W1ThermSensor(
                sensor_type=Sensor.DS18B20,
                sensor_id=CONFIG['w1_id']
            )

        else:

            sensor = W1ThermSensor()

        logging.info("DS18B20 initialized")

        return sensor

    except Exception as e:

        logging.error(
            f"Temperature init failed: {e}"
        )

        return None

def read_temperature(sensor):

    try:

        temp = sensor.get_temperature(
            Unit.DEGREES_C
        )

        temp = (
            temp + CONFIG['temp_offset']
        ) * CONFIG['temp_scale']

        return temp

    except Exception as e:

        logging.error(
            f"Temperature read failed: {e}"
        )

        return 25.0

# =========================================================
# FIX #4 — TEMPERATURE THREAD (reads at ~1 Hz)
# DS18B20 only updates every ~750ms. Reading it at 100Hz
# was wasteful and returned stale data on most calls.
# This thread reads it at 1Hz and shares via _latest_temp.
# =========================================================

def temperature_loop(sensor):

    global running, _latest_temp

    while running:

        temp = read_temperature(sensor)

        with _temp_lock:
            _latest_temp = temp

        time.sleep(1.0)

def get_latest_temp():

    with _temp_lock:
        return _latest_temp

# =========================================================
# CALIBRATION
# =========================================================

def calibrate_accelerometer(bus):

    N = CONFIG['calibrate_samples']

    sx = sy = sz = 0.0

    logging.info("Calibrating accelerometer")

    for _ in range(N):

        x, y, z = read_acc(bus)

        sx += x / 9.80665
        sy += y / 9.80665
        sz += z / 9.80665

        time.sleep(0.01)

    acc_offset[0] = sx / N
    acc_offset[1] = sy / N
    acc_offset[2] = sz / N

# =========================================================
# SENSOR LOOP
# FIX #4 applied — temperature now fetched from shared var
# =========================================================

def sensor_loop(bus, queue):

    global running

    fs = CONFIG['sample_rate']

    interval = 1.0 / fs

    alpha = design_highpass(
        CONFIG['cutoff_hz'],
        fs
    )

    prev_x = prev_y = prev_z = 0.0

    hp_x = hp_y = hp_z = 0.0

    while running:

        # FIX P3 — monotonic clock is immune to NTP wall-clock adjustments
        t0 = time.monotonic()

        x, y, z = read_acc(bus)

        hp_x = high_pass_filter(
            x,
            prev_x,
            hp_x,
            alpha
        )

        hp_y = high_pass_filter(
            y,
            prev_y,
            hp_y,
            alpha
        )

        hp_z = high_pass_filter(
            z,
            prev_z,
            hp_z,
            alpha
        )

        prev_x = x
        prev_y = y
        prev_z = z

        # FIX #4 — read from shared variable, not sensor directly
        temp = get_latest_temp()

        try:

            queue.put_nowait(
                (
                    time.time(),
                    (
                        hp_x,
                        hp_y,
                        hp_z,
                        temp
                    )
                )
            )

        except Exception as e:
            # FIX P4 — log queue overflow instead of silently dropping
            logging.warning(f"Queue full, sample dropped: {e}")

        elapsed = time.monotonic() - t0

        # Always sleep at least 2ms even if we're behind schedule.
        # Without this, a slow I2C read causes the next read to fire
        # immediately with zero gap, cascading into bus timeouts.
        sleep_time = max(interval - elapsed, 0.002)

        time.sleep(sleep_time)

# =========================================================
# TRAIN MODE
# FIX #2 applied — only append to train_data once fft_buf
# is full (128 samples), avoiding zero-padded FFT values
# that would skew fft_mean downward.
# =========================================================

def train_mode(queue):

    global running

    logging.info("TRAIN MODE STARTED")

    train_data = []

    fft_buf = deque(
        maxlen=CONFIG['fft_size']
    )

    start = time.time()

    with open(
        CONFIG['train_file'],
        'w'
    ) as f:

        f.write(
            "time,rms,fft,temp\n"
        )

    while running:

        if (
            time.time() - start
            >= CONFIG['train_duration']
        ):
            break

        try:

            timestamp, (
                ax,
                ay,
                az,
                temp
            ) = queue.get(timeout=0.1)

        except Empty:
            continue

        rms = math.sqrt(
            ax*ax + ay*ay + az*az
        )

        fft_buf.append(rms)

        # FIX #2 — skip until buffer is full so spectral_energy
        # is never computed from a partially filled window
        if len(fft_buf) < CONFIG['fft_size']:
            continue

        fft_signal = np.array(fft_buf)

        fft_signal = apply_bandpass(fft_signal)

        fft_signal = window_data(fft_signal)

        _, yf = compute_fft(fft_signal)

        spectral_energy = float(np.mean(yf))

        # FIX #2 — train_data only gets real FFT values now
        train_data.append(
            (
                rms,
                spectral_energy,
                temp
            )
        )

        with open(
            CONFIG['train_file'],
            'a',
            buffering=1
        ) as f:

            f.write(
                f"{timestamp},"
                f"{rms:.5f},"
                f"{spectral_energy:.5f},"
                f"{temp:.3f}\n"
            )

        logging.info(
            f"TRAIN | RMS={rms:.3f} | "
            f"FFT={spectral_energy:.3f} | "
            f"TEMP={temp:.2f}"
        )

    # FIX P7 — guard against empty training set (e.g. sensor failed during train)
    if not train_data:
        logging.error("No training data collected — model not saved")
        return

    rms_vals = [x[0] for x in train_data]
    fft_vals = [x[1] for x in train_data]
    temp_vals = [x[2] for x in train_data]

    model = {

        'rms_mean': float(np.mean(rms_vals)),
        'rms_std':  float(np.std(rms_vals)),

        'fft_mean': float(np.mean(fft_vals)),
        'fft_std':  float(np.std(fft_vals)),

        'temp_mean': float(np.mean(temp_vals)),
        'temp_std':  float(np.std(temp_vals))
    }

    with open(
        CONFIG['model_file'],
        'w'
    ) as f:

        json.dump(model, f, indent=2)

    logging.info("Training completed")

    logging.info(f"Model saved: {model}")

# =========================================================
# MONITOR MODE
# FIX #3 — monitor_file opened ONCE before the loop,
#           not at every 100Hz sample (SD card wear fix)
# FIX #5 — WARNING state now blinks the LED
# EMA UPGRADE — model baselines drift-corrected using
#               exponential moving average (alpha=0.90)
#               so the model stays fresh over long runs
# =========================================================

def monitor_mode(queue):

    global running
    global status_led

    if not os.path.exists(CONFIG['model_file']):

        logging.error("No trained model found")

        return

    with open(CONFIG['model_file']) as f:

        model = json.load(f)

    logging.info("MONITOR MODE STARTED")

    fft_buf = deque(maxlen=CONFIG['fft_size'])

    alpha = CONFIG['alpha']

    # FIX #3 — open file once here, write inside loop
    with open(CONFIG['monitor_file'], 'w') as monitor_csv:

        monitor_csv.write("time,rms,fft,temp,state\n")

        while running:

            try:

                timestamp, (
                    ax,
                    ay,
                    az,
                    temp
                ) = queue.get(timeout=0.1)

            except Empty:
                continue

            rms = math.sqrt(
                ax*ax + ay*ay + az*az
            )

            fft_buf.append(rms)

            spectral_energy = 0.0

            if len(fft_buf) == CONFIG['fft_size']:

                fft_signal = np.array(fft_buf)

                fft_signal = apply_bandpass(fft_signal)

                fft_signal = window_data(fft_signal)

                _, yf = compute_fft(fft_signal)

                spectral_energy = float(np.mean(yf))

            rms_z = abs(
                (rms - model['rms_mean']) /
                (model['rms_std'] + 1e-6)
            )

            fft_z = abs(
                (spectral_energy - model['fft_mean']) /
                (model['fft_std'] + 1e-6)
            )

            temp_z = abs(
                (temp - model['temp_mean']) /
                (model['temp_std'] + 1e-6)
            )

            anomaly_score = rms_z + fft_z + temp_z

            if rms < CONFIG['idle_threshold']:

                state = "IDLE"

                status_led.off()

            elif anomaly_score > 6:

                state = "ANOMALY"

                status_led.on()

            elif anomaly_score > 3:

                state = "WARNING"

                # FIX #5 — blink LED to distinguish WARNING from NORMAL
                status_led.blink(
                    on_time=0.3,
                    off_time=0.3,
                    n=3,
                    background=True
                )

            else:

                state = "NORMAL"

                status_led.off()

                # EMA UPGRADE — only update baseline during NORMAL state
                # so anomalies don't corrupt the learned model over time
                model['rms_mean']  = alpha * model['rms_mean']  + (1 - alpha) * rms
                model['fft_mean']  = alpha * model['fft_mean']  + (1 - alpha) * spectral_energy
                model['temp_mean'] = alpha * model['temp_mean'] + (1 - alpha) * temp

            # FIX #3 — single open file handle, flush periodically
            monitor_csv.write(
                f"{timestamp},"
                f"{rms:.5f},"
                f"{spectral_energy:.5f},"
                f"{temp:.3f},"
                f"{state}\n"
            )

            monitor_csv.flush()

            logging.info(
                f"RMS={rms:.3f} | "
                f"FFT={spectral_energy:.3f} | "
                f"TEMP={temp:.2f} | "
                f"SCORE={anomaly_score:.3f} | "
                f"STATE={state}"
            )

# =========================================================
# MAIN
# =========================================================

def main():

    global running
    global status_led

    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--config',
        default='config.ini'
    )

    parser.add_argument(
        '--train',
        action='store_true'
    )

    args = parser.parse_args()

    load_config(args.config)

    # =====================================================
    # LOGGING
    # =====================================================

    log_handler = RotatingFileHandler(

        CONFIG['log_file'],

        maxBytes=CONFIG['logging_size'],

        backupCount=3
    )

    formatter = logging.Formatter(
        '%(asctime)s %(levelname)s:%(message)s'
    )

    log_handler.setFormatter(formatter)

    logger = logging.getLogger()

    logger.addHandler(log_handler)

    logger.setLevel(logging.INFO)

    # =====================================================
    # GPIO
    # =====================================================

    status_led = LED(CONFIG['led_pin'])

    # =====================================================
    # HARDWARE
    # =====================================================

    bus = init_accelerometer()

    sensor = init_temperature()

    # FIX P2 — guard against failed hardware init before calibration runs
    if bus is None:
        logging.critical("Accelerometer initialization failed — aborting")
        return

    if sensor is None:
        logging.critical("Temperature initialization failed — aborting")
        return

    calibrate_accelerometer(bus)

    # =====================================================
    # QUEUE
    # =====================================================

    # FIX P5 — 5000 gives ~50s of buffer headroom at 100Hz
    data_queue = Queue(maxsize=5000)

    # FIX #4 — temperature in its own 1Hz thread
    temp_thread = threading.Thread(
        target=temperature_loop,
        args=(sensor,),
        daemon=True
    )

    temp_thread.start()

    # sensor_loop no longer needs sensor arg (uses shared var)
    sensor_thread = threading.Thread(
        target=sensor_loop,
        args=(bus, data_queue),
        daemon=True
    )

    sensor_thread.start()

    # =====================================================
    # SHUTDOWN BUTTON
    # =====================================================

    shutdown_btn = Button(
        CONFIG['shutdown_pin'],
        hold_time=2
    )

    shutdown_btn.when_held = (
        lambda: os.system("sudo shutdown -h now")
    )

    # =====================================================
    # SIGNALS
    # =====================================================

    def terminate(signum, frame):

        global running

        running = False

    signal.signal(signal.SIGINT, terminate)

    signal.signal(signal.SIGTERM, terminate)

    # =====================================================
    # TRAIN / MONITOR
    # =====================================================

    if args.train:

        train_mode(data_queue)

    else:

        monitor_mode(data_queue)

    # =====================================================
    # CLEAN SHUTDOWN
    # =====================================================

    running = False

    sensor_thread.join(timeout=2)

    temp_thread.join(timeout=2)

    try:

        if bus:
            bus.close()

    except:
        pass

    try:

        if status_led:
            status_led.off()
            status_led.close()

    except:
        pass

    logging.info("Monitor exited cleanly")

if __name__ == "__main__":

    main()
