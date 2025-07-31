import nidaqmx
from nidaqmx.constants import AcquisitionType, TerminalConfiguration
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time

# ---- CONFIGURATION ----
device = "Dev2"
sample_rate = 25000  # Hz
duration = 7  # seconds
freq = 1  # Hz
ao_amplitude = 3

# ---- SIGNAL GENERATION ----
def create_sine(amplitude, freq, sample_rate, signal_type='bipolar'):
    t = np.linspace(0, 1, sample_rate, endpoint=False)
    signal = 2 * amplitude * np.sin(np.pi * freq * t)
    return np.tile(signal, duration)

ao0_wave = create_sine(ao_amplitude, freq, sample_rate)
ao1_wave = create_sine(ao_amplitude, freq, sample_rate)

# ---- TASK SETUP ----
with nidaqmx.Task() as ao_task, nidaqmx.Task() as ai_task:
    # AO
    ao_task.ao_channels.add_ao_voltage_chan(f"{device}/ao0")
    ao_task.ao_channels.add_ao_voltage_chan(f"{device}/ao1")
    ao_task.timing.cfg_samp_clk_timing(
        sample_rate,
        sample_mode=AcquisitionType.CONTINUOUS,
        samps_per_chan=sample_rate
    )

    # AI
    ai_task.ai_channels.add_ai_voltage_chan(f"{device}/ai0", terminal_config=TerminalConfiguration.RSE)
    ai_task.ai_channels.add_ai_voltage_chan(f"{device}/ai1", terminal_config=TerminalConfiguration.RSE)
    ai_task.ai_channels.add_ai_voltage_chan(f"{device}/ai2", terminal_config=TerminalConfiguration.RSE)
    ai_task.timing.cfg_samp_clk_timing(
        sample_rate,
        sample_mode=AcquisitionType.CONTINUOUS,
        samps_per_chan=sample_rate
    )

    # Start AI
    ai_task.start()

    # Start AO
    ao_task.write(np.vstack((ao0_wave, ao1_wave)), auto_start=True)

    # ---- DATA COLLECTION ----
    collected = []
    start_time = time.time()

    while time.time() - start_time < duration:
        try:
            data = ai_task.read(number_of_samples_per_channel=100, timeout=2)
            collected.append(np.array(data))
        except nidaqmx.errors.DaqError as e:
            print(f"DAQ Error: {e}")
            break

    # Stop tasks
    ai_task.stop()
    ao_task.stop()

# ---- PROCESS DATA ----
data_array = np.hstack(collected)  # shape: (3, N)
samples_total = data_array.shape[1]

dx = data_array[0]
dy = data_array[1]
summed = data_array[2]

# Avoid division by zero
summed[summed == 0] = np.nan
dist_x = dx / summed
dist_y = dy / summed

# Stats
#var_x = np.nanvar(dist_x)
#var_y = np.nanvar(dist_y)

#print(f"Variance X: {var_x:.6e} m²")
#print(f"Variance Y: {var_y:.6e} m²")

# ---- TIME VECTOR ----
t = np.linspace(0, duration, samples_total, endpoint=False)
'''
# ---- SAVE TO CSV ----
df = pd.DataFrame({
    "Time (s)": t,
    "AI0 (dx)": dx,
    "AI1 (dy)": dy,
    "AI2 (sum)": summed,
    "Dist_X (m)": dist_x,
    "Dist_Y (m)": dist_y
})
csv_filename = "Dev2_180mA_Force_data.csv"
df.to_csv(csv_filename, index=False)
print(f"Data saved to: {csv_filename}")

# ---- PLOT ----
plt.figure(figsize=(10, 5))
plt.plot(t, dx, label="AI0 (dx)")
plt.plot(t, dy, label="AI1 (dy)")
plt.plot(t, summed, label="AI2 (sum)")
plt.xlabel("Time (s)")
plt.ylabel("Voltage (V)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

plt.figure()
plt.plot(t, dist_x, label="X Distance")
plt.plot(t, dist_y, label="Y Distance")
plt.xlabel("Time (s)")
plt.ylabel("Voltage (V)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
'''