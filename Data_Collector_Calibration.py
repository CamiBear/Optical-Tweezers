import nidaqmx
from nidaqmx.constants import AcquisitionType, TerminalConfiguration
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time

# Device CONFIGURATION
device = "Dev2"
sample_rate = 25000  # Hz
duration = 7  # seconds
freq = 1  # Hz

# TASK SETUP 
with nidaqmx.Task() as ao_task, nidaqmx.Task() as ai_task:

    # Analog Input
    ai_task.ai_channels.add_ai_voltage_chan(f"{device}/ai0", terminal_config=TerminalConfiguration.RSE)
    ai_task.ai_channels.add_ai_voltage_chan(f"{device}/ai1", terminal_config=TerminalConfiguration.RSE)
    ai_task.ai_channels.add_ai_voltage_chan(f"{device}/ai2", terminal_config=TerminalConfiguration.RSE)
    ai_task.timing.cfg_samp_clk_timing(
        sample_rate,
        sample_mode=AcquisitionType.CONTINUOUS,
        samps_per_chan=sample_rate
    )

    # Start recording
    ai_task.start()

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

# ---- TIME VECTOR ----
t = np.linspace(0, duration, samples_total, endpoint=False)

# ---- SAVE TO CSV ----
df = pd.DataFrame({
    "Time (s)": t,
    "AI0 (dx)": dx,
    "AI1 (dy)": dy,
    "AI2 (sum)": summed,
    "Dist_X (m)": dist_x,
    "Dist_Y (m)": dist_y
})
csv_filename = "Dev2_600mA_data.csv"
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
