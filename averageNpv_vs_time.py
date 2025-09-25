import hdf5plugin
import h5py
import numpy as np
import matplotlib.pyplot as plt

# ======= HDF5 input file =======
file_name = "new_Data/data_Run_2016_283408_longest.h5"

# ======= Load NPV data from HDF5 =======
with h5py.File(file_name, "r") as h5file:
    npv_data = h5file["PV_npvsGood"][:]

# ======= Remove events with NPV == 0 =======
npv_data = npv_data[npv_data > 0]

# ======= Parameters =======
chunk_size = 100000  # number of events per chunk
num_chunks = len(npv_data) // chunk_size
time_fraction = np.linspace(0, 1, num_chunks, endpoint=True)

# ======= Compute average NPV per chunk =======
avg_npv = []
for i in range(num_chunks):
    start, end = i * chunk_size, (i + 1) * chunk_size
    chunk = npv_data[start:end]
    avg_npv.append(np.mean(chunk))

# ======= Create figure =======
fig, ax = plt.subplots(figsize=(8, 6))

# Scatter plot with colormap, no colorbar
sc = ax.scatter(time_fraction, avg_npv, c=time_fraction,
                cmap="viridis", s=50, edgecolor="black")

# ======= Axis labels and formatting =======
ax.set_xlabel("Time (Fraction of Run)", fontsize=14)
ax.set_ylabel("Average NPV", fontsize=14)
ax.tick_params(labelsize=13)
ax.grid(False)

fig.tight_layout()
fig.savefig("NPV_vs_time.png", dpi=300)
plt.show()
