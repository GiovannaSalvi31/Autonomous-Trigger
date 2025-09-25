import h5py
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import hdf5plugin
from matplotlib.lines import Line2D

# ======= Function to manually calculate HT =======
def load_and_calculate_ht(filename):
    with h5py.File(filename, "r") as f:
        n_events = f["j0Eta"].shape[0]
        n_jets = 8  # up to j7
        ht_values = np.zeros(n_events)

        for i in range(n_jets):
            eta = f[f"j{i}Eta"][:]
            pt = f[f"j{i}Pt"][:]
            # selection: pt > 20 and |eta| < 2.5
            mask = (pt > 20) & (np.abs(eta) < 2.5)
            ht_values += pt * mask

    return ht_values

# ======= Load data =======
file_name = "new_Data/data_Run_2016_283408_longest.h5"
Ht_manual = load_and_calculate_ht(file_name)

# ======= Parameters =======
chunk_size = 50000
num_chunks = len(Ht_manual) // chunk_size
time_fraction = np.linspace(0, 1, len(Ht_manual), endpoint=False)

# ======= DataFrame for violin plot =======
df = pd.DataFrame({
    "HT": Ht_manual,
    "time_frac": time_fraction
})

n_time_bins = 10
df["time_bin"] = pd.cut(df["time_frac"], bins=n_time_bins)

# ======= Colormap and binning =======
cmap = plt.cm.viridis
norm = plt.Normalize(vmin=0, vmax=1)

ht_min, ht_max = 0, 250
num_bins_ht = 12
ht_bins = np.linspace(ht_min, ht_max, num_bins_ht + 1)

# ================= FIGURE (a) Histogram =================
fig, ax1 = plt.subplots(figsize=(7, 6))

# Loop over chunks of events and plot histogram outlines
for i in range(num_chunks):
    start, end = i*chunk_size, (i+1)*chunk_size
    chunk = Ht_manual[start:end]
    frac = i / num_chunks
    color = cmap(norm(frac))
    ax1.hist(chunk, bins=ht_bins, histtype="step", 
             color=color, linewidth=1.5)

ax1.set_xlabel("HT", fontsize=14)
ax1.set_ylabel("Number of Events", fontsize=14)
ax1.set_xlim(ht_min, ht_max)
ax1.tick_params(labelsize=13)
ax1.grid(False)

# Colorbar showing run time fraction
sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])
cbar1 = fig.colorbar(sm, ax=ax1)
cbar1.set_label("Time (Fraction of Run)", fontsize=13)

fig.tight_layout()
fig.savefig("Fig2a_HT_histogram.png", dpi=300)
plt.close(fig)

# ================= FIGURE (b) Violin =================
fig, ax2 = plt.subplots(figsize=(7, 6))

sns.violinplot(
    x="time_bin", y="HT", data=df,
    density_norm="width", inner=None, cut=0,
    hue="time_bin", legend=False,
    palette="viridis", ax=ax2,
    width=0.6
)

# Colors for quartiles
median_color = "#4e8b50"
quartile_color = "#d98c4c"

# Draw quartiles and medians for each time bin
for i, (bin_label, group) in enumerate(df.groupby("time_bin", observed=True)):
    q1 = group["HT"].quantile(0.25)
    q2 = group["HT"].quantile(0.50)
    q3 = group["HT"].quantile(0.75)
    ax2.hlines([q1], i-0.25, i+0.25, color=quartile_color, linewidth=2)
    ax2.hlines([q2], i-0.25, i+0.25, color=median_color, linewidth=2)
    ax2.hlines([q3], i-0.25, i+0.25, color=quartile_color, linewidth=2)

# Custom legend for quartiles
legend_elements = [
    Line2D([0], [0], color=quartile_color, lw=2, label="Q1 / Q3"),
    Line2D([0], [0], color=median_color, lw=2, label="Median")
]
ax2.legend(handles=legend_elements, title="Quartiles", loc="upper right")

# Axis labels and formatting
ax2.set_xticks(np.arange(n_time_bins))
ax2.set_xticklabels([f"{b.left:.1f}–{b.right:.1f}" for b in df["time_bin"].cat.categories])
ax2.set_xlabel("Time (Fraction of Run)", fontsize=14)
ax2.set_ylabel("HT", fontsize=14)
ax2.set_ylim(ht_min, ht_max)
ax2.tick_params(labelsize=10)
ax2.grid(False)

fig.tight_layout()
fig.savefig("Fig2b_HT_violin.png", dpi=300)
plt.close(fig)
