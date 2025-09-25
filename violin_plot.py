import h5py
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import hdf5plugin
from matplotlib.lines import Line2D

# ======= Funzione per calcolare HT manualmente =======
def load_and_calculate_ht(filename):
    with h5py.File(filename, "r") as f:
        n_events = f["j0Eta"].shape[0]
        n_jets = 8  # fino a j7
        ht_values = np.zeros(n_events)

        for i in range(n_jets):
            eta = f[f"j{i}Eta"][:]
            pt = f[f"j{i}Pt"][:]
            mask = (pt > 20) & (np.abs(eta) < 2.5)
            ht_values += pt * mask

    return ht_values

# ======= Carica dati =======
file_name = "new_Data/data_Run_2016_283408_longest.h5"
Ht_manual = load_and_calculate_ht(file_name)

# ======= Parameters =======
chunk_size = 50000
num_chunks = len(Ht_manual) // chunk_size
time_fraction = np.linspace(0, 1, len(Ht_manual), endpoint=False)

# ======= DataFrame for violin =======
df = pd.DataFrame({
    "HT": Ht_manual,
    "time_frac": time_fraction
})

n_time_bins = 10
df["time_bin"] = pd.cut(df["time_frac"], bins=n_time_bins)

# ======= Colormap =======
cmap = plt.cm.viridis
norm = plt.Normalize(vmin=0, vmax=1)

ht_min, ht_max = 0, 250
num_bins_ht = 12
ht_bins = np.linspace(ht_min, ht_max, num_bins_ht + 1)

# ================= FIGURE (a) Histogram =================
fig, ax1 = plt.subplots(figsize=(7, 6))

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

# Colori per quartili
median_color = "#4e8b50"
quartile_color = "#d98c4c"

for i, (bin_label, group) in enumerate(df.groupby("time_bin")):
    q1 = group["HT"].quantile(0.25)
    q2 = group["HT"].quantile(0.50)
    q3 = group["HT"].quantile(0.75)
    ax2.hlines([q1], i-0.25, i+0.25, color=quartile_color, linewidth=2)
    ax2.hlines([q2], i-0.25, i+0.25, color=median_color, linewidth=2)
    ax2.hlines([q3], i-0.25, i+0.25, color=quartile_color, linewidth=2)

legend_elements = [
    Line2D([0], [0], color=quartile_color, lw=2, label="Q1 / Q3"),
    Line2D([0], [0], color=median_color, lw=2, label="Median")
]
ax2.legend(handles=legend_elements, title="Quartiles", loc="upper right")

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


#import hdf5plugin
#import h5py
#import numpy as np
#import matplotlib.pyplot as plt
#import seaborn as sns
#import pandas as pd
#
#
## ======= Funzione per calcolare HT manualmente =======
#def load_and_calculate_ht(filename):
#    with h5py.File(filename, "r") as f:
#        n_events = f["j0Eta"].shape[0]
#        n_jets = 8  # ci sono fino a j7
#
#        ht_values = np.zeros(n_events)
#
#        for i in range(n_jets):
#            eta = f[f"j{i}Eta"][:]   # float32
#            pt  = f[f"j{i}Pt"][:]
#
#            # selezione: pt > 20, |eta| < 2.5
#            mask = (pt > 20) & (np.abs(eta) < 2.5)
#            ht_values += pt * mask
#
#    return ht_values
#
#
## ======= Carica HT manuale =======
#file_name = "new_Data/data_Run_2016_283408_longest.h5"
#Ht_manual = load_and_calculate_ht(file_name)
#
## ======= Parameters =======
#chunk_size = 50000
#num_chunks = len(Ht_manual) // chunk_size
#time_fraction = np.linspace(0, 1, len(Ht_manual), endpoint=False)
#
## ======= DataFrame for violin =======
#df = pd.DataFrame({
#    "HT": Ht_manual,
#    "time_frac": time_fraction
#})
#
## dividiamo il run in 10 bin temporali (da 0 a 1)
#n_time_bins = 10
#df["time_bin"] = pd.cut(df["time_frac"], bins=n_time_bins)
#
## ======= Figure =======
#fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
#
## --- Subplot (a): unfilled histogram with lower binning ---
#num_bins_ht = 12  # binning più grosso
#ht_min, ht_max = 0, 250
#ht_bins = np.linspace(ht_min, ht_max, num_bins_ht + 1)
#
#cmap = plt.cm.viridis
#norm = plt.Normalize(vmin=0, vmax=1)
#
#for i in range(num_chunks):
#    start, end = i*chunk_size, (i+1)*chunk_size
#    chunk = Ht_manual[start:end]
#    frac = i / num_chunks
#    color = cmap(norm(frac))
#    # hist con solo contorno (unfilled)
#    ax1.hist(chunk, bins=ht_bins, histtype="step", 
#             color=color, linewidth=1.5)
#
#ax1.set_xlabel("HT", fontsize=14)
#ax1.set_ylabel("Number of Events", fontsize=14)
#ax1.set_xlim(ht_min, ht_max)
#ax1.tick_params(labelsize=13)
#ax1.grid(False)
#
#sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
#sm.set_array([])
#cbar1 = fig.colorbar(sm, ax=ax1)
#cbar1.set_label("Time fraction of run", fontsize=13)
#
## --- Subplot (b): violin plot ---
#
#
## --- Subplot (b): violin plot senza linee interne ---
#sns.violinplot(
#    x="time_bin", y="HT", data=df,
#    density_norm="width", inner=None, cut=0,
#    hue="time_bin", legend=False,
#    palette="viridis", ax=ax2,
#    width=0.6
#)
## Colori più soft
#median_color = "#4e8b50"   # verde spento
#quartile_color = "#d98c4c" # arancio spento
#
#for i, (bin_label, group) in enumerate(df.groupby("time_bin")):
#    q1 = group["HT"].quantile(0.25)
#    q2 = group["HT"].quantile(0.50)
#    q3 = group["HT"].quantile(0.75)
#
#    ax2.hlines([q1], i-0.25, i+0.25, color=quartile_color, linewidth=2)
#    ax2.hlines([q2], i-0.25, i+0.25, color=median_color, linewidth=2)
#    ax2.hlines([q3], i-0.25, i+0.25, color=quartile_color, linewidth=2)
#
## Aggiungi legenda
#from matplotlib.lines import Line2D
#legend_elements = [
#    Line2D([0], [0], color=quartile_color, lw=2, label="Q1 / Q3"),
#    Line2D([0], [0], color=median_color, lw=2, label="Median")
#]
#ax2.legend(handles=legend_elements, title="Quartiles", loc="upper right")
#
## Etichette e pulizia
#ax2.set_xticks(np.arange(n_time_bins))
#ax2.set_xticklabels([f"{b.left:.1f}–{b.right:.1f}" for b in df["time_bin"].cat.categories])
#ax2.set_xlabel("Time fraction of run", fontsize=14)
#ax2.set_ylabel("HT (manual calc, pt>20, |eta|<2.5)", fontsize=14)
#ax2.set_ylim(ht_min, ht_max)
#ax2.tick_params(labelsize=10)
#ax2.grid(False)   # <<-- tolta la griglia
#
#
#
## ======= Title & layout =======
#fig.suptitle("Figure 1: Calibrated HT distribution and evolution with time", fontsize=16)
#fig.tight_layout(rect=[0,0,1,0.96])
#
#fig.savefig("Fig1_HT_manual_violin.png", dpi=300)
#plt.show()

