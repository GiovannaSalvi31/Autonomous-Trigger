import matplotlib.pyplot as plt
import numpy as np
import h5py
import hdf5plugin

# ======= Sort jets in each event by pT (descending) =======
def sort_obj(data_):
    for index in np.arange(data_.shape[0]):
        ele = data_[index, :, :]
        ele = ele.T
        sorted_indices = np.argsort(ele[2])[::-1]
        sorted_array = ele[:, sorted_indices]
        sorted_array = sorted_array.T
        data_[index, :, :] = sorted_array
    return data_

# ======= Plot helper: normalized histograms to unit maximum =======
def plot_normed_to_max(ax, arrays, labels, colors, bins, xlabel):
    hists = [np.histogram(a, bins=bins, density=True)[0] for a in arrays]
    ymax = max((h.max() for h in hists if h.size), default=1.0)
    for a, lab, col in zip(arrays, labels, colors):
        y, edges = np.histogram(a, bins=bins, density=True)
        ax.stairs(y, edges, label=lab, color=col, linewidth=1.8)
    ax.set_xlim(bins[0], bins[-1])
    ax.set_ylim(0, 1.05)
    ax.set_xlabel(xlabel, fontsize=22)
    ax.set_ylabel("Density", fontsize=22)
    ax.legend(fontsize=20, frameon=True)

# ======= Plot helper: step histogram =======
def plot_step_hist(ax, arr, bins, label, color, density=True):
    y, edges = np.histogram(arr, bins=bins, density=density)
    ax.stairs(y, edges, label=label, color=color, linewidth=1.8)

# ======= Load and preprocess HDF5 file =======
def process_h5_file(input_filename):
    with h5py.File(input_filename, 'r') as h5_file:
        n_events = h5_file['j0Eta'].shape[0]
        print('n_events:', n_events)
        n_jets = 8
        n_features = 4

        # Use all events
        n_selected = n_events
        data_array = np.zeros((n_selected, n_jets, n_features), dtype=np.float32)

        # Fill the array with eta, phi, pt
        for i in range(n_jets):
            data_array[:, i, 0] = h5_file[f'j{i}Eta'][:] + 5   # Eta (shifted)
            data_array[:, i, 1] = h5_file[f'j{i}Phi'][:] + np.pi  # Phi (shifted)
            data_array[:, i, 2] = h5_file[f'j{i}Pt'][:]   # Pt

        # Load number of primary vertices
        npvsGood_smr1_values = h5_file['PV_npvsGood'][:]

        # Sort jets in each event by pT
        sorted_data_array = sort_obj(data_array)

        # Compute HT manually (pt>20, |eta|<2.5)
        Ht_values = np.zeros(n_selected)
        for i in range(n_selected):
            ht = 0
            for j in range(n_jets):
                pt = sorted_data_array[i, j, 2]
                eta = sorted_data_array[i, j, 0] - 5  # undo shift
                if pt > 20 and abs(eta) < 2.5:
                    ht += pt
                else:
                    # Mask invalid jets
                    sorted_data_array[i, j, 2] = 0.0
                    sorted_data_array[i, j, 0] = -1
                    sorted_data_array[i, j, 1] = -1
            Ht_values[i] = ht

        # Remove events with npv == 0
        non_zero_mask = npvsGood_smr1_values > 0
        sorted_data_array = sorted_data_array[non_zero_mask]
        Ht_values = Ht_values[non_zero_mask]
        npvsGood_smr1_values = npvsGood_smr1_values[non_zero_mask]

        # Store NPV values in the last column (as "time" proxy)
        sorted_data_array[:, :, 3] = npvsGood_smr1_values[:, np.newaxis]

        # Apply masks to jets with pt == 0
        zero_pt_mask = sorted_data_array[:, :, 2] == 0
        sorted_data_array[:, :, 0][zero_pt_mask] = -1
        sorted_data_array[:, :, 1][zero_pt_mask] = -1

        # Remove first entry (technical fix)
        sorted_data_array = np.delete(sorted_data_array, 0, axis=0)
        Ht_values = Ht_values[1:]

        print(sorted_data_array[0, :, 2])

        return sorted_data_array, Ht_values

# ======= Compute missing HT (vector sum of jets) =======
def compute_missing_ht(jets):
    px = np.sum(jets[:, :, 2] * np.cos(jets[:, :, 1]), axis=1)
    py = np.sum(jets[:, :, 2] * np.sin(jets[:, :, 1]), axis=1)
    return np.sqrt(px**2 + py**2)

# ======= Load datasets =======
mc_bkg_jets, mc_bkg_ht = process_h5_file("new_Data/minimumbias_2016_1.h5")
mc_ttbar_jets, mc_ttbar_ht = process_h5_file("new_Data/ttbar_2016_1.h5")
mc_aa_jets, mc_aa_ht = process_h5_file("new_Data/HToAATo4B.h5")
mc_data_jets, mc_data_ht = process_h5_file("new_Data/data_Run_2016_283408_longest.h5")
print('all data loaded')

# ======= Compute missing HT =======
mc_bkg_missing_ht = compute_missing_ht(mc_bkg_jets)
mc_ttbar_missing_ht = compute_missing_ht(mc_ttbar_jets)
mc_aa_missing_ht = compute_missing_ht(mc_aa_jets)
mc_data_missing_ht = compute_missing_ht(mc_data_jets)
print('missing Ht calculated')

# ======= Compute number of jets per event =======
mc_bkg_njets = np.sum(mc_bkg_jets[:, :, 2] > 0, axis=1)
mc_ttbar_njets = np.sum(mc_ttbar_jets[:, :, 2] > 0, axis=1)
mc_aa_njets = np.sum(mc_aa_jets[:, :, 2] > 0, axis=1)
mc_data_njets = np.sum(mc_data_jets[:, :, 2] > 0, axis=1)
print('njets done')

# ======= Apply masks for jets with pT > 0 =======
bkg_mask   = mc_bkg_jets[:, :, 2] > 0
ttbar_mask = mc_ttbar_jets[:, :, 2] > 0
aa_mask    = mc_aa_jets[:, :, 2] > 0
data_mask  = mc_data_jets[:, :, 2] > 0

# ======= Extract pT arrays for jets =======
mc_bkg_pt  = mc_bkg_jets[:, :, 2][bkg_mask]
mc_ttbar_pt  = mc_ttbar_jets[:, :, 2][ttbar_mask]
mc_aa_pt  = mc_aa_jets[:, :, 2][aa_mask]
mc_data_pt  = mc_data_jets[:, :, 2][data_mask]

print('all pt extracted')

# ===================== PLOTTING =====================
ht_bins  = np.linspace(0, 700, 16)
mht_bins = np.linspace(0, 175, 8)
pt_bins  = np.linspace(0, 200, 10)
nj_bins  = np.arange(-0.5, 8.5 + 1, 1.0)

COLORS = {
    "data":   "firebrick",
    "minbias":"royalblue",
    "ttbar":  "goldenrod",
    "hToAA":  "seagreen"
}

labels = ["2016 Zerobias Data", "MinBias", "TTbar", "HToAATo4B"]
colors = [COLORS["data"], COLORS["minbias"], COLORS["ttbar"], COLORS["hToAA"]]

# --- 1) HT Distribution ---
fig, ax = plt.subplots(figsize=(8,6))
plot_normed_to_max(
    ax,
    [mc_data_ht, mc_bkg_ht, mc_ttbar_ht, mc_aa_ht],
    labels, colors, ht_bins,
    r"$H_T$ [GeV]"
)
ax.set_xlim(0, 700)
ax.set_ylim(0, 0.02)
ax.tick_params(axis='both', which='major', labelsize=16)
plt.tight_layout()
plt.savefig("paper/HT_distribution.png")
plt.close()

# --- 2) Number of Jets per Event ---
fig, ax = plt.subplots(figsize=(8,6))
plot_step_hist(ax, mc_data_njets,  nj_bins, "2016 Zerobias Data", COLORS["data"])
plot_step_hist(ax, mc_bkg_njets,   nj_bins, "MinBias",   COLORS["minbias"])
plot_step_hist(ax, mc_ttbar_njets, nj_bins, "TTbar",     COLORS["ttbar"])
plot_step_hist(ax, mc_aa_njets,    nj_bins, "HToAATo4B", COLORS["hToAA"])
ax.set_xlabel("Number of Jets", fontsize=22)
ax.set_ylabel("Density", fontsize=22)
ax.set_xticks(range(0, 9))
ax.legend(fontsize=18, frameon=True)
ax.tick_params(axis='both', which='major', labelsize=16)
plt.tight_layout()
plt.savefig("paper/Njets_distribution.png")
plt.close()

# --- 3) Missing HT ---
fig, ax = plt.subplots(figsize=(8,6))
plot_normed_to_max(
    ax,
    [mc_data_missing_ht, mc_bkg_missing_ht, mc_ttbar_missing_ht, mc_aa_missing_ht],
    labels, colors, mht_bins,
    r"Missing $H_T$ [GeV]"
)
ax.set_xlim(0, 175)
ax.set_ylim(0, 0.04)
ax.tick_params(axis='both', which='major', labelsize=16)
plt.tight_layout()
plt.savefig("paper/MissingHT_distribution.png")
plt.close()

# --- 4) Jet pT ---
fig, ax = plt.subplots(figsize=(8,6))
plot_normed_to_max(
    ax,
    [mc_data_pt, mc_bkg_pt, mc_ttbar_pt, mc_aa_pt],
    labels, colors, pt_bins,
    r"$p_T$ [GeV]"
)
ax.set_xlim(0, 200)
ax.set_ylim(0, 0.035)
ax.tick_params(axis='both', which='major', labelsize=16)
plt.tight_layout()
plt.savefig("paper/jetPt_distribution.png")
plt.close()
# =================== END PLOTTING ===================
