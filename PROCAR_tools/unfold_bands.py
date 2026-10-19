import matplotlib.pyplot as plt
import numpy as np
import os
import re

# =========================
# 1. File paths
# =========================
filepath = '.'
filename_bands = os.path.join(filepath, 'BAND.dat')
filename_K = os.path.join(filepath, 'KLABELS')
filename_gap = os.path.join(filepath, 'BAND_GAP')  # <-- add this

# =========================
# 2. Read Fermi energy (if BAND_GAP exists)
# =========================
fermi_from_gap = None

# If no BAND_GAP, assume energies already referenced to EF=0
FERMI_E = 0.0
E_SHIFT = fermi_from_gap if fermi_from_gap is not None else 0.0  # subtract this from eigenvalues

# =========================
# 3. Initialize containers
# =========================
band_data = []
current_band_k = []
current_band_e = []
nkpts = 0
nbands = 0

# =========================
# 4. Parse BAND.dat
# =========================
with open(filename_bands, 'r') as f:
    for line in f:
        if 'NKPTS' in line and 'NBANDS' in line:
            nums = re.findall(r'\d+', line)
            if len(nums) >= 2:
                nkpts = int(nums[0])
                nbands = int(nums[1])
            continue
        
        # Identify the start of a new band block

        if line.startswith('# Band-Index:'):
            if current_band_k:
                band_data.append({
                    'k': np.array(current_band_k),
                    'e': np.array(current_band_e)
                })
            current_band_k = []
            current_band_e = []
            continue

        if not line.startswith('#') and line.strip():
            parts = line.split()
            if len(parts) == 2:
                try:
                    k_val = float(parts[0])
                    e_val = float(parts[1]) - E_SHIFT  # <-- shift energies so EF is at 0
                    current_band_k.append(k_val)
                    current_band_e.append(e_val)
                except ValueError:
                    continue

if current_band_k:
    band_data.append({'k': np.array(current_band_k), 'e': np.array(current_band_e)})
# After finishing the BAND.dat parsing (right after the final append)
nbands = len(band_data)

# =========================
# 5. Parse KLABELS
# =========================
tick_coords = []
tick_labels = []
with open(filename_K, 'r') as f:
    for line in f:
        parts = line.split()
        if not parts or 'K-Label' in line or '*' in line:
            continue
        try:
            label = parts[0]
            coord = float(parts[1])
            if label.upper() == 'GAMMA':
                label = r'$\Gamma$'
            tick_labels.append(label)
            tick_coords.append(coord)
        except (ValueError, IndexError):
            continue

# =========================
# 6. Bandgap at center k (Γ if present)
# =========================
def _interp_at_k(k_arr: np.ndarray, e_arr: np.ndarray, k0: float):
    if k_arr.size < 2:
        return None
    kmin, kmax = float(np.min(k_arr)), float(np.max(k_arr))
    if not (min(kmin, kmax) <= k0 <= max(kmin, kmax)):
        return None

    # Make x increasing for interpolation
    if k_arr[0] > k_arr[-1]:
        k_use = k_arr[::-1]
        e_use = e_arr[::-1]
    else:
        k_use = k_arr
        e_use = e_arr

    if np.any(np.diff(k_use) < 0):
        idx = np.argsort(k_use)
        k_use = k_use[idx]
        e_use = e_use[idx]

    return float(np.interp(k0, k_use, e_use))

def compute_gap_at_k(bands, k0: float, fermi: float = 0.0):
    e_at_k = []
    for b in bands:
        val = _interp_at_k(b['k'], b['e'], k0)
        if val is not None:
            e_at_k.append(val)
    if not e_at_k:
        return None, None, None

    e_at_k = np.array(e_at_k)
    below = e_at_k[e_at_k <= fermi + 1e-12]
    above = e_at_k[e_at_k >= fermi - 1e-12]

    if below.size == 0 or above.size == 0:
        return None, (float(np.max(below)) if below.size else None), (float(np.min(above)) if above.size else None)

    vbm = float(np.max(below))
    cbm = float(np.min(above))
    eg = cbm - vbm
    if eg <= 1e-6:
        return None, vbm, cbm
    return eg, vbm, cbm

# Choose center k = Γ if present, else k-midpoint
all_k = np.concatenate([b['k'] for b in band_data]) if band_data else np.array([0.0, 1.0])
k_center = None
for lab, kc in zip(tick_labels, tick_coords):
    if lab == r'$\Gamma$' or lab.upper() == 'GAMMA':
        k_center = float(kc)
        break
if k_center is None:
    k_center = float(0.5 * (all_k.min() + all_k.max()))

Eg, VBM, CBM = compute_gap_at_k(band_data, k_center, fermi=FERMI_E)

# =========================
# 7. Plot
# =========================
plt.figure(figsize=(10, 7))

for band in band_data:
    plt.plot(band['k'], band['e'], color='black', linewidth=1.0)

for coord in tick_coords:
    plt.axvline(x=coord, color='gray', linestyle='--', linewidth=0.8, alpha=0.5)

plt.xticks(tick_coords, tick_labels)
plt.axhline(FERMI_E, color='red', linestyle='-', linewidth=0.8)
plt.ylabel('Energy (eV)', fontsize=12)
plt.title(f'Electronic Band Structure: {nbands} Bands', fontsize=14)

plt.xlim(all_k.min(), all_k.max())

# =========================
# 8. Bandgap annotation (offset label + white box)
# =========================
if Eg is not None and VBM is not None and CBM is not None:
    xspan = (all_k.max() - all_k.min())
    dx_seg = 0.06 * xspan
    x1, x2 = k_center - dx_seg, k_center + dx_seg

    plt.hlines([VBM, CBM], x1, x2, linestyles=':', linewidth=1.4)

    plt.annotate(
        '',
        xy=(k_center, CBM),
        xytext=(k_center, VBM),
        arrowprops=dict(arrowstyle='<->', linewidth=1.5)
    )

    # Offset the text slightly (right + a touch up)
    dx_text = 0.05 * xspan
    y_text = 0.5 * (VBM + CBM) + 0.02 * (CBM - VBM)

    plt.text(
        k_center + dx_text,
        y_text,
        f'{Eg:.2f} eV',
        ha='left',
        va='center',
        fontsize=12,
        bbox=dict(boxstyle='round,pad=0.25', facecolor='white', edgecolor='black', alpha=0.95)
    )

plt.tight_layout()
plt.ylim(-3,3)
plt.show()
