# Import libraries
import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import chi2

# Load the dataset
data = sio.loadmat("data/Yonghe_Modal_FDD.mat")
print(data.keys())

# Inspect the dataset
freq_fdd = data['freq_fdd']
print(freq_fdd.shape)

# Split the dataset into healthy (Train + Val) and Damaged set
n_modes, n_total = freq_fdd.shape
n_healthy = 192
n_train = int(0.75 * n_healthy)

Freq_U = freq_fdd[:, :n_healthy]
Freq_D = freq_fdd[:, n_healthy:]
X_train = Freq_U[:, :n_train]
X_val = Freq_U[:, n_train:]

# Inspect the covariance structure between modes (training data only) --
Sigma_display = np.cov(X_train)
mode_labels = ['Mode1', 'Mode2', 'Mode3', 'Mode4']
print("\nCovariance matrix (training data):")
print(f"{'':8}" + "".join(f"{m:>10}" for m in mode_labels))
for i, row in enumerate(Sigma_display):
    print(f"{mode_labels[i]:8}" + "".join(f"{v:10.5f}" for v in row))

# Define the chi square control limit
threshold_99 = chi2.ppf(0.99, df=n_modes)

# Figure A: Mode 1 vs Mode 2 scatter -- shows the correlation
fig, ax = plt.subplots(figsize=(7, 6))
ax.scatter(X_train[0], X_train[1], color='tab:blue', alpha=0.6, label='Train (healthy)', s=25)
ax.scatter(X_val[0], X_val[1], color='tab:green', alpha=0.6, marker='+', s=60, label='Validation (healthy)')
ax.scatter(Freq_D[0], Freq_D[1], color='tab:red', alpha=0.8, marker='x', s=60, label='Damaged')

corr = np.corrcoef(X_train[0], X_train[1])[0, 1]
ax.set_xlabel('Mode 1 frequency (Hz)')
ax.set_ylabel('Mode 2 frequency (Hz)')
ax.set_title(f'Mode 1 vs Mode 2 (training correlation r = {corr:.2f})')
ax.legend()
plt.tight_layout()
plt.savefig('figures/A_mode1_vs_mode2_correlation.png', dpi=300)
plt.show()

def mahalanobis_batch(X, mu, sigma_inv):
    D2 = np.zeros(X.shape[1])
    for i in range(X.shape[1]):
        diff = X[:, i] - mu
        D2[i] = diff @ sigma_inv @ diff
    return D2

# Method 1: Fixed baseline 
# Fixed baseline (single mu/Sigma from all training data)
mu = X_train.mean(axis=1)
Sigma_inv = np.linalg.inv(np.cov(X_train))
D2_train_fixed = mahalanobis_batch(X_train, mu, Sigma_inv)
D2_val_fixed = mahalanobis_batch(X_val, mu, Sigma_inv)
D2_damaged_fixed = mahalanobis_batch(Freq_D, mu, Sigma_inv)

# Figure B: Mahalanobis Distance 
fig, ax = plt.subplots(figsize=(9, 5))
 
# x-axis positions: train samples first, then val, then damaged, in order
n1 = len(D2_train_fixed)
n2 = n1 + len(D2_val_fixed)
n3 = n2 + len(D2_damaged_fixed)
 
ax.plot(range(1, n1+1), D2_train_fixed, '.', color='tab:blue', label='Train (healthy)')
ax.plot(range(n1+1, n2+1), D2_val_fixed, '+', color='tab:green', label='Validation (healthy)')
ax.plot(range(n2+1, n3+1), D2_damaged_fixed, 'x', color='tab:red', label='Damaged (test)')
 
# Vertical dotted lines marking where each group ends
ax.axvline(n1 + 0.5, linestyle=':', color='k')
ax.axvline(n2 + 0.5, linestyle=':', color='k')
 
# Horizontal threshold line
ax.axhline(threshold_99, linestyle='--', color='gray',
           label=f'99% chi-square threshold ({threshold_99:.2f})')
ax.set_xlabel('Sample index')
ax.set_ylabel('Mahalanobis $D^2$')
ax.legend()
plt.tight_layout()
plt.savefig('figures/B_fixed_baseline_mahalanobis.png', dpi=300)
plt.show()

# Quick summary stats
false_alarms = (D2_val_fixed > threshold_99).sum()
detected = (D2_damaged_fixed > threshold_99).sum()
print(f"\nFalse alarms on validation (healthy) set: {false_alarms}/{len(D2_val_fixed)}")
print(f"Damaged samples correctly flagged: {detected}/{len(D2_damaged_fixed)}")

# Figure C: Raw frequency drift, all 4 modes
fig, axes = plt.subplots(2, 2, figsize=(11, 7))
titles = ['(a) Mode 1', '(b) Mode 2', '(c) Mode 3', '(d) Mode 4']
for i, ax in enumerate(axes.flat):
    ax.plot(range(1, n_healthy+1), freq_fdd[i, :n_healthy], '.', color='tab:blue',
            markersize=6, label='Healthy (days 1-8)')
    ax.plot(range(n_healthy+1, n_total+1), freq_fdd[i, n_healthy:], 'x', color='tab:red',
            markersize=6, label='Damaged (day 9)')
    ax.axvline(n_healthy + 0.5, linestyle=':', color='k', alpha=0.6)
    ax.set_xlabel('Sample'); ax.set_ylabel('Frequency (Hz)')
    ax.set_title(titles[i])
    ax.legend(fontsize=8)
plt.tight_layout()
plt.savefig('figures/C_raw_frequency_drift.png', dpi=300)
plt.show()

# Method 2: Naive rolling window 
# Naive rolling window (local mu/Sigma, updates every step)
def rolling_mahalanobis(X, window_size=20):
    D2_values, valid_indices = [], []
    for i in range(window_size, X.shape[1]):
        window = X[:, i - window_size:i]
        current_sample = X[:, i]
        l_mu = window.mean(axis=1)
        l_sigma_inv = np.linalg.inv(np.cov(window))
        diff = current_sample - l_mu
        D2_values.append(diff @ l_sigma_inv @ diff)
        valid_indices.append(i)
    return np.array(D2_values), np.array(valid_indices)

full_sequence = np.concatenate([X_train, X_val, Freq_D], axis=1)
D2_full_rol, idx_full_rol = rolling_mahalanobis(full_sequence, window_size=20)
train_mask = idx_full_rol < 144
val_mask = (idx_full_rol >= 144) & (idx_full_rol < 192)
damaged_mask = idx_full_rol >= 192
D2_train_rol = D2_full_rol[train_mask]
D2_val_rol = D2_full_rol[val_mask]
D2_damaged_rol = D2_full_rol[damaged_mask]

# Quick summary stats
false_alarms_rol  = (D2_val_rol > threshold_99).sum()
detected_rol = (D2_damaged_rol > threshold_99).sum()
print(f"\nFalse alarms on validation (healthy) set: {false_alarms_rol}/{len(D2_val_rol)}")
print(f"Damaged samples correctly flagged: {detected_rol}/{len(D2_damaged_rol)}")

# Figure D: Naive rolling window's damage-only D2 -- shows the collapse
def rolling_mahalanobis(X, window_size=20):
    D2_values, valid_indices = [], []
    for i in range(window_size, X.shape[1]):
        window = X[:, i - window_size:i]
        current_sample = X[:, i]
        l_mu = window.mean(axis=1)
        l_sigma_inv = np.linalg.inv(np.cov(window))
        diff = current_sample - l_mu
        D2_values.append(diff @ l_sigma_inv @ diff)
        valid_indices.append(i)
    return np.array(D2_values), np.array(valid_indices)

full_sequence = np.concatenate([X_train, X_val, Freq_D], axis=1)
D2_full_rol, idx_full_rol = rolling_mahalanobis(full_sequence, window_size=20)
damaged_mask = idx_full_rol >= 192
D2_damaged_rol = D2_full_rol[damaged_mask]

fig, ax = plt.subplots(figsize=(9, 5))
damage_sample_num = np.arange(len(D2_damaged_rol))
ax.plot(damage_sample_num, D2_damaged_rol, 'o-', color='tab:red')
ax.axhline(threshold_99, linestyle='--', color='gray', label=f'99% threshold ({threshold_99:.1f})')
ax.set_xlabel('Damage sample # (in order, day 9)')
ax.set_ylabel('D² (naive rolling window)')
ax.set_title('Naive Rolling Window: D² Collapses After the First 1-2 Damaged Samples')
ax.legend()
plt.tight_layout()
plt.savefig('figures/D_naive_rolling_collapse.png', dpi=300)
plt.show()

# Method 3: Clean-buffer rolling
# Clean-buffer rolling (local mu/Sigma, only updates on non-flagged samples)
def clean_buffer_mahalanobis(X_train, X_rest, window_size=20, threshold=threshold_99):
    buffer = [X_train[:, i] for i in range(X_train.shape[1] - window_size, X_train.shape[1])]
    D2_values, flagged = [], []
    for i in range(X_rest.shape[1]):
        current_sample = X_rest[:, i]
        window_arr = np.array(buffer).T
        l_mu = window_arr.mean(axis=1)
        l_sigma_inv = np.linalg.inv(np.cov(window_arr))
        diff = current_sample - l_mu
        D2 = diff @ l_sigma_inv @ diff
        D2_values.append(D2)
        is_flagged = D2 > threshold
        flagged.append(is_flagged)
        if not is_flagged:
            buffer.pop(0)
            buffer.append(current_sample)
    return np.array(D2_values), np.array(flagged)

X_rest = np.concatenate([X_val, Freq_D], axis=1)
D2_clean_full, _ = clean_buffer_mahalanobis(X_train, X_rest, window_size=20)
D2_val_clean = D2_clean_full[:48]
D2_damaged_clean = D2_clean_full[48:]
# Training portion for clean-buffer: same as fixed (buffer seeded from last 20 train samples,
# training D2 not separately evaluated in this method -- use fixed-baseline train D2 for display)
D2_train_clean = D2_train_fixed

# Quick summary stats
false_alarms_clean = (D2_val_clean > threshold_99).sum()
detected_clean = (D2_damaged_clean > threshold_99).sum()
print(f"\nFalse alarms on validation (healthy) set: {false_alarms_clean}/{len(D2_val_rol)}")
print(f"Damaged samples correctly flagged: {detected_clean}/{len(D2_damaged_rol)}")

# Method 4: Linear detrend
# Linear detrend (remove per-mode linear trend, then fixed Mahalanobis) 
x_train_idx = np.arange(n_train)
x_full_idx = np.arange(n_total)
trend_full = np.zeros((n_modes, n_total))
for m in range(n_modes):
    coeffs = np.polyfit(x_train_idx, X_train[m, :], deg=1)
    trend_full[m, :] = np.polyval(coeffs, x_full_idx)
    print(f"Mode {m+1}: slope={coeffs[0]:.6f}, intercept={coeffs[1]:.4f}")
freq_detrended = freq_fdd - trend_full
X_train_dt = freq_detrended[:, :n_train]
X_val_dt = freq_detrended[:, n_train:n_healthy]
X_damaged_dt = freq_detrended[:, n_healthy:]
mu_dt = X_train_dt.mean(axis=1)
Sigma_inv_dt = np.linalg.inv(np.cov(X_train_dt))
D2_train_dt = mahalanobis_batch(X_train_dt, mu_dt, Sigma_inv_dt)
D2_val_dt = mahalanobis_batch(X_val_dt, mu_dt, Sigma_inv_dt)
D2_damaged_dt = mahalanobis_batch(X_damaged_dt, mu_dt, Sigma_inv_dt)

# Quick summary stats
false_alarms_dt = (D2_val_dt > threshold_99).sum()
detected_dt = (D2_damaged_dt > threshold_99).sum()
print(f"\nDETREND - False alarms (val): {false_alarms_dt}/{len(D2_val_dt)}")
print(f"DETREND - Damaged detected:   {detected_dt}/{len(D2_damaged_dt)}")

# Figure E: 2x2 grid, one panel per method
methods = [
    ("1. Fixed Baseline", D2_train_fixed, D2_val_fixed, D2_damaged_fixed),
    ("2. Naive Rolling Window", D2_train_rol, D2_val_rol, D2_damaged_rol),
    ("3. Clean-Buffer Rolling", D2_train_clean, D2_val_clean, D2_damaged_clean),
    ("4. Linear Detrend", D2_train_dt, D2_val_dt, D2_damaged_dt),
]

fig, axes = plt.subplots(2, 2, figsize=(13, 9))
for ax, (title, d2_tr, d2_va, d2_da) in zip(axes.flat, methods):
    n1, n2 = len(d2_tr), len(d2_tr) + len(d2_va)
    ax.plot(range(1, n1+1), d2_tr, '.', color='tab:blue', label='Train', alpha=0.7)
    ax.plot(range(n1+1, n2+1), d2_va, '+', color='tab:green', label='Val (healthy)')
    ax.plot(range(n2+1, n2+len(d2_da)+1), d2_da, 'x', color='tab:red', label='Damaged')
    ax.axhline(threshold_99, linestyle='--', color='gray', alpha=0.8)
    ax.axvline(n1+0.5, linestyle=':', color='k', alpha=0.5)
    ax.axvline(n2+0.5, linestyle=':', color='k', alpha=0.5)

    fa = (d2_va > threshold_99).sum()
    det = (d2_da > threshold_99).sum()
    ax.set_title(f"{title}\nFalse alarms: {fa}/{len(d2_va)}  |  Detected: {det}/{len(d2_da)}")
    ax.set_xlabel('Sample'); ax.set_ylabel('D²')
    ax.legend(fontsize=8)
    ax.set_ylim(0, max(60, d2_da.max()*1.1))

plt.tight_layout()
plt.savefig('figures/E_all_methods_comparison.png', dpi=300)
plt.show()

print(f"\n{'Method':<25}{'False alarms':<16}{'Detected':<12}")
print("-"*53)
for title, d2_tr, d2_va, d2_da in methods:
    fa = (d2_va > threshold_99).sum()
    det = (d2_da > threshold_99).sum()
    print(f"{title:<25}{f'{fa}/{len(d2_va)}':<16}{f'{det}/{len(d2_da)}':<12}")
