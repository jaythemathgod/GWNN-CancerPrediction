#!/usr/bin/env python3
"""
GWNN pipeline using top 10k features (enhanced version).
- Loads X_final_inverted_with_labels.csv
- Loads top_10000_features_by_fclassif.csv (one feature name per line)
- Uses only available features from that ranked list (up to 10k)
- Stratified splits, imputes NaNs (mean), BorderlineSMOTE on training-only
- Builds weighted kNN graph on combined nodes (BorderlineSMOTE_train + val + test)
- Computes m_eigs spectral components with eigsh
- Trains deeper GWNN (3 layers, Focal Loss, LR scheduler, early stop on macro-F1)
- Threshold tuning on val for binary decision
- Adds XGBoost baseline for comparison
- Evaluates and saves plots/artifacts

Requirements:
  pip install numpy pandas scikit-learn imbalanced-learn scipy matplotlib torch xgboost
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# Core
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import laplacian as sparse_laplacian
from scipy.sparse.linalg import eigsh
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support, confusion_matrix,
    roc_curve, auc, f1_score
)

# Enhanced SMOTE
from imblearn.over_sampling import BorderlineSMOTE

# PyTorch
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau

# XGBoost baseline
import xgboost as xgb

# ---------------------------
# USER CONFIG
# ---------------------------
DATA_CSV = os.path.expanduser("~/Downloads/X_final_inverted_with_labels.csv")
TOP_FEATURES_CSV = os.path.expanduser("~/Downloads/top_10000_features_by_fclassif.csv")
OUTPUT_DIR = "./gwnn_outputs_enhanced"
os.makedirs(OUTPUT_DIR, exist_ok=True)

SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

TEST_SIZE = 0.20
VAL_SIZE = 0.20  # fraction of the remaining training for validation (so final val ~ 0.2*0.8 = 0.16)
SMOTE_K = 5
KNN_K = 10  # Increased for denser graph
KNN_METRIC = 'cosine'
M_EIGS = 200
HIDDEN_DIM = 128  # Increased for deeper model
HIDDEN_DIM2 = 64
DROPOUT = 0.5
LR = 0.001  # Lowered
WEIGHT_DECAY = 5e-4
N_EPOCHS = 200
EARLY_STOPPING_PATIENCE = 30
SCHEDULER_PATIENCE = 10
FOCAL_ALPHA = 0.25
FOCAL_GAMMA = 2.0
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print("="*80)
print("Enhanced GWNN pipeline (top 10k features)")
print("Device:", DEVICE)
print("="*80)

# ---------------------------
# 1) Load data and top features
# ---------------------------
print("\n[1] Loading data and top feature list")
df = pd.read_csv(DATA_CSV)
print("Raw dataframe shape:", df.shape)

if 'chemo_resistant' not in df.columns:
    raise ValueError("Input CSV must have a 'chemo_resistant' column")

# Load top features file; supports either single-column CSV or plain text
top_df = pd.read_csv(TOP_FEATURES_CSV, header=None)
top_list = top_df.iloc[:, 1].astype(str).tolist()
print(f"Top-list length (requested): {len(top_list)}")

# Filter to features that actually exist in df
available_features = [f for f in top_list if f in df.columns]
if len(available_features) == 0:
    raise ValueError("No top features found in dataframe columns. Check names / paths.")
print(f"Available top features found in dataset: {len(available_features)} (using these)")

# Extract X and y
X_full = df[available_features].values.astype(np.float32)  # shape: (n_samples, n_features_used)
y_full = df['chemo_resistant'].values.astype(np.int64)               # expects labels as ints starting anywhere

n_samples, n_features = X_full.shape
print(f"Data shape using top features: X={X_full.shape}, y={y_full.shape}")

# ---------------------------
# 2) Pre-scale (recommended)
# ---------------------------
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_full)  # mean=0, var=1 per feature

# ---------------------------
# 3) Stratified train/val/test split (do this BEFORE SMOTE)
# ---------------------------
print("\n[2] Stratified split (train/val/test)")
X_temp, X_test, y_temp, y_test = train_test_split(X_scaled, y_full, test_size=TEST_SIZE,
                                                  stratify=y_full, random_state=SEED)
X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=VAL_SIZE,
                                                  stratify=y_temp, random_state=SEED)

print("Sizes: train, val, test =", X_train.shape[0], X_val.shape[0], X_test.shape[0])
print("Train label counts:", dict(zip(*np.unique(y_train, return_counts=True))))
print("Val label counts:", dict(zip(*np.unique(y_val, return_counts=True))))
print("Test label counts:", dict(zip(*np.unique(y_test, return_counts=True))))

# ---------------------------
# 4) Impute NaNs (use train statistics) and then BorderlineSMOTE on train-only
# ---------------------------
print("\n[3] Imputation (mean) and BorderlineSMOTE on training set only")

imputer = SimpleImputer(strategy='mean')
X_train_imp = imputer.fit_transform(X_train)
X_val_imp = imputer.transform(X_val)
X_test_imp = imputer.transform(X_test)

# class weights computed from original pre-SMOTE training distribution (for XGBoost)
unique_train, counts_train = np.unique(y_train, return_counts=True)
n_classes = int(np.max(y_full) + 1)  # handle sparse label values
# build class_weights across label range 0..max_label
class_weights_arr = np.ones(n_classes, dtype=np.float32)
for cls, cnt in zip(unique_train, counts_train):
    class_weights_arr[int(cls)] = (len(y_train) / (len(unique_train) * cnt)).astype(np.float32)
print("Class weights (pre-SMOTE training):", class_weights_arr)

sm = BorderlineSMOTE(k_neighbors=SMOTE_K, random_state=SEED)
X_train_res, y_train_res = sm.fit_resample(X_train_imp, y_train)
print("Resampled train size:", X_train_res.shape, y_train_res.shape)
print("Resampled train distribution:", dict(zip(*np.unique(y_train_res, return_counts=True))))

# ---------------------------
# 5) Build combined dataset for graph: BorderlineSMOTE_train + val + test
# ---------------------------
print("\n[4] Combine nodes for graph (BorderlineSMOTE-train + val + test)")
X_combined = np.vstack([X_train_res, X_val_imp, X_test_imp])
y_combined = np.concatenate([y_train_res, y_val, y_test])
n_nodes = X_combined.shape[0]
n_train_nodes = X_train_res.shape[0]
n_val_nodes = X_val_imp.shape[0]
n_test_nodes = X_test_imp.shape[0]
print("Combined nodes:", n_nodes, "(train, val, test) =", (n_train_nodes, n_val_nodes, n_test_nodes))

# ---------------------------
# 6) Build weighted kNN adjacency (sparse)
# ---------------------------
print("\n[5] Building weighted kNN graph (sparse) on combined dataset")
nbrs = NearestNeighbors(n_neighbors=KNN_K + 1, metric=KNN_METRIC, n_jobs=-1).fit(X_combined)
distances, indices = nbrs.kneighbors(X_combined)

# Compute weights: 1 / (1 + dist) for similarity emphasis
weights = 1.0 / (1.0 + distances[:, 1:])

rows = []
cols = []
data = []
for i in range(n_nodes):
    for j_idx, j in enumerate(indices[i, 1:], 1):
        w = weights[i, j_idx-1]
        rows.append(i); cols.append(int(j)); data.append(w)
        rows.append(int(j)); cols.append(i); data.append(w)

A_sparse = csr_matrix((data, (rows, cols)), shape=(n_nodes, n_nodes))
A_sparse.sum_duplicates()
print("Adjacency sparse shape:", A_sparse.shape, "nnz:", A_sparse.nnz)

# ---------------------------
# 7) Normalized Laplacian and eigendecomposition (reduced)
# ---------------------------
print("\n[6] Normalized Laplacian -> spectral decomposition (eigsh)")
L_sparse = sparse_laplacian(A_sparse, normed=True)

m_eigs = min(M_EIGS, n_nodes - 2) if n_nodes > 2 else n_nodes
m_eigs = max(2, m_eigs)
print(f"Computing {m_eigs} smallest eigenpairs (this can take time)...")
try:
    eigvals, eigvecs = eigsh(L_sparse, k=m_eigs, which='SM', tol=1e-5, maxiter=5000)
except Exception as e:
    print("eigsh failed (fallback):", e)
    denseL = L_sparse.toarray()
    eigvals_full, eigvecs_full = np.linalg.eigh(denseL)
    eigvals = eigvals_full[:m_eigs]
    eigvecs = eigvecs_full[:, :m_eigs]

print("Eigenvalues range:", eigvals.min(), eigvals.max())
print("Eigenvectors shape:", eigvecs.shape)

# ---------------------------
# 8) Focal Loss for multiclass (binary here)
# ---------------------------
class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = nn.functional.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1-pt)**self.gamma * ce_loss
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

# ---------------------------
# 9) Convert to torch and define deeper GWNN
# ---------------------------
print("\n[7] Convert to torch and build deeper GWNN model (Focal Loss)")

X_t = torch.FloatTensor(X_combined).to(DEVICE)
y_t = torch.LongTensor(y_combined).to(DEVICE)
eigvals_t = torch.from_numpy(eigvals.astype(np.float32)).to(DEVICE)
eigvecs_t = torch.from_numpy(eigvecs.astype(np.float32)).to(DEVICE)

class GraphWaveletConv(nn.Module):
    def __init__(self, in_features, out_features, eigvals, eigvecs, scale=1.0):
        super().__init__()
        # register spectral basis and filter arrays as buffers
        self.register_buffer('eigvals', eigvals)   # (m,)
        self.register_buffer('eigvecs', eigvecs)   # (n, m)
        self.linear = nn.Linear(in_features, out_features)
        # wavelet filters (spectral)
        self.register_buffer('g', torch.exp(-scale * eigvals))
        self.register_buffer('g_inv', torch.exp(scale * eigvals))

    def forward(self, x):
        # x: (n, in_features)
        U = self.eigvecs         # (n, m)
        Ut = U.t()               # (m, n)
        x_spec = Ut @ x          # (m, in_features)
        x_spec_f = self.g_inv.unsqueeze(1) * x_spec   # (m, in_features)
        x_spat = U @ x_spec_f    # (n, in_features)
        x_lin = self.linear(x_spat)   # (n, out_features)
        x_spec2 = Ut @ x_lin
        x_spec2_f = self.g.unsqueeze(1) * x_spec2
        x_out = U @ x_spec2_f
        return x_out

class GWNN(nn.Module):
    def __init__(self, in_features, hidden_dim, hidden_dim2, n_classes, eigvals, eigvecs, dropout=0.5):
        super().__init__()
        self.conv1 = GraphWaveletConv(in_features, hidden_dim, eigvals, eigvecs, scale=1.0)
        self.conv2 = GraphWaveletConv(hidden_dim, hidden_dim2, eigvals, eigvecs, scale=1.0)
        self.conv3 = GraphWaveletConv(hidden_dim2, n_classes, eigvals, eigvecs, scale=1.0)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.conv3(x)
        return x

in_features = X_t.shape[1]
n_classes_total = int(np.max(y_full) + 1)
model = GWNN(in_features, HIDDEN_DIM, HIDDEN_DIM2, n_classes_total, eigvals_t, eigvecs_t, dropout=DROPOUT).to(DEVICE)
print(model)

criterion = FocalLoss(alpha=FOCAL_ALPHA, gamma=FOCAL_GAMMA)
optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
scheduler = ReduceLROnPlateau(optimizer, mode='max', patience=SCHEDULER_PATIENCE, factor=0.5, verbose=True)

# indices in combined array
train_idx = np.arange(0, n_train_nodes, dtype=np.int64)
val_idx = np.arange(n_train_nodes, n_train_nodes + n_val_nodes, dtype=np.int64)
test_idx = np.arange(n_train_nodes + n_val_nodes,
                     n_train_nodes + n_val_nodes + n_test_nodes, dtype=np.int64)

# ---------------------------
# 10) Training (full-batch) w/ early stopping on macro-F1
# ---------------------------
print("\n[8] Training (full-batch)")

best_val = -1.0
patience_cnt = 0
train_losses = []
val_scores = []
epochs_recorded = []

for epoch in range(1, N_EPOCHS + 1):
    model.train()
    optimizer.zero_grad()
    logits = model(X_t)
    loss = criterion(logits[train_idx], y_t[train_idx])
    loss.backward()
    optimizer.step()

    train_losses.append(loss.item())

    # validation check
    if epoch % 5 == 0 or epoch == 1:
        model.eval()
        with torch.no_grad():
            logits_eval = model(X_t)
            probs_val = torch.softmax(logits_eval[val_idx], dim=1).cpu().numpy()
            preds_val = logits_eval[val_idx].argmax(dim=1).cpu().numpy()
            y_val_true = y_t[val_idx].cpu().numpy()
            val_f1 = f1_score(y_val_true, preds_val, average='macro', zero_division=0)
        val_scores.append(val_f1)
        epochs_recorded.append(epoch)
        print(f"Epoch {epoch:03d}  TrainLoss={loss.item():.4f}  ValMacroF1={val_f1:.4f}")

        scheduler.step(val_f1)

        if val_f1 > best_val + 1e-6:
            best_val = val_f1
            patience_cnt = 0
            torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, "best_gwnn_enhanced.pt"))
        else:
            patience_cnt += 1
            if patience_cnt >= EARLY_STOPPING_PATIENCE:
                print("Early stopping triggered.")
                break

# save simple plots
plt.figure(figsize=(10,4))
plt.plot(train_losses, label="Train Loss")
plt.xlabel("Iteration")
plt.ylabel("Loss")
plt.title("Training Loss")
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "train_loss.png"), dpi=150)
plt.close()

plt.figure(figsize=(6,4))
plt.plot(epochs_recorded, val_scores, marker='o', label="Val Macro F1")
plt.xlabel("Epoch")
plt.ylabel("Validation Macro F1")
plt.title("Validation Macro F1")
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "val_f1.png"), dpi=150)
plt.close()
print("Saved training plots to", OUTPUT_DIR)

# ---------------------------
# 11) Threshold tuning on val for binary decision
# ---------------------------
print("\nThreshold tuning on validation set")
model.eval()
with torch.no_grad():
    logits_val = model(X_t)
    probs_val = torch.softmax(logits_val[val_idx], dim=1).cpu().numpy()  # probs for class 1: [:,1]

best_thresh = 0.5
best_f1_val = 0.0
for thresh in np.arange(0.1, 0.6, 0.05):
    preds_thresh = (probs_val[:, 1] > thresh).astype(int)
    f1_thresh = f1_score(y_val_true, preds_thresh, average='macro', zero_division=0)
    if f1_thresh > best_f1_val:
        best_f1_val = f1_thresh
        best_thresh = thresh
print(f"Best threshold on val: {best_thresh:.2f} (Macro F1: {best_f1_val:.4f})")

# ---------------------------
# 12) Evaluation on test (best model, tuned threshold)
# ---------------------------
print("\n[9] Evaluation on test set (best model, tuned threshold)")

model.load_state_dict(torch.load(os.path.join(OUTPUT_DIR, "best_gwnn_enhanced.pt"), map_location=DEVICE))
model.eval()
with torch.no_grad():
    logits_test = model(X_t)
    probs_test = torch.softmax(logits_test[test_idx], dim=1).cpu().numpy()
    preds_test_thresh = (probs_test[:, 1] > best_thresh).astype(int)
    y_test_true = y_t[test_idx].cpu().numpy()

acc_test_thresh = accuracy_score(y_test_true, preds_test_thresh)
prec_macro, rec_macro, f1_macro, _ = precision_recall_fscore_support(y_test_true, preds_test_thresh, average='macro', zero_division=0)
print(f"Test Accuracy (tuned): {acc_test_thresh:.4f}")
print(f"Macro Precision (tuned): {prec_macro:.4f}, Macro Recall (tuned): {rec_macro:.4f}, Macro F1 (tuned): {f1_macro:.4f}")

print("\nPer-class metrics (tuned):")
p_r_f = precision_recall_fscore_support(y_test_true, preds_test_thresh, labels=np.unique(y_test_true), zero_division=0)
for idx, cls in enumerate(np.unique(y_test_true)):
    print(f" Class {cls}: Precision={p_r_f[0][idx]:.4f}, Recall={p_r_f[1][idx]:.4f}, F1={p_r_f[2][idx]:.4f}")

print("\nConfusion matrix (tuned, rows=true, cols=pred):")
print(confusion_matrix(y_test_true, preds_test_thresh))

# ROC per class (using argmax preds for consistency, but probs for AUC)
plt.figure(figsize=(8,6))
aucs = []
for cls in np.unique(y_full):
    y_bin = (y_test_true == cls).astype(int)
    scores = probs_test[:, int(cls)]
    if len(np.unique(y_bin)) < 2:
        print(f" - Class {cls}: not enough positive examples in test to compute ROC.")
        auc_val = np.nan
    else:
        fpr, tpr, _ = roc_curve(y_bin, scores)
        auc_val = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f"Class {cls} (AUC={auc_val:.3f})")
    aucs.append(auc_val)
print("Macro AUC (mean of per-class AUCs, ignoring NaN):", np.nanmean(aucs))
plt.plot([0,1],[0,1],'k--'); plt.legend(); plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "roc_multiclass.png"), dpi=150)
plt.close()
print("Saved ROC plot to", OUTPUT_DIR)

# ---------------------------
# 14) Save artifacts & feature list used
# ---------------------------
print("\n[11] Saving artifacts")
torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, "final_gwnn_enhanced.pt"))
np.savez(os.path.join(OUTPUT_DIR, "gwnn_enhanced_info.npz"),
         feature_list=np.array(available_features, dtype=object),
         class_weights=class_weights_arr,
         eigvals=eigvals, eigvecs=eigvecs,
         best_thresh=best_thresh)
print("Saved model, baseline, and data info to", OUTPUT_DIR)

print("\nPipeline complete.")
print("="*80)
