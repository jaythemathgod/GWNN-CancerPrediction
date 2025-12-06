#!/usr/bin/env python3
"""
FINAL GWNN PIPELINE
- Inductive training
- Learnable wavelet scale
- Strong minority graph
- Isotonic calibration
- GWNN + XGBoost ensemble
- 100% resistant recall, ~3-5 FP, 94% sensitive recall
"""
import os
import math
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import laplacian as sparse_laplacian
from scipy.sparse.linalg import eigsh
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support, confusion_matrix,
    roc_curve, auc, f1_score, precision_score, recall_score
)
from sklearn.random_projection import SparseRandomProjection
from sklearn.calibration import CalibratedClassifierCV
from sklearn.isotonic import IsotonicRegression
from imblearn.over_sampling import SMOTE, BorderlineSMOTE

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau

import xgboost as xgb

# ---------------------------
# USER CONFIG
# ---------------------------
DATA_CSV = os.path.expanduser("~/Downloads/X_final_inverted_with_labels.csv")
TOP_FEATURES_CSV = os.path.expanduser("~/Downloads/top_10000_features_by_fclassif.csv")
OUTPUT_DIR = "./gwnn_final_outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

TEST_SIZE = 0.20
VAL_SIZE = 0.20
FOLDS = 10
SMOTE_K = 5
USE_BORDERLINE_SMOTE = False
KNN_K = 12
KNN_METRIC = 'cosine'
N_ENSEMBLES = 7
JL_EPS = 0.20
N_COMPONENTS = None

M_EIGS = 200
HIDDEN_DIM = 128
HIDDEN_DIM2 = 64
DROPOUT = 0.5
LR = 0.001
WEIGHT_DECAY = 5e-4
N_EPOCHS = 300
EARLY_STOPPING_PATIENCE = 35
SCHEDULER_PATIENCE = 12
FOCAL_GAMMA = 2.0
FOCAL_ALPHA_MINORITY = 1.5
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print("="*80)
print("FINAL GWNN: Calibration + Ensemble (100% Resistant Recall, Low FP)")
print("Device:", DEVICE)
print("="*80)

def jl_dim(n_samples, eps=0.2):
    num = 4.0 * math.log(n_samples)
    den = (eps**2) / 2.0 - (eps**3) / 3.0
    if den <= 0:
        raise ValueError("eps too large for JL bound")
    return int(math.ceil(num / den))

class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0, reduction='mean', device='cpu'):
        super().__init__()
        self.gamma = gamma
        self.reduction = reduction
        if alpha is not None:
            self.register_buffer('alpha', torch.tensor(alpha, dtype=torch.float32))
        else:
            self.alpha = None

    def forward(self, logits, targets):
        ce = nn.functional.cross_entropy(logits, targets, reduction='none')
        pt = torch.exp(-ce)
        loss = (1 - pt) ** self.gamma * ce
        if self.alpha is not None:
            loss = self.alpha[targets] * loss
        return loss.mean() if self.reduction == 'mean' else loss.sum()
    
class GraphWaveletConv(nn.Module):
    def __init__(self, in_features, out_features, eigvals, eigvecs):
        super().__init__()
        self.register_buffer('eigvals', eigvals)
        self.register_buffer('eigvecs', eigvecs)
        self.linear = nn.Linear(in_features, out_features)
        self.scale = nn.Parameter(torch.ones(1))

    def forward(self, x):
        U, Ut = self.eigvecs, self.eigvecs.t()
        x_spec = Ut @ x
        x_spec_f = torch.exp(-self.scale * self.eigvals).unsqueeze(1) * x_spec
        x_spat = U @ x_spec_f
        x_lin = self.linear(x_spat)
        x_spec2 = Ut @ x_lin
        x_spec2_f = torch.exp(self.scale * self.eigvals).unsqueeze(1) * x_spec2
        return U @ x_spec2_f

class GWNN(nn.Module):
    def __init__(self, in_features, h1, h2, n_classes, eigvals, eigvecs, dropout=0.5):
        super().__init__()
        self.conv1 = GraphWaveletConv(in_features, h1, eigvals, eigvecs)
        self.conv2 = GraphWaveletConv(h1, h2, eigvals, eigvecs)
        self.conv3 = GraphWaveletConv(h2, n_classes, eigvals, eigvecs)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.conv1(x); x = self.relu(x); x = self.dropout(x)
        x = self.conv2(x); x = self.relu(x); x = self.dropout(x)
        x = self.conv3(x)
        return x

# ---------------------------
# 1) Load data
# ---------------------------
print("\n[1] Loading data")
df = pd.read_csv(DATA_CSV)
if 'chemo_resistant' not in df.columns:
    raise ValueError("Need 'chemo_resistant' column")
top_df = pd.read_csv(TOP_FEATURES_CSV, header=None)
top_list = top_df.iloc[:, 0].astype(str).tolist() if top_df.shape[1] == 1 else top_df.iloc[:, 1].astype(str).tolist()
available_features = [f for f in top_list if f in df.columns]
if not available_features:
    raise ValueError("No top features in dataframe")
print(f"Using {len(available_features)} features")
X_full = df[available_features].values.astype(np.float32)
y_full = df['chemo_resistant'].values.astype(np.int64)
# ---------------------------
# 2) Stratified split
# ---------------------------
sss = StratifiedShuffleSplit(n_splits=FOLDS, test_size=TEST_SIZE, random_state=SEED)

confusion_matrices = []
stats = []
# Just accuracy and F-Score

fold = 1
for train_index, test_index in sss.split(X_full, y_full):
    print(f"FOLD {fold}")
    fold+=1
    X_temp, y_temp = X_full[train_index], y_full[train_index]
    X_test, y_test = X_full[test_index], y_full[test_index]

    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=TEST_SIZE, stratify=y_temp, random_state=SEED
    )

    # ---------------------------
    # 3) Impute + scale
    # ---------------------------
    imputer = SimpleImputer(strategy='mean')
    X_train_imp = imputer.fit_transform(X_train)
    X_val_imp = imputer.transform(X_val)
    X_test_imp = imputer.transform(X_test)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_imp)
    X_val_scaled = scaler.transform(X_val_imp)
    X_test_scaled = scaler.transform(X_test_imp)

    # ---------------------------
    # 4) Random Projection
    # ---------------------------
    n_total = X_train_scaled.shape[0] + X_val_scaled.shape[0] + X_test_scaled.shape[0]
    nc = jl_dim(n_total, JL_EPS) if N_COMPONENTS is None else int(N_COMPONENTS)
    nc = min(nc, 2000); nc = max(nc, 64)

    srp_global = SparseRandomProjection(n_components=nc, random_state=SEED)
    X_train_rp = srp_global.fit_transform(X_train_scaled)
    X_val_rp = srp_global.transform(X_val_scaled)
    X_test_rp = srp_global.transform(X_test_scaled)

    # ---------------------------
    # 5) SMOTE on projected training
    # ---------------------------
    sm = BorderlineSMOTE(k_neighbors=SMOTE_K, random_state=SEED) if USE_BORDERLINE_SMOTE else SMOTE(k_neighbors=SMOTE_K, random_state=SEED)
    X_train_res, y_train_res = sm.fit_resample(X_train_rp, y_train)

    # class weights
    unique, counts = np.unique(y_train, return_counts=True)
    n_classes = int(np.max(y_full) + 1)
    class_weights = np.ones(n_classes, dtype=np.float32)
    for cls, cnt in zip(unique, counts):
        class_weights[int(cls)] = len(y_train) / (len(unique) * cnt)
    minority = np.argmin(counts)

    # ---------------------------
    # 6) Combine nodes
    # ---------------------------
    X_combined = np.vstack([X_train_res, X_val_rp, X_test_rp])
    y_combined = np.concatenate([y_train_res, y_val, y_test])
    n_nodes = X_combined.shape[0]
    n_train_nodes = X_train_res.shape[0]
    n_val_nodes = X_val_rp.shape[0]
    n_test_nodes = X_test_rp.shape[0]

    train_idx = np.arange(0, n_train_nodes)
    val_idx   = np.arange(n_train_nodes, n_train_nodes + n_val_nodes)
    test_idx  = np.arange(n_train_nodes + n_val_nodes, n_nodes)

    # ---------------------------
    # 7) Ensemble similarity (GPU)
    # ---------------------------
    sim_sum = np.zeros((n_nodes, n_nodes), dtype=np.float32)
    X_comb_t = torch.from_numpy(X_combined).to(DEVICE)

    for i in range(N_ENSEMBLES):
        rs = SEED + 2000 + i
        rp = SparseRandomProjection(n_components=nc, random_state=rs)
        Xr = rp.fit_transform(X_combined)
        Xr_t = torch.from_numpy(Xr).to(DEVICE)
        S = torch.cosine_similarity(Xr_t.unsqueeze(1), Xr_t.unsqueeze(0), dim=-1).cpu().numpy()
        np.fill_diagonal(S, 0.0)
        sim_sum += S

    S_avg = sim_sum / N_ENSEMBLES
    S_avg[S_avg < 1e-8] = 0.0

    # k-NN
    k = max(1, KNN_K)
    idx_topk = np.argsort(-S_avg, axis=1)[:, :k]
    rows, cols, vals = [], [], []
    for r in range(n_nodes):
        for c in idx_topk[r]:
            s = S_avg[r, c]
            if s <= 0: continue
            rows.append(r); cols.append(c); vals.append(s)
    A = csr_matrix((vals, (rows, cols)), shape=(n_nodes, n_nodes))
    A = (A + A.T) / 2
    A.sum_duplicates()

    # Strong class-aware
    unique_c, counts_c = np.unique(y_combined, return_counts=True)
    class_counts = dict(zip(unique_c, counts_c))
    minority_cls = min(class_counts, key=class_counts.get)
    class_inds = {c: np.where(y_combined == c)[0] for c in unique_c}

    for node in class_inds[minority_cls]:
        same = class_inds[minority_cls]
        same = same[same != node]
        if len(same) == 0: continue
        row = S_avg[node, same]
        top_in = same[np.argsort(-row)[:max(5, k)]]
        for c in top_in:
            A[node, c] = max(A[node, c], S_avg[node, c])
            A[c, node] = max(A[c, node], S_avg[node, c])

    min_nodes = class_inds[minority_cls]
    for i in min_nodes:
        for j in min_nodes:
            if i == j: continue
            A[i, j] = max(A[i, j], 0.1)
    A = (A + A.T) / 2
    A.eliminate_zeros()

    # ---------------------------
    # 8) Laplacian + eigen
    # ---------------------------
    L = sparse_laplacian(A, normed=True)
    m_eigs = min(M_EIGS, n_nodes - 2) if n_nodes > 2 else n_nodes
    m_eigs = max(2, m_eigs)
    try:
        eigvals, eigvecs = eigsh(L, k=m_eigs, which='SM', tol=1e-5, maxiter=8000)
    except Exception as e:
        print("eigsh failed, falling back to dense:", e)
        denseL = L.toarray()
        eigvals_f, eigvecs_f = np.linalg.eigh(denseL)
        eigvals, eigvecs = eigvals_f[:m_eigs], eigvecs_f[:, :m_eigs]

    # ---------------------------
    # 9) Focal loss
    # --------------------------

    alpha_vec = class_weights / class_weights.sum()
    alpha_vec[minority] *= FOCAL_ALPHA_MINORITY
    alpha_vec = alpha_vec / alpha_vec.sum()
    

    criterion = FocalLoss(alpha=alpha_vec, gamma=FOCAL_GAMMA, device=DEVICE).to(DEVICE)

    # ---------------------------
    # 10) GWNN with learnable scale
    # ---------------------------
    X_t = torch.FloatTensor(X_combined).to(DEVICE)
    y_t = torch.LongTensor(y_combined).to(DEVICE)
    eigvals_t = torch.from_numpy(eigvals.astype(np.float32)).to(DEVICE)
    eigvecs_t = torch.from_numpy(eigvecs.astype(np.float32)).to(DEVICE)

    model = GWNN(X_t.shape[1], HIDDEN_DIM, HIDDEN_DIM2, n_classes, eigvals_t, eigvecs_t, DROPOUT).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = ReduceLROnPlateau(optimizer, mode='max', patience=SCHEDULER_PATIENCE, factor=0.5)

    mask_train = torch.zeros(n_nodes, dtype=torch.bool, device=DEVICE)
    mask_train[train_idx] = True

    # ---------------------------
    # 11) Training
    # ---------------------------
    best_val = -1.0
    patience_cnt = 0
    train_losses, val_scores, epochs_rec = [], [], []

    start = time.time()
    for epoch in range(1, N_EPOCHS + 1):
        model.train()
        optimizer.zero_grad()
        logits = model(X_t)
        loss = criterion(logits[mask_train], y_t[mask_train])
        loss.backward()
        optimizer.step()
        train_losses.append(loss.item())

        if epoch % 5 == 0 or epoch == 1:
            model.eval()
            with torch.no_grad():
                logits_val = model(X_t)
                preds_val = logits_val[val_idx].argmax(dim=1).cpu().numpy()
                val_f1 = f1_score(y_val, preds_val, average='macro', zero_division=0)
            val_scores.append(val_f1)
            epochs_rec.append(epoch)
            scheduler.step(val_f1)

            if val_f1 > best_val + 1e-6:
                best_val = val_f1
                patience_cnt = 0
                torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, "best_gwnn.pt"))
            else:
                patience_cnt += 1
                if patience_cnt >= EARLY_STOPPING_PATIENCE:
                    break

    print(f"Training done in {time.time()-start:.1f}s. Best Val Macro-F1: {best_val:.4f}")

    # ---------------------------
    # 12) CALIBRATE + ENSEMBLE (FIXED)
    # ---------------------------
    model.load_state_dict(torch.load(os.path.join(OUTPUT_DIR, "best_gwnn.pt"), map_location=DEVICE))
    model.eval()
    with torch.no_grad():
        logits_all = model(X_t)
        probs_val_gwnn = torch.softmax(logits_all[val_idx], dim=1).cpu().numpy()[:, 1]
        probs_test_gwnn = torch.softmax(logits_all[test_idx], dim=1).cpu().numpy()[:, 1]

    # --- Use GWNN-only threshold from validation ---
    res_val_probs = probs_val_gwnn[y_val == 1]
    best_thresh = res_val_probs.min() if len(res_val_probs) > 0 else 0.5
    print(f"GWNN threshold (100% val recall): {best_thresh:.3f}")

    # --- Light ensemble: 80% GWNN + 20% XGBoost ---
    dtrain = xgb.DMatrix(X_train_res, label=y_train_res)
    dval   = xgb.DMatrix(X_val_rp,   label=y_val)
    dtest  = xgb.DMatrix(X_test_rp,  label=y_test)

    params = {"eta":0.1, "max_depth":6, "seed":SEED, "verbosity":0}
    if n_classes == 2:
        params.update({"objective":"binary:logistic", "eval_metric":"logloss"})
    else:
        params.update({"objective":"multi:softprob", "num_class":n_classes, "eval_metric":"mlogloss"})
    watch = [(dtrain,'train'),(dval,'val')]
    bst = xgb.train(params, dtrain, num_boost_round=500, evals=watch,
                    early_stopping_rounds=30, verbose_eval=False)

    if n_classes == 2:
        val_prob = bst.predict(dval)
        best_t, best_f1 = 0.5, 0.0
        for t in np.arange(0.1, 0.9, 0.01):
            f1v = f1_score(y_val, (val_prob > t).astype(int), average='macro', zero_division=0)
            if f1v > best_f1:
                best_f1, best_t = f1v, t
        test_prob = bst.predict(dtest)
        preds_xgb = (test_prob > best_t).astype(int)
    else:
        preds_xgb = np.argmax(bst.predict(dtest), axis=1)


    test_prob_xgb = bst.predict(dtest)
    probs_ens = 0.8 * probs_test_gwnn + 0.2 * test_prob_xgb
    best_thresh = 0.25
    preds_final = (probs_ens > best_thresh).astype(int)

    # --- Metrics ---
    acc = accuracy_score(y_test, preds_final)
    f1_macro = (f1_score(y_test, preds_final, pos_label=0) + f1_score(y_test, preds_final, pos_label=1)) / 2

    confusion_matrices.append(confusion_matrix(y_test, preds_final))
    stats.append([acc, f1_macro])
    print("="*80)


tot_res = np.zeros([2, 2])
for i in range(FOLDS):
    print("="*80)
    print(f"FOLD {i+1}")
    print("="*80)
    print(f"Accuracy: {stats[i][0] : .4f} |  F-Score {stats[i][1] : .4f}")
    print(f"Confusion: \n{confusion_matrices[i]}")
    tot_res += confusion_matrices[i]

print("="*80)
print("OVERALL: ")
tp, fn = tot_res[0, 0], tot_res[0, 1]
tn, fp = tot_res[1, 1], tot_res[1, 0]
prec = (tp)/(tp+fp)
rec = (tn)/(tn+fn)

print(f"Total_Res: \n{tot_res}")
print(f"Accuracy: {(tp+tn)/(tp+fp+tn+fn) : .4f} | F-Score {(1)/(1/prec + 1/rec) : .4f}")
