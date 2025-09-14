import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.impute import KNNImputer
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, accuracy_score, confusion_matrix
import scipy.sparse as sp
from scipy.sparse.linalg import eigsh
import matplotlib.pyplot as plt
import seaborn as sns
from imblearn.over_sampling import SMOTE
import warnings
warnings.filterwarnings('ignore')

class GraphWaveletLayer(nn.Module):
    """
    Graph Wavelet Convolution Layer based on Xu et al. (ICLR 2019)
    Implements spectral graph wavelets with learnable filters, vectorized for efficiency
    """
    def __init__(self, in_channels, out_channels, num_wavelets=4, K=64):
        super(GraphWaveletLayer, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_wavelets = num_wavelets
        self.K = K  # Number of eigenvectors

        # Learnable wavelet parameters
        self.wavelet_scales = nn.Parameter(torch.linspace(0.1, 2.0, num_wavelets))
        self.wavelet_filters = nn.Parameter(torch.randn(num_wavelets, in_channels, out_channels))
        self.bias = nn.Parameter(torch.zeros(out_channels))

        # Learnable kernel parameters for g(λ)
        self.kernel_params = nn.Parameter(torch.randn(num_wavelets, 3))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.wavelet_filters)
        nn.init.uniform_(self.wavelet_scales, 0.1, 2.0)
        nn.init.normal_(self.kernel_params, 0, 0.1)

    def raised_cosine_kernel(self, eigenvals, scale, params):
        """Raised cosine wavelet kernel g(λ)"""
        a, b, c = params
        lambda_scaled = eigenvals / scale
        kernel = torch.zeros_like(lambda_scaled)
        mask = (lambda_scaled >= a) & (lambda_scaled <= b)
        kernel[mask] = 0.5 * (1 + torch.cos(np.pi * (lambda_scaled[mask] - a) / (b - a)))
        return kernel + c * torch.exp(-lambda_scaled**2)

    def compute_wavelet_operator(self, eigenvals, eigenvecs):
        """Compute graph wavelet operator ψ(Λ) = diag((g(Λ) - γ g(Λ/s))^2)"""
        gamma = 0.5
        wavelet_ops = []
        for i, scale in enumerate(self.wavelet_scales):
            g_lambda = self.raised_cosine_kernel(eigenvals, scale, self.kernel_params[i])
            g_lambda_scaled = self.raised_cosine_kernel(eigenvals, scale * 2.0, self.kernel_params[i])
            psi_lambda = (g_lambda - gamma * g_lambda_scaled) ** 2
            psi_diag = torch.diag(psi_lambda)
            wavelet_op = eigenvecs @ psi_diag @ eigenvecs.T
            wavelet_ops.append(wavelet_op)
        return torch.stack(wavelet_ops, dim=0)

    def forward(self, x, eigenvals, eigenvecs):
        """Forward pass: Vectorized wavelet convolution"""
        num_nodes, in_channels = x.size()
        device = x.device
        wavelet_ops = self.compute_wavelet_operator(eigenvals.to(device), eigenvecs.to(device))

        # Vectorized computation across all wavelets
        wavelet_x = torch.einsum('wnm,mi->wni', wavelet_ops, x)  # [num_wavelets, num_nodes, in_channels]
        spectral_x = torch.einsum('kn,wni->wki', eigenvecs.T, wavelet_x)  # [num_wavelets, K, in_channels]
        filtered = torch.einsum('wki,wio->wko', spectral_x, self.wavelet_filters)  # [num_wavelets, K, out_channels]
        spatial_output = torch.einsum('nk,wko->nwo', eigenvecs, filtered)  # [num_nodes, num_wavelets, out_channels]
        output = spatial_output.mean(dim=1) + self.bias  # [num_nodes, out_channels]
        return output

class GWNN(nn.Module):
    """Graph Wavelet Neural Network for chemotherapy resistance prediction"""
    def __init__(self, input_dim, hidden_dim=128, num_classes=1, num_wavelets=4, K=64, dropout=0.5):
        super(GWNN, self).__init__()
        self.gwnn1 = GraphWaveletLayer(input_dim, hidden_dim, num_wavelets, K)
        self.gwnn2 = GraphWaveletLayer(hidden_dim, hidden_dim, num_wavelets, K)
        self.gwnn3 = GraphWaveletLayer(hidden_dim, hidden_dim, num_wavelets, K)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.bn3 = nn.BatchNorm1d(hidden_dim)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, num_classes)
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, eigenvals, eigenvecs, mask):
        """Forward pass with node mask for transductive learning"""
        num_nodes = x.size(0)
        device = x.device
        idx = torch.nonzero(mask).squeeze()

        x = self.gwnn1(x, eigenvals, eigenvecs)
        x_masked = x[idx]
        x_masked = self.bn1(x_masked)
        x_masked = F.relu(x_masked)
        x_masked = self.dropout(x_masked)
        x_full = torch.zeros(num_nodes, x_masked.size(1), device=device)
        x_full[idx] = x_masked
        x = x_full

        x = self.gwnn2(x, eigenvals, eigenvecs)
        x_masked = x[idx]
        x_masked = self.bn2(x_masked)
        x_masked = F.relu(x_masked)
        x_masked = self.dropout(x_masked)
        x_full = torch.zeros(num_nodes, x_masked.size(1), device=device)
        x_full[idx] = x_masked
        x = x_full

        x = self.gwnn3(x, eigenvals, eigenvecs)
        x_masked = x[idx]
        x_masked = self.bn3(x_masked)
        x_masked = F.relu(x_masked)
        x_masked = self.dropout(x_masked)

        x_masked = self.mlp(x_masked)
        return x_masked

def load_and_preprocess_data(csv_path='tcga_rna_chemo.csv'):
    """Load and preprocess TCGA RNA-seq data or generate synthetic data"""
    try:
        df = pd.read_csv(csv_path)
        print(f"Loaded {len(df)} samples from {csv_path}")
    except FileNotFoundError:
        print(f"CSV not found. Generating synthetic data for demonstration...")
        df = generate_synthetic_data(n_samples=32, n_genes=100)
        df = extend_dataset(df, target_size=286)
        print(f"Generated {len(df)} synthetic samples")

    patient_ids = df['patient_id'].values
    target = df['chemo_resistant'].values
    rna_features = df.drop(['patient_id', 'chemo_resistant'], axis=1)

    imputer = KNNImputer(n_neighbors=5)
    rna_imputed = imputer.fit_transform(rna_features)
    scaler = MinMaxScaler()
    rna_scaled = scaler.fit_transform(rna_imputed)

    print(f"RNA features shape: {rna_scaled.shape}")
    print(f"Target distribution: {np.bincount((target * 2).astype(int)) / 2}")
    return rna_scaled, target, patient_ids

def generate_synthetic_data(n_samples=32, n_genes=100):
    """Generate synthetic RNA-seq data"""
    np.random.seed(42)
    data = []
    for i in range(n_samples):
        patient_id = f"TCGA-04-{1330+i:04d}"
        rna_values = np.random.lognormal(0, 2, n_genes) * np.random.exponential(10, n_genes)
        rna_values[rna_values < 0.01] = 0
        missing_mask = np.random.random(n_genes) < 0.05
        rna_values[missing_mask] = np.nan
        resistance_prob = np.random.random()
        chemo_resistant = 0.0 if resistance_prob < 0.4 else 0.5 if resistance_prob < 0.7 else 1.0
        row = {'patient_id': patient_id, 'chemo_resistant': chemo_resistant}
        for j in range(n_genes):
            row[f'RNA_GENE_{j:03d}'] = rna_values[j]
        data.append(row)
    return pd.DataFrame(data)

def extend_dataset(df, target_size=286):
    """Extend dataset to target size using SMOTE-like technique"""
    current_size = len(df)
    if current_size >= target_size:
        return df
    numerical_cols = [col for col in df.columns if col not in ['patient_id', 'chemo_resistant']]
    X = df[numerical_cols].fillna(0).values
    y = df['chemo_resistant'].values
    n_synthetic = target_size - current_size
    smote = SMOTE(random_state=42, k_neighbors=min(5, len(df)-1))
    y_discrete = (y * 2).astype(int)
    try:
        X_synthetic, y_synthetic = smote.fit_resample(X, y_discrete)
        X_new = X_synthetic[current_size:][:n_synthetic]
        y_new = y_synthetic[current_size:][:n_synthetic] / 2.0
        new_rows = []
        for i, (x_row, y_val) in enumerate(zip(X_new, y_new)):
            row = {'patient_id': f"SYNTH-{i:04d}", 'chemo_resistant': y_val}
            for j, col in enumerate(numerical_cols):
                row[col] = x_row[j]
            new_rows.append(row)
        return pd.concat([df, pd.DataFrame(new_rows)], ignore_index=True)
    except Exception as e:
        print(f"SMOTE failed: {e}. Using duplication with noise.")
        extended_df = df.copy()
        while len(extended_df) < target_size:
            sample_idx = np.random.choice(len(df))
            new_row = df.iloc[sample_idx].copy()
            new_row['patient_id'] = f"DUP-{len(extended_df):04d}"
            for col in numerical_cols:
                if pd.notna(new_row[col]):
                    new_row[col] *= np.random.normal(1.0, 0.05)
            extended_df = pd.concat([extended_df, new_row.to_frame().T], ignore_index=True)
        return extended_df[:target_size]

def build_knn_graph(features, k=10):
    """Build k-NN graph based on cosine similarity"""
    print(f"Building {k}-NN graph...")
    nbrs = NearestNeighbors(n_neighbors=k+1, metric='cosine', algorithm='brute').fit(features)
    distances, indices = nbrs.kneighbors(features)
    edges = []
    for i in range(len(features)):
        for j in range(1, k+1):
            edges.extend([[i, indices[i, j]], [indices[i, j], i]])
        edges.append([i, i])
    edges = list(set(tuple(edge) for edge in edges))
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    print(f"Graph constructed with {edge_index.size(1)} edges")
    return edge_index

def compute_laplacian_eigenvectors(edge_index, num_nodes, K=64):
    """Compute normalized Laplacian eigenvectors and eigenvalues"""
    print(f"Computing top {K} eigenvectors...")
    adj_matrix = sp.coo_matrix((np.ones(edge_index.size(1)), (edge_index[0], edge_index[1])),
                              shape=(num_nodes, num_nodes)).tocsr()
    degrees = np.array(adj_matrix.sum(axis=1)).flatten()
    D_inv_sqrt = sp.diags(1.0 / np.sqrt(np.maximum(degrees, 1e-8)))
    I = sp.eye(num_nodes)
    L_norm = I - D_inv_sqrt @ adj_matrix @ D_inv_sqrt
    try:
        eigenvals, eigenvecs = eigsh(L_norm, k=min(K, num_nodes-1), which='SM')
        eigenvals = torch.tensor(eigenvals, dtype=torch.float32)
        eigenvecs = torch.tensor(eigenvecs, dtype=torch.float32)
    except Exception as e:
        print(f"Eigendecomposition failed: {e}. Using identity.")
        eigenvals = torch.linspace(0, 2, min(K, num_nodes-1))
        eigenvecs = torch.eye(num_nodes)[:, :min(K, num_nodes-1)]
    print(f"Computed {len(eigenvals)} eigenvalues")
    return eigenvals, eigenvecs

def train_gwnn(model, X, y, eigenvals, eigenvecs, train_mask, val_mask, num_epochs=200, patience=20):
    """Train GWNN in transductive setting"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    X = X.to(device)
    y = y.to(device)
    eigenvals = eigenvals.to(device)
    eigenvecs = eigenvecs.to(device)
    train_mask = train_mask.to(device)
    val_mask = val_mask.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=5e-4)
    criterion = nn.MSELoss()

    train_losses, val_losses = [], []
    best_val_loss = float('inf')
    patience_counter = 0

    print(f"Training on {device} for {num_epochs} epochs...")
    for epoch in range(num_epochs):
        model.train()
        optimizer.zero_grad()
        out = model(X, eigenvals, eigenvecs, train_mask).squeeze()
        loss = criterion(out, y[train_mask])
        loss.backward()
        optimizer.step()
        train_losses.append(loss.item())

        model.eval()
        with torch.no_grad():
            out_val = model(X, eigenvals, eigenvecs, val_mask).squeeze()
            val_loss = criterion(out_val, y[val_mask])
            val_losses.append(val_loss.item())

        if epoch % 10 == 0:
            print(f'Epoch {epoch:03d}: Train Loss={train_losses[-1]:.4f}, Val Loss={val_losses[-1]:.4f}')

        if val_losses[-1] < best_val_loss:
            best_val_loss = val_losses[-1]
            patience_counter = 0
            torch.save(model.state_dict(), 'best_gwnn_model.pth')
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f'Early stopping at epoch {epoch}')
                break

    model.load_state_dict(torch.load('best_gwnn_model.pth'))
    return train_losses, val_losses

def evaluate_model(model, X, y, eigenvals, eigenvecs, test_mask, patient_ids_test):
    """Evaluate the trained model"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.eval()
    X = X.to(device)
    eigenvals = eigenvals.to(device)
    eigenvecs = eigenvecs.to(device)
    test_mask = test_mask.to(device)

    with torch.no_grad():
        predictions = model(X, eigenvals, eigenvecs, test_mask).squeeze().cpu().numpy()
    true_labels = y[test_mask].cpu().numpy()

    mse = mean_squared_error(true_labels, predictions)
    mae = mean_absolute_error(true_labels, predictions)
    r2 = r2_score(true_labels, predictions)

    pred_discrete = np.round(predictions * 2) / 2
    pred_discrete = np.clip(pred_discrete, 0, 1)

    true_labels_int = (true_labels * 2).astype(int)
    pred_discrete_int = (pred_discrete * 2).astype(int)
    accuracy = accuracy_score(true_labels_int, pred_discrete_int)

    print(f"Test MSE: {mse:.4f}")
    print(f"Test MAE: {mae:.4f}")
    print(f"Test R²: {r2:.4f}")
    print(f"Discretized Accuracy: {accuracy:.4f}")

    results_df = pd.DataFrame({
        'patient_id': patient_ids_test,
        'true': true_labels,
        'pred': predictions
    })
    results_df.to_csv('gwnn_predictions.csv', index=False)
    print("Predictions saved to gwnn_predictions.csv")

    return mse, mae, r2, accuracy, predictions, true_labels

def plot_results(train_losses, val_losses, predictions, true_labels):
    """Plot training curves, predictions, and confusion matrix"""
    plt.figure(figsize=(15, 5))

    plt.subplot(1, 3, 1)
    plt.plot(train_losses, label='Training Loss', alpha=0.7)
    plt.plot(val_losses, label='Validation Loss', alpha=0.7)
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.title('Training Curves')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.subplot(1, 3, 2)
    plt.scatter(true_labels, predictions, alpha=0.6)
    plt.plot([0, 1], [0, 1], 'r--', alpha=0.8)
    plt.xlabel('True Values')
    plt.ylabel('Predictions')
    plt.title('Predictions vs True Values')
    plt.grid(True, alpha=0.3)

    plt.subplot(1, 3, 3)
    pred_discrete = np.round(predictions * 2) / 2
    pred_discrete = np.clip(pred_discrete, 0, 1)
    cm = confusion_matrix(true_labels, pred_discrete, labels=[0.0, 0.5, 1.0])
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Not Resistant', 'Partial', 'Resistant'],
                yticklabels=['Not Resistant', 'Partial', 'Resistant'])
    plt.title('Confusion Matrix')
    plt.ylabel('True')
    plt.xlabel('Predicted')

    plt.tight_layout()
    plt.savefig('gwnn_results.png', dpi=300, bbox_inches='tight')
    plt.show()

def main():
    """Main execution pipeline"""
    print("=== Graph Wavelet Neural Network for Chemotherapy Resistance Prediction ===")

    # Load and preprocess data
    X, y, patient_ids = load_and_preprocess_data()

    # Train/test split (70/30 = 200/86)
    y_discrete = (y * 2).astype(int)
    X_train, X_test, y_train, y_test, ids_train, ids_test = train_test_split(
        X, y, patient_ids, test_size=0.3, random_state=42, stratify=y_discrete
    )
    X_train, X_val, y_train, y_val, ids_train, ids_val = train_test_split(
        X_train, y_train, ids_train, test_size=0.2, random_state=42, stratify=(y_train * 2).astype(int)
    )

    print(f"Dataset splits - Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")

    # Build graph
    edge_index = build_knn_graph(X, k=10)
    eigenvals, eigenvecs = compute_laplacian_eigenvectors(edge_index, len(X), K=64)

    # Create masks
    train_mask = torch.zeros(len(X), dtype=torch.bool)
    val_mask = torch.zeros(len(X), dtype=torch.bool)
    test_mask = torch.zeros(len(X), dtype=torch.bool)
    train_indices, val_indices, test_indices = [], [], []
    for i, pid in enumerate(patient_ids):
        if pid in ids_train:
            train_indices.append(i)
        elif pid in ids_val:
            val_indices.append(i)
        else:
            test_indices.append(i)
    train_mask[train_indices] = True
    val_mask[val_indices] = True
    test_mask[test_indices] = True

    # Convert to tensors
    X = torch.tensor(X, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.float32)

    # Initialize model
    model = GWNN(input_dim=X.shape[1], hidden_dim=128, num_classes=1, num_wavelets=4, K=64, dropout=0.5)
    print(f"Model initialized with {sum(p.numel() for p in model.parameters())} parameters")

    # Train model
    train_losses, val_losses = train_gwnn(model, X, y, eigenvals, eigenvecs, train_mask, val_mask)

    # Evaluate
    mse, mae, r2, accuracy, predictions, true_labels = evaluate_model(
        model, X, y, eigenvals, eigenvecs, test_mask, ids_test
    )

    # Plot results
    plot_results(train_losses, val_losses, predictions, true_labels)

    print("\n" + "="*60)
    print("GWNN Training Complete!")
    print(f"Model trained. Test MSE: {mse:.4f}, R²: {r2:.4f}")
    print(f"Test MAE: {mae:.4f}, Discretized Accuracy: {accuracy:.4f}")
    print("="*60)

    return model, mse, r2

if __name__ == "__main__":
    model, mse, r2 = main()
