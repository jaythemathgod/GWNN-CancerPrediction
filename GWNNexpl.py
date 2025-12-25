import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import shap


def plot_xgb_feature_importance(bst, max_features=20):
    importance = bst.get_score(importance_type='gain')
    if len(importance) == 0:
        print("No feature importance available.")
        return

    imp_df = pd.DataFrame({
        'feature': list(importance.keys()),
        'gain': list(importance.values())
    }).sort_values('gain', ascending=False)

    plt.figure(figsize=(6,4))
    plt.barh(imp_df['feature'][:max_features], imp_df['gain'][:max_features])
    plt.gca().invert_yaxis()
    plt.xlabel("Gain")
    plt.title("XGBoost Feature Importance (Gain)")
    plt.tight_layout()
    plt.show()


def plot_xgb_shap(bst, X):
    explainer = shap.TreeExplainer(bst)
    shap_values = explainer.shap_values(X)
    shap.summary_plot(shap_values, X, show=False)
    plt.tight_layout()
    plt.show()



def plot_gwnn_mean_saliency(model, X_t, test_idx, top_k=20):
    model.eval()
    X_t.requires_grad_(True)

    logits = model(X_t)
    logits[:, 1].sum().backward()

    saliency = X_t.grad[test_idx].abs().detach().cpu().numpy()
    mean_sal = saliency.mean(axis=0)

    idx = np.argsort(mean_sal)[-top_k:]

    plt.figure(figsize=(6,4))
    plt.bar(range(top_k), mean_sal[idx])
    plt.xticks(range(top_k), idx, rotation=90)
    plt.ylabel("Mean |Gradient|")
    plt.title("GWNN Mean Saliency (Test Nodes)")
    plt.tight_layout()
    plt.show()


def explain_node_neighbors(A, y, node, top_k=10):
    neighbors = A[node].nonzero()[1]

    if len(neighbors) == 0:
        return pd.DataFrame(columns=["neighbor_id", "edge_weight", "label"])

    weights = A[node, neighbors].toarray().ravel()

    order = np.argsort(-weights)[:top_k]
    return pd.DataFrame({
        "neighbor_id": neighbors[order],
        "edge_weight": weights[order],
        "label": y[neighbors[order]]
    })


def plot_node_neighbors(A, y, node, top_k=10):
    df = explain_node_neighbors(A, y, node, top_k)

    if len(df) == 0:
        print("No neighbors found for node", node)
        return df

    plt.figure(figsize=(5,4))
    plt.scatter(
        range(len(df)),
        df["edge_weight"],
        c=df["label"],
        cmap="coolwarm",
        s=60
    )
    plt.xlabel("Neighbor Rank")
    plt.ylabel("Edge Weight")
    plt.title(f"Influential Neighbors (Node {node})")
    plt.tight_layout()
    plt.show()

    return df


def plot_wavelet_scales(model):
    scales = [p.item() for n, p in model.named_parameters() if "scale" in n]
    if len(scales) == 0:
        print("No wavelet scales found.")
        return

    plt.figure(figsize=(4,3))
    plt.hist(scales, bins=10)
    plt.xlabel("Learned Scale Value")
    plt.ylabel("Count")
    plt.title("Learned Graph Wavelet Scales")
    plt.tight_layout()
    plt.show()


def plot_probability_threshold(probs, y_true, threshold):
    plt.figure(figsize=(5,4))
    plt.hist(probs[y_true==0], bins=30, alpha=0.6, label="Sensitive")
    plt.hist(probs[y_true==1], bins=30, alpha=0.6, label="Resistant")
    plt.axvline(threshold, color="red", linestyle="--", label="Threshold")
    plt.xlabel("Predicted Probability")
    plt.ylabel("Count")
    plt.legend()
    plt.title("Prediction Distribution with Decision Threshold")
    plt.tight_layout()
    plt.show()


print("Running explainability diagnostics...")

plot_xgb_feature_importance(bst)
plot_gwnn_mean_saliency(model, X_t, test_idx)

df_neighbors = plot_node_neighbors(A, y_combined, node=test_idx[0])
display(df_neighbors)

plot_wavelet_scales(model)

plot_probability_threshold(probs_test_gwnn, y_test, best_thresh)
