"""
evaluate.py
-----------
Évaluation complète du modèle de détection de mélanome.

Produit :
    - Courbe ROC + AUC
    - Matrice de confusion
    - Rapport de classification complet
    - Recherche du seuil optimal (maximise le F1-score)
    - Métriques médicales : sensibilité, spécificité, VPP, VPN

En MedTech, l'évaluation est aussi importante que l'entraînement.
Un modèle avec AUC=0.87 mais seuil mal choisi peut avoir
une sensibilité de 40% → inutilisable cliniquement.
"""

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    auc,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)

from dataset import get_dataloaders
from model import get_model


# ─── Collecte des prédictions ─────────────────────────────────────────────────

def get_predictions(model, loader, device: str) -> tuple[np.ndarray, np.ndarray]:
    """
    Fait tourner le modèle sur tout un DataLoader et collecte
    les probabilités prédites et les vraies étiquettes.

    Returns:
        all_probs  : probabilités sigmoid [0, 1] pour chaque image
        all_labels : vraies étiquettes (0=bénin, 1=mélanome)
    """
    model.eval()
    all_probs  = []
    all_labels = []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)

            logits = model(images)                          # [batch, 1]
            probs  = torch.sigmoid(logits).cpu().numpy()   # → [0, 1]

            all_probs.extend(probs.flatten().tolist())
            all_labels.extend(labels.numpy().flatten().tolist())

    return np.array(all_probs), np.array(all_labels)


# ─── Seuil optimal ────────────────────────────────────────────────────────────

def find_optimal_threshold(labels: np.ndarray, probs: np.ndarray) -> float:
    """
    Trouve le seuil de décision qui maximise le F1-score.

    Par défaut on utilise 0.5 mais ce n'est jamais optimal
    sur un dataset déséquilibré.

    Sur ISIC 2020, le seuil optimal est souvent autour de 0.3-0.4
    car les mélanomes sont rares → le modèle est "timide".

    Returns:
        best_threshold : seuil optimal
    """
    precisions, recalls, thresholds = precision_recall_curve(labels, probs)

    # F1 = 2 * (precision * recall) / (precision + recall)
    # On calcule le F1 pour chaque seuil possible
    f1_scores = 2 * (precisions[:-1] * recalls[:-1]) / (
        precisions[:-1] + recalls[:-1] + 1e-8  # +1e-8 évite division par 0
    )

    best_idx       = np.argmax(f1_scores)
    best_threshold = thresholds[best_idx]
    best_f1        = f1_scores[best_idx]

    print(f"[Threshold] Seuil optimal : {best_threshold:.3f} "
          f"(F1={best_f1:.4f} vs F1@0.5={f1_score(labels, probs >= 0.5):.4f})")

    return float(best_threshold)


# ─── Métriques médicales ──────────────────────────────────────────────────────

def medical_metrics(labels: np.ndarray, preds: np.ndarray) -> dict:
    """
    Calcule les métriques cliniquement pertinentes.

    Glossaire médical vs ML :
        Sensibilité  = Recall    = TP / (TP + FN)  → % mélanomes détectés
        Spécificité  = TNR       = TN / (TN + FP)  → % bénins correctement rejetés
        VPP          = Precision = TP / (TP + FP)  → si positif, prob que ce soit vrai
        VPN                      = TN / (TN + FN)  → si négatif, prob que ce soit vrai

    En dépistage du mélanome :
        PRIORITÉ 1 → sensibilité élevée (ne pas rater un mélanome)
        PRIORITÉ 2 → spécificité correcte (ne pas alarmer inutilement)

    Un dermatologue humain atteint ~80% de sensibilité.
    Objectif modèle : > 85% sensibilité.
    """
    tn, fp, fn, tp = confusion_matrix(labels, preds).ravel()

    sensitivity  = tp / (tp + fn + 1e-8)   # recall / sensibilité
    specificity  = tn / (tn + fp + 1e-8)   # spécificité
    ppv          = tp / (tp + fp + 1e-8)   # valeur prédictive positive
    npv          = tn / (tn + fn + 1e-8)   # valeur prédictive négative
    accuracy     = (tp + tn) / (tp + tn + fp + fn)
    f1           = f1_score(labels, preds)

    metrics = {
        "TP": int(tp), "FP": int(fp),
        "TN": int(tn), "FN": int(fn),
        "Sensibilité (Recall)":    sensitivity,
        "Spécificité":             specificity,
        "VPP (Precision)":         ppv,
        "VPN":                     npv,
        "Accuracy":                accuracy,
        "F1-Score":                f1,
    }

    return metrics


# ─── Courbe ROC ───────────────────────────────────────────────────────────────

def plot_roc_curve(labels: np.ndarray, probs: np.ndarray,
                   save_dir: str = "outputs") -> float:
    """
    Trace et sauvegarde la courbe ROC.

    La courbe ROC montre le trade-off entre :
        - TPR (sensibilité) sur l'axe Y
        - FPR (1 - spécificité) sur l'axe X

    L'AUC (aire sous la courbe) résume la performance globale :
        0.5 = modèle aléatoire
        0.7 = acceptable
        0.85 = bon
        0.95 = excellent

    Returns:
        auc_score : valeur AUC-ROC
    """
    os.makedirs(save_dir, exist_ok=True)

    fpr, tpr, thresholds = roc_curve(labels, probs)
    auc_score = roc_auc_score(labels, probs)

    fig, ax = plt.subplots(figsize=(8, 6))

    # Courbe ROC du modèle
    ax.plot(fpr, tpr, color="steelblue", lw=2,
            label=f"Modèle (AUC = {auc_score:.4f})")

    # Ligne de référence : modèle aléatoire (AUC = 0.5)
    ax.plot([0, 1], [0, 1], color="gray", lw=1,
            linestyle="--", label="Aléatoire (AUC = 0.5)")

    ax.set_xlabel("Taux de Faux Positifs (1 - Spécificité)", fontsize=12)
    ax.set_ylabel("Taux de Vrais Positifs (Sensibilité)", fontsize=12)
    ax.set_title("Courbe ROC — Détection Mélanome", fontsize=14)
    ax.legend(loc="lower right", fontsize=11)
    ax.grid(alpha=0.3)

    path = os.path.join(save_dir, "roc_curve.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[Plot] Courbe ROC sauvegardée → {path}")

    return auc_score


# ─── Matrice de confusion ─────────────────────────────────────────────────────

def plot_confusion_matrix(labels: np.ndarray, preds: np.ndarray,
                          save_dir: str = "outputs"):
    """
    Trace et sauvegarde la matrice de confusion avec annotations claires.

    Lecture de la matrice (binaire) :
        Ligne 0 = vrais bénins    → [TN, FP]
        Ligne 1 = vrais mélanomes → [FN, TP]

    FN (Faux Négatifs) = mélanomes ratés → le pire cas en médecine
    """
    os.makedirs(save_dir, exist_ok=True)

    cm = confusion_matrix(labels, preds)

    fig, ax = plt.subplots(figsize=(6, 5))

    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    plt.colorbar(im, ax=ax)

    classes = ["Bénin (0)", "Mélanome (1)"]
    tick_marks = np.arange(len(classes))
    ax.set_xticks(tick_marks)
    ax.set_xticklabels(classes, fontsize=11)
    ax.set_yticks(tick_marks)
    ax.set_yticklabels(classes, fontsize=11)

    # Annotations dans chaque cellule
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            label = "TN" if (i==0 and j==0) else \
                    "FP" if (i==0 and j==1) else \
                    "FN ⚠️" if (i==1 and j==0) else "TP"
            ax.text(j, i, f"{label}\n{cm[i, j]}",
                    ha="center", va="center", fontsize=12,
                    color="white" if cm[i, j] > thresh else "black")

    ax.set_ylabel("Vraie étiquette", fontsize=12)
    ax.set_xlabel("Prédiction", fontsize=12)
    ax.set_title("Matrice de Confusion", fontsize=14)

    path = os.path.join(save_dir, "confusion_matrix.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[Plot] Matrice de confusion sauvegardée → {path}")


# ─── Rapport complet ──────────────────────────────────────────────────────────

def full_evaluation(
    checkpoint_path: str,
    csv_path: str,
    images_dir: str,
    batch_size: int = 32,
    num_workers: int = 0,
    save_dir: str = "outputs",
):
    """
    Évaluation complète à partir d'un checkpoint sauvegardé.

    Étapes :
        1. Charge le modèle depuis le checkpoint
        2. Collecte les prédictions sur le test set
        3. Trouve le seuil optimal
        4. Calcule toutes les métriques médicales
        5. Génère les graphiques (ROC + confusion matrix)
        6. Sauvegarde un rapport CSV
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(save_dir, exist_ok=True)

    # ── Chargement modèle ────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  Évaluation Complète — Mélanome Detector")
    print(f"{'='*60}\n")

    checkpoint = torch.load(checkpoint_path, map_location=device)

    model = get_model(
        pretrained=False,      # les poids viennent du checkpoint
        freeze_backbone=False, # on dégèle tout pour l'inférence
        device=device,
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    print(f"Checkpoint chargé : epoch {checkpoint['epoch']} "
          f"— AUC val = {checkpoint['auc']:.4f}\n")

    # ── DataLoader test ──────────────────────────────────────────────────────
    loaders, _ = get_dataloaders(
        csv_path=csv_path,
        images_dir=images_dir,
        batch_size=batch_size,
        num_workers=num_workers,
    )

    # ── Prédictions ──────────────────────────────────────────────────────────
    print("Calcul des prédictions sur le test set...")
    probs, labels = get_predictions(model, loaders["test"], device)

    # ── AUC-ROC ──────────────────────────────────────────────────────────────
    auc_score = plot_roc_curve(labels, probs, save_dir)
    print(f"\n  AUC-ROC : {auc_score:.4f}")

    # ── Seuil optimal ────────────────────────────────────────────────────────
    threshold = find_optimal_threshold(labels, probs)
    preds     = (probs >= threshold).astype(int)

    # ── Matrice de confusion ─────────────────────────────────────────────────
    plot_confusion_matrix(labels, preds, save_dir)

    # ── Métriques médicales ──────────────────────────────────────────────────
    metrics = medical_metrics(labels, preds)

    print(f"\n{'─'*40}")
    print(f"  Métriques au seuil {threshold:.3f}")
    print(f"{'─'*40}")
    for name, value in metrics.items():
        if isinstance(value, float):
            print(f"  {name:<30} : {value:.4f}")
        else:
            print(f"  {name:<30} : {value}")

    # ── Rapport sklearn ──────────────────────────────────────────────────────
    print(f"\n{classification_report(labels, preds, target_names=['Bénin', 'Mélanome'])}")

    # ── Sauvegarde CSV ───────────────────────────────────────────────────────
    report = {"AUC-ROC": auc_score, "Threshold": threshold, **metrics}
    df_report = pd.DataFrame([report])
    report_path = os.path.join(save_dir, "evaluation_report.csv")
    df_report.to_csv(report_path, index=False)
    print(f"[Report] Rapport sauvegardé → {report_path}")

    return auc_score, threshold, metrics


# ─── Point d'entrée ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    """
    Lance l'évaluation complète depuis la racine du projet :
        python src/evaluate.py
    """
    full_evaluation(
        checkpoint_path="checkpoints/best_model.pt",
        csv_path="data/ISIC_2020_Training_GroundTruth.csv",
        images_dir="data/images",
        batch_size=32,
        num_workers=0,
        save_dir="outputs",
    )