"""
train.py
--------
Boucle d'entraînement complète pour le classifieur de mélanome.

Stratégie en 2 phases :
    Phase 1 : backbone gelé, seul le classifier est entraîné (LR élevé, ~5 epochs)
    Phase 2 : fine-tuning des derniers blocs du backbone (LR très bas, ~10 epochs)

Fonctionnalités :
    - Mixed precision (fp16) → 2x plus rapide sur GPU, 2x moins de VRAM
    - Early stopping → arrête si l'AUC ne s'améliore plus
    - Sauvegarde du meilleur checkpoint (meilleur AUC-ROC val)
    - Class weights dans la loss → gère le déséquilibre 2%/98%
    - Logging clair epoch par epoch
"""

import os
import time

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from torch.cuda.amp import GradScaler, autocast  # mixed precision

from dataset import get_dataloaders
from model import get_model


# ─── Configuration ────────────────────────────────────────────────────────────

CONFIG = {
    # Données
    "csv_path":    "data/ISIC_2020_Training_GroundTruth.csv",
    "images_dir":  "data/images",

    # Modèle
    "model_name":  "efficientnet_b3",
    "dropout":     0.3,

    # Phase 1 : backbone gelé
    "lr_phase1":   1e-3,    # LR élevé car seul le petit classifier est entraîné
    "epochs_phase1": 5,

    # Phase 2 : fine-tuning
    "lr_phase2":   1e-5,    # LR très bas pour ne pas détruire les features pré-entraînées
    "epochs_phase2": 15,
    "unfreeze_layers": 3,   # nombre de blocs du backbone à dégeler

    # Entraînement
    "batch_size":  32,
    "num_workers": 0,       # 0 sur Windows pour éviter les erreurs multiprocessing
    "seed":        42,

    # Early stopping
    "patience":    5,       # arrête si pas d'amélioration après 5 epochs

    # Sauvegarde
    "checkpoint_dir": "checkpoints",
    "checkpoint_name": "best_model.pt",
}


# ─── Calcul du pos_weight ─────────────────────────────────────────────────────

def get_pos_weight(train_df, device: str) -> torch.Tensor:
    """
    Calcule le poids positif pour BCEWithLogitsLoss.

    pos_weight = nb_bénins / nb_mélanomes
    → la loss pénalise ~49x plus les erreurs sur les mélanomes
    → compense le déséquilibre 2%/98% dans la loss elle-même

    Combiné avec le WeightedRandomSampler du DataLoader,
    on attaque le déséquilibre sur deux fronts.
    """
    n_pos = train_df["target"].sum()          # nb mélanomes
    n_neg = len(train_df) - n_pos             # nb bénins
    pos_weight = torch.tensor([n_neg / n_pos], dtype=torch.float32).to(device)

    print(f"[Loss] pos_weight = {pos_weight.item():.1f} "
          f"(bénins:{n_neg} / mélanomes:{n_pos})")
    return pos_weight


# ─── Évaluation sur val/test ──────────────────────────────────────────────────

def evaluate(model, loader, criterion, device: str) -> tuple[float, float]:
    """
    Évalue le modèle sur un DataLoader (val ou test).

    Returns:
        avg_loss : loss moyenne sur tout le loader
        auc      : AUC-ROC score (métrique principale)
    """
    model.eval()  # mode évaluation : désactive Dropout et BatchNorm stochastique

    total_loss = 0.0
    all_labels = []
    all_probs  = []

    with torch.no_grad():  # pas de calcul de gradient → plus rapide, moins de VRAM
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)

            logits = model(images)
            loss   = criterion(logits, labels)
            total_loss += loss.item()

            # Sigmoid → probabilité entre 0 et 1 (pour AUC-ROC)
            probs = torch.sigmoid(logits).cpu().numpy()
            all_probs.extend(probs.flatten())
            all_labels.extend(labels.cpu().numpy().flatten())

    avg_loss = total_loss / len(loader)

    # AUC-ROC : 0.5 = aléatoire, 1.0 = parfait
    # Nécessite au moins une instance de chaque classe
    try:
        auc = roc_auc_score(all_labels, all_probs)
    except ValueError:
        auc = 0.0  # cas rare : pas de positif dans le batch

    return avg_loss, auc


# ─── Boucle d'entraînement d'une epoch ───────────────────────────────────────

def train_one_epoch(model, loader, criterion, optimizer, scaler, device: str) -> float:
    """
    Entraîne le modèle sur une epoch complète.

    Utilise mixed precision (autocast) :
    - Les calculs forward se font en float16 (plus rapide sur GPU)
    - Les gradients sont mis à l'échelle (GradScaler) pour éviter l'underflow
    - Les poids sont mis à jour en float32 (précision maintenue)

    Returns:
        avg_loss : loss moyenne sur l'epoch
    """
    model.train()  # mode entraînement : active Dropout et BatchNorm

    total_loss = 0.0

    for batch_idx, (images, labels) in enumerate(loader):
        images = images.to(device)
        labels = labels.to(device)

        # ── Zero grad ────────────────────────────────────────────────────────
        # OBLIGATOIRE : PyTorch accumule les gradients par défaut
        # Si on ne remet pas à 0, les gradients s'accumulent sur plusieurs batchs
        optimizer.zero_grad()

        # ── Forward pass avec mixed precision ────────────────────────────────
        with autocast():
            logits = model(images)             # [batch, 1]
            loss   = criterion(logits, labels) # scalaire

        # ── Backward pass ─────────────────────────────────────────────────────
        # scaler.scale() multiplie la loss pour éviter l'underflow en fp16
        scaler.scale(loss).backward()

        # Gradient clipping : évite les gradients explosifs
        # Si la norme des gradients > 1.0, on la ramène à 1.0
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        # ── Mise à jour des poids ─────────────────────────────────────────────
        scaler.step(optimizer)    # optimizer.step() via scaler
        scaler.update()           # ajuste le facteur d'échelle

        total_loss += loss.item()

        # Log tous les 100 batchs
        if (batch_idx + 1) % 100 == 0:
            print(f"  Batch {batch_idx+1}/{len(loader)} — loss: {loss.item():.4f}")

    return total_loss / len(loader)


# ─── Sauvegarde du checkpoint ─────────────────────────────────────────────────

def save_checkpoint(model, optimizer, epoch: int, auc: float, config: dict):
    """
    Sauvegarde le modèle quand l'AUC s'améliore.

    Le checkpoint contient :
    - Les poids du modèle (model_state_dict)
    - L'état de l'optimizer (pour reprendre l'entraînement)
    - L'epoch et l'AUC (pour le logging)
    - La config complète (pour reproduire)
    """
    os.makedirs(config["checkpoint_dir"], exist_ok=True)
    path = os.path.join(config["checkpoint_dir"], config["checkpoint_name"])

    torch.save({
        "epoch":              epoch,
        "auc":                auc,
        "model_state_dict":   model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "config":             config,
    }, path)

    print(f"  ✅ Checkpoint sauvegardé (AUC={auc:.4f}) → {path}")


# ─── Entraînement complet ─────────────────────────────────────────────────────

def train(config: dict):
    """
    Entraînement complet en 2 phases avec early stopping.

    Phase 1 : backbone gelé  → entraîne le classifier
    Phase 2 : fine-tuning    → affine les derniers blocs du backbone
    """

    # ── Setup ─────────────────────────────────────────────────────────────────
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n{'='*60}")
    print(f"  Mélanome Detector — Entraînement")
    print(f"  Device : {device}")
    print(f"{'='*60}\n")

    # ── DataLoaders ───────────────────────────────────────────────────────────
    loaders, train_df = get_dataloaders(
        csv_path=config["csv_path"],
        images_dir=config["images_dir"],
        batch_size=config["batch_size"],
        num_workers=config["num_workers"],
        seed=config["seed"],
    )

    # ── Modèle ────────────────────────────────────────────────────────────────
    model = get_model(
        model_name=config["model_name"],
        pretrained=True,
        dropout_rate=config["dropout"],
        freeze_backbone=True,   # Phase 1 : backbone gelé
        device=device,
    )

    # ── Loss avec class weights ───────────────────────────────────────────────
    pos_weight = get_pos_weight(train_df, device)
    criterion  = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    # ── GradScaler pour mixed precision ──────────────────────────────────────
    # Uniquement utile sur GPU ; sur CPU c'est un no-op
    scaler = GradScaler()

    # ── Tracking ──────────────────────────────────────────────────────────────
    best_auc       = 0.0
    patience_count = 0

    # ══════════════════════════════════════════════════════════════════════════
    # PHASE 1 : Backbone gelé — entraîne uniquement le classifier
    # ══════════════════════════════════════════════════════════════════════════
    print(f"\n{'─'*60}")
    print(f"  PHASE 1 : Backbone gelé — LR={config['lr_phase1']}")
    print(f"  Epochs : {config['epochs_phase1']}")
    print(f"{'─'*60}\n")

    # Optimizer ne voit que les paramètres entraînables (requires_grad=True)
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=config["lr_phase1"],
        weight_decay=1e-4,   # L2 regularisation légère
    )

    # Scheduler cosinus : LR diminue doucement en suivant une courbe cosinus
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=config["epochs_phase1"]
    )

    for epoch in range(1, config["epochs_phase1"] + 1):
        t0 = time.time()

        train_loss = train_one_epoch(model, loaders["train"], criterion,
                                     optimizer, scaler, device)
        val_loss, val_auc = evaluate(model, loaders["val"], criterion, device)

        scheduler.step()  # met à jour le LR selon le scheduler cosinus

        elapsed = time.time() - t0
        print(f"Epoch {epoch:02d}/{config['epochs_phase1']} "
              f"| Train Loss: {train_loss:.4f} "
              f"| Val Loss: {val_loss:.4f} "
              f"| Val AUC: {val_auc:.4f} "
              f"| LR: {scheduler.get_last_lr()[0]:.2e} "
              f"| {elapsed:.0f}s")

        # Sauvegarde si meilleur AUC
        if val_auc > best_auc:
            best_auc = val_auc
            patience_count = 0
            save_checkpoint(model, optimizer, epoch, val_auc, config)
        else:
            patience_count += 1
            print(f"  ⚠️  Pas d'amélioration ({patience_count}/{config['patience']})")

        # Early stopping
        if patience_count >= config["patience"]:
            print(f"\n⛔ Early stopping déclenché à l'epoch {epoch}")
            break

    # ══════════════════════════════════════════════════════════════════════════
    # PHASE 2 : Fine-tuning — dégel des derniers blocs
    # ══════════════════════════════════════════════════════════════════════════
    print(f"\n{'─'*60}")
    print(f"  PHASE 2 : Fine-tuning — LR={config['lr_phase2']}")
    print(f"  Dégel des {config['unfreeze_layers']} derniers blocs")
    print(f"  Epochs : {config['epochs_phase2']}")
    print(f"{'─'*60}\n")

    # Dégel des derniers blocs du backbone
    model.unfreeze_backbone(n_layers=config["unfreeze_layers"])

    # Reset patience pour la phase 2
    patience_count = 0

    # Nouvel optimizer avec LR très bas pour le backbone dégelé
    # weight_decay plus fort pour régulariser le backbone
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=config["lr_phase2"],
        weight_decay=1e-3,
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=config["epochs_phase2"]
    )

    for epoch in range(1, config["epochs_phase2"] + 1):
        t0 = time.time()

        train_loss = train_one_epoch(model, loaders["train"], criterion,
                                     optimizer, scaler, device)
        val_loss, val_auc = evaluate(model, loaders["val"], criterion, device)

        scheduler.step()

        elapsed = time.time() - t0
        print(f"Epoch {epoch:02d}/{config['epochs_phase2']} "
              f"| Train Loss: {train_loss:.4f} "
              f"| Val Loss: {val_loss:.4f} "
              f"| Val AUC: {val_auc:.4f} "
              f"| LR: {scheduler.get_last_lr()[0]:.2e} "
              f"| {elapsed:.0f}s")

        if val_auc > best_auc:
            best_auc = val_auc
            patience_count = 0
            save_checkpoint(model, optimizer, epoch, val_auc, config)
        else:
            patience_count += 1
            print(f"  ⚠️  Pas d'amélioration ({patience_count}/{config['patience']})")

        if patience_count >= config["patience"]:
            print(f"\n⛔ Early stopping déclenché à l'epoch {epoch}")
            break

    # ══════════════════════════════════════════════════════════════════════════
    # ÉVALUATION FINALE SUR LE TEST SET
    # ══════════════════════════════════════════════════════════════════════════
    print(f"\n{'='*60}")
    print(f"  ÉVALUATION FINALE — Test Set")
    print(f"{'='*60}")

    # Charge le meilleur checkpoint sauvegardé pendant l'entraînement
    best_path = os.path.join(config["checkpoint_dir"], config["checkpoint_name"])
    checkpoint = torch.load(best_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    print(f"Meilleur modèle : epoch {checkpoint['epoch']} "
          f"— AUC val = {checkpoint['auc']:.4f}")

    test_loss, test_auc = evaluate(model, loaders["test"], criterion, device)
    print(f"\n  Test Loss : {test_loss:.4f}")
    print(f"  Test AUC  : {test_auc:.4f}")
    print(f"\n  Meilleur AUC val : {best_auc:.4f}")
    print(f"{'='*60}\n")

    return model, best_auc, test_auc


# ─── Point d'entrée ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    """
    Lance l'entraînement complet.
    Depuis la racine du projet :
        python src/train.py
    """
    model, best_val_auc, test_auc = train(CONFIG)