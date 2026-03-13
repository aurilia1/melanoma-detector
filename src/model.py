"""
model.py
--------
Définition du modèle EfficientNet-B3 fine-tuné pour la détection de mélanome.

Stratégie de fine-tuning :
- Phase 1 : on gèle le backbone, on entraîne uniquement le classifier (head)
- Phase 2 : on dégèle les dernières couches du backbone (fine-tuning progressif)

Pourquoi EfficientNet-B3 ?
- Meilleur compromis accuracy/vitesse pour de l'inférence smartphone
- Pré-entraîné sur ImageNet → features visuelles générales réutilisables
- Utilisé dans les papiers ISIC récents avec AUC > 0.87
"""

import torch
import torch.nn as nn
import timm  # bibliothèque de modèles pré-entraînés (EfficientNet, ViT, etc.)


# ─── Modèle ───────────────────────────────────────────────────────────────────

class MelanomaClassifier(nn.Module):
    """
    Classifieur binaire mélanome / bénin basé sur EfficientNet-B3.

    Architecture :
        EfficientNet-B3 (backbone pré-entraîné ImageNet)
            └── classifier maison :
                    Linear → BN → ReLU → Dropout → Linear(1)

    Le head custom remplace la tête de classification originale d'ImageNet (1000 classes)
    par une sortie binaire (1 logit) compatible avec BCEWithLogitsLoss.

    Args:
        model_name      : Nom du modèle timm (défaut : efficientnet_b3)
        pretrained      : Utiliser les poids ImageNet pré-entraînés
        dropout_rate    : Taux de dropout dans le classifier (régularisation)
        freeze_backbone : Si True, gèle le backbone au démarrage (Phase 1)
    """

    def __init__(
        self,
        model_name: str = "efficientnet_b3",
        pretrained: bool = True,
        dropout_rate: float = 0.3,
        freeze_backbone: bool = True,
    ):
        super().__init__()

        # ── Chargement du backbone pré-entraîné ─────────────────────────────
        # num_classes=0 → timm retire la tête de classification originale
        # On récupère uniquement le feature extractor (backbone)
        self.backbone = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=0,       # pas de tête → on en met une custom
            global_pool="avg",   # Global Average Pooling en sortie du backbone
        )

        # Dimension de sortie du backbone (1536 pour EfficientNet-B3)
        in_features = self.backbone.num_features

        # ── Tête de classification custom ───────────────────────────────────
        # Linear(1536→256) → BatchNorm → ReLU → Dropout → Linear(256→1)
        # BatchNorm stabilise l'entraînement avec peu de données
        # Dropout prévient l'overfitting (dataset médical = souvent petit)
        self.classifier = nn.Sequential(
            nn.Linear(in_features, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_rate),
            nn.Linear(256, 1),   # 1 logit → BCEWithLogitsLoss
        )

        # ── Gel du backbone (Phase 1) ────────────────────────────────────────
        # On commence par entraîner uniquement le classifier
        # Le backbone pré-entraîné est trop "fragile" pour un LR élevé
        if freeze_backbone:
            self.freeze_backbone()

    def freeze_backbone(self):
        """
        Gèle tous les paramètres du backbone.
        Utilisé en Phase 1 : seul le classifier est entraîné.
        """
        for param in self.backbone.parameters():
            param.requires_grad = False
        print("[Model] Backbone gelé — seul le classifier est entraînable.")

    def unfreeze_backbone(self, n_layers: int = 3):
        """
        Dégèle les n derniers blocs du backbone pour le fine-tuning (Phase 2).
        On dégèle progressivement pour éviter de détruire les features pré-entraînées.

        Args:
            n_layers : Nombre de blocs à dégeler depuis la fin
        """
        # Récupère tous les blocs enfants du backbone
        children = list(self.backbone.children())

        # Dégèle uniquement les n derniers
        for child in children[-n_layers:]:
            for param in child.parameters():
                param.requires_grad = True

        print(f"[Model] {n_layers} derniers blocs du backbone dégelés pour fine-tuning.")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x : Tensor image [batch, 3, 224, 224]

        Returns:
            logits : Tensor [batch, 1]  (pas de sigmoid → BCEWithLogitsLoss s'en charge)
        """
        # Extraction des features par le backbone
        features = self.backbone(x)   # → [batch, 1536]

        # Classification
        logits = self.classifier(features)   # → [batch, 1]

        return logits


# ─── Utilitaires ──────────────────────────────────────────────────────────────

def get_model(
    model_name: str = "efficientnet_b3",
    pretrained: bool = True,
    dropout_rate: float = 0.3,
    freeze_backbone: bool = True,
    device: str = None,
) -> MelanomaClassifier:
    """
    Factory function : crée et envoie le modèle sur le bon device.

    Args:
        device : "cuda", "cpu", ou None (auto-détection)

    Returns:
        model sur le device approprié
    """

    # Auto-détection GPU / CPU
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"[Model] Device : {device}")
    print(f"[Model] Backbone : {model_name} (pretrained={pretrained})")

    model = MelanomaClassifier(
        model_name=model_name,
        pretrained=pretrained,
        dropout_rate=dropout_rate,
        freeze_backbone=freeze_backbone,
    )

    model = model.to(device)

    # Résumé des paramètres
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[Model] Paramètres totaux    : {total:,}")
    print(f"[Model] Paramètres entraînables : {trainable:,} ({100*trainable/total:.1f}%)")

    return model


def load_checkpoint(model: MelanomaClassifier, checkpoint_path: str, device: str = "cpu"):
    """
    Charge un checkpoint sauvegardé pendant l'entraînement.

    Args:
        model           : Instance du modèle (architecture déjà créée)
        checkpoint_path : Chemin vers le fichier .pt
        device          : Device cible

    Returns:
        model avec les poids chargés
        metadata du checkpoint (epoch, AUC, etc.)
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])

    print(f"[Model] Checkpoint chargé : {checkpoint_path}")
    print(f"        Epoch : {checkpoint.get('epoch', '?')}")
    print(f"        AUC-ROC val : {checkpoint.get('auc', '?'):.4f}")

    return model, checkpoint


# ─── Test rapide ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    """
    Test minimal : vérifie que le forward pass tourne sans erreur.
    Lance avec : python src/model.py
    """

    # Création du modèle (Phase 1 : backbone gelé)
    model = get_model(
        model_name="efficientnet_b3",
        pretrained=False,    # False pour le test (pas besoin de télécharger les poids)
        freeze_backbone=True,
    )

    # Simule un batch de 4 images 224x224
    dummy_input = torch.randn(4, 3, 224, 224)
    output = model(dummy_input)

    print(f"\nInput shape  : {dummy_input.shape}")   # [4, 3, 224, 224]
    print(f"Output shape : {output.shape}")           # [4, 1]
    print(f"Output logits : {output.squeeze().tolist()}")

    # Test Phase 2 : dégel des 3 derniers blocs
    print("\n── Phase 2 : unfreeze ──")
    model.unfreeze_backbone(n_layers=3)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Paramètres entraînables après unfreeze : {trainable:,}")