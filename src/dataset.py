"""
dataset.py
----------
Pipeline de chargement et préparation du dataset ISIC 2020 pour la détection de mélanome.

Points clés :
- Split stratifié train/val/test (80/10/10)
- Augmentations adaptées aux lésions cutanées (smartphone)
- WeightedRandomSampler pour gérer le déséquilibre ~2% mélanome
- Gestion des images corrompues
- Seed globale pour reproductibilité
"""

import os
import random

import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

import albumentations as A
from albumentations.pytorch import ToTensorV2


# ─── Reproductibilité ─────────────────────────────────────────────────────────

def seed_everything(seed: int = 42):
    """
    Fixe toutes les seeds pour garantir la reproductibilité des expériences.
    Important en MedTech pour valider et comparer les résultats.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ─── Augmentations ────────────────────────────────────────────────────────────

def get_transforms(phase: str) -> A.Compose:
    """
    Retourne le pipeline d'augmentation selon la phase.

    - Train : augmentations agressives pour simuler la variabilité des photos smartphone
      (angle, lumière, distance, qualité capteur)
    - Val / Test : uniquement resize + normalisation (pas d'aléatoire)

    Normalisation ImageNet (mean/std) car EfficientNet est pré-entraîné sur ImageNet.
    """

    if phase == "train":
        return A.Compose([
            # Resize standard EfficientNet-B3
            A.Resize(224, 224),

            # Flips : les lésions n'ont pas d'orientation canonique
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),

            # Shift / Scale / Rotate : simule les variations de prise de vue smartphone
            A.ShiftScaleRotate(
                shift_limit=0.05,
                scale_limit=0.1,
                rotate_limit=30,
                p=0.5
            ),

            # Bruit gaussien : simule les artefacts capteur bas de gamme
            A.GaussNoise(p=0.2),

            # Couleur : variations d'éclairage, teinte de peau, balance des blancs
            A.ColorJitter(
                brightness=0.2,
                contrast=0.2,
                saturation=0.2,
                hue=0.1,
                p=0.5
            ),

            # Normalisation ImageNet obligatoire avant EfficientNet
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),

            # Conversion numpy → tensor PyTorch [C, H, W]
            ToTensorV2(),
        ])

    else:
        # Val et Test : pas d'augmentation, juste resize + normalisation
        return A.Compose([
            A.Resize(224, 224),
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
            ToTensorV2(),
        ])


# ─── Dataset ──────────────────────────────────────────────────────────────────

class MelanomaDataset(Dataset):
    """
    Dataset PyTorch pour les images ISIC 2020.

    Args:
        df          : DataFrame avec colonnes 'image_name' et 'target'
        images_dir  : Dossier contenant les fichiers .jpg
        transform   : Pipeline Albumentations à appliquer
    """

    def __init__(self, df: pd.DataFrame, images_dir: str, transform=None):
        # reset_index pour éviter les problèmes d'indexation après split
        self.df = df.reset_index(drop=True)
        self.images_dir = images_dir
        self.transform = transform

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]

        # Construction du chemin avec f-string (plus lisible que +)
        img_path = os.path.join(self.images_dir, f"{row['image_name']}.jpg")

        # Chargement robuste : certaines images ISIC sont corrompues
        # On remplace par un tenseur noir plutôt que de crasher l'entraînement
        try:
            image = np.array(Image.open(img_path).convert("RGB"))
        except Exception:
            print(f"[WARNING] Image corrompue ou manquante : {img_path}")
            image = np.zeros((224, 224, 3), dtype=np.uint8)

        # Label float32 + unsqueeze(0) → shape [1] pour BCEWithLogitsLoss
        # BCEWithLogitsLoss attend pred:[batch,1] et label:[batch,1]
        label = torch.tensor(
            row["target"],
            dtype=torch.float32
        ).unsqueeze(0)

        # Application des augmentations Albumentations
        if self.transform:
            image = self.transform(image=image)["image"]

        return image, label


# ─── WeightedSampler ──────────────────────────────────────────────────────────

def get_weighted_sampler(train_df: pd.DataFrame) -> WeightedRandomSampler:
    """
    Crée un WeightedRandomSampler pour compenser le déséquilibre de classe.

    Dataset ISIC 2020 : ~2% mélanome / ~98% bénin
    Sans correction → le modèle prédit tout bénin → accuracy=98% mais inutile.

    Le sampler sur-échantillonne les mélanomes pour équilibrer les batchs.
    """
    class_counts = train_df["target"].value_counts()

    # Poids inversement proportionnel à la fréquence de chaque classe
    weights = 1.0 / class_counts

    # Poids par sample
    sample_weights = train_df["target"].map(weights).values

    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True  # nécessaire pour sur-échantillonner la classe rare
    )

    return sampler


# ─── DataLoaders ──────────────────────────────────────────────────────────────

def get_dataloaders(
    csv_path: str,
    images_dir: str,
    batch_size: int = 32,
    num_workers: int = 4,
    seed: int = 42
) -> tuple[dict, pd.DataFrame]:
    """
    Construit les DataLoaders train/val/test à partir du CSV ISIC.

    Args:
        csv_path    : Chemin vers ISIC_2020_Training_GroundTruth.csv
        images_dir  : Dossier contenant les images .jpg
        batch_size  : Taille des batchs (32 recommandé pour EfficientNet-B3)
        num_workers : Threads de chargement parallèle
        seed        : Seed pour reproductibilité du split

    Returns:
        loaders     : Dict {"train", "val", "test"} → DataLoader
        train_df    : DataFrame d'entraînement (utile pour class weights loss)
    """

    # Fixe les seeds avant tout
    seed_everything(seed)

    df = pd.read_csv(csv_path)

    # ── Split stratifié ──────────────────────────────────────────────────────
    # stratify=df["target"] garantit le même ratio mélanome/bénin dans chaque split
    # Sans ça, par malchance un split pourrait avoir 0 mélanome
    train_df, temp_df = train_test_split(
        df,
        test_size=0.2,
        stratify=df["target"],
        random_state=seed
    )
    val_df, test_df = train_test_split(
        temp_df,
        test_size=0.5,
        stratify=temp_df["target"],
        random_state=seed
    )

    # ── Stats du dataset ─────────────────────────────────────────────────────
    print("─" * 50)
    print(f"Train : {len(train_df):>6} images  |  mélanomes : {train_df['target'].sum():>4}  "
          f"({train_df['target'].mean():.1%})")
    print(f"Val   : {len(val_df):>6} images  |  mélanomes : {val_df['target'].sum():>4}  "
          f"({val_df['target'].mean():.1%})")
    print(f"Test  : {len(test_df):>6} images  |  mélanomes : {test_df['target'].sum():>4}  "
          f"({test_df['target'].mean():.1%})")
    print("─" * 50)

    # ── Datasets ─────────────────────────────────────────────────────────────
    datasets = {
        "train": MelanomaDataset(train_df, images_dir, get_transforms("train")),
        "val":   MelanomaDataset(val_df,   images_dir, get_transforms("val")),
        "test":  MelanomaDataset(test_df,  images_dir, get_transforms("val")),
    }

    # ── Sampler pour le train uniquement ─────────────────────────────────────
    # Val et Test doivent rester non modifiés (distribution réelle)
    train_sampler = get_weighted_sampler(train_df)

    # ── DataLoaders ──────────────────────────────────────────────────────────
    loaders = {}

    loaders["train"] = DataLoader(
        datasets["train"],
        batch_size=batch_size,
        sampler=train_sampler,       # remplace shuffle=True
        num_workers=num_workers,
        pin_memory=True,             # accélère le transfert CPU → GPU
        persistent_workers=True,     # évite de re-créer les workers à chaque epoch
        prefetch_factor=2,           # précharge 2 batchs en avance
    )

    for phase in ["val", "test"]:
        loaders[phase] = DataLoader(
            datasets[phase],
            batch_size=batch_size,
            shuffle=False,           # pas de shuffle pour val/test
            num_workers=num_workers,
            pin_memory=True,
            persistent_workers=True,
            prefetch_factor=2,
        )

    return loaders, train_df


# ─── Test rapide ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    """
    Test minimal : vérifie que le pipeline tourne sans erreur.
    Lance avec : python src/dataset.py
    """
    loaders, train_df = get_dataloaders(
        csv_path="data/ISIC_2020_Training_GroundTruth.csv",
        images_dir="data/images",
        batch_size=32,
        num_workers=0,   # 0 pour le debug Windows (évite les erreurs multiprocessing)
    )

    # Vérifie la forme d'un batch
    images, labels = next(iter(loaders["train"]))
    print(f"Batch images : {images.shape}")   # attendu : [32, 3, 224, 224]
    print(f"Batch labels : {labels.shape}")   # attendu : [32, 1]
    print(f"Exemple labels : {labels[:8].squeeze().tolist()}")