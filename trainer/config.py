# config.py - Handles imports & global settings
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import KFold
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time
import os
import psutil
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, roc_curve, auc, top_k_accuracy_score
import json
import load_dataset

# Global Configurations
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_CLASSES = 10
NUM_EPOCHS = 20
BATCH_SIZE = 32
K_FOLDS = 5
