import json
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold, cross_val_score, learning_curve
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC



Region = 'forest' # 'forest', 'nonforest'


fft_dataframe = pd.read_parquet(f'data/cc/cc_data_intra_{Region}.parquet')
X = fft_dataframe.drop('results', axis = 1)
y = fft_dataframe.results

with open("hyper_param_intra.json", "r") as f:
    hp = json.load(f)

# SVM
print(f"Using hyperparams -> C={hp['SVM'][Region]['C']}, gamma={hp['SVM'][Region]['gamma']}")
print(f"Input shape: {X.shape}")
max_amp = np.max(X.values)
min_amp = np.min(X.values)
print(f"Maximum amplitude in input: {max_amp}")
print(f"Minimum amplitude in input: {min_amp}")

unique, counts = np.unique(y, return_counts=True)
print("Class distribution:", dict(zip(unique, counts)))

X_scaled = X

svm_classifier = SVC(random_state=42, probability=True, class_weight='balanced')
kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
param_grid_svm = {'C': hp['SVM'][Region]['C'],  'gamma': hp['SVM'][Region]['gamma'],  'kernel': ['rbf']} 

X_train_full, X_test, y_train_full, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42, stratify=y)
X_train, X_val, y_train, y_val = train_test_split(X_train_full, y_train_full, test_size=0.2, random_state=42, stratify=y_train_full)

print(f"Training set size: {len(X_train)}")
print(f"Validation set size: {len(X_val)}")
print(f"Test set size: {len(X_test)}")

grid_search_svm = GridSearchCV(svm_classifier, param_grid_svm, cv=kfold, scoring='accuracy', n_jobs=-1)
grid_search_svm.fit(X_train, y_train)
best_svm = grid_search_svm.best_estimator_
print("Best parameters for SVM: ", grid_search_svm.best_params_)

svm_pred = best_svm.predict(X_test)

print(f"\nSVM Classification Report for test_size={0.2}:")
print(classification_report(y_test, svm_pred))
print("Confusion Matrix for SVM:")
cm = confusion_matrix(y_test, svm_pred)
print(cm)

disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=best_svm.classes_)
disp.plot(cmap="Blues")
plt.tight_layout()
out_path = Path(f"results/intra/CM_{Region}_svm.svg")
disp.figure_.savefig(out_path, format="svg", bbox_inches="tight")

svm_cv_scores = cross_val_score(best_svm, X_train_full, y_train_full, cv=kfold, scoring='accuracy')
print(f"SVM Cross-Validation Accuracy for test_size={0.2}: {np.mean(svm_cv_scores):.4f}")

svm_prob = best_svm.predict_proba(X_test)[:, 1]
svm_fpr, svm_tpr, _ = roc_curve(y_test, svm_prob, pos_label = fft_dataframe['results'].unique()[1])
svm_auc = auc(svm_fpr, svm_tpr)
print('AUC = ', svm_auc)

plt.figure()
plt.plot(svm_fpr, svm_tpr, label=f'SVM (AUC = {svm_auc:.2f})', color='g')
plt.plot([0, 1], [0, 1], 'k--')  
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend(loc='lower right')
plt.tight_layout()
out_path = Path(f"results/intra/roc_{Region}_svm.svg")
plt.savefig(out_path, format="svg", bbox_inches="tight")
plt.close()


# RF
print(
    f"Using hyperparams -> "
    f"n_estimators={hp['RF'][Region]['n_estimators']}, "
    f"max_depth={hp['RF'][Region]['max_depth']}, "
    f"min_samples_split={hp['RF'][Region]['min_samples_split']}, "
    f"min_samples_leaf={hp['RF'][Region]['min_samples_leaf']}, "
    f"max_features={hp['RF'][Region]['max_features']}"
)

print(f"Input shape: {X.shape}")
max_amp = np.max(X.values)
min_amp = np.min(X.values)
print(f"Maximum amplitude in input: {max_amp}")
print(f"Minimum amplitude in input: {min_amp}")

unique, counts = np.unique(y, return_counts=True)
print("Class distribution:", dict(zip(unique, counts)))

X_scaled = X

X_train_full, X_test, y_train_full, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42, stratify=y)
X_train, X_val, y_train, y_val = train_test_split(X_train_full, y_train_full, test_size=0.2, random_state=42, stratify=y_train_full)

print(f"Training set size: {len(X_train)} (forest and non-forest combined)")
print(f"Validation set size: {len(X_val)} (forest and non-forest combined)")
print(f"Test set size: {len(X_test)} (forest and non-forest combined)")

rf_classifier = RandomForestClassifier(random_state=42)
kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
param_grid_rf = {
    'n_estimators': hp['RF'][Region]['n_estimators'], 
    'max_depth': hp['RF'][Region]['max_depth'],  
    'min_samples_split': hp['RF'][Region]['min_samples_split'], 
    'min_samples_leaf': hp['RF'][Region]['min_samples_leaf'],
    'max_features': hp['RF'][Region]['max_features'],
    'class_weight': ['balanced']
}

grid_search_rf = GridSearchCV(rf_classifier, param_grid_rf, cv=kfold, scoring='accuracy', n_jobs=-1)
grid_search_rf.fit(X_train, y_train)
best_rf = grid_search_rf.best_estimator_
print("Best parameters for RF: ", grid_search_rf.best_params_)

rf_pred = best_rf.predict(X_test)

print(f"\nRF Classification Report for test_size={0.2}:")
print(classification_report(y_test, rf_pred))
print("Confusion Matrix for RF:")
cm = confusion_matrix(y_test, rf_pred)
print(cm)

disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=best_rf.classes_)
disp.plot(cmap="Blues")
plt.tight_layout()
out_path = Path(f"results/intra/CM_{Region}_rf.svg")
disp.figure_.savefig(out_path, format="svg", bbox_inches="tight")

rf_cv_scores = cross_val_score(best_rf, X_train_full, y_train_full, cv=kfold, scoring='accuracy')
print(f"RF Cross-Validation Accuracy for test_size={0.2}: {np.mean(rf_cv_scores):.4f}")

rf_prob = best_rf.predict_proba(X_test)[:, 1]
rf_fpr, rf_tpr, _ = roc_curve(y_test, rf_prob, pos_label = fft_dataframe['results'].unique()[1])
rf_auc = auc(rf_fpr, rf_tpr)
print('AUC = ', rf_auc)

plt.figure()
plt.plot(rf_fpr, rf_tpr, label=f'RF (AUC = {rf_auc:.2f})', color='g')
plt.plot([0, 1], [0, 1], 'k--')  
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend(loc='lower right')
plt.tight_layout()
out_path = Path(f"results/intra/roc_{Region}_rf.svg")
plt.savefig(out_path, format="svg", bbox_inches="tight")
plt.close()