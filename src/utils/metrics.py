import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    confusion_matrix
)
import time

def compute_multiclass_specificity(y_true, y_pred, classes):
    """
    Computes macro-averaged specificity across all classes.
    For each class c:
    TN = sum of all confusion matrix elements not in row c and not in col c
    FP = sum of column c excluding diagonal
    Specificity_c = TN / (TN + FP)
    """
    cm = confusion_matrix(y_true, y_pred, labels=classes)
    n_classes = len(classes)
    if n_classes == 2:
        # Binary case: cm is [[TN, FP], [FN, TP]]
        tn, fp, fn, tp = cm.ravel()
        return float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0

    specificities = []
    for c in range(n_classes):
        tp = cm[c, c]
        fn = np.sum(cm[c, :]) - tp
        fp = np.sum(cm[:, c]) - tp
        tn = np.sum(cm) - (tp + fn + fp)
        spec = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
        specificities.append(spec)
    return float(np.mean(specificities))

def evaluate_model(model, X_train, y_train, X_test, y_test):
    """
    Evaluates a model across 6 key classification metrics:
    1. Accuracy
    2. Precision (macro)
    3. Recall / Sensitivity (macro)
    4. Specificity (macro)
    5. F1-Score (macro)
    6. AUC-ROC (multi-class OVR or binary)
    """
    t0 = time.perf_counter()
    model.fit(X_train, y_train)
    t_train = time.perf_counter() - t0

    t1 = time.perf_counter()
    y_pred = model.predict(X_test)
    t_infer = time.perf_counter() - t1

    classes = np.unique(np.concatenate([y_train, y_test]))

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average='macro', zero_division=0)
    rec = recall_score(y_test, y_pred, average='macro', zero_division=0)
    spec = compute_multiclass_specificity(y_test, y_pred, classes)
    f1 = f1_score(y_test, y_pred, average='macro', zero_division=0)

    # AUC-ROC computation
    auc_roc = 0.5
    try:
        if hasattr(model, 'predict_proba'):
            y_prob = model.predict_proba(X_test)
            if len(classes) == 2:
                # Binary
                if y_prob.shape[1] == 2:
                    auc_roc = float(roc_auc_score(y_test, y_prob[:, 1]))
                else:
                    auc_roc = float(roc_auc_score(y_test, y_prob))
            else:
                # Multiclass OVR
                auc_roc = float(roc_auc_score(y_test, y_prob, multi_class='ovr', labels=classes))
        else:
            auc_roc = float(acc)
    except Exception:
        auc_roc = float(acc)

    # Count hyperboxes
    total_boxes = 0
    if hasattr(model, 'hyperboxes'):
        total_boxes = len(model.hyperboxes)
    elif hasattr(model, 'models_'):
        for sub_m in model.models_:
            if hasattr(sub_m, 'hyperboxes'):
                total_boxes += len(sub_m.hyperboxes)

    return {
        'accuracy': float(acc),
        'precision': float(prec),
        'recall_sensitivity': float(rec),
        'specificity': float(spec),
        'f1_score': float(f1),
        'auc_roc': float(auc_roc),
        'train_time_sec': float(t_train),
        'infer_time_sec': float(t_infer),
        'total_hyperboxes': int(total_boxes)
    }
