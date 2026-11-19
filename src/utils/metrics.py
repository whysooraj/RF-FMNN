import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import time

def evaluate_model(model, X_train, y_train, X_test, y_test):
    """
    Evaluates a model and returns performance metrics, runtime, and box count.
    """
    t0 = time.perf_counter()
    model.fit(X_train, y_train)
    t_train = time.perf_counter() - t0

    t1 = time.perf_counter()
    y_pred = model.predict(X_test)
    t_infer = time.perf_counter() - t1

    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='macro', zero_division=0)
    prec = precision_score(y_test, y_pred, average='macro', zero_division=0)
    rec = recall_score(y_test, y_pred, average='macro', zero_division=0)

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
        'f1_macro': float(f1),
        'precision_macro': float(prec),
        'recall_macro': float(rec),
        'train_time_sec': float(t_train),
        'infer_time_sec': float(t_infer),
        'total_hyperboxes': int(total_boxes)
    }
