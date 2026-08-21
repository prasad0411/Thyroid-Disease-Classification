"""
Leak-free evaluation: RFE fit on training fold only.
Outputs per-class metrics, calibration, and confusion matrix.
"""
import json
import numpy as np
import pandas as pd
from datetime import datetime

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.feature_selection import RFE
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.metrics import (classification_report, confusion_matrix,
                             brier_score_loss, log_loss, accuracy_score)
from sklearn.calibration import calibration_curve

from config import *
from data_generator import generate_medical_dataset

try:
    from imblearn.over_sampling import SMOTE
    SMOTE_OK = True
except ImportError:
    SMOTE_OK = False

try:
    import xgboost as xgb
    XGB_OK = True
except ImportError:
    XGB_OK = False


def main():
    df = generate_medical_dataset()
    X = df.drop('target', axis=1)
    y = LabelEncoder().fit_transform(df['target'])
    le = LabelEncoder().fit(df['target'])

    # SPLIT FIRST — no test data touches feature selection
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y)

    # RFE on TRAIN ONLY
    sel = RFE(RandomForestClassifier(n_estimators=100,
              random_state=RANDOM_STATE, n_jobs=-1),
              n_features_to_select=N_FEATURES_SELECTED, step=1)
    sel.fit(X_tr, y_tr)
    feats = X_tr.columns[sel.support_].tolist()
    X_tr, X_te = X_tr[feats], X_te[feats]

    scaler = StandardScaler()
    X_trs = scaler.fit_transform(X_tr)
    X_tes = scaler.transform(X_te)

    if SMOTE_OK:
        X_trs, y_tr = SMOTE(random_state=RANDOM_STATE).fit_resample(X_trs, y_tr)

    models = {}
    if XGB_OK:
        models['XGBoost'] = xgb.XGBClassifier(**XGBOOST_PARAMS).fit(X_trs, y_tr)
    models['RandomForest'] = RandomForestClassifier(**RANDOMFOREST_PARAMS).fit(X_trs, y_tr)
    if len(models) > 1:
        models['Ensemble'] = VotingClassifier(
            estimators=list(models.items()), voting='soft', n_jobs=-1).fit(X_trs, y_tr)

    report = {'timestamp': datetime.now().strftime('%Y%m%d_%H%M%S'),
              'note': 'RFE fit on training fold only (no leakage)',
              'selected_features': feats, 'models': {}}

    for name, m in models.items():
        pred = m.predict(X_tes)
        proba = m.predict_proba(X_tes)
        cls_rep = classification_report(y_te, pred, target_names=le.classes_,
                                        output_dict=True, zero_division=0)
        # calibration: one-vs-rest per class
        calib = {}
        for i, c in enumerate(le.classes_):
            yt = (y_te == i).astype(int)
            calib[c] = {'brier': float(brier_score_loss(yt, proba[:, i]))}
        report['models'][name] = {
            'accuracy': float(accuracy_score(y_te, pred)),
            'log_loss': float(log_loss(y_te, proba)),
            'per_class': {c: {k: float(v) for k, v in cls_rep[c].items()}
                          for c in le.classes_},
            'calibration_brier': calib,
            'confusion_matrix': confusion_matrix(y_te, pred).tolist(),
        }
        print(f"\n{name}: acc={report['models'][name]['accuracy']:.4f} "
              f"logloss={report['models'][name]['log_loss']:.4f}")
        for c in le.classes_:
            p = report['models'][name]['per_class'][c]
            print(f"  {c:14s} P={p['precision']:.3f} R={p['recall']:.3f} "
                  f"F1={p['f1-score']:.3f} brier={calib[c]['brier']:.4f}")

    OUTPUTS_DIR.mkdir(exist_ok=True, parents=True)
    with open(OUTPUTS_DIR / 'eval_report.json', 'w') as f:
        json.dump(report, f, indent=2)
    print(f"\nSaved: {OUTPUTS_DIR / 'eval_report.json'}")


if __name__ == '__main__':
    main()
