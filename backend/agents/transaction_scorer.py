"""
TransactionScorer — ML layer for per-transaction fraud scoring.

Trains an XGBoost (or RandomForest fallback) classifier on the Kaggle
Credit Card Fraud dataset. Scores every incoming transaction 0-100
BEFORE Ring Scout sees it, so Ring Scout can combine ML signal with
graph topology for higher-confidence detection.

Dataset columns expected:
  transaction_id, amount, transaction_hour, merchant_category,
  foreign_transaction, location_mismatch, device_trust_score,
  velocity_last_24h, cardholder_age, is_fraud
"""
import os
import pickle
import asyncio
import numpy as np
import pandas as pd
from agents.base_agent import BaseAgent

try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import (
    classification_report, roc_auc_score,
    precision_score, recall_score, f1_score,
    confusion_matrix
)

try:
    from imblearn.over_sampling import SMOTE
    SMOTE_AVAILABLE = True
except ImportError:
    SMOTE_AVAILABLE = False


MODEL_DIR   = 'models'
MODEL_PATH  = os.path.join(MODEL_DIR, 'fraud_scorer.pkl')
SCALER_PATH = os.path.join(MODEL_DIR, 'scaler.pkl')
ENCODER_PATH= os.path.join(MODEL_DIR, 'encoder.pkl')
METRICS_PATH= os.path.join(MODEL_DIR, 'metrics.json')


class TransactionScorer(BaseAgent):
    """
    Per-transaction ML fraud scorer.

    Two operating modes:
      1. Trained mode  — loads saved model, scores in microseconds
      2. Untrained mode — falls back to heuristic scoring using raw features
    """

    FEATURE_COLS = [
        'amount', 'transaction_hour',
        'foreign_transaction', 'location_mismatch',
        'device_trust_score', 'velocity_last_24h', 'cardholder_age',
        # one-hot encoded merchant categories appended at runtime
    ]

    MERCHANT_CATEGORIES = [
        'Electronics', 'Groceries', 'Travel',
        'Entertainment', 'Retail', 'Gas', 'Restaurant',
    ]

    def __init__(self):
        super().__init__('Transaction Scorer')
        self.model         = None
        self.scaler        = None
        self.label_encoder = None
        self.trained       = False
        self.metrics       = {}
        self._load_if_exists()

    # ── BaseAgent contract ──────────────────────────────────────────────
    async def execute(self, input_data: dict) -> dict:
        if 'transactions' in input_data:
            scored = await asyncio.get_event_loop().run_in_executor(
                None, lambda: self.score_batch(input_data['transactions'])
            )
            flagged = sum(1 for t in scored if t.get('ml_flagged'))
            self.log(f'Scored {len(scored)} transactions — ML flagged: {flagged}')
            return {'transactions': scored, 'ml_flagged_count': flagged}
        else:
            return self.score_transaction(input_data)

    # ── Training ────────────────────────────────────────────────────────
    def train(self, csv_path: str = 'data/creditcard.csv') -> dict:
        self.log(f'Loading dataset: {csv_path}')
        df = pd.read_csv(csv_path)
        self.log(f'Dataset: {len(df):,} rows — {df["is_fraud"].sum()} fraud, '
                 f'{(df["is_fraud"]==0).sum():,} legitimate')

        X, y = self._prepare_features(df, fit=True)

        # Split FIRST — test set must never see SMOTE synthetic samples
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        # SMOTE only on training data
        if SMOTE_AVAILABLE:
            self.log('Applying SMOTE to training set only...')
            sm = SMOTE(random_state=42, k_neighbors=min(5, sum(y_train==1)-1))
            X_train, y_train = sm.fit_resample(X_train, y_train)
            self.log(f'After SMOTE: {len(X_train):,} training samples')
        else:
            self.log('imblearn not installed — skipping SMOTE')

        # ── Train primary model ──────────────────────────────────────
        if XGBOOST_AVAILABLE:
            self.log('Training XGBoost classifier...')
            self.model = XGBClassifier(
                n_estimators=300,
                max_depth=6,
                learning_rate=0.08,
                subsample=0.8,
                colsample_bytree=0.8,
                scale_pos_weight=1,
                use_label_encoder=False,
                eval_metric='logloss',
                random_state=42,
                n_jobs=-1,
            )
        else:
            self.log('XGBoost not available — training GradientBoosting...')
            self.model = XGBClassifier(
                n_estimators=300,
                max_depth=6,
                learning_rate=0.08,
                subsample=0.8,
                colsample_bytree=0.8,
                scale_pos_weight=1,
                eval_metric='logloss',
                random_state=42,
                n_jobs=-1,
            )

        self.model.fit(X_train, y_train)

        # ── Evaluate ────────────────────────────────────────────────
        y_pred  = self.model.predict(X_test)
        y_proba = self.model.predict_proba(X_test)[:, 1]
        auc     = roc_auc_score(y_test, y_proba)
        prec    = precision_score(y_test, y_pred, zero_division=0)
        rec     = recall_score(y_test, y_pred, zero_division=0)
        f1      = f1_score(y_test, y_pred, zero_division=0)
        cm      = confusion_matrix(y_test, y_pred).tolist()

        self.metrics = {
            'auc_roc':   round(auc,  4),
            'precision': round(prec, 4),
            'recall':    round(rec,  4),
            'f1':        round(f1,   4),
            'confusion_matrix': cm,
            'model_type': 'XGBoost' if XGBOOST_AVAILABLE else 'GradientBoosting',
            'n_features': X.shape[1],
            'n_train':    len(X_train),
        }

        self.log(f'Training complete.')
        self.log(f'  AUC-ROC:   {auc:.4f}')
        self.log(f'  Precision: {prec:.4f}')
        self.log(f'  Recall:    {rec:.4f}')
        self.log(f'  F1:        {f1:.4f}')
        self.log('\n' + classification_report(y_test, y_pred,
                                              target_names=['Legit', 'Fraud']))

        # ── Save ─────────────────────────────────────────────────────
        os.makedirs(MODEL_DIR, exist_ok=True)
        with open(MODEL_PATH,   'wb') as f: pickle.dump(self.model,         f)
        with open(SCALER_PATH,  'wb') as f: pickle.dump(self.scaler,        f)
        with open(ENCODER_PATH, 'wb') as f: pickle.dump(self.label_encoder, f)

        import json
        with open(METRICS_PATH, 'w') as f:
            json.dump(self.metrics, f, indent=2)

        self.trained = True
        self.log(f'Model saved to {MODEL_DIR}/')
        return self.metrics

    # ── Scoring ─────────────────────────────────────────────────────────
    def score_transaction(self, txn: dict) -> dict:
        """Score one transaction. Returns dict with fraud_score (0-100) added."""
        if not self.trained:
            score = self._heuristic_score(txn)
            return {**txn, 'fraud_score': score, 'ml_flagged': score > 60,
                    'ml_source': 'heuristic'}

        features = self._extract_single(txn)
        proba    = float(self.model.predict_proba([features])[0][1])
        score    = min(int(proba * 100), 100)
        return {**txn, 'fraud_score': score, 'ml_flagged': score > 60,
                'ml_probability': proba, 'ml_source': 'xgboost'}

    def score_batch(self, transactions: list) -> list:
        """Score a list of transactions efficiently."""
        if not self.trained:
            return [self.score_transaction(t) for t in transactions]

        matrix = np.array([self._extract_single(t) for t in transactions])
        probas  = self.model.predict_proba(matrix)[:, 1]
        return [
            {**txn, 'fraud_score': min(int(p * 100), 100),
             'ml_flagged': p > 0.6, 'ml_probability': float(p),
             'ml_source': 'xgboost'}
            for txn, p in zip(transactions, probas)
        ]

    # ── Feature engineering ─────────────────────────────────────────────
    def _prepare_features(self, df: pd.DataFrame, fit: bool = False):
        """
        Full feature pipeline for training.
        One-hot encodes merchant_category, scales continuous features.
        """
        df = df.copy()

        # Encode merchant_category
        self.label_encoder = LabelEncoder()
        if 'merchant_category' in df.columns:
            df['merchant_category_enc'] = self.label_encoder.fit_transform(
                df['merchant_category'].fillna('Unknown')
            )
        else:
            df['merchant_category_enc'] = 0

        # Hour cyclical encoding — captures midnight wrap-around
        df['hour_sin'] = np.sin(2 * np.pi * df.get('transaction_hour', 12) / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df.get('transaction_hour', 12) / 24)

        # Risk interaction features
        df['risk_score'] = (
            df.get('location_mismatch', 0).astype(float) * 30 +
            df.get('foreign_transaction', 0).astype(float) * 20 +
            (100 - df.get('device_trust_score', 100).clip(0, 100)) * 0.3 +
            df.get('velocity_last_24h', 0).clip(0, 20) * 2
        )

        # Amount bins — fraud tends to cluster at specific ranges
        df['amount_log']  = np.log1p(df['amount'])
        df['is_high_amt'] = (df['amount'] > 500).astype(int)
        df['is_micro_amt']= (df['amount'] < 10).astype(int)

        feature_cols = [
            'amount', 'amount_log', 'is_high_amt', 'is_micro_amt',
            'hour_sin', 'hour_cos',
            'foreign_transaction', 'location_mismatch',
            'device_trust_score', 'velocity_last_24h', 'cardholder_age',
            'merchant_category_enc', 'risk_score',
        ]
        # Fill any missing columns with 0
        for col in feature_cols:
            if col not in df.columns:
                df[col] = 0

        X = df[feature_cols].fillna(0).values

        self.scaler = StandardScaler()
        X_scaled    = self.scaler.fit_transform(X)

        y = df['is_fraud'].values
        return X_scaled, y

    def _extract_single(self, txn: dict) -> list:
        """Extract feature vector from a single transaction dict."""
        amount   = float(txn.get('amount', 0))
        hour     = float(txn.get('transaction_hour', 12))
        foreign  = float(txn.get('foreign_transaction', 0))
        loc_mis  = float(txn.get('location_mismatch', 0))
        dev_trust= float(txn.get('device_trust_score', 100))
        velocity = float(txn.get('velocity_last_24h', 1))
        age      = float(txn.get('cardholder_age', 35))
        merchant = str(txn.get('merchant_category', 'Unknown'))

        # Encode merchant
        try:
            m_enc = float(self.label_encoder.transform([merchant])[0])
        except (ValueError, AttributeError):
            m_enc = 0.0

        hour_sin  = np.sin(2 * np.pi * hour / 24)
        hour_cos  = np.cos(2 * np.pi * hour / 24)
        risk      = loc_mis * 30 + foreign * 20 + (100 - dev_trust) * 0.3 + min(velocity, 20) * 2
        amount_log= np.log1p(amount)
        is_high   = float(amount > 500)
        is_micro  = float(amount < 10)

        raw = np.array([[
            amount, amount_log, is_high, is_micro,
            hour_sin, hour_cos,
            foreign, loc_mis, dev_trust, velocity, age,
            m_enc, risk,
        ]])

        if self.scaler:
            return self.scaler.transform(raw)[0].tolist()
        return raw[0].tolist()

    def _heuristic_score(self, txn: dict) -> int:
        """Rule-based score when model not trained yet."""
        score = 0
        if txn.get('location_mismatch'):        score += 30
        if txn.get('foreign_transaction'):      score += 20
        dev = float(txn.get('device_trust_score', 100))
        if dev < 30:                            score += 30
        elif dev < 60:                          score += 15
        vel = int(txn.get('velocity_last_24h', 1))
        if vel > 10:                            score += 20
        elif vel > 5:                           score += 10
        amt = float(txn.get('amount', 0))
        if 400 <= amt <= 499:                   score += 25
        if amt > 2000:                          score += 10
        return min(score, 100)

    # ── Persistence ─────────────────────────────────────────────────────
    def _load_if_exists(self):
        if (os.path.exists(MODEL_PATH) and
                os.path.exists(SCALER_PATH) and
                os.path.exists(ENCODER_PATH)):
            try:
                with open(MODEL_PATH,   'rb') as f: self.model         = pickle.load(f)
                with open(SCALER_PATH,  'rb') as f: self.scaler        = pickle.load(f)
                with open(ENCODER_PATH, 'rb') as f: self.label_encoder = pickle.load(f)
                self.trained = True
                if os.path.exists(METRICS_PATH):
                    import json
                    with open(METRICS_PATH) as f:
                        self.metrics = json.load(f)
                self.log(f'Loaded pre-trained model. '
                         f'AUC: {self.metrics.get("auc_roc", "unknown")}')
            except Exception as e:
                self.log(f'Failed to load model: {e}')
                self.trained = False
        else:
            self.log('No pre-trained model found — call train() to train.')