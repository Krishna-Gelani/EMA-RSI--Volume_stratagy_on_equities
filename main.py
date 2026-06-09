# =========================================================
# SETTINGS
# =========================================================

TRAIN_PATH = r"bajajfin_equity_1hour_start_2022.csv"
VAL_PATH   = r"bajajfin_equity_1hour_start_2023.csv"
TEST_PATH  = r"bajajfin_ohlcv_1hour_start_2024.csv"

THRESHOLD = 0.70  # 🔥 Higher precision
OUTPUT_PATH = "trades_output.csv"

# =========================================================
# IMPORTS
# =========================================================

import numpy as np
import pandas as pd
import lightgbm as lgb

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, roc_auc_score
)

# =========================================================
# LOAD DATA
# =========================================================

train_df = pd.read_csv(TRAIN_PATH)
val_df   = pd.read_csv(VAL_PATH)
test_df  = pd.read_csv(TEST_PATH)

# =========================================================
# FEATURE ENGINEERING
# =========================================================

def add_features(df):

    df["EMA20"] = df["CLOSE"].ewm(span=20).mean()
    df["EMA50"] = df["CLOSE"].ewm(span=50).mean()

    df["compression"] = abs(df["EMA20"] - df["EMA50"]) / df["CLOSE"]

    df["range"] = df["HIGH"] - df["LOW"]
    df["range_ma"] = df["range"].rolling(10).mean()

    df["volatility"] = df["CLOSE"].pct_change().rolling(10).std()

    df["trend"] = df["EMA20"] - df["EMA50"]

    df["slope20"] = df["EMA20"].diff()
    df["slope50"] = df["EMA50"].diff()

    df["dist_ema20"] = (df["CLOSE"] - df["EMA20"]) / df["EMA20"]
    df["dist_ema50"] = (df["CLOSE"] - df["EMA50"]) / df["EMA50"]

    return df

train_df = add_features(train_df)
val_df   = add_features(val_df)
test_df  = add_features(test_df)

# =========================================================
# TARGET (VOLATILITY ADAPTIVE)
# =========================================================

def add_target(df):

    future = df["CLOSE"].shift(-5)
    ret = (future - df["CLOSE"]) / df["CLOSE"]

    vol = df["volatility"]

    df["Target"] = np.where(
        ret > 1.2 * vol,
        1, 0
    )

    return df

train_df = add_target(train_df)
val_df   = add_target(val_df)
test_df  = add_target(test_df)

# =========================================================
# FEATURES
# =========================================================

features = [
    "CLOSE","EMA20","EMA50","compression","range","range_ma",
    "volatility","trend","slope20","slope50",
    "dist_ema20","dist_ema50","VOLUME"
]

# =========================================================
# CLEAN DATA
# =========================================================

train_df = train_df.dropna()
val_df   = val_df.dropna()
test_df  = test_df.dropna()

# =========================================================
# DATA PREPARATION
# =========================================================

X_train = train_df[features]
y_train = train_df["Target"]

X_val = val_df[features]
y_val = val_df["Target"]

X_test = test_df[features]
y_test = test_df["Target"]

# =========================================================
# MODEL (FIXED)
# =========================================================

model = lgb.LGBMClassifier(
    n_estimators=500,
    learning_rate=0.03,
    max_depth=7,
    num_leaves=50,
    subsample=0.8,
    colsample_bytree=0.8,
    class_weight="balanced",
    verbosity=-1
)

model.fit(
    X_train, y_train,
    eval_set=[(X_val, y_val)],
    eval_metric="auc"
)

# =========================================================
# PREDICTIONS
# =========================================================

pred_prob = model.predict_proba(X_test)[:, 1]
pred = (pred_prob > THRESHOLD).astype(int)

# =========================================================
# MODEL PERFORMANCE
# =========================================================

accuracy = accuracy_score(y_test, pred)
precision = precision_score(y_test, pred)
recall = recall_score(y_test, pred)
f1 = f1_score(y_test, pred)
roc_auc = roc_auc_score(y_test, pred_prob)

cm = confusion_matrix(y_test, pred)
tn, fp, fn, tp = cm.ravel()

print("\n================ MODEL PERFORMANCE ================\n")

print(f"Accuracy        : {accuracy:.4f}")
print(f"Precision       : {precision:.4f}")
print(f"Recall          : {recall:.4f}")
print(f"F1 Score        : {f1:.4f}")
print(f"ROC-AUC Score   : {roc_auc:.4f}")

print("\nConfusion Matrix:")
print(cm)

print("\nInterpretation:")
print(f"True Positives (Correct Buy)   : {tp}")
print(f"False Positives (Bad Buy)      : {fp}")
print(f"False Negatives (Missed Trade) : {fn}")
print(f"True Negatives (Correct Skip)  : {tn}")

# =========================================================
# SIGNAL GENERATION (FINAL BEST VERSION)
# =========================================================

test_df = test_df.reset_index(drop=True)
test_df["prob"] = pred_prob

# Volatility filter
test_df["vol_ma"] = test_df["volatility"].rolling(20).mean()

test_df["Buy_Signal"] = np.where(
    (test_df["prob"] > THRESHOLD) &
    (test_df["EMA20"] > test_df["EMA50"]) &   # trend confirmation
    (test_df["volatility"] > test_df["vol_ma"]),  # volatility filter
    1, 0
)

# =========================================================
# DETAILED BACKTEST
# =========================================================

trades = []

for i in range(len(test_df) - 6):

    if test_df.loc[i, "Buy_Signal"] != 1:
        continue

    signal_row = test_df.loc[i]

    # -----------------------------------
    # ENTRY
    # -----------------------------------
    entry_idx = i + 1
    entry_row = test_df.loc[entry_idx]

    entry_price = entry_row["CLOSE"]

    # -----------------------------------
    # EXIT LOGIC
    # Best price within next 5 candles
    # -----------------------------------

    future = test_df.loc[entry_idx:entry_idx + 4]

    exit_idx = future["CLOSE"].idxmax()
    exit_row = test_df.loc[exit_idx]

    exit_price = exit_row["CLOSE"]

    pnl = exit_price - entry_price
    ret = pnl / entry_price * 100

    # -----------------------------------
    # MFE / MAE
    # -----------------------------------

    max_high = future["HIGH"].max()
    min_low = future["LOW"].min()

    mfe = max_high - entry_price
    mae = min_low - entry_price

    # -----------------------------------
    # HOLDING PERIOD
    # -----------------------------------

    holding_candles = exit_idx - entry_idx + 1

    trades.append({

        # =============================
        # SIGNAL CANDLE
        # =============================

        "Signal DateTime":
            signal_row["DATETIME"],

        "Signal Open":
            signal_row["OPEN"],

        "Signal High":
            signal_row["HIGH"],

        "Signal Low":
            signal_row["LOW"],

        "Signal Close":
            signal_row["CLOSE"],

        "Signal Volume":
            signal_row["VOLUME"],

        "Model Probability":
            signal_row["prob"],

        # =============================
        # ENTRY
        # =============================

        "Entry DateTime":
            entry_row["DATETIME"],

        "Entry Open":
            entry_row["OPEN"],

        "Entry High":
            entry_row["HIGH"],

        "Entry Low":
            entry_row["LOW"],

        "Entry Close":
            entry_row["CLOSE"],

        "Entry Volume":
            entry_row["VOLUME"],

        "Entry Price":
            entry_price,

        # =============================
        # EXIT
        # =============================

        "Exit DateTime":
            exit_row["DATETIME"],

        "Exit Open":
            exit_row["OPEN"],

        "Exit High":
            exit_row["HIGH"],

        "Exit Low":
            exit_row["LOW"],

        "Exit Close":
            exit_row["CLOSE"],

        "Exit Volume":
            exit_row["VOLUME"],

        "Exit Price":
            exit_price,

        # =============================
        # TRADE STATS
        # =============================

        "Holding Candles":
            holding_candles,

        "PnL":
            pnl,

        "Return %":
            ret,

        "MFE":
            mfe,

        "MAE":
            mae,

        "Exit Reason":
            "Highest Close Next 5 Bars"
    })

trades_df = pd.DataFrame(trades)
# =========================================================
# SAVE EXCEL
# =========================================================

OUTPUT_PATH = "Detailed_Trade_Log.xlsx"

summary_df = pd.DataFrame({
    "Metric": [
        "Total Trades",
        "Winning Trades",
        "Losing Trades",
        "Win Rate %",
        "Total PnL"
    ],
    "Value": [
        len(trades_df),
        len(trades_df[trades_df["PnL"] > 0]),
        len(trades_df[trades_df["PnL"] <= 0]),
        round(
            len(trades_df[trades_df["PnL"] > 0])
            / max(len(trades_df), 1) * 100,
            2
        ),
        round(trades_df["PnL"].sum(), 2)
    ]
})

feature_importance = pd.DataFrame({
    "Feature": features,
    "Importance": model.feature_importances_
}).sort_values(
    by="Importance",
    ascending=False
)

trades_df.to_csv(
    "Detailed_Trade_Log.csv",
    index=False
)
print(f"\nSaved Excel Report: {OUTPUT_PATH}")
