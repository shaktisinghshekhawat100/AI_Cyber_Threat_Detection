import os
import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

RANDOM_STATE = 42
DATA_PATH = "data/cyber_threat_dataset.csv"
MODEL_PATH = "models/cyber_threat_model.joblib"

os.makedirs("data", exist_ok=True)
os.makedirs("models", exist_ok=True)

def generate_dataset(n_per_class=700, seed=RANDOM_STATE):
    rng = np.random.default_rng(seed)
    rows = []

    for _ in range(n_per_class):
        rows.append([
            rng.gamma(2.2, 1.8), rng.choice(["TCP","UDP"], p=[.75,.25]),
            int(rng.lognormal(7.0,.9)), int(rng.lognormal(7.2,.9)),
            int(rng.integers(5,180)), int(rng.integers(1024,65535)),
            int(rng.choice([53,80,443,22,8080])), int(rng.integers(0,2)),
            rng.uniform(.1,8), int(rng.integers(0,3)), int(rng.integers(100,9000)), "Normal"
        ])

    for _ in range(n_per_class):
        rows.append([
            rng.uniform(0,1.5), rng.choice(["TCP","UDP","ICMP"]),
            int(rng.integers(100,2500)), int(rng.integers(0,500)),
            int(rng.integers(150,2500)), int(rng.integers(1024,65535)),
            int(rng.choice([80,443,53])), int(rng.integers(0,2)),
            rng.uniform(80,500), int(rng.integers(0,4)), int(rng.integers(50,2000)), "DoS"
        ])

    for _ in range(n_per_class):
        rows.append([
            rng.uniform(0,3), rng.choice(["TCP","UDP","ICMP"]),
            int(rng.integers(40,1200)), int(rng.integers(0,700)),
            int(rng.integers(10,120)), int(rng.integers(1024,65535)),
            int(rng.integers(1,65535)), int(rng.integers(0,2)),
            rng.uniform(20,180), int(rng.integers(0,3)), int(rng.integers(20,1500)), "Port Scan"
        ])

    for _ in range(n_per_class):
        rows.append([
            rng.uniform(.2,8), "TCP",
            int(rng.integers(100,2500)), int(rng.integers(100,3000)),
            int(rng.integers(5,100)), int(rng.integers(1024,65535)),
            int(rng.choice([21,22,23,3389])), int(rng.integers(4,15)),
            rng.uniform(2,45), int(rng.integers(5,25)), int(rng.integers(100,5000)), "Brute Force"
        ])

    for _ in range(n_per_class):
        rows.append([
            rng.uniform(.1,10), "TCP",
            int(rng.integers(200,6000)), int(rng.integers(100,10000)),
            int(rng.integers(5,150)), int(rng.integers(1024,65535)),
            int(rng.choice([80,443,8080])), int(rng.integers(0,5)),
            rng.uniform(1,25), int(rng.integers(0,8)), int(rng.integers(1000,20000)), "Web Attack"
        ])

    columns = ["duration","protocol","src_bytes","dst_bytes","packets","src_port",
               "dst_port","failed_logins","connection_rate","login_attempts","payload_size","label"]
    return pd.DataFrame(rows, columns=columns).sample(frac=1, random_state=seed).reset_index(drop=True)

def main():
    df = generate_dataset()
    df.to_csv(DATA_PATH, index=False)

    X, y = df.drop(columns="label"), df["label"]
    categorical = ["protocol"]
    numeric = [c for c in X.columns if c not in categorical]

    pre = ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical),
        ("num", "passthrough", numeric)
    ])

    clf = RandomForestClassifier(
        n_estimators=250, max_depth=16, min_samples_leaf=2,
        class_weight="balanced", random_state=RANDOM_STATE, n_jobs=-1
    )
    pipe = Pipeline([("preprocessor", pre), ("classifier", clf)])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=.2, stratify=y, random_state=RANDOM_STATE
    )
    pipe.fit(X_train, y_train)
    pred = pipe.predict(X_test)
    print("Accuracy:", round(accuracy_score(y_test, pred), 4))
    print(classification_report(y_test, pred))
    joblib.dump(pipe, MODEL_PATH)
    print("Saved:", MODEL_PATH)
    print("Saved:", DATA_PATH)

if __name__ == "__main__":
    main()
