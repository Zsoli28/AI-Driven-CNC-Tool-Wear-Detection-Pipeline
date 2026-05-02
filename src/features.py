# src/features.py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns

def get_data(csv_path=r"data/CNC_Milling_Data/merged_data.csv",
             test_size=0.2,
             random_state=42,
             mode='snapshot',
             window_size=50):
    
    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        print(f"❌ HIBA: A '{csv_path}' fájl nem található.")
        return

    # 🧹 2. Szűrés
    df = df[(df["M1_CURRENT_FEEDRATE"] != 50) &
            (df["X1_ActualPosition"] != 198)]
    active_processes = ["Layer 1 Up", "Layer 1 Down", "Layer 2 Up", "Layer 2 Down", "Layer 3 Up", "Layer 3 Down"]
    df = df[df['Machining_Process'].isin(active_processes)].copy()
    
    # Címke kódolás
    sensor_cols = [c for c in df.columns if any(k in c for k in ["_OutputPower", "_OutputCurrent", "_ActualVelocity", "_ActualAcceleration"])]
    le = LabelEncoder()
    df_processed = df[sensor_cols + ['tool_condition', 'No']].copy()
    df_processed['tool_condition_encoded'] = le.fit_transform(df_processed['tool_condition'].astype(str))
    

    # Kísérletek listája
    experiments = df_processed['No'].unique()
    
    train_exps, test_exps = train_test_split(experiments, test_size=test_size, random_state=random_state)
    
    print(f"📊 Kísérlet-alapú szétválasztás (Leakage protection):")
    print(f"   Tanító kísérletek: {train_exps}")
    print(f"   Teszt kísérletek: {test_exps}")

    if mode == 'snapshot':
        # 2D MÓD
        train_df = df_processed[df_processed['No'].isin(train_exps)]
        test_df = df_processed[df_processed['No'].isin(test_exps)]
        
        X_train = train_df[sensor_cols]
        y_train = train_df['tool_condition_encoded']
        X_test = test_df[sensor_cols]
        y_test = test_df['tool_condition_encoded']
        
        return X_train, X_test, y_train, y_test, le

    elif mode == 'timeseries':
        # 3D MÓD - ABLAKOZÁS
        # Segédfüggvény az ablakozáshoz
        def create_windows(dataset_df):
            X_wins = []
            y_wins = []
            for exp_id in dataset_df['No'].unique():
                group = dataset_df[dataset_df['No'] == exp_id]
                feats = group[sensor_cols].values
                targets = group['tool_condition_encoded'].values
                
                # Ha a kísérlet rövidebb mint az ablak, kihagyjuk
                if len(group) < window_size:
                    continue
                    
                for i in range(len(group) - window_size + 1):
                    X_wins.append(feats[i : i + window_size])
                    y_wins.append(targets[i + window_size - 1])
            return np.array(X_wins), np.array(y_wins)

        # Külön generáljuk a Train és Test ablakokat a szétválasztott kísérletekből
        print("⏳ Ablakok generálása a TANÍTÓ kísérletekből...")
        train_df = df_processed[df_processed['No'].isin(train_exps)]
        X_train, y_train = create_windows(train_df)
        
        print("⏳ Ablakok generálása a TESZT kísérletekből...")
        test_df = df_processed[df_processed['No'].isin(test_exps)]
        X_test, y_test = create_windows(test_df)

        print(f"✅ Kész. X_train: {X_train.shape}, X_test: {X_test.shape}")
        
        return X_train, X_test, y_train, y_test, le
    
    else:
        raise ValueError(f"Ismeretlen mód: '{mode}'")