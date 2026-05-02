import os
import pandas as pd

# --- Állítsd be az elérési útvonalat ---
DATA_DIR = r"data/CNC_Milling_Data"
train_path = os.path.join(DATA_DIR, "train.csv")

# --- Metaadatok betöltése ---
train_df = pd.read_csv(train_path)

all_data = []

# --- Végigmegyünk az összes experiment fájlon ---
for i in range(1, 19):
    exp_name = f"experiment_{i:02d}.csv"
    exp_path = os.path.join(DATA_DIR, exp_name)
    if not os.path.exists(exp_path):
        print(f"⚠️ {exp_name} nem található, kihagyva.")
        continue

    exp_df = pd.read_csv(exp_path)
    meta = train_df[train_df["No"] == i].iloc[0].to_dict()

    # Metaadatokat hozzáfűzzük minden sorhoz
    for key, val in meta.items():
        exp_df[key] = val

    all_data.append(exp_df)
    print(f"✅ Összefűzve: {exp_name}")

# --- Egyesítés és mentés ---
merged_df = pd.concat(all_data, ignore_index=True)
output_path = os.path.join(DATA_DIR, "merged_data.csv")
merged_df.to_csv(output_path, index=False)

print(f"\n💾 Mentve: {output_path}")
print(f"Sorok száma: {len(merged_df)}, oszlopok száma: {len(merged_df.columns)}")
print("Elérhető címkék:", merged_df["tool_condition"].unique())
