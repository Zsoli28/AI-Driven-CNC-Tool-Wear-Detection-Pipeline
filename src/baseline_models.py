# src/baseline_models.py

import pandas as pd
from features import get_data
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

# ÚJ IMPORTOK
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

def main():
    print("🚀 Adatok betöltése...")
    # 1. Adatok betöltése
    # Fontos: X_train és X_test mostantól Pandas DataFrame-ek!
    X_train, X_test, y_train, y_test, le = get_data()

    print("✅ Adatok betöltve.")
    print("-" * 30)

    # 2. Modellek definiálása
    lr_pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('model', LogisticRegression(max_iter=1000, random_state=42))
    ])

    models = {
        "Logistic Regression": lr_pipeline,
        "Random Forest": RandomForestClassifier(random_state=42),
    }
    
    # 3. Modellek tanítása és értékelése
    for name, model in models.items():
        print(f"=== {name} ===")
        
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        target_names = le.classes_
        print(classification_report(y_test, y_pred, target_names=target_names))

        # 📊 ÚJ VIZUALIZÁCIÓ: KONFÚZIÓS MÁTRIX
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=target_names, yticklabels=target_names)
        plt.title(f'Konfúziós Mátrix - {name}', fontsize=16)
        plt.ylabel('Valós Címke', fontsize=12)
        plt.xlabel('Jósolt Címke', fontsize=12)
        plt.tight_layout()
        
        # Fájlnév ékezetek és szóközök nélkül
        figname_cm = name.lower().replace(' ', '_') + '_confusion_matrix.png'
        plt.savefig(figname_cm)
        plt.close()
        print(f"📊 '{figname_cm}' mentve.")

        # 📊 ÚJ VIZUALIZÁCIÓ: JELLEMZŐ-FONTOSSÁG (Csak RF esetén)
        if name == "Random Forest":
            try:
                # A Pipeline miatt a 'model' maga a Pipeline objektum
                # De az RF-et nem tettük Pipeline-ba, így közvetlenül elérjük
                importances = model.feature_importances_
                feature_names = X_train.columns
                
                feat_imp = pd.Series(importances, index=feature_names).sort_values()
                
                plt.figure(figsize=(10, 8))
                feat_imp.plot(kind='barh')
                plt.title('Jellemző-Fontosság (Random Forest)', fontsize=16)
                plt.xlabel('Fontosság', fontsize=12)
                plt.tight_layout()
                plt.savefig('feature_importance.png')
                plt.close()
                print(f"📊 'feature_importance.png' mentve.")

            except Exception as e:
                print(f"Hiba a Jellemző-fontosság ábra készítésekor: {e}")

        print("-" * 30)


if __name__ == "__main__":
    main()