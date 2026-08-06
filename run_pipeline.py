from __future__ import annotations

from pathlib import Path

from mice_protein_pipeline.data_processing import create_group_datasets, load_dataset, split_features_target
from mice_protein_pipeline.models import evaluate_classifiers, train_random_forest, train_xgboost


def main() -> None:
    data_path = Path("Data_Cortex_Nuclear.csv")
    df = load_dataset(data_path)
    print("Dataset loaded and preprocessed.")

    groups = create_group_datasets(df)
    for name, group_df in groups.items():
        X, y = split_features_target(group_df)
        print(f"\n[{name}]")
        print(evaluate_classifiers(X, y).head())

        model = train_random_forest(X, y)
        preds = model.predict(X)
        print("RandomForest training accuracy:", round((preds == y).mean(), 4))

    full_features = [
        "DYRK1A_N", "ITSN1_N", "NR1_N", "NR2A_N", "pAKT_N", "pCAMKII_N", "pCREB_N",
        "pJNK_N", "pNR2A_N", "pRSK_N", "AKT_N", "BRAF_N", "CREB_N", "ERK_N", "MEK_N",
        "TRKA_N", "RSK_N", "APP_N", "Bcatenin_N", "AMPKA_N", "NR2B_N", "pNUMB_N",
        "TIAM1_N", "pP70S6_N", "pPKCG_N", "S6_N", "ADARB1_N", "AcetylH3K9_N",
        "RRP1_N", "ERBB4_N", "nNOS_N", "Tau_N", "GFAP_N", "GluR3_N", "GluR4_N",
        "IL1B_N", "SNCA_N", "Ubiquitin_N", "SHH_N", "pCFOS_N", "EGR1_N", "H3MeK4_N",
        "CaNA_N"
    ]
    selected_df = df[full_features + ["mice_class"]].copy()
    X_selected = selected_df.drop(columns=["mice_class"])
    y_selected = selected_df["mice_class"]
    print("\nSelected features dataset ready.")
    xgb_model = train_xgboost(X_selected, y_selected)
    print("XGBoost model trained:", type(xgb_model).__name__)


if __name__ == "__main__":
    main()
