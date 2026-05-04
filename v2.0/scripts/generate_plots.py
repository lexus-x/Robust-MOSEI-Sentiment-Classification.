import json
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from pathlib import Path

def generate_plots():
    summary_path = Path("v2.0/outputs/thesis_pack/comparison_summary.json")
    out_dir = Path("v2.0/outputs/thesis_pack/figures")
    out_dir.mkdir(parents=True, exist_ok=True)
    
    with open(summary_path) as f:
        data = json.load(f)
        
    # 1. Robustness Gap Plot
    families = []
    gaps = []
    for row in data.get("family_summary", []):
        families.append(row["family"].replace("_", " ").title())
        gaps.append(row["weighted_f1_gap_vs_baseline"])
        
    plt.figure(figsize=(10, 6))
    sns.barplot(x=gaps, y=families, palette="viridis")
    plt.axvline(0.02, color='red', linestyle='--', label='Mean Tolerance (0.02)')
    plt.axvline(0.03, color='darkred', linestyle=':', label='Worst-Case Tolerance (0.03)')
    plt.title("Weighted-F1 Gap (Baseline vs EIDMSA) by Corruption Family")
    plt.xlabel("F1 Gap (Lower is better)")
    plt.ylabel("Corruption Family")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "robustness_gap.png", dpi=300)
    plt.close()
    
    # 2. Calibration (ECE) Comparison
    ece_data = []
    for row in data.get("family_summary", []):
        fam = row["family"].replace("_", " ").title()
        ece_data.append({"Family": fam, "Model": "Transformer (Baseline)", "ECE": row["baseline_ece"]})
        ece_data.append({"Family": fam, "Model": "EIDMSA (Proposed)", "ECE": row["compact_ece"]})
    
    df_ece = pd.DataFrame(ece_data)
    plt.figure(figsize=(10, 6))
    sns.barplot(data=df_ece, x="ECE", y="Family", hue="Model", palette="coolwarm")
    plt.title("Expected Calibration Error (ECE) Comparison")
    plt.xlabel("ECE (Lower is better)")
    plt.ylabel("Corruption Family")
    plt.legend(title="Model")
    plt.tight_layout()
    plt.savefig(out_dir / "calibration_ece.png", dpi=300)
    plt.close()

    # 3. Abstention Mechanics
    # Extract mean accuracy vs abstention accuracy for EIDMSA
    records = data.get("condition_comparison", [])
    acc = []
    abs_acc = []
    for r in records:
        if r["role_compact"] == "compact" and pd.notnull(r.get("abstention_accuracy_compact")):
            acc.append(r["accuracy_compact"])
            abs_acc.append(r["abstention_accuracy_compact"])
            
    if acc:
        mean_acc = sum(acc)/len(acc)
        mean_abs_acc = sum(abs_acc)/len(abs_acc)
        plt.figure(figsize=(6, 5))
        sns.barplot(x=["Forced Prediction", "With 20% Abstention"], y=[mean_acc, mean_abs_acc], palette="Set2")
        plt.title("EIDMSA Accuracy: Forced vs Abstention")
        plt.ylabel("Accuracy")
        plt.ylim(0.5, 0.7)
        plt.tight_layout()
        plt.savefig(out_dir / "abstention_accuracy.png", dpi=300)
        plt.close()

if __name__ == "__main__":
    generate_plots()
    print("Plots generated in v2.0/outputs/thesis_pack/figures/")
