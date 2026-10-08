import os
import csv
import matplotlib.pyplot as plt

def generate_summary_plots(patient_dir, output_png):
    dice_file = os.path.join(patient_dir, "DICE_scores.txt")
    nta_file = os.path.join(patient_dir, "nta.txt")
    vol_file = os.path.join(patient_dir, "TumorVolumes.txt")

    times_dice, dice_vals = [], []
    if os.path.exists(dice_file):
        with open(dice_file, 'r') as f:
            reader = csv.reader(f)
            next(reader, None)
            for row in reader:
                if row:
                    times_dice.append(float(row[0]))
                    dice_vals.append(float(row[1]))

    times_nta, nta_model, nta_mri = [], [], []
    if os.path.exists(nta_file):
        with open(nta_file, 'r') as f:
            reader = csv.reader(f)
            next(reader, None)
            for row in reader:
                if row:
                    times_nta.append(float(row[0]))
                    nta_model.append(float(row[1]))
                    nta_mri.append(float(row[2]))

    times_vol, vols = [], []
    if os.path.exists(vol_file):
        with open(vol_file, 'r') as f:
            reader = csv.reader(f)
            next(reader, None)
            for row in reader:
                if row:
                    times_vol.append(float(row[0]))
                    vols.append(float(row[1]))

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    # DICE plot
    if times_dice:
        axes[0].plot(times_dice[:-1], dice_vals[:-1], 'o-', color='#1f77b4', label='Calibration Days')
        axes[0].plot(times_dice[-1:], dice_vals[-1:], 's', color='#d62728', markersize=9, label='Forecast Day (Day 9)')
        axes[0].axhline(y=dice_vals[-1], color='#d62728', linestyle='--', alpha=0.5)
        axes[0].set_title(f'DICE Score over Time\n(Forecast Day 9 DICE: {dice_vals[-1]:.4f})', fontsize=11, fontweight='bold')
        axes[0].set_xlabel('Scan Day (Time)')
        axes[0].set_ylabel('DICE Similarity Index')
        axes[0].set_ylim(0.7, 1.05)
        axes[0].grid(True, linestyle=':', alpha=0.6)
        axes[0].legend()

    # NTA plot
    if times_nta:
        axes[1].plot(times_nta, nta_model, 'o-', color='#2ca02c', label='Model Predicted NTA')
        axes[1].plot(times_nta, nta_mri, '^--', color='#ff7f0e', label='True MRI NTA')
        axes[1].set_title('Normalized Tumor Area (NTA)', fontsize=11, fontweight='bold')
        axes[1].set_xlabel('Scan Day (Time)')
        axes[1].set_ylabel('Normalized Tumor Area')
        axes[1].grid(True, linestyle=':', alpha=0.6)
        axes[1].legend()

    # Volume plot
    if times_vol:
        axes[2].plot(times_vol, vols, 'd-', color='#9467bd')
        axes[2].set_title('Estimated Tumor Volume over Time', fontsize=11, fontweight='bold')
        axes[2].set_xlabel('Scan Day (Time)')
        axes[2].set_ylabel('Volume')
        axes[2].grid(True, linestyle=':', alpha=0.6)

    plt.tight_layout()
    plt.savefig(output_png, dpi=200)
    plt.close()
    print(f"Summary plot saved to {output_png}")

if __name__ == "__main__":
    generate_summary_plots("BayesianInference/results/Patient_001", "BayesianInference/results/Patient_001/metrics_summary.png")
