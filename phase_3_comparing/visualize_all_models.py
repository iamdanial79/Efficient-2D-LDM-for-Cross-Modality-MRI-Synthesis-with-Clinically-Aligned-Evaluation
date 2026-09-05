import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from scipy.stats import norm
import scipy.stats as stats
import pandas as pd
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
# ==========================================
# CLINICAL METRICS
# ==========================================
def compute_pixel_metrics(real_img, gen_img):
    """Compute MSE, MAE, PSNR, SSIM between real and generated images."""
    # Flatten arrays
    real_flat = real_img.flatten()
    gen_flat = gen_img.flatten()
    
    # MSE
    mse = np.mean((real_flat - gen_flat) ** 2)
    
    # MAE
    mae = np.mean(np.abs(real_flat - gen_flat))
    
    # PSNR (skimage expects images in [0,1] range)
    psnr = peak_signal_noise_ratio(real_img, gen_img, data_range=1.0)
    
    # SSIM
    ssim = structural_similarity(real_img, gen_img, data_range=1.0)
    
    return {'mse': mse, 'mae': mae, 'psnr': psnr, 'ssim': ssim}


def calculate_clinical_metrics(real_img, gen_img, mask_img):
    """Computes gSNR, gCNR, d' for a single slice, for both real and generated images."""
    fg_mask = mask_img > 0
    bg_mask = mask_img == 0

    if np.sum(fg_mask) == 0 or np.sum(bg_mask) == 0:
        return {"gSNR_real": np.nan, "gSNR_gen": np.nan,
                "gCNR_real": np.nan, "gCNR_gen": np.nan,
                "d_prime_real": np.nan, "d_prime_gen": np.nan}

    real_fg = real_img[fg_mask].flatten()
    real_bg = real_img[bg_mask].flatten()
    gen_fg = gen_img[fg_mask].flatten()
    gen_bg = gen_img[bg_mask].flatten()

    def compute_metrics(fg, bg):
        mu_fg, mu_bg = np.mean(fg), np.mean(bg)
        std_fg, std_bg = np.std(fg), np.std(bg)
        pooled_noise = np.sqrt(0.5 * (std_fg**2 + std_bg**2))
        if pooled_noise == 0:
            return np.nan, np.nan, np.nan
        signal_diff = np.abs(mu_fg - mu_bg)
        gsnr = signal_diff / pooled_noise
        d_prime = gsnr
        z_score = signal_diff / np.sqrt(std_fg**2 + std_bg**2)
        gcnr = 1 - 2 * norm.cdf(-z_score / 2)
        return gsnr, gcnr, d_prime

    gsnr_real, gcnr_real, d_prime_real = compute_metrics(real_fg, real_bg)
    gsnr_gen, gcnr_gen, d_prime_gen = compute_metrics(gen_fg, gen_bg)

    return {
        "gSNR_real": gsnr_real, "gSNR_gen": gsnr_gen,
        "gCNR_real": gcnr_real, "gCNR_gen": gcnr_gen,
        "d_prime_real": d_prime_real, "d_prime_gen": d_prime_gen
    }


def summarize(values_dict):
    """Turns {key: [values...]} into {key: {mean, std, ci_low, ci_high, n}}."""
    final = {}
    for key, values in values_dict.items():
        arr = np.array(values, dtype=float)
        valid = arr[~np.isnan(arr)]
        if len(valid) == 0:
            continue
        mean, std = np.mean(valid), np.std(valid)
        try:
            ci = stats.t.interval(0.95, len(valid) - 1, loc=mean, scale=stats.sem(valid))
            final[key] = {'mean': mean, 'std': std, 'ci_low': ci[0], 'ci_high': ci[1], 'n': len(valid)}
        except Exception:
            final[key] = {'mean': mean, 'std': std, 'n': len(valid)}
    return final


# ==========================================
# MAIN VISUALIZER / AGGREGATOR
# ==========================================
class FullComparisonVisualizer:
    def __init__(self, save_plots=True, max_plots=20):
        self.models = {
            'LDM_scaled_final': Path("evaluation_results_diffusion_best_claude2/generated_samples"),
            'LDM_scaled_basic': Path("evaluation_results_scaled_basic/generated_samples"),
            'LDM_basic': Path("evaluation_results_Linear/generated_samples"),
            'GAN (Pix2Pix)': Path("pix2pix_baseline"),
            'Transformer': Path("transformer_full_val_eval"),
        }
        self.output_dir = Path("final_visualizations_7")
        self.output_dir.mkdir(exist_ok=True)
        self.save_plots = save_plots
        self.max_plots = max_plots  # cap how many per-patient PNGs we save (full val set can be huge)

    def load_data(self, patient_name):
        data = {}
        for name, path in self.models.items():
            pt_file = path / f"{patient_name}.pt"
            if pt_file.exists():
                data[name] = torch.load(pt_file, weights_only=False)
        return data

    def plot_comparison(self, patient_name, data):
        """Builds the qualitative comparison figure. Returns per-model clinical metrics dict.

        FIXED: pixel metrics (mse/mae/psnr/ssim) are now guaranteed to be
        computed here too, not just in the "skip plotting" branch of
        generate_all(). Previously, a model's .pt file without
        precomputed 'metrics' (e.g. the GAN/Transformer baselines) would
        silently get NO pixel metrics for any slice that happened to be
        one of the first `max_plots` processed, since compute_pixel_metrics
        was only ever called in the else-branch of generate_all(). That
        produced a smaller, non-random n for those models' PSNR/SSIM/MSE/MAE
        averages (e.g. 196 vs 216) without any warning.
        """
        model_names = list(data.keys())
        n_models = len(model_names)

        gt = data[model_names[0]]['real_flair']
        t1ce = data[model_names[0]]['real_t1ce']
        mask = data[model_names[0]]['mask']

        clinical_per_model = {}
        for name in model_names:
            # ADDED: always ensure pixel metrics exist, plotted or not.
            if 'metrics' not in data[name]:
                data[name]['metrics'] = compute_pixel_metrics(
                    gt, data[name]['synthetic_flair']
                )

            clinical_per_model[name] = calculate_clinical_metrics(
                gt, data[name]['synthetic_flair'], mask
            )

        if not self.save_plots:
            return clinical_per_model

        fig = plt.figure(figsize=(5 * n_models + 4, 16))
        gs_main = GridSpec(3, n_models + 1, figure=fig, width_ratios=[1] * n_models + [1.2],
                            hspace=0.35, wspace=0.15, height_ratios=[1, 1, 1.2])

        # Row 1: generated outputs
        for i, name in enumerate(model_names):
            ax = fig.add_subplot(gs_main[0, i])
            ax.imshow(data[name]['synthetic_flair'], cmap='gray', vmin=0, vmax=1)
            ax.set_title(f'{name}\n(Output)', fontsize=11, fontweight='bold', color='blue')
            ax.axis('off')

        # Row 2: error maps
        for i, name in enumerate(model_names):
            ax = fig.add_subplot(gs_main[1, i])
            error_map = np.abs(gt - data[name]['synthetic_flair'])
            ax.imshow(error_map, cmap='hot', vmin=0, vmax=0.3)
            ax.set_title('|GT - Output|\n(Error Map)', fontsize=11, color='red')
            ax.axis('off')

        # Last column: GT and T1ce reference
        ax_gt = fig.add_subplot(gs_main[0, n_models])
        ax_gt.imshow(gt, cmap='gray', vmin=0, vmax=1)
        ax_gt.set_title('Ground Truth\n(Real FLAIR)', fontsize=11, fontweight='bold', color='green')
        ax_gt.axis('off')

        ax_input = fig.add_subplot(gs_main[1, n_models])
        ax_input.imshow(t1ce, cmap='gray', vmin=0, vmax=1)
        ax_input.set_title('Input\n(T1ce)', fontsize=11)
        ax_input.axis('off')

        # Row 3: clinical analysis
        gs_bottom = GridSpecFromSubplotSpec(1, 3, subplot_spec=gs_main[2, :], wspace=0.3)

        # Histogram
        ax_hist = fig.add_subplot(gs_bottom[0, 0])
        ax_hist.hist(gt.flatten(), bins=50, alpha=0.6, label='GT', color='green')
        for name in model_names:
            ax_hist.hist(data[name]['synthetic_flair'].flatten(), bins=50, alpha=0.3, label=name)
        ax_hist.legend(fontsize=8)
        ax_hist.set_title("Intensity Distribution", fontweight='bold')
        ax_hist.set_xlim(0, 1)

        # Tumor mask
        ax_mask = fig.add_subplot(gs_bottom[0, 1])
        mask_display = np.zeros((*mask.shape, 3))
        mask_display[:, :, 0] = (mask > 0) & (mask != 2)
        mask_display[:, :, 1] = (mask == 2)
        ax_mask.imshow(mask_display)
        ax_mask.set_title("Tumor Mask\n(Red: TC, Green: ED)", fontsize=10)
        ax_mask.axis('off')

        # Tumor vs background error bars
        ax_clinical = fig.add_subplot(gs_bottom[0, 2])
        error_tumor, error_bg = [], []
        for name in model_names:
            err_map = np.abs(gt - data[name]['synthetic_flair'])
            tumor_pixels = err_map[mask > 0]
            bg_pixels = err_map[mask == 0]
            error_tumor.append(np.mean(tumor_pixels) if len(tumor_pixels) > 0 else 0)
            error_bg.append(np.mean(bg_pixels) if len(bg_pixels) > 0 else 0)

        x = np.arange(n_models)
        width = 0.35
        bars1 = ax_clinical.bar(x - width / 2, error_tumor, width, label='Error inside Tumor', color='red', alpha=0.7)
        bars2 = ax_clinical.bar(x + width / 2, error_bg, width, label='Error in Background', color='blue', alpha=0.7)
        ax_clinical.set_xticks(x)
        ax_clinical.set_xticklabels(model_names, rotation=15, ha='right', fontsize=9)
        ax_clinical.set_ylabel('Mean Absolute Error')
        ax_clinical.set_title('Error: Tumor vs Background', fontweight='bold')
        ax_clinical.legend(fontsize=8)
        for bar in bars1:
            ax_clinical.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f'{bar.get_height():.3f}',
                              ha='center', va='bottom', fontsize=7)
        for bar in bars2:
            ax_clinical.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f'{bar.get_height():.3f}',
                              ha='center', va='bottom', fontsize=7)

        # Clinical metrics text box (gCNR / gSNR / d')
        clinical_text = "gCNR / gSNR / d' (generated):   "
        clinical_text += "   |   ".join(
            f"{name}: gCNR={clinical_per_model[name]['gCNR_gen']:.3f}, gSNR={clinical_per_model[name]['gSNR_gen']:.3f}"
            for name in model_names
        )
        fig.text(0.5, 0.005, clinical_text, ha='center', fontsize=9, family='monospace')

        plt.suptitle(f'Patient/Slice: {patient_name}', fontsize=14, fontweight='bold', y=0.98)
        plt.savefig(self.output_dir / f"{patient_name}_full_analysis.png", dpi=200, bbox_inches='tight', facecolor='white')
        plt.close()

        return clinical_per_model

    def generate_all(self):
        base_dir = None
        for name, path in self.models.items():
            if path.exists() and list(path.glob("*.pt")):
                base_dir = path
                print(f"Found available results in: {path.name}")
                break

        if not base_dir:
            print("Error: No output folders with .pt files found!")
            return

        pt_files = sorted(base_dir.glob("*.pt"))
        first_file = pt_files[0].stem
        available_models = list(self.load_data(first_file).keys())
        print(f"Found {len(pt_files)} patient/slice files")
        print(f"Models available: {available_models}\n")

        # ADDED: per-model file-count report. base_dir's file list drives
        # everything downstream; if a model's folder is simply missing a
        # .pt file for a given patient_name, that slice silently doesn't
        # contribute to that model's stats with no warning anywhere else
        # in this script. This makes any *genuine* missing-file gap
        # visible instead of looking identical to a metrics bug.
        print("=" * 70)
        print("PER-MODEL FILE AVAILABILITY (relative to base_dir file list)")
        print("=" * 70)
        expected_names = {p.stem for p in pt_files}
        for name, path in self.models.items():
            if not path.exists():
                print(f"  {name:20s}: folder not found ({path})")
                continue
            found_names = {p.stem for p in path.glob("*.pt")}
            matched = expected_names & found_names
            missing = expected_names - found_names
            status = "OK" if len(missing) == 0 else f"MISSING {len(missing)}"
            print(f"  {name:20s}: {len(matched)}/{len(expected_names)} files  [{status}]")
            if missing and len(missing) <= 10:
                print(f"    missing: {sorted(missing)}")
            elif missing:
                print(f"    missing: {sorted(missing)[:10]} ... (+{len(missing) - 10} more)")
        print("=" * 70 + "\n")

        # Aggregation storage: per model, per metric, list of values across ALL slices
        aggregated = {name: {
            "gCNR_gen": [], "gSNR_gen": [], "d_prime_gen": [],
            "gCNR_real": [], "gSNR_real": [], "d_prime_real": [],
            "mse": [], "mae": [], "psnr": [], "ssim": []
        } for name in available_models}

        plots_saved = 0

        for pt_file in pt_files:
            patient_name = pt_file.stem
            data = self.load_data(patient_name)
            if len(data) < 1:
                continue

            # Only render a PNG for a limited subset (full val set can be hundreds/thousands of slices)
            should_plot = self.save_plots and plots_saved < self.max_plots
            if should_plot:
                clinical = self.plot_comparison(patient_name, data)
                plots_saved += 1
            else:
                # FIXED: compute pixel metrics for every model here too
                # (previously this was the ONLY branch that computed them,
                # so slices routed to plot_comparison() instead never got
                # pixel metrics unless they were already precomputed in
                # the .pt file -- see plot_comparison() docstring above).
                gt = data[list(data.keys())[0]]['real_flair']
                for name in data.keys():
                    gen_img = data[name]['synthetic_flair']

                    # Compute metrics if not already stored
                    if 'metrics' not in data[name]:
                        data[name]['metrics'] = compute_pixel_metrics(gt, gen_img)
                mask = data[list(data.keys())[0]]['mask']
                clinical = {name: calculate_clinical_metrics(gt, data[name]['synthetic_flair'], mask)
                            for name in data}

            for name, c in clinical.items():
                if name not in aggregated:
                    continue
                for k in ["gCNR_gen", "gSNR_gen", "d_prime_gen", "gCNR_real", "gSNR_real", "d_prime_real"]:
                    if not np.isnan(c[k]):
                        aggregated[name][k].append(c[k])

                # Also grab pre-computed pixel metrics if the .pt file stored them (LDM does; baselines may not)
                if 'metrics' in data[name]:
                    for k in ["mse", "mae", "psnr", "ssim"]:
                        if k in data[name]['metrics']:
                            aggregated[name][k].append(data[name]['metrics'][k])

        # ---- Print + save summary table ----
        print("\n" + "=" * 70)
        print(f"CLINICAL METRICS SUMMARY (across {len(pt_files)} slices)")
        print("=" * 70)

        rows = []
        for name, vals in aggregated.items():
            summary = summarize(vals)
            print(f"\n{name}:")
            row = {"model": name}
            for k, s in summary.items():
                print(f"  {k}: {s['mean']:.4f} ± {s['std']:.4f}  (n={s['n']})")
                row[f"{k}_mean"] = s['mean']
                row[f"{k}_std"] = s['std']
                row[f"{k}_n"] = s['n']
            rows.append(row)

        df = pd.DataFrame(rows)
        df.to_csv(self.output_dir / "clinical_metrics_summary.csv", index=False)
        print(f"\nSaved summary table -> {self.output_dir / 'clinical_metrics_summary.csv'}")
        print(f"Saved {plots_saved} example comparison figures -> {self.output_dir}")


if __name__ == "__main__":
    # save_plots=True renders PNGs for the first `max_plots` slices (visual inspection);
    # all slices still get aggregated into the numeric summary regardless.
    viz = FullComparisonVisualizer(save_plots=True, max_plots=20)
    viz.generate_all()
    print("\nDone!")