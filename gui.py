# gui.py
import sys
from pathlib import Path
import numpy as np
import tifffile
from scipy.signal import wiener as _wiener
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QListWidget, QFileDialog, QLabel, QSpinBox, QCheckBox
)
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.widgets import RectangleSelector
from matplotlib.patches import Rectangle
import matplotlib.pyplot as plt
import pandas as pd

# your processing module
import pipeline


# ---------------------------
# Utilities
# ---------------------------

def _to_uint8(img):
    """Scale to uint8 using robust percentiles."""
    arr = np.asarray(img)
    if arr.dtype == np.uint8:
        return arr
    lo, hi = np.percentile(arr, 1), np.percentile(arr, 99.5)
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        lo, hi = float(arr.min()), float(arr.max() if arr.max() > arr.min() else arr.min() + 1e-6)
    arr = np.clip((arr - lo) / (hi - lo), 0, 1)
    return (arr * 255).astype(np.uint8)


def _as_CHW(raw, n_ch_guess=32):
    """
    Normalize input stack to (C,H,W); also extract brightfield if present (n+1 channels).
    Accepts stacks like (H,W,C), (C,H,W), (T,Z,C,Y,X)... Squeezes and drops extra dims.
    """
    arr = np.asarray(raw)
    arr = np.squeeze(arr)

    # Prefer explicit count from pipeline.channel_lambdas if present
    try:
        n_ch_guess = len(getattr(pipeline, 'channel_lambdas'))
    except Exception:
        pass

    # Find plausible channel axis (equals n_ch_guess or n_ch_guess+1)
    ch_axis = None
    for ax, L in enumerate(arr.shape):
        if L == n_ch_guess or L == (n_ch_guess + 1):
            ch_axis = ax
            break
    if ch_axis is None:
        ch_axis = 0  # fallback

    # Split brightfield if n+1 channels, assume last along channel axis
    has_bf = (arr.shape[ch_axis] == n_ch_guess + 1)
    if has_bf:
        slc_ch = [slice(None)] * arr.ndim
        slc_bf = [slice(None)] * arr.ndim
        slc_ch[ch_axis] = slice(0, n_ch_guess)
        slc_bf[ch_axis] = slice(n_ch_guess, n_ch_guess + 1)
        chan = arr[tuple(slc_ch)]
        bf = np.squeeze(arr[tuple(slc_bf)])
    else:
        chan = arr
        bf = None

    # Move channel axis to front
    if ch_axis != 0:
        chan = np.moveaxis(chan, ch_axis, 0)

    # Drop extra dims (e.g., T/Z)
    while chan.ndim > 3:
        chan = chan[:, 0, ...]  # take first index along extra dims

    # If single plane (H,W), make it (1,H,W)
    if chan.ndim == 2:
        chan = chan[np.newaxis, ...]

    return chan.astype(np.float32, copy=False), (bf.astype(np.float32) if bf is not None else None)


def _wiener_per_channel(stack_CHW, ksize=(5, 5)):
    """Apply 2D Wiener per channel; returns CHW."""
    filt = [_wiener(img, ksize) for img in stack_CHW]
    return np.stack(filt, axis=0)


def _imsave_png(path, array_uint8_or_float01):
    """Save an image as PNG using matplotlib (no extra deps)."""
    arr = np.asarray(array_uint8_or_float01)
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 1)
    plt.imsave(str(path), arr)


# ---------------------------
# GUI
# ---------------------------

class PhasorGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Phasor Unmixing GUI")
        self.selected_files = []
        self.selector = None
        self.sel_rect_artist = None

        self.channels = None       # (C,H,W) float32
        self.rgb_img = None        # (H,W,3) float or uint8
        self.int_img = None        # (H,W)

        self.ax_select = None
        self.ax_phasor_sel = None
        self.ax_sum_sel = None

        self.current_outdir = None
        self.sel_idx = 0

        self._build_ui()
        self._load_defaults()

    def _build_ui(self):
        cw = QWidget()
        self.setCentralWidget(cw)
        main_layout = QHBoxLayout(cw)

        # Control panel
        ctrl = QVBoxLayout()
        main_layout.addLayout(ctrl, 1)

        btn_open = QPushButton("Select LSM/TIFF Files")
        btn_open.clicked.connect(self._on_select_files)
        ctrl.addWidget(btn_open)

        self.file_list = QListWidget()
        ctrl.addWidget(self.file_list)

        ctrl.addWidget(QLabel("Harmonic:"))
        self.spin_harm = QSpinBox()
        self.spin_harm.setRange(1, 4)
        ctrl.addWidget(self.spin_harm)

        self.chk_wiener = QCheckBox("Apply Wiener Filter")
        ctrl.addWidget(self.chk_wiener)

        ctrl.addWidget(QLabel("Display Options:"))
        self.chk_components = QCheckBox("Components")
        ctrl.addWidget(self.chk_components)
        self.chk_phasor = QCheckBox("Phasor Plot")
        ctrl.addWidget(self.chk_phasor)
        self.chk_spectral = QCheckBox("Spectral RGB")
        ctrl.addWidget(self.chk_spectral)
        self.chk_select = QCheckBox("Interactive Selection")
        ctrl.addWidget(self.chk_select)

        # Auto-save toggle
        self.chk_autosave = QCheckBox("Auto-save outputs")
        self.chk_autosave.setChecked(True)
        ctrl.addWidget(self.chk_autosave)

        btn_run = QPushButton("Run on Selected File")
        btn_run.clicked.connect(self._on_run)
        ctrl.addWidget(btn_run)
        ctrl.addStretch()

        # Display panel
        self.canvas = FigureCanvas(Figure())
        main_layout.addWidget(self.canvas, 3)

    def _load_defaults(self):
        self.spin_harm.setValue(getattr(pipeline, 'NTH_HARMONIC', 1))
        self.chk_wiener.setChecked(getattr(pipeline, 'WIENER', False))
        self.chk_components.setChecked(True)
        self.chk_phasor.setChecked(False)
        self.chk_spectral.setChecked(False)
        self.chk_select.setChecked(False)

    def _on_select_files(self):
        files, _ = QFileDialog.getOpenFileNames(
            self, "Select LSM/TIFF files", str(Path.home()),
            "Images (*.lsm *.tiff *.tif)"
        )
        if files:
            self.selected_files = files
            self.file_list.clear()
            for f in files:
                self.file_list.addItem(Path(f).name)

    def _prepare_outdir(self, filepath: Path) -> Path:
        outdir = filepath.parent / f"{filepath.stem}_outputs"
        outdir.mkdir(exist_ok=True, parents=True)
        return outdir

    def _save_overview(self, outdir: Path, tag="overview"):
        fig = self.canvas.figure
        fig.savefig(outdir / f"{tag}.png", dpi=200, bbox_inches='tight')

    def _save_components(self, outdir: Path, comps: np.ndarray):
        # Save stack as multi-page TIFF
        tifffile.imwrite(str(outdir / "components.tif"), comps.astype(np.float32), imagej=True)
        # Save per-component PNGs with dye names if available
        dye_list = getattr(pipeline, 'DYE_LIST', None)
        for j in range(comps.shape[0]):
            base = f"component_{j+1}"
            if isinstance(dye_list, (list, tuple)) and j < len(dye_list):
                base = dye_list[j]
            png = _to_uint8(comps[j])
            _imsave_png(outdir / f"{j+1:02d}_{base}.png", png)

    def _save_spectral_rgb(self, outdir: Path, rgb: np.ndarray, filename="spectral_rgb.png"):
        if rgb.dtype != np.uint8:
            norm = rgb / max(np.percentile(rgb, 99.5), 1e-6)
            norm = np.clip(norm, 0, 1)
            png = (norm * 255).astype(np.uint8)
        else:
            png = rgb
        _imsave_png(outdir / filename, png)

    def _save_phasor_plot(self, outdir: Path, g: np.ndarray, s: np.ndarray, tag="phasor"):
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.hist2d(g.ravel(), (-s).ravel(), bins=100, range=[[-1, 1], [-1, 1]], cmap='nipy_spectral')
        ax.add_patch(plt.Circle((0, 0), 1, fill=False, color='white'))
        ax.set_aspect('equal', adjustable='box')
        ax.set_xlim([-1, 1]); ax.set_ylim([-1, 1])
        ax.set_title('Phasor Plot')
        fig.savefig(outdir / f"{tag}.png", dpi=200, bbox_inches='tight')
        plt.close(fig)

    def _compute_rgb(self):
        """Compute spectral RGB with a robust default for lambdas; return None if unavailable."""
        try:
            lambdas = getattr(pipeline, 'channel_lambdas', np.linspace(405, 750, self.channels.shape[0]))
            if not hasattr(pipeline, 'SpectralStack2RGB'):
                return None
            rgb = pipeline.SpectralStack2RGB(self.channels, lambdas)
            return rgb
        except Exception:
            return None

    def _on_run(self):
        # Remove old selector & rectangle
        if self.selector:
            self.selector.disconnect_events()
            self.selector = None
        self.sel_rect_artist = None

        idx = self.file_list.currentRow()
        if idx < 0:
            return
        filepath = Path(self.selected_files[idx])

        # Update pipeline config
        pipeline.NTH_HARMONIC = self.spin_harm.value()
        pipeline.WIENER = self.chk_wiener.isChecked()

        # Load data (robust to dimension order)
        raw = tifffile.imread(str(filepath))
        self.channels, _brightfield = _as_CHW(raw, n_ch_guess=32)

        # Prepare intensity and filtered data
        self.int_img = np.sum(self.channels, axis=0)
        if self.chk_wiener.isChecked():
            try:
                filt = _wiener_per_channel(self.channels, ksize=(5, 5))
            except Exception as e:
                print(f"Wiener failed ({e}); falling back to unfiltered.")
                filt = self.channels
        else:
            filt = self.channels

        # Compute RGB regardless so selection uses RGB if possible
        self.rgb_img = self._compute_rgb()

        want_comps   = self.chk_components.isChecked()
        want_phasor  = self.chk_phasor.isChecked()
        want_spectral= self.chk_spectral.isChecked()
        want_select  = self.chk_select.isChecked()
        if not any([want_comps, want_phasor, want_spectral, want_select]):
            want_comps = True

        # Output dir (and selection counter reset)
        self.current_outdir = self._prepare_outdir(filepath) if self.chk_autosave.isChecked() else None
        self.sel_idx = 0

        # Compute components if needed
        comps = pipeline.process_file(filepath) if want_comps else None

        # Count primary and interactive plots
        n_primary = (comps.shape[0] if want_comps else 0) + (1 if want_phasor else 0) + (1 if want_spectral else 0)
        n_interactive = 3 if want_select else 0

        # Determine grid
        rows = 2 if want_select else 1
        cols = max(n_primary, n_interactive) if want_select else max(n_primary, 1)

        # Create subplots
        fig = self.canvas.figure
        fig.clf()
        axes = fig.subplots(rows, cols, squeeze=False)

        # Row 0: primary views
        ax_flat = axes[0]
        ax_idx = 0

        # Components
        if want_comps and comps is not None:
            for j in range(comps.shape[0]):
                if ax_idx >= cols:
                    break
                ax = ax_flat[ax_idx]
                v99 = np.percentile(comps[j], 99)
                ax.imshow(comps[j], vmax=v99, cmap='gray')
                title = f"Component {j+1}"
                if hasattr(pipeline, 'DYE_LIST') and j < len(pipeline.DYE_LIST):
                    title = pipeline.DYE_LIST[j]
                ax.set_title(title)
                ax.axis('off')
                ax_idx += 1

        # Phasor
        g = s = None
        if want_phasor and ax_idx < cols:
            ax = ax_flat[ax_idx]
            g, s = pipeline.phasor_transform(filt, n_harm=self.spin_harm.value(), axis=0)
            ax.hist2d(g.ravel(), (-s).ravel(), bins=100, range=[[-1, 1], [-1, 1]], cmap='nipy_spectral')
            ax.set_title('Phasor Plot')
            ax.axis('off')
            ax.add_patch(plt.Circle((0, 0), 1, fill=False, color='white'))
            ax.set_box_aspect(1)
            ax_idx += 1

        # Spectral
        if want_spectral and ax_idx < cols and self.rgb_img is not None:
            ax = ax_flat[ax_idx]
            show = self.rgb_img
            if show.dtype != np.uint8:
                show = np.clip(show / max(np.percentile(show, 99.5), 1e-6), 0, 1)
            ax.imshow(show)
            ax.set_title('Spectral RGB')
            ax.axis('off')
            ax_idx += 1

        # Hide unused primaries
        for k in range(ax_idx, cols):
            axes[0, k].set_visible(False)

        # Row 1: interactive selection
        if want_select:
            ax1, ax2, ax3 = axes[1]

            # Use RGB for selection if available; otherwise grayscale intensity
            if self.rgb_img is not None:
                show = self.rgb_img
                if show.dtype != np.uint8:
                    show = np.clip(show / max(np.percentile(show, 99.5), 1e-6), 0, 1)
                ax1.imshow(show)
                ax1.set_title('Select Region (RGB)')
            else:
                ax1.imshow(self.int_img, cmap='gray')
                ax1.set_title('Select Region (Intensity)')
            ax1.axis('off')

            ax2.set_title('Selected Phasor')
            ax2.axis('off')

            ax3.set_title('Intensity Sum')
            ax3.axis('off')

            # Create selector on selection axis
            self.ax_select = ax1
            self.ax_phasor_sel = ax2
            self.ax_sum_sel = ax3
            self.selector = RectangleSelector(
                ax1, self._on_select_region,
                useblit=True, interactive=True
            )

        fig.tight_layout()
        self.canvas.draw_idle()

        # === Auto-save outputs (non-interactive) ===
        if self.current_outdir is not None:
            # Save overview
            self._save_overview(self.current_outdir, tag="overview")

            # Save components
            if want_comps and comps is not None:
                self._save_components(self.current_outdir, comps)

            # Save spectral RGB if computed
            if (want_spectral or want_select) and self.rgb_img is not None:
                self._save_spectral_rgb(self.current_outdir, self.rgb_img, filename="spectral_rgb.png")

            # Save phasor
            if want_phasor and g is not None and s is not None:
                self._save_phasor_plot(self.current_outdir, g, s, tag="phasor")

            # Save intensity sum
            _imsave_png(self.current_outdir / "intensity_sum.png", _to_uint8(self.int_img))

            # Save raw multi-channel stack
            try:
                tifffile.imwrite(str(self.current_outdir / "raw_channels.tif"),
                                 self.channels.astype(np.float32), imagej=True)
            except Exception:
                pass

    def _on_select_region(self, eclick, erelease):
        if eclick.xdata is None or erelease.xdata is None:
            return  # clicked outside axes

        # Coordinates
        x1, x2 = sorted([int(eclick.xdata), int(erelease.xdata)])
        y1, y2 = sorted([int(eclick.ydata), int(erelease.ydata)])
        if (y2 - y1) <= 0 or (x2 - x1) <= 0:
            return

        sel = self.channels[:, y1:y2, x1:x2]

        # Draw/refresh rectangle overlay on selection axis
        if self.sel_rect_artist is not None:
            try:
                self.sel_rect_artist.remove()
            except Exception:
                pass
        self.sel_rect_artist = Rectangle(
            (x1, y1), x2 - x1, y2 - y1,
            fill=False, linewidth=2, edgecolor='yellow'
        )
        self.ax_select.add_artist(self.sel_rect_artist)

        # Phasor of selection
        g_sel, s_sel = pipeline.phasor_transform(sel, n_harm=self.spin_harm.value(), axis=0)

        # Intensity mask (above-mean within selection)
        sel_int = self.int_img[y1:y2, x1:x2]
        mask = sel_int > np.mean(sel_int)

        if mask.shape == g_sel.shape[1:]:
            g_vals = g_sel[:, mask].ravel()
            s_vals = -s_sel[:, mask].ravel()
        else:
            g_vals = g_sel.ravel()
            s_vals = (-s_sel).ravel()

        g_corr = pd.DataFrame(g_vals, columns=["G"])
        s_corr = pd.DataFrame(s_vals, columns=["S"])

        # Update phasor plot
        ax2 = self.ax_phasor_sel
        ax2.clear()
        ax2.hist2d(g_vals, s_vals, bins=100, range=[[-1, 1], [-1, 1]], cmap='nipy_spectral')
        ax2.add_patch(plt.Circle((0, 0), 1, fill=False, color='white'))
        ax2.set_box_aspect(1)
        ax2.set_xlim([-1, 1]); ax2.set_ylim([-1, 1])
        ax2.set_title('Selected Phasor')

        # Update sum plot
        ax3 = self.ax_sum_sel
        ax3.clear()
        sum_vals = sel.sum(axis=(1, 2))
        ax3.plot(sum_vals, marker='o')
        ax3.set_title('Intensity Sum')
        ax3.set_xlabel('Channel')
        ax3.set_ylabel('Sum (a.u.)')

        # === Save selection artifacts ===
        if self.current_outdir is not None:
            self.sel_idx += 1
            sel_tag = f"selection_{self.sel_idx:02d}"

            # CSVs
            pd.DataFrame(sum_vals, columns=["sum"]).to_csv(self.current_outdir / f"{sel_tag}_sum_vals.csv", index=False)
            g_corr.to_csv(self.current_outdir / f"{sel_tag}_G_coordinate.csv", index=False)
            s_corr.to_csv(self.current_outdir / f"{sel_tag}_S_coordinate.csv", index=False)

            # Selected phasor image
            self._save_phasor_plot(self.current_outdir, g_sel, s_sel, tag=f"{sel_tag}_phasor")

            # Selected sum plot
            fig_sum = plt.figure()
            axp = fig_sum.add_subplot(111)
            axp.plot(sum_vals, marker='o')
            axp.set_xlabel('Channel'); axp.set_ylabel('Sum (a.u.)')
            axp.set_title('Selected Region Sum per Channel')
            fig_sum.savefig(self.current_outdir / f"{sel_tag}_sum_plot.png", dpi=200, bbox_inches='tight')
            plt.close(fig_sum)

            # Save selected crop as multi-channel TIFF
            try:
                tifffile.imwrite(str(self.current_outdir / f"{sel_tag}_crop.tif"),
                                 sel.astype(np.float32), imagej=True)
            except Exception:
                pass

            # Save intensity crop
            _imsave_png(self.current_outdir / f"{sel_tag}_intensity.png", _to_uint8(sel.sum(axis=0)))

            # Save RGB annotated + crop if available
            if self.rgb_img is not None:
                # Full RGB with rectangle overlay
                fig_rgb = plt.figure()
                axr = fig_rgb.add_subplot(111)
                show = self.rgb_img
                if show.dtype != np.uint8:
                    show = np.clip(show / max(np.percentile(show, 99.5), 1e-6), 0, 1)
                axr.imshow(show)
                axr.add_patch(Rectangle((x1, y1), x2 - x1, y2 - y1,
                                        fill=False, linewidth=2, edgecolor='yellow'))
                axr.axis('off')
                fig_rgb.savefig(self.current_outdir / f"{sel_tag}_rgb_annotated.png", dpi=200, bbox_inches='tight')
                plt.close(fig_rgb)

                # RGB crop
                rgb_crop = show[y1:y2, x1:x2]
                _imsave_png(self.current_outdir / f"{sel_tag}_rgb_crop.png", (rgb_crop * 255).astype(np.uint8) if rgb_crop.dtype != np.uint8 else rgb_crop)

        self.canvas.draw_idle()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = PhasorGUI()
    win.show()
    sys.exit(app.exec_())
