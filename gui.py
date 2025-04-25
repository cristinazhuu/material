
# gui.py
import sys
from pathlib import Path
import numpy as np
import tifffile
from scipy.signal import wiener
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QListWidget, QFileDialog, QLabel, QSpinBox, QCheckBox
)
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.widgets import RectangleSelector
import matplotlib.pyplot as plt
import pandas as pd
# your processing module
import pipeline

class PhasorGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Phasor Unmixing GUI")
        self.selected_files = []
        self.selector = None
        self.channels = None
        self.int_img = None
        self.ax_select = self.ax_phasor_sel = self.ax_sum_sel = None
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

    def _on_run(self):
        # Remove old selector if any
        if self.selector:
            self.selector.disconnect_events()
            self.selector = None

        idx = self.file_list.currentRow()
        if idx < 0:
            return
        filepath = Path(self.selected_files[idx])

        # Update pipeline config
        pipeline.NTH_HARMONIC = self.spin_harm.value()
        pipeline.WIENER = self.chk_wiener.isChecked()

        # Load data
        raw = tifffile.imread(str(filepath))
        if raw.shape[0] == 33:
            self.channels = raw[:32]
            brightfield = raw[-1]
        else:
            self.channels = raw
            brightfield = np.zeros(raw.shape[1:])

        # Prepare intensity and filtered data
        self.int_img = np.sum(self.channels, axis=0)
        filt = wiener(self.channels, (1,5,5)) if self.chk_wiener.isChecked() else self.channels

        want_comps   = self.chk_components.isChecked()
        want_phasor  = self.chk_phasor.isChecked()
        want_spectral= self.chk_spectral.isChecked()
        want_select  = self.chk_select.isChecked()
        if not any([want_comps, want_phasor, want_spectral, want_select]):
            want_comps = True

        # Compute components if needed
        comps = pipeline.process_file(filepath) if want_comps else None

        # Count primary plots and interactive plots
        n_primary = (comps.shape[0] if want_comps else 0) + (1 if want_phasor else 0) + (1 if want_spectral else 0)
        n_interactive = 3 if want_select else 0

        # Determine grid: two rows if interactive else one
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
        if want_comps:
            for j in range(comps.shape[0]):
                if ax_idx >= cols: break
                ax = ax_flat[ax_idx]
                v99 = np.percentile(comps[j], 99)
                ax.imshow(comps[j], vmax=v99, cmap='gray')
                ax.set_title(pipeline.DYE_LIST[j])
                ax.axis('off')
                ax_idx += 1
        # Phasor
        if want_phasor and ax_idx < cols:
            ax = ax_flat[ax_idx]
            g, s = pipeline.phasor_transform(filt, n_harm=self.spin_harm.value(), axis=0)
            ax.hist2d(g.ravel(), (-s).ravel(), bins=100, range=[[-1,1],[-1,1]], cmap='nipy_spectral')
            ax.set_title('Phasor Plot')
            ax.axis('off')
            ax.add_patch(plt.Circle((0, 0), 1, fill=False, color='white'))
            ax.set_box_aspect(1)
            ax_idx += 1
        # Spectral
        if want_spectral and ax_idx < cols:
            ax = ax_flat[ax_idx]
            rgb = pipeline.SpectralStack2RGB(self.channels, pipeline.channel_lambdas)
            ax.imshow(rgb/np.percentile(rgb,99.5))
            ax.set_title('Spectral RGB')
            ax.axis('off')
            ax_idx += 1
        # Hide unused
        for k in range(ax_idx, cols):
            axes[0,k].set_visible(False)

        # Row 1: interactive if requested
        if want_select:
            ax1, ax2, ax3 = axes[1]
            ax1.imshow(self.int_img, cmap='gray')
            ax1.set_title('Select Region')
            ax1.axis('off')

            ax2.set_title('Selected Phasor')
            ax2.axis('off')

            ax3.set_title('Intensity Sum')
            ax3.axis('off')

            # Create selector on raw image axis
            self.ax_select = ax1
            self.ax_phasor_sel = ax2
            self.ax_sum_sel = ax3
            self.selector = RectangleSelector(
                ax1, self._on_select_region,
                useblit=True, interactive=True
            )

        fig.tight_layout()
        self.canvas.draw_idle()

    def _on_select_region(self, eclick, erelease):
        # Coordinates
        x1, x2 = sorted([int(eclick.xdata), int(erelease.xdata)])
        y1, y2 = sorted([int(eclick.ydata), int(erelease.ydata)])
        sel = self.channels[:, y1:y2, x1:x2]

        # Phasor of selection
        g_sel, s_sel = pipeline.phasor_transform(sel, n_harm=self.spin_harm.value(), axis=0)
        mask = self.int_img[y1:y2, x1:x2] > np.mean(self.int_img[y1:y2, x1:x2])
        if mask.shape == g_sel.shape[1:]:
            c = g_sel[:, mask].ravel()
            s_vals = -s_sel[:, mask].ravel()
        else:
            g_vals = g_sel.ravel(); s_vals = -s_sel.ravel()

        g_corr = pd.DataFrame(g_vals)
        s_corr = pd.DataFrame(s_vals)

        # Update phasor plot
        ax2 = self.ax_phasor_sel
        ax2.clear()
        ax2.hist2d(g_vals, s_vals, bins=100, range=[[-1,1],[-1,1]], cmap='nipy_spectral')
        ax2.add_patch(plt.Circle((0,0),1, fill=False, color='white'))
        ax2.set_box_aspect(1)

        # Update sum plot
        ax3 = self.ax_sum_sel
        ax3.clear()
        sum_vals = sel.sum(axis=(1,2))
        ax3.plot(sum_vals, marker='o')
        print(f'Spectral Data: {sum_vals}')
        sum_vals = pd.DataFrame(sum_vals)
        sum_vals.to_csv('sum_vals.csv')

        print(f'Phasor Data G: {g_corr}')
        g_corr.to_csv('G_coordinate.csv', index=False)
        print(f'Phasor Data S: {s_corr}')
        s_corr.to_csv('S_coordinate.csv')




        self.canvas.draw_idle()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = PhasorGUI()
    win.show()
    sys.exit(app.exec_())

