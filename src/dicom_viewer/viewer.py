"""Interactive orthogonal (axial/coronal/sagittal) viewer for a DICOM MRI series.

Built for `dataset/MRI-with-electrodes/DCM`, which holds a post-implantation
head MRI (visible electrode-shaft artifacts) alongside the sEEG channel
montage in `sEEG-HFOs-8_montage.txt`. That montage file only names bipolar
channel pairs (e.g. `PM3-4`) — it carries no 3-D coordinates — so this viewer
does not attempt to overlay electrode positions; it just lets you page
through the volume and see the implant artifacts directly in the images.

The three views are simple axis-aligned reslices of the acquired voxel grid
(not a true patient-space reformat via the full direction-cosine matrix), so
they only line up with clinical anatomical planes when the acquisition is
close to axial. On this dataset the tilt is small (a few degrees), so the
approximation is good enough for visual review; a heavily oblique
acquisition would need real 3-D resampling to look right.
"""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import matplotlib
import numpy as np
import pydicom


def _ensure_interactive_backend() -> str:
    """Force a GUI backend known to handle widget *dragging*, not just clicks.

    A plain `import matplotlib.pyplot` can silently resolve to a
    non-interactive backend (`agg`, or the Jupyter `inline`/`module://
    matplotlib_inline...` one some IDE run modes inject) — the figure then
    renders as a dead, static image with no crosshair or slice interaction
    at all. macOS's native `MacOSX` backend avoids that, but its handling of
    continuous mouse-drag motion (which `Slider` widgets need — press, then
    move while still held) is known to be flaky: clicking can register while
    dragging the handle does nothing, which looks exactly like a frozen
    widget. `TkAgg` doesn't have that problem, so prefer it whenever
    tkinter is actually available; fall back through the rest only if not.
    """
    for candidate in ("TkAgg", "QtAgg", "Qt5Agg", "MacOSX"):
        try:
            matplotlib.use(candidate, force=True)
            return candidate
        except Exception:
            continue

    raise RuntimeError(
        "No interactive matplotlib GUI backend is available in this environment "
        "(tried TkAgg/QtAgg/Qt5Agg/MacOSX). This viewer needs a real display — "
        "run it in a local terminal/Run-and-Debug session, not a headless one, "
        "or install python-tk (or one of PyQt5/PySide6)."
    )


_ensure_interactive_backend()
import matplotlib.pyplot as plt  # noqa: E402  (backend must be chosen first)
from matplotlib.widgets import Button, Slider  # noqa: E402


@dataclass
class Series:
    series_number: str
    description: str
    files: list[Path]  # sorted into slice order
    rows: int
    columns: int
    pixel_spacing: tuple[float, float]  # (row spacing mm, column spacing mm)


def find_dicom_files(directory: Path) -> list[Path]:
    """Recursively collect every parseable DICOM file under `directory`.

    Files here carry no `.dcm` extension, so filtering has to be by content
    (a valid DICOM header), not by name.
    """
    files = []
    for path in sorted(directory.rglob("*")):
        if not path.is_file():
            continue
        try:
            pydicom.dcmread(path, stop_before_pixels=True)
        except Exception:
            continue
        files.append(path)
    return files


def _instance_sort_key(ds: pydicom.Dataset, fallback_index: int):
    if getattr(ds, "InstanceNumber", None) not in (None, ""):
        return (0, int(ds.InstanceNumber))
    return (1, fallback_index)


def group_series(files: list[Path]) -> dict[str, Series]:
    """Bucket files into series keyed by (SeriesNumber, SeriesDescription).

    `SeriesInstanceUID` would be the textbook grouping key, but on this
    dataset it comes out unique per instance (an anonymization artifact) so
    grouping by it would put every slice in its own single-slice "series".
    SeriesNumber/SeriesDescription is what actually stays constant across
    the slices of one acquisition here.
    """
    buckets: dict[tuple[str, str], list[tuple[Path, pydicom.Dataset]]] = defaultdict(list)
    for path in files:
        ds = pydicom.dcmread(path, stop_before_pixels=True)
        key = (str(ds.get("SeriesNumber", "0")), str(ds.get("SeriesDescription", "")))
        buckets[key].append((path, ds))

    series_map: dict[str, Series] = {}
    for (series_number, description), items in buckets.items():
        indexed = list(enumerate(items))
        indexed.sort(key=lambda entry: _instance_sort_key(entry[1][1], entry[0]))
        items = [item for _, item in indexed]
        ref = items[0][1]
        series_map[series_number] = Series(
            series_number=series_number,
            description=description,
            files=[path for path, _ in items],
            rows=int(ref.Rows),
            columns=int(ref.Columns),
            pixel_spacing=tuple(float(x) for x in ref.PixelSpacing),
        )
    return series_map


def load_series(series: Series) -> tuple[np.ndarray, tuple[float, float, float]]:
    """Load one series into a (z, y, x) volume plus (dz, dy, dx) spacing in mm."""
    datasets = [pydicom.dcmread(f) for f in series.files]

    slope = float(getattr(datasets[0], "RescaleSlope", 1) or 1)
    intercept = float(getattr(datasets[0], "RescaleIntercept", 0) or 0)
    volume = np.stack([ds.pixel_array.astype(np.float32) for ds in datasets], axis=0)
    volume = volume * slope + intercept

    positions = [
        np.array(ds.ImagePositionPatient, dtype=float)
        for ds in datasets
        if getattr(ds, "ImagePositionPatient", None) is not None
    ]
    if len(positions) >= 2:
        diffs = np.linalg.norm(np.diff(positions, axis=0), axis=1)
        slice_spacing = float(np.median(diffs))
    else:
        slice_spacing = float(
            getattr(datasets[0], "SpacingBetweenSlices", None)
            or getattr(datasets[0], "SliceThickness", 1.0)
        )

    row_spacing, col_spacing = series.pixel_spacing
    return volume, (slice_spacing, row_spacing, col_spacing)


class OrthogonalViewer:
    """Three linked orthogonal views of one volume in a single figure.

    - Click the ◀/▶ buttons under a view to step that view's slice one at a
      time — a plain click-release, so it doesn't depend on continuous
      mouse-drag tracking actually reaching the canvas (unlike the slider,
      which some backends handle unreliably for real drag gestures).
    - Drag the slider under a view to jump to a slice directly.
    - Scroll over a view (or hover it and press +/-/arrow keys) also pages
      through its slices.
    - Click in a view to move the shared crosshair; the other two views jump
      to the clicked position and redraw.
    """

    def __init__(self, volume: np.ndarray, spacing: tuple[float, float, float], title: str = ""):
        self.volume = volume
        self.dz, self.dy, self.dx = spacing
        self.nz, self.ny, self.nx = volume.shape
        self.z = self.nz // 2
        self.y = self.ny // 2
        self.x = self.nx // 2
        self.vmin, self.vmax = np.percentile(volume, [1, 99])

        self.fig, (self.ax_axial, self.ax_coronal, self.ax_sagittal) = plt.subplots(1, 3, figsize=(14, 6))
        self.fig.subplots_adjust(bottom=0.2)
        self.fig.suptitle(title)
        for ax in (self.ax_axial, self.ax_coronal, self.ax_sagittal):
            ax.set_facecolor("black")

        self.im_axial = self.ax_axial.imshow(
            self._axial_slice(), cmap="gray", vmin=self.vmin, vmax=self.vmax,
            extent=[0, self.nx * self.dx, self.ny * self.dy, 0],
        )
        self.ax_axial.set_xlabel("x (mm)")
        self.ax_axial.set_ylabel("y (mm)")

        self.im_coronal = self.ax_coronal.imshow(
            self._coronal_slice(), cmap="gray", vmin=self.vmin, vmax=self.vmax,
            extent=[0, self.nx * self.dx, self.nz * self.dz, 0],
        )
        self.ax_coronal.set_xlabel("x (mm)")
        self.ax_coronal.set_ylabel("z (mm)")

        self.im_sagittal = self.ax_sagittal.imshow(
            self._sagittal_slice(), cmap="gray", vmin=self.vmin, vmax=self.vmax,
            extent=[0, self.ny * self.dy, self.nz * self.dz, 0],
        )
        self.ax_sagittal.set_xlabel("y (mm)")
        self.ax_sagittal.set_ylabel("z (mm)")

        self._crosshair_lines: dict[plt.Axes, tuple] = {}
        for ax in (self.ax_axial, self.ax_coronal, self.ax_sagittal):
            vline = ax.axvline(0, color="yellow", linewidth=0.8)
            hline = ax.axhline(0, color="yellow", linewidth=0.8)
            self._crosshair_lines[ax] = (vline, hline)

        self._build_controls()
        self._connect_events()
        self._refresh_all()

    def _axis_for(self, name: str):
        return {"axial": self.ax_axial, "coronal": self.ax_coronal, "sagittal": self.ax_sagittal}[name]

    def _build_controls(self):
        specs = [
            ("axial", self.ax_axial, self.nz - 1, self.z),
            ("coronal", self.ax_coronal, self.ny - 1, self.y),
            ("sagittal", self.ax_sagittal, self.nx - 1, self.x),
        ]
        self.sliders: dict[str, Slider] = {}
        self.buttons: dict[str, tuple[Button, Button]] = {}
        for name, ax, max_index, init in specs:
            pos = ax.get_position()
            button_w = pos.width * 0.09

            prev_ax = self.fig.add_axes([pos.x0, 0.07, button_w, 0.05])
            next_ax = self.fig.add_axes([pos.x0 + pos.width - button_w, 0.07, button_w, 0.05])
            slider_ax = self.fig.add_axes(
                [pos.x0 + button_w * 1.3, 0.085, pos.width - 2 * button_w * 1.3, 0.02]
            )
            for a in (prev_ax, next_ax, slider_ax):
                a.set_navigate(False)  # exclude from toolbar pan/zoom targeting

            prev_button = Button(prev_ax, "\N{BLACK LEFT-POINTING TRIANGLE}")
            next_button = Button(next_ax, "\N{BLACK RIGHT-POINTING TRIANGLE}")
            prev_button.on_clicked(lambda _event, name=name: self._on_button(name, -1))
            next_button.on_clicked(lambda _event, name=name: self._on_button(name, 1))
            self.buttons[name] = (prev_button, next_button)

            slider = Slider(slider_ax, "", 0, max(max_index, 1), valinit=init, valstep=1, color="tab:orange")
            slider.valtext.set_visible(False)  # index is already shown in the pane's title; avoids overlapping the ▶ button
            slider.on_changed(lambda val, name=name: self._on_slider(name, val))
            self.sliders[name] = slider

    def _on_button(self, name: str, step: int):
        if self._step_axis(self._axis_for(name), step):
            self._refresh_all()

    def _on_slider(self, name: str, val: float):
        idx = int(val)
        if name == "axial":
            self.z = idx
        elif name == "coronal":
            self.y = idx
        elif name == "sagittal":
            self.x = idx
        self._refresh_all()

    def _sync_sliders(self):
        for name, idx in (("axial", self.z), ("coronal", self.y), ("sagittal", self.x)):
            slider = self.sliders[name]
            if int(slider.val) != idx:
                slider.eventson = False
                slider.set_val(idx)
                slider.eventson = True

    # -- slice extraction -------------------------------------------------

    def _axial_slice(self) -> np.ndarray:
        return self.volume[self.z, :, :]

    def _coronal_slice(self) -> np.ndarray:
        return self.volume[:, self.y, :]

    def _sagittal_slice(self) -> np.ndarray:
        return self.volume[:, :, self.x]

    # -- drawing ------------------------------------------------------------

    def _refresh_all(self):
        self.im_axial.set_data(self._axial_slice())
        self.im_coronal.set_data(self._coronal_slice())
        self.im_sagittal.set_data(self._sagittal_slice())

        ax_v, ax_h = self._crosshair_lines[self.ax_axial]
        ax_v.set_xdata([self.x * self.dx, self.x * self.dx])
        ax_h.set_ydata([self.y * self.dy, self.y * self.dy])

        co_v, co_h = self._crosshair_lines[self.ax_coronal]
        co_v.set_xdata([self.x * self.dx, self.x * self.dx])
        co_h.set_ydata([self.z * self.dz, self.z * self.dz])

        sa_v, sa_h = self._crosshair_lines[self.ax_sagittal]
        sa_v.set_xdata([self.y * self.dy, self.y * self.dy])
        sa_h.set_ydata([self.z * self.dz, self.z * self.dz])

        self.ax_axial.set_title(f"Axial  z={self.z}/{self.nz - 1}")
        self.ax_coronal.set_title(f"Coronal  y={self.y}/{self.ny - 1}")
        self.ax_sagittal.set_title(f"Sagittal  x={self.x}/{self.nx - 1}")
        self._sync_sliders()
        self.fig.canvas.draw_idle()

    # -- interaction ----------------------------------------------------

    def _connect_events(self):
        self._hover_ax = self.ax_axial
        self.fig.canvas.mpl_connect("scroll_event", self._on_scroll)
        self.fig.canvas.mpl_connect("button_press_event", self._on_click)
        self.fig.canvas.mpl_connect("axes_enter_event", self._on_axes_enter)
        self.fig.canvas.mpl_connect("key_press_event", self._on_key)

    def _on_axes_enter(self, event):
        if event.inaxes is not None:
            self._hover_ax = event.inaxes

    def _step_axis(self, ax, step: int) -> bool:
        if ax is self.ax_axial:
            self.z = int(np.clip(self.z + step, 0, self.nz - 1))
        elif ax is self.ax_coronal:
            self.y = int(np.clip(self.y + step, 0, self.ny - 1))
        elif ax is self.ax_sagittal:
            self.x = int(np.clip(self.x + step, 0, self.nx - 1))
        else:
            return False
        return True

    def _on_scroll(self, event):
        if event.inaxes is None:
            return
        step = 1 if event.button == "up" else -1
        if self._step_axis(event.inaxes, step):
            self._refresh_all()

    def _on_key(self, event):
        # Scroll-wheel events don't always reach matplotlib the same way
        # across backends/remote displays, so arrow keys/+/- are a backup
        # way to step through slices of whichever pane the mouse is over.
        step_keys = {"up": 1, "right": 1, "+": 1, "=": 1, "down": -1, "left": -1, "-": -1}
        if event.key not in step_keys:
            return
        if self._step_axis(self._hover_ax, step_keys[event.key]):
            self._refresh_all()

    def _on_click(self, event):
        if event.inaxes is None or event.xdata is None or event.ydata is None:
            return
        if event.inaxes is self.ax_axial:
            self.x = int(np.clip(round(event.xdata / self.dx), 0, self.nx - 1))
            self.y = int(np.clip(round(event.ydata / self.dy), 0, self.ny - 1))
        elif event.inaxes is self.ax_coronal:
            self.x = int(np.clip(round(event.xdata / self.dx), 0, self.nx - 1))
            self.z = int(np.clip(round(event.ydata / self.dz), 0, self.nz - 1))
        elif event.inaxes is self.ax_sagittal:
            self.y = int(np.clip(round(event.xdata / self.dy), 0, self.ny - 1))
            self.z = int(np.clip(round(event.ydata / self.dz), 0, self.nz - 1))
        else:
            return
        self._refresh_all()


def _print_series_table(series_map: dict[str, Series]):
    print(f"{'series':>6}  {'slices':>6}  description")
    for number, series in sorted(series_map.items(), key=lambda kv: kv[0]):
        print(f"{number:>6}  {len(series.files):>6}  {series.description}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dicom_dir", type=Path, help="Directory containing a DICOM series (searched recursively)")
    parser.add_argument("--series", help="Series number to load (see --list-series for the choices)")
    parser.add_argument("--list-series", action="store_true", help="List the series found and exit")
    args = parser.parse_args(argv)

    files = find_dicom_files(args.dicom_dir)
    if not files:
        print(f"No DICOM files found under {args.dicom_dir}", file=sys.stderr)
        return 1

    series_map = group_series(files)

    if args.list_series:
        _print_series_table(series_map)
        return 0

    if args.series is not None:
        if args.series not in series_map:
            print(f"No series numbered {args.series!r} found. Choices:", file=sys.stderr)
            _print_series_table(series_map)
            return 1
        series = series_map[args.series]
    elif len(series_map) == 1:
        series = next(iter(series_map.values()))
    else:
        print(f"{len(series_map)} series found under {args.dicom_dir} — pick one with --series:", file=sys.stderr)
        _print_series_table(series_map)
        return 1

    volume, spacing = load_series(series)
    OrthogonalViewer(volume, spacing, title=f"Series {series.series_number}: {series.description}")
    print(f"matplotlib backend: {matplotlib.get_backend()}", flush=True)
    print(
        "Click the </> buttons or drag a slider under a view to change its slice "
        "(or scroll/press +-/arrows over a pane); click inside a view to move the crosshair.",
        flush=True,
    )
    plt.show()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
