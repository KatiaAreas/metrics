"""
uncertainty_analysis.py
=======================
Three-stage uncertainty analysis for the OptiTrack → Camera reprojection pipeline.

Stage 1 – Model → OptiTrack   : SVD rigid-body fitting residuals (mm)
Stage 2 – OptiTrack → Camera  : Rigid-body tracking jitter (mm, °)
Stage 3 – Camera projection   : Reprojection error vs 2D keypoints (px)

UncertaintyPlotter
------------------
Displays a 4-panel diagnostic figure in its **own cv2 window** ("Uncertainty Analysis").
Call  plotter.update(record)  every N frames — the window refreshes automatically.
No overlay on the video frame.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.backends.backend_agg import FigureCanvasAgg
import numpy as np
import numpy.typing as npt


# ---------------------------------------------------------------------------
# Quality thresholds (px)
# ---------------------------------------------------------------------------
THRESH_GOOD   = 2.0   # green dashed line
THRESH_ACCEPT = 5.0   # orange dashed line


# =============================================================================
# Data-classes
# =============================================================================

@dataclass
class TransformationUncertainty:
    """Uncertainty metrics for one pipeline stage."""
    stage_name: str
    # 3D fit quality
    mean_error_3d_m:   Optional[float] = None
    std_error_3d_m:    Optional[float] = None
    max_error_3d_m:    Optional[float] = None
    point_errors_3d_m: Optional[npt.NDArray[np.float64]] = None
    condition_number:  Optional[float] = None
    residual_norm:     Optional[float] = None
    # Tracking
    translation_error_m: Optional[float] = None
    rotation_error_deg:  Optional[float] = None
    # Reprojection
    mean_reprojection_error_px: Optional[float] = None
    std_reprojection_error_px:  Optional[float] = None
    max_reprojection_error_px:  Optional[float] = None
    reprojection_errors_px: Optional[npt.NDArray[np.float64]] = None
    num_points: int = 0
    notes: str = ""


@dataclass
class UncertaintyReport:
    """Complete pipeline report for one frame."""
    model_to_optitrack:  Optional[TransformationUncertainty] = None
    optitrack_to_camera: Optional[TransformationUncertainty] = None
    camera_projection:   Optional[TransformationUncertainty] = None
    total_reprojection_error_px: Optional[float] = None
    # Cache for downstream use
    points_3d_camera_last: Optional[npt.NDArray[np.float64]] = None

    def to_dict(self) -> Dict:
        out: Dict = {}
        for name in ("model_to_optitrack", "optitrack_to_camera", "camera_projection"):
            u: Optional[TransformationUncertainty] = getattr(self, name)
            if u is None:
                continue
            out[name] = {
                "mean_error_3d_mm":           (u.mean_error_3d_m  or 0) * 1000,
                "std_error_3d_mm":            (u.std_error_3d_m   or 0) * 1000,
                "max_error_3d_mm":            (u.max_error_3d_m   or 0) * 1000,
                "translation_error_mm":       (u.translation_error_m or 0) * 1000,
                "rotation_error_deg":          u.rotation_error_deg,
                "mean_reprojection_error_px":  u.mean_reprojection_error_px,
                "std_reprojection_error_px":   u.std_reprojection_error_px,
                "max_reprojection_error_px":   u.max_reprojection_error_px,
                "condition_number":            u.condition_number,
                "num_points":                  u.num_points,
                "notes":                       u.notes,
            }
        out["total_reprojection_error_px"] = self.total_reprojection_error_px
        return out

    def save_json(self, path: Path) -> None:
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)
        print(f"[OK] Report saved → {path}")


# =============================================================================
# UncertaintyAnalyzer
# =============================================================================

class UncertaintyAnalyzer:
    """
    Analyzes uncertainty at each stage of the projection pipeline.

    Parameters
    ----------
    transformer : PyramidTransformer
    calib_data  : CalibData
    verbose     : bool
    """

    def __init__(self, transformer, calib_data, verbose: bool = True):
        self.transformer = transformer
        self.calib_data  = calib_data
        self.verbose     = verbose
        self.report      = UncertaintyReport()

    # ------------------------------------------------------------------
    # Stage 1 – SVD fitting: model → OptiTrack
    # ------------------------------------------------------------------
    def analyze_model_to_optitrack(
        self,
        marker_positions_m: Dict[str, npt.NDArray],
        matching: Dict[str, int],
    ) -> TransformationUncertainty:
        u = TransformationUncertainty(
            stage_name="model_to_optitrack",
            num_points=len(matching),
        )

        cids   = list(matching.values())
        pts_w  = self.transformer.points_m[cids]
        R_pyr  = self.transformer.R_pyramid
        origin = self.transformer.pyramid_origin_m
        pts_p  = (R_pyr.T @ (pts_w - origin).T).T
        opti   = np.array([marker_positions_m[n] for n in matching.keys()])
        fitted = self.transformer.transform_pyramid_to_optitrack(pts_p)

        errs = np.linalg.norm(opti - fitted, axis=1)
        u.point_errors_3d_m = errs
        u.mean_error_3d_m   = float(np.mean(errs))
        u.std_error_3d_m    = float(np.std(errs))
        u.max_error_3d_m    = float(np.max(errs))
        u.residual_norm      = float(np.linalg.norm(opti - fitted))

        c0 = pts_p - pts_p.mean(0)
        c1 = opti  - opti.mean(0)
        _, S, _ = np.linalg.svd(c0.T @ c1)
        u.condition_number = float(S[0] / S[-1]) if S[-1] > 1e-10 else float("inf")
        u.notes = f"SVD fit over {len(matching)} markers."

        if self.verbose:
            print(f"  [Stage 1] SVD mean={u.mean_error_3d_m*1e3:.3f}mm "
                  f"max={u.max_error_3d_m*1e3:.3f}mm  cond={u.condition_number:.1f}")

        self.report.model_to_optitrack = u
        return u

    # ------------------------------------------------------------------
    # Stage 2 – Tracking jitter: OptiTrack → camera
    # ------------------------------------------------------------------
    def analyze_optitrack_to_camera(
        self,
        rb_data: Dict,
        frame_id: int,
        jitter_window: int = 5,
    ) -> TransformationUncertainty:
        u = TransformationUncertainty(stage_name="optitrack_to_camera")

        u.translation_error_m = self._jitter_translation(rb_data, frame_id, jitter_window)
        u.rotation_error_deg  = self._jitter_rotation(rb_data, frame_id, jitter_window)

        T_Lens = rb_data["Lens_RB"][frame_id].get_transform()
        RT = np.linalg.inv(T_Lens @ self.calib_data.RT)
        u.condition_number = float(np.linalg.cond(RT))

        if self.verbose:
            print(f"  [Stage 2] jitter trans={u.translation_error_m*1e3:.3f}mm "
                  f"rot={u.rotation_error_deg:.4f}°")

        self.report.optitrack_to_camera = u
        return u

    def _jitter_translation(self, rb_data, frame_id, win, key="Pyramid_RB"):
        frames = rb_data[key]
        n = len(frames)
        pos = [
            frames[f].get_transform()[:3, 3]
            for f in range(max(0, frame_id - win), min(n, frame_id + win + 1))
            if frames[f].data.is_visible
        ]
        if len(pos) < 3:
            return 0.001
        diffs = np.diff(np.array(pos), axis=0)
        return float(np.linalg.norm(np.std(diffs, axis=0)))

    def _jitter_rotation(self, rb_data, frame_id, win, key="Pyramid_RB"):
        frames = rb_data[key]
        n = len(frames)
        rots = [
            frames[f].get_transform()[:3, :3]
            for f in range(max(0, frame_id - win), min(n, frame_id + win + 1))
            if frames[f].data.is_visible
        ]
        if len(rots) < 3:
            return 0.05
        angles = [
            float(np.degrees(np.arccos(
                np.clip((np.trace(rots[i + 1] @ rots[i].T) - 1) / 2, -1, 1)
            )))
            for i in range(len(rots) - 1)
        ]
        return float(np.std(angles))

    # ------------------------------------------------------------------
    # Stage 3 – Camera projection / reprojection vs 2D keypoints
    # ------------------------------------------------------------------
    def analyze_camera_projection(
        self,
        pts_3d_camera: npt.NDArray,
        pts_2d_observed: Optional[npt.NDArray] = None,
        R_cor: Optional[npt.NDArray] = None,
    ) -> TransformationUncertainty:
        u = TransformationUncertainty(
            stage_name="camera_projection",
            num_points=len(pts_3d_camera),
        )

        p2d, _ = cv2.projectPoints(
            pts_3d_camera, np.zeros(3), np.zeros(3),
            self.calib_data.K, self.calib_data.dist_coeffs,
        )
        p2d = p2d.reshape(-1, 2)

        if R_cor is not None:
            h = np.hstack([p2d, np.ones((len(p2d), 1))]).T
            p2d = (R_cor @ h)[:2, :].T

        if pts_2d_observed is not None:
            n = min(len(p2d), len(pts_2d_observed))
            if n > 0:
                errs = np.linalg.norm(p2d[:n] - pts_2d_observed[:n], axis=1)
                u.reprojection_errors_px      = errs
                u.mean_reprojection_error_px  = float(np.mean(errs))
                u.std_reprojection_error_px   = float(np.std(errs))
                u.max_reprojection_error_px   = float(np.max(errs))

        u.condition_number = float(np.linalg.cond(self.calib_data.K))
        u.notes = f"K cond={u.condition_number:.2f}"

        if self.verbose and u.mean_reprojection_error_px is not None:
            print(f"  [Stage 3] reproj mean={u.mean_reprojection_error_px:.2f}px "
                  f"max={u.max_reprojection_error_px:.2f}px")

        self.report.camera_projection = u
        return u

    # ------------------------------------------------------------------
    # Full pipeline
    # ------------------------------------------------------------------
    def analyze_full_pipeline(
        self,
        rb_data: Dict,
        marker_positions_m: Dict[str, npt.NDArray],
        matching: Dict[str, int],
        frame_id: int = 0,
        keypoints_2d: Optional[npt.NDArray] = None,
        R_cor: Optional[npt.NDArray] = None,
    ) -> UncertaintyReport:
        # Stage 1
        self.analyze_model_to_optitrack(marker_positions_m, matching)

        # Stage 2
        self.analyze_optitrack_to_camera(rb_data, frame_id)

        # Build 3-D points in camera frame for stage 3
        # Same computation as utils.py — no invented helper method needed
        _pts_w  = self.transformer.points_m                   # all 22 points (world coords)
        _R_pyr  = self.transformer.R_pyramid
        _origin = self.transformer.pyramid_origin_m
        pts_pyr  = (_R_pyr.T @ (_pts_w - _origin).T).T        # world → pyramid frame
        pts_pyr  = pts_pyr[:18]                               # keep only points 0-17
        pts_opti = self.transformer.transform_pyramid_to_optitrack(pts_pyr)

        T_Pyr  = rb_data["Pyramid_RB"][frame_id].get_transform()
        T_Lens = rb_data["Lens_RB"][frame_id].get_transform()
        RT     = np.linalg.inv(T_Lens @ self.calib_data.RT)

        n = pts_opti.shape[0]
        pts_w = (T_Pyr @ np.hstack([pts_opti, np.ones((n, 1))]).T).T[:, :3]
        pts_c = (RT[:3, :3] @ pts_w.T + RT[:3, 3:4]).T
        self.report.points_3d_camera_last = pts_c.copy()

        # Stage 3
        self.analyze_camera_projection(pts_c, keypoints_2d, R_cor)

        if (keypoints_2d is not None
                and self.report.camera_projection is not None
                and self.report.camera_projection.mean_reprojection_error_px is not None):
            self.report.total_reprojection_error_px = (
                self.report.camera_projection.mean_reprojection_error_px
            )

        return self.report

    def print_summary(self) -> None:
        print("\n" + "=" * 55 + "\nUNCERTAINTY SUMMARY\n" + "=" * 55)
        u1 = self.report.model_to_optitrack
        if u1 and u1.mean_error_3d_m is not None:
            q = "[OK]" if u1.mean_error_3d_m < 0.002 else "[!]" if u1.mean_error_3d_m < 0.005 else "[X]"
            print(f"{q} SVD:       {u1.mean_error_3d_m*1e3:.3f} ± {u1.std_error_3d_m*1e3:.3f} mm")
        u2 = self.report.optitrack_to_camera
        if u2:
            print(f"     Jitter: trans={u2.translation_error_m*1e3:.3f}mm  rot={u2.rotation_error_deg:.4f}°")
        u3 = self.report.camera_projection
        if u3 and u3.mean_reprojection_error_px is not None:
            q = "[OK]" if u3.mean_reprojection_error_px < 2 else "[!]" if u3.mean_reprojection_error_px < 5 else "[X]"
            print(f"{q} Reproj:    {u3.mean_reprojection_error_px:.2f} ± {u3.std_reprojection_error_px:.2f} px")


# =============================================================================
# FrameRecord  – lightweight snapshot pushed into UncertaintyPlotter
# =============================================================================

@dataclass
class FrameRecord:
    """All uncertainty numbers for one frame, ready to plot."""
    frame_id: int
    # Stage 1
    svd_mean_mm:   float = 0.0
    svd_std_mm:    float = 0.0
    svd_max_mm:    float = 0.0
    svd_cond:      float = 0.0
    # Stage 2
    trans_jitter_mm: float = 0.0
    rot_jitter_deg:  float = 0.0
    # Stage 3
    reproj_mean_px: Optional[float] = None
    reproj_std_px:  Optional[float] = None
    reproj_max_px:  Optional[float] = None
    # Derived: predicted RSS error (mm → px using rough focal-length factor)
    predicted_rss_px: float = 0.0


def frame_record_from_analyzer(frame_id: int, analyzer: UncertaintyAnalyzer) -> FrameRecord:
    """Build a FrameRecord from the latest report in an UncertaintyAnalyzer."""
    r  = analyzer.report
    u1 = r.model_to_optitrack
    u2 = r.optitrack_to_camera
    u3 = r.camera_projection

    svd_mean = (u1.mean_error_3d_m or 0.0) * 1e3 if u1 else 0.0
    svd_std  = (u1.std_error_3d_m  or 0.0) * 1e3 if u1 else 0.0
    svd_max  = (u1.max_error_3d_m  or 0.0) * 1e3 if u1 else 0.0
    svd_cond = (u1.condition_number or 0.0)       if u1 else 0.0

    t_jit = (u2.translation_error_m or 0.0) * 1e3 if u2 else 0.0
    r_jit = (u2.rotation_error_deg  or 0.0)        if u2 else 0.0

    reproj_mean = u3.mean_reprojection_error_px if u3 else None
    reproj_std  = u3.std_reprojection_error_px  if u3 else None
    reproj_max  = u3.max_reprojection_error_px  if u3 else None

    # Rough focal-length estimate to convert mm → px for RSS display
    K = analyzer.calib_data.K
    focal_px_per_mm = float(np.mean([K[0, 0], K[1, 1]])) / 1000.0
    rss_px = np.sqrt(
        (svd_mean * focal_px_per_mm) ** 2 +
        (t_jit    * focal_px_per_mm) ** 2
    )

    return FrameRecord(
        frame_id       = frame_id,
        svd_mean_mm    = svd_mean,
        svd_std_mm     = svd_std,
        svd_max_mm     = svd_max,
        svd_cond       = svd_cond,
        trans_jitter_mm= t_jit,
        rot_jitter_deg  = r_jit,
        reproj_mean_px  = reproj_mean,
        reproj_std_px   = reproj_std,
        reproj_max_px   = reproj_max,
        predicted_rss_px= float(rss_px),
    )


# =============================================================================
# UncertaintyPlotter  – separate cv2 window, 4 panels
# =============================================================================

class UncertaintyPlotter:
    """
    4-panel diagnostic plot shown in a **dedicated cv2 window**.

    Panels
    ------
    1  Stage 1 – SVD 3D error (mm)  [rolling time-series]
    2  Stage 2 – Tracking jitter: translation (mm) and rotation (°) [rolling]
    3  Stage 3 – Reprojection error (px) vs good/accept thresholds [rolling]
    4  Error summary bar chart (current frame snapshot)

    Usage
    -----
        plotter = UncertaintyPlotter()
        # inside video loop, every N frames:
        rec = frame_record_from_analyzer(frame_id, analyzer)
        plotter.update(rec)
        cv2.waitKey(1)   # allows cv2 to process window events
    """

    _BG    = "#1a1a2e"
    _PANEL = "#0f0f23"
    _SPINE = "#444466"
    _TICK  = "#aaaacc"
    _GREEN = "#2ecc71"
    _AMBER = "#f39c12"
    _RED   = "#e74c3c"
    _BLUE  = "#3498db"
    _PURP  = "#9b59b6"
    _WHITE = "#ffffff"

    def __init__(
        self,
        history_len: int = 200,
        fig_w_px: int = 1100,
        fig_h_px: int = 650,
        window_name: str = "Uncertainty Analysis",
    ):
        self.history:     List[FrameRecord] = []
        self.history_len  = history_len
        self.window_name  = window_name
        self._dpi         = 100

        fw = fig_w_px / self._dpi
        fh = fig_h_px / self._dpi

        self.fig = plt.figure(figsize=(fw, fh), facecolor=self._BG)
        self.fig.suptitle(
            "Uncertainty Analysis  ·  OptiTrack → Camera Pipeline",
            fontsize=11, fontweight="bold", color="white", y=0.98,
        )
        gs = gridspec.GridSpec(
            2, 2, figure=self.fig,
            hspace=0.50, wspace=0.38,
            left=0.07, right=0.97, top=0.92, bottom=0.08,
        )
        self.axes = [self.fig.add_subplot(gs[r, c]) for r in range(2) for c in range(2)]
        for ax in self.axes:
            self._style_ax(ax)

        self.canvas = FigureCanvasAgg(self.fig)

        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.window_name, fig_w_px, fig_h_px)

    # ------------------------------------------------------------------
    def _style_ax(self, ax) -> None:
        ax.set_facecolor(self._PANEL)
        for sp in ax.spines.values():
            sp.set_color(self._SPINE)
        ax.tick_params(colors=self._TICK, labelsize=8)
        ax.yaxis.label.set_color(self._TICK)
        ax.xaxis.label.set_color(self._TICK)
        ax.title.set_color("white")

    def _ann(self, ax, text: str, color: str = "#ffcc00") -> None:
        ax.annotate(
            text, xy=(0.97, 0.95), xycoords="axes fraction",
            ha="right", va="top", fontsize=7.5, color=color,
            bbox=dict(boxstyle="round,pad=0.3", fc=self._BG, ec=color, alpha=0.85),
        )

    def _thresholds(self, ax) -> None:
        ax.axhline(THRESH_GOOD,   color=self._GREEN, linestyle="--", lw=1, alpha=0.7,
                   label=f"Good   ({THRESH_GOOD}px)")
        ax.axhline(THRESH_ACCEPT, color=self._AMBER, linestyle="--", lw=1, alpha=0.7,
                   label=f"Accept ({THRESH_ACCEPT}px)")

    # ------------------------------------------------------------------
    def update(self, record: FrameRecord) -> None:
        """Append a FrameRecord, redraw, and refresh the cv2 window."""
        self.history.append(record)
        if len(self.history) > self.history_len:
            self.history = self.history[-self.history_len:]
        self._redraw()
        self._push()

    def _push(self) -> None:
        self.canvas.draw()
        buf = self.canvas.buffer_rgba()
        img = np.asarray(buf, dtype=np.uint8)
        bgr = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
        cv2.imshow(self.window_name, bgr)

    def close(self) -> None:
        cv2.destroyWindow(self.window_name)
        plt.close(self.fig)

    # ------------------------------------------------------------------
    def _redraw(self) -> None:
        for ax in self.axes:
            ax.cla()
            self._style_ax(ax)
        if not self.history:
            return
        frames = [r.frame_id for r in self.history]
        self._panel1_svd(self.axes[0], frames)
        self._panel2_jitter(self.axes[1], frames)
        self._panel3_reproj(self.axes[2], frames)
        self._panel4_budget(self.axes[3])

    # ------------------------------------------------------------------
    # Panel 1 – Stage 1: SVD 3D fitting error
    # ------------------------------------------------------------------
    def _panel1_svd(self, ax, frames: List[int]) -> None:
        ax.set_title("Stage 1 · SVD 3D Fitting Error", fontsize=9, fontweight="bold", pad=4)
        ax.set_xlabel("Frame", fontsize=8)
        ax.set_ylabel("Error (mm)", fontsize=8)

        mean = [r.svd_mean_mm for r in self.history]
        lo   = [r.svd_mean_mm - r.svd_std_mm for r in self.history]
        hi   = [r.svd_mean_mm + r.svd_std_mm for r in self.history]
        mx   = [r.svd_max_mm  for r in self.history]

        fa = np.array(frames)
        ax.fill_between(fa, lo, hi, alpha=0.2, color=self._BLUE, label="±1σ")
        ax.plot(fa, mean, color=self._BLUE,  lw=1.5, label="Mean")
        ax.plot(fa, mx,   color=self._AMBER, lw=1.0, linestyle=":", label="Max")

        # Quality thresholds in mm
        ax.axhline(2.0, color=self._GREEN, linestyle="--", lw=1, alpha=0.7, label="Good (2mm)")
        ax.axhline(5.0, color=self._AMBER, linestyle="--", lw=1, alpha=0.7, label="Accept (5mm)")

        last = self.history[-1]
        c = (self._GREEN if last.svd_mean_mm < 2 else
             self._AMBER if last.svd_mean_mm < 5 else self._RED)
        self._ann(ax, f"Now:  {last.svd_mean_mm:.3f} mm\n"
                      f"Max:  {last.svd_max_mm:.3f} mm\n"
                      f"Cond: {last.svd_cond:.1f}", color=c)

        ax.legend(fontsize=7, loc="upper left", labelcolor="white",
                  facecolor=self._BG, edgecolor=self._SPINE)
        ax.grid(axis="y", alpha=0.12, color=self._SPINE)

    # ------------------------------------------------------------------
    # Panel 2 – Stage 2: Tracking jitter
    # ------------------------------------------------------------------
    def _panel2_jitter(self, ax, frames: List[int]) -> None:
        ax.set_title("Stage 2 · OptiTrack Tracking Jitter", fontsize=9, fontweight="bold", pad=4)
        ax.set_xlabel("Frame", fontsize=8)

        fa    = np.array(frames)
        trans = [r.trans_jitter_mm for r in self.history]
        rot   = [r.rot_jitter_deg  for r in self.history]

        ax_r = ax.twinx()
        ax_r.set_facecolor(self._PANEL)
        ax_r.tick_params(colors=self._TICK, labelsize=8)
        ax_r.yaxis.label.set_color(self._TICK)

        ax.plot(fa,   trans, color=self._BLUE, lw=1.5, label="Trans jitter (mm)")
        ax_r.plot(fa, rot,   color=self._PURP, lw=1.5, linestyle="--", label="Rot jitter (°)")

        ax.set_ylabel("Translation jitter (mm)", fontsize=8)
        ax_r.set_ylabel("Rotation jitter (°)", fontsize=8)

        ax.axhline(0.5, color=self._GREEN, linestyle="--", lw=1, alpha=0.6)
        ax.axhline(2.0, color=self._AMBER, linestyle="--", lw=1, alpha=0.6)

        last = self.history[-1]
        c = self._GREEN if last.trans_jitter_mm < 0.5 else \
            self._AMBER if last.trans_jitter_mm < 2.0 else self._RED
        self._ann(ax, f"Trans: {last.trans_jitter_mm:.3f} mm\n"
                      f"Rot:   {last.rot_jitter_deg:.4f} °", color=c)

        lines1, lbl1 = ax.get_legend_handles_labels()
        lines2, lbl2 = ax_r.get_legend_handles_labels()
        ax.legend(lines1 + lines2, lbl1 + lbl2,
                  fontsize=7, loc="upper left", labelcolor="white",
                  facecolor=self._BG, edgecolor=self._SPINE)
        ax.grid(axis="y", alpha=0.12, color=self._SPINE)

    # ------------------------------------------------------------------
    # Panel 3 – Stage 3: Reprojection error
    # ------------------------------------------------------------------
    def _panel3_reproj(self, ax, frames: List[int]) -> None:
        ax.set_title("Stage 3 · Reprojection Error  vs  Keypoints", fontsize=9,
                     fontweight="bold", pad=4)
        ax.set_xlabel("Frame", fontsize=8)
        ax.set_ylabel("Error (px)", fontsize=8)

        fa   = np.array(frames)
        mean = [r.reproj_mean_px if r.reproj_mean_px is not None else np.nan
                for r in self.history]
        mx   = [r.reproj_max_px  if r.reproj_max_px  is not None else np.nan
                for r in self.history]
        pred = [r.predicted_rss_px for r in self.history]

        has_real = any(not np.isnan(v) for v in mean)

        ax.plot(fa, pred, color=self._WHITE,  lw=1.2, linestyle=":", alpha=0.7,
                label="Predicted RSS (px)")
        if has_real:
            ax.plot(fa, mean, color=self._RED,  lw=1.8, label="Measured mean (px)")
            ax.plot(fa, mx,   color=self._AMBER, lw=1.0, linestyle=":", label="Measured max (px)")

        self._thresholds(ax)

        last = self.history[-1]
        if last.reproj_mean_px is not None:
            c = (self._GREEN if last.reproj_mean_px < THRESH_GOOD else
                 self._AMBER if last.reproj_mean_px < THRESH_ACCEPT else self._RED)
            self._ann(ax, f"Measured: {last.reproj_mean_px:.2f} px\n"
                          f"Max:      {last.reproj_max_px:.2f} px\n"
                          f"Predicted:{last.predicted_rss_px:.2f} px", color=c)
        else:
            self._ann(ax, f"No keypoints loaded\n"
                          f"Predicted RSS: {last.predicted_rss_px:.2f} px",
                      color=self._TICK)

        ax.legend(fontsize=7, loc="upper left", labelcolor="white",
                  facecolor=self._BG, edgecolor=self._SPINE)
        ax.grid(axis="y", alpha=0.12, color=self._SPINE)

    # ------------------------------------------------------------------
    # Panel 4 – Error budget snapshot (current frame)
    # ------------------------------------------------------------------
    def _panel4_budget(self, ax) -> None:
        ax.set_title("Error Budget · Current Frame Snapshot", fontsize=9,
                     fontweight="bold", pad=4)

        last = self.history[-1]
        K    = None  # focal info already baked into predicted_rss_px

        labels  = ["SVD\n(mm)", "Trans\njitter (mm)", "Rot\njitter (°)",
                   "Reproj\nmean (px)", "Predicted\nRSS (px)"]
        values  = [last.svd_mean_mm, last.trans_jitter_mm, last.rot_jitter_deg,
                   last.reproj_mean_px if last.reproj_mean_px is not None else 0.0,
                   last.predicted_rss_px]
        colors  = [self._BLUE, self._PURP, self._AMBER, self._RED, self._WHITE]

        bars = ax.bar(labels, values, color=colors, alpha=0.75, edgecolor=self._SPINE, zorder=3)
        ax.set_ylabel("Value (see label for unit)", fontsize=8)
        ax.grid(axis="y", alpha=0.15, color=self._SPINE, zorder=0)

        for bar, val in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + max(values) * 0.02,
                f"{val:.3f}",
                ha="center", va="bottom", fontsize=7.5, color="white",
            )

        # Colour-code reproj bar
        if last.reproj_mean_px is not None:
            c = (self._GREEN if last.reproj_mean_px < THRESH_GOOD else
                 self._AMBER if last.reproj_mean_px < THRESH_ACCEPT else self._RED)
            status = ("GOOD" if last.reproj_mean_px < THRESH_GOOD else
                      "ACCEPT" if last.reproj_mean_px < THRESH_ACCEPT else "POOR")
            self._ann(ax, f"Frame {last.frame_id}\nStatus: {status}", color=c)
        else:
            self._ann(ax, f"Frame {last.frame_id}\n(no keypoints)", color=self._TICK)