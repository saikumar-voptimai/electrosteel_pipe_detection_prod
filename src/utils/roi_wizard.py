from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import cv2
import numpy as np
import yaml

Point = Tuple[int, int]

Polygon4 = List[Point] # List of 4 (x,y) points

@dataclass
class ROISpec:
  """
  Specification for a single ROI. Name  of the ROI, color for display, and help text.
  """
  name: str
  color_bgr: Tuple[int, int, int] # since cv2 uses BGR
  help_text: str

@dataclass
class ROIWizardConfig:
  """
  Configuration for the ROI Redraw Wizard.
  """
  video_source: int | str = 0 # default to webcam
  rois_path: str = "config/rois.yaml"
  window_name: str = "ROI Redraw Wizard - q=quit, u=undo, n=next, c=clear, ENTER=submit"
  # Optional override: if provided, the wizard will use this frame as the base image
  # instead of opening the video_source. Useful for live GigE pipelines.
  first_frame: Optional[np.ndarray] = None
  alpha_fill: float = 0.22
  edge_thickness: int = 4
  point_radius: int = 5
  max_display_width: int = 1280
  max_display_height: int = 720
  warmup_frames: int = 10 # grab a stable camera frame or skip using next
  font_scale: float = 0.7
  font_thickness: int = 2


class ROIRedrawWizard:
  """
  Interactive wizard to redraw ROIs on a video frame and save to YAML file.
  - captures first usable frame from the camera/video_source
  - collects 4-point polygons for each ROI in specs order
  - displays completed polygons with transparent fill + thick edges + labels
  - saves to YAML (pixel coordinates in ORIGINAL frame space)
  """
  def __init__(self, cfg: ROIWizardConfig, specs: List[ROISpec]) -> None:
    self.cfg = cfg
    self.specs = specs

    self._frame_orig: Optional[np.ndarray] = None
    self._frame_disp: Optional[np.ndarray] = None
    self._scale: float = 1.0 # disp = orig * scale

    self._idx: int = 0
    self._current_points_disp: List[Point] = []

    # Stored ROIs (orig coordinates) + *(disp coordinates for visualization)
    self._rois_orig: Dict[str, Polygon4] = {}
    self._rois_disp: Dict[str, Polygon4] = {}

  def run(self) -> None:
    """
    Runs the ROI redraw wizard.
    """
    frame = self._capture_first_frame()
    self._set_display_frame(frame)

    cv2.namedWindow(self.cfg.window_name, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(self.cfg.window_name, self._on_mouse)

    while True:
      vis = self._render()
      cv2.imshow(self.cfg.window_name, vis)
      key = cv2.waitKey(20) & 0xFF

      if key == ord('q'):  # quit
        cv2.destroyWindow(self.cfg.window_name)
        raise RuntimeError("ROI redraw cancelled by user (q). No changes saved.")
      
      if key == ord('u'):  # undo
        if self._current_points_disp:
          self._current_points_disp.pop()
      
      if key == ord("c"): # clear current
        self._current_points_disp.clear()
      
      if key in (13, 10): # enter key
        if len(self._current_points_disp) == 4:
          self._accept_current_roi()
          if self._idx >= len(self.specs):
            break # All rois done
    cv2.destroyWindow(self.cfg.window_name)
    return self._rois_orig
  
  def save(self) -> None:
    """
    Saves the current ROIs to a YAML file.
    """
    rois = self.run()
    out_path = Path(self.cfg.rois_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Save simple mapping: roi_name ->[[x,y],...]
    payload = {name: [[int(x), int(y)] for (x, y) in pts] for name, pts in rois.items()}
    with out_path.open("w", encoding="utf-8") as f:
      yaml.safe_dump(payload, f, sort_keys=False)

  # Internals
  def _capture_first_frame(self) -> np.ndarray:
    """
    Captures first usable frame from video source. Can skip ahead on 'n' key.
    """
    if self.cfg.first_frame is not None:
      return self.cfg.first_frame

    cap = cv2.VideoCapture(self.cfg.video_source)
    if not cap.isOpened():
      raise RuntimeError(f"Cannot open video source: {self.cfg.video_source}")
    
    frame: Optional[np.ndarray] = None
    for _ in range(max(1, self.cfg.warmup_frames)):
      ok, frm = cap.read()
      if ok:
        cv2.imshow(self.cfg.window_name, frm)
        key = cv2.waitKey(20) & 0xFF
        #TODO: User can press 'n' to skip ahead to next frame if first is bad

        if key == ord('n'):  # skip ahead
          continue
        frame = frm
        cv2.destroyWindow(self.cfg.window_name)
    return frame
  
  def _set_display_frame(self, frame_orig: np.ndarray) -> None:
    self._frame_orig = frame_orig
    h, w = frame_orig.shape[:2]
    
    scale_w = self.cfg.max_display_width / float(w)
    scale_h = self.cfg.max_display_height / float(h)
    scale = min(1.0, scale_w, scale_h)

    self._scale = scale
    if scale < 1.0:
      new_w = int(w * scale)
      new_h = int(h * scale)
      self._frame_disp = cv2.resize(frame_orig, (new_w, new_h), interpolation=cv2.INTER_AREA)
    else:
      self._frame_disp = frame_orig.copy()

  def _on_mouse(self, event: int, x: int, y: int, flags: int, param) -> None:
    """
    Mouse callback to collect points for current ROI.
    """
    if event != cv2.EVENT_LBUTTONDOWN:
      return
    
    # Accept only 4 points
    if len(self._current_points_disp) >= 4:
      return
    self._current_points_disp.append((int(x), int(y)))

  def _accept_current_roi(self) -> None:
    """
    Accept the current ROI points and moves to next ROI.
    """
    spec = self.specs[self._idx]
    pts_disp = list(self._current_points_disp)
    pts_orig = [self._disp_to_orig(p) for p in pts_disp]

    self._rois_disp[spec.name] = pts_disp
    self._rois_orig[spec.name] = pts_orig

    self._current_points_disp.clear()
    self._idx += 1
  
  def _disp_to_orig(self, p: Point) -> Point:
    """
    Converts a point from display coordinates to original frame coordinates.
    """
    if self._scale <= 0:
      return p
    ox = int(round(p[0] / self._scale))
    oy = int(round(p[1] / self._scale))
    return (ox, oy)
  
  def _render(self) -> np.ndarray:
    """
    Renders the current state of the ROI selection interface.
    """
    assert self._frame_disp is not None
    base = self._frame_disp.copy()
    overlay = base.copy()

    # Draw completed ROIs with transparent fill
    for spec in self.specs:
      if spec.name not in self._rois_disp:
        continue
      pts = np.array(self._rois_disp[spec.name], dtype=np.int32)
      cv2.fillPoly(overlay, [pts], spec.color_bgr)

    # Blend overlay
    cv2.addWeighted(overlay, self.cfg.alpha_fill, base, 1.0 - self.cfg.alpha_fill, 0, base)

    # Thick edges + labels
    for spec in self.specs:
      if spec.name not in self._rois_disp:
        continue
      pts = np.array(self._rois_disp[spec.name], dtype=np.int32)
      cv2.polylines(base, [pts], isClosed=True, color=spec.color_bgr, thickness=self.cfg.edge_thickness)

      # Label near first point
      x0, y0 = self._rois_disp[spec.name][0]
      cv2.putText(base, 
                 spec.name, 
                 (x0 + 6, y0 - 6),
                 cv2.FONT_HERSHEY_SIMPLEX,
                 self.cfg.font_scale,
                 spec.color_bgr,
                 self.cfg.font_thickness
      )
    # Draw current ROI points/lines
    if self._idx < len(self.specs):
      spec = self.specs[self._idx]
      pts = self._current_points_disp

      for (px, py) in pts:
        cv2.circle(base, (px, py), self.cfg.point_radius, spec.color_bgr, -1)

      if len(pts) >= 2:
        # Show as closed too (preview)
        pts_np = np.array(pts, dtype=np.int32)
        cv2.polylines(base, [pts_np], isClosed=True, color=spec.color_bgr, thickness=self.cfg.edge_thickness)

      # Helper text
      self._draw_help_text(base, spec)
    
    # Progress footer
    self._draw_progress(base)
    return base
  
  def _draw_help_text(self, img: np.ndarray, spec: ROISpec) -> None:
    """
    Draws help text for current ROI.
    """
    lines = [
      f"Draw ROI: {spec.name}",
      spec.help_text,
      "",
      "Controls: LeftClick=point | u=undo | c=clear | ENTER=accept (needs 4 points) | q=quit | n=next"
    ]
    y = 30
    for line in lines:
      cv2.putText(img, line, (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
      y += 28

  def _draw_progress(self, img: np.ndarray) -> None:
    """
    Draws progress footer.
    """
    done = self._idx
    total = len(self.specs)
    msg = f"Progress: {done} / {total} ROIs completed"
    h, _w = img.shape[:2]
    cv2.putText(img, msg, (20, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)


def default_roi_specs() -> List[ROISpec]:
  """
  Specifies the default ROIs needed for the pipe tracking system.
  """
  #TODO: Use enums or constants for ROI names
  return [
    ROISpec("roi_loadcell", (0,255, 255), "Loadcell zone: trigger PLC when ELIGIBLE pipe enters this ROI."),
    ROISpec("roi_caster5_origin", (255, 255, 0), "Caster 5 origin zone: includes, caster5, trolley-end-area upto gate1."),
    ROISpec("roi_left_origin",     (0, 165, 255), "Left origin/exclusion: pipes here are ignored. If originating here origin='other'."),
    ROISpec("roi_right_origin",    (255, 0, 255), "Right origin/exclusion: pipes here are ignored. If originating here origin='other'."),
    ROISpec("roi_safety_critical", (0, 0, 255),   "Safety zone: human detection inside this ROI is flagged/used for occlusion checks."),
    ROISpec("roi_gate1_closed",    (255, 0, 0),   "Gate1 closed reference ROI (4-pt box). Used as baseline for geometry checks."),
    ROISpec("roi_gate1_open",      (0, 255, 0),   "Gate1 open ROI tall-narrow: gate should land here when open."),
    ROISpec("roi_gate2_closed",    (128, 0, 0),   "Gate2 closed reference ROI (4-pt box). Used as baseline for geometry checks."),
    ROISpec("roi_gate2_open",      (0, 128, 0),   "Gate2 open ROI tall-narrow: gate should land here when open.")
    ]

                                                   

