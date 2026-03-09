from __future__ import annotations

"""
Video Stitcher App 

Highlights
- Loads MP4 clips from a folder.
- Extracts first/last frame thumbnails and lightweight frame signatures.
- Matches end -> start frames to suggest sequence links.
- Can guess a start clip when none is known.
- Supports batch selection rules:
    - Fixed batch size (1-4)
    - Time-gap batching
- Lets the user compare a clip's last frame against top candidate next clips.
- Timeline can be auto-built, reordered, moved to top/bottom, cleared, and saved.
- Project settings/timeline can be saved to and loaded from JSON.
- Stitches chosen clips using ffmpeg concat with stream-copy fallback to re-encode.
- Optional looped backing music can be mixed under the final video.

Dependencies:
    pip install customtkinter pillow opencv-python
Also requires ffmpeg in PATH.
"""

import json
import queue
import threading
from concurrent.futures import ThreadPoolExecutor
import os
import shutil
import subprocess
import tempfile
import time
import ctypes
from ctypes import wintypes
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import customtkinter as ctk
import tkinter as tk
from tkinter import Canvas, filedialog, messagebox

try:
    import cv2
except Exception:  # pragma: no cover
    cv2 = None

try:
    from PIL import Image, ImageTk, ImageOps
except Exception:  # pragma: no cover
    Image = None
    ImageTk = None
    ImageOps = None


APP_TITLE = "Video Stitcher"
THUMB_SIZE = (150, 84)
COMPARE_SIZE = (330, 186)
PREVIEW_MAX = (720, 405)
TIMELINE_THUMB = (120, 68)
TIMELINE_SLOT_WIDTH = 150
CARD_WIDTH = 360
CARD_HEIGHT = 290
CARDS_PER_ROW = 3
CACHE_DIR_NAME = ".video_stitcher_cache"
PROJECT_FILE_NAME = "video_stitcher_project.json"
APP_STATE_FILE = Path.home() / ".video_stitcher_app_state.json"

ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")


@dataclass(slots=True)
class FrameSignature:
    small_gray: List[int]
    hist: List[float]
    edge_density: float


@dataclass(slots=True)
class VideoClip:
    path: Path
    created_ts: float
    modified_ts: float
    first_frame_path: Path
    last_frame_path: Path
    duration_sec: float = 0.0
    fps: float = 0.0
    width: int = 0
    height: int = 0
    first_sig: Optional[FrameSignature] = None
    last_sig: Optional[FrameSignature] = None
    batch_group: int = -1
    best_next_name: str = ""
    best_next_score: float = 0.0
    best_prev_name: str = ""
    best_prev_score: float = 0.0

    @property
    def name(self) -> str:
        return self.path.name

    @property
    def created_dt(self) -> datetime:
        return datetime.fromtimestamp(self.created_ts)

    @property
    def modified_dt(self) -> datetime:
        return datetime.fromtimestamp(self.modified_ts)


@dataclass(slots=True)
class LinkScore:
    src_name: str
    dst_name: str
    score: float


class VideoAnalyzer:
    def __init__(self) -> None:
        self.cv2_ok = cv2 is not None and Image is not None

    def ensure_dependencies(self) -> None:
        if not self.cv2_ok:
            raise RuntimeError(
                "OpenCV and Pillow are required. Install: pip install opencv-python pillow customtkinter"
            )

    def extract_clip_data(self, video_path: Path, cache_dir: Path) -> Optional[VideoClip]:
        self.ensure_dependencies()
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            return None

        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        duration = (frame_count / fps) if fps and frame_count else 0.0

        first_frame = self._read_frame(cap, 0)
        last_frame = self._read_frame(cap, max(frame_count - 1, 0))
        cap.release()

        if first_frame is None or last_frame is None:
            return None

        safe_stem = "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in video_path.stem)
        first_frame_path = cache_dir / f"{safe_stem}__first.jpg"
        last_frame_path = cache_dir / f"{safe_stem}__last.jpg"
        cv2.imwrite(str(first_frame_path), first_frame)
        cv2.imwrite(str(last_frame_path), last_frame)

        created_ts = self._get_created_ts(video_path)
        modified_ts = self._get_modified_ts(video_path)

        first_sig = self.make_signature(first_frame)
        last_sig = self.make_signature(last_frame)

        return VideoClip(
            path=video_path,
            created_ts=created_ts,
            modified_ts=modified_ts,
            first_frame_path=first_frame_path,
            last_frame_path=last_frame_path,
            duration_sec=duration,
            fps=fps,
            width=width,
            height=height,
            first_sig=first_sig,
            last_sig=last_sig,
        )

    def _read_frame(self, cap, index: int):
        cap.set(cv2.CAP_PROP_POS_FRAMES, index)
        ok, frame = cap.read()
        return frame if ok else None

    def _get_created_ts(self, path: Path) -> float:
        if os.name == "nt":
            win_times = self._get_windows_file_times(path)
            if win_times is not None:
                return win_times[0]
        stat = path.stat()
        if os.name == "nt":
            return stat.st_ctime
        return min(stat.st_ctime, stat.st_mtime)

    def _get_modified_ts(self, path: Path) -> float:
        if os.name == "nt":
            win_times = self._get_windows_file_times(path)
            if win_times is not None:
                return win_times[1]
        return path.stat().st_mtime

    def _get_windows_file_times(self, path: Path) -> Optional[Tuple[float, float]]:
        if os.name != "nt":
            return None
        try:
            class FILETIME(ctypes.Structure):
                _fields_ = [("dwLowDateTime", wintypes.DWORD), ("dwHighDateTime", wintypes.DWORD)]

            class WIN32_FILE_ATTRIBUTE_DATA(ctypes.Structure):
                _fields_ = [
                    ("dwFileAttributes", wintypes.DWORD),
                    ("ftCreationTime", FILETIME),
                    ("ftLastAccessTime", FILETIME),
                    ("ftLastWriteTime", FILETIME),
                    ("nFileSizeHigh", wintypes.DWORD),
                    ("nFileSizeLow", wintypes.DWORD),
                ]

            GetFileAttributesExW = ctypes.windll.kernel32.GetFileAttributesExW
            GetFileAttributesExW.argtypes = [wintypes.LPCWSTR, wintypes.INT, ctypes.c_void_p]
            GetFileAttributesExW.restype = wintypes.BOOL

            data = WIN32_FILE_ATTRIBUTE_DATA()
            ok = GetFileAttributesExW(str(path), 0, ctypes.byref(data))
            if not ok:
                return None

            def filetime_to_ts(ft: FILETIME) -> float:
                ticks = (ft.dwHighDateTime << 32) | ft.dwLowDateTime
                if ticks <= 0:
                    return 0.0
                return max(0.0, (ticks - 116444736000000000) / 10000000.0)

            created = filetime_to_ts(data.ftCreationTime)
            modified = filetime_to_ts(data.ftLastWriteTime)
            return created, modified
        except Exception:
            return None

    def make_signature(self, bgr_frame) -> FrameSignature:
        gray = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2GRAY)
        resized = cv2.resize(gray, (16, 16), interpolation=cv2.INTER_AREA)
        hist = cv2.calcHist([gray], [0], None, [16], [0, 256]).flatten()
        total = float(hist.sum() or 1.0)
        hist = [float(v / total) for v in hist]
        edges = cv2.Canny(gray, 80, 160)
        edge_density = float(edges.mean() / 255.0)
        return FrameSignature(
            small_gray=[int(v) for v in resized.flatten().tolist()],
            hist=hist,
            edge_density=edge_density,
        )

    def compare(self, a: FrameSignature, b: FrameSignature) -> float:
        pix_delta = sum(abs(x - y) for x, y in zip(a.small_gray, b.small_gray)) / (16 * 16 * 255)
        hist_delta = sum(abs(x - y) for x, y in zip(a.hist, b.hist)) / 2.0
        edge_delta = abs(a.edge_density - b.edge_density)
        score = 1.0 - ((pix_delta * 0.65) + (hist_delta * 0.25) + (edge_delta * 0.10))
        return max(0.0, min(1.0, score))


class FFmpegStitcher:
    def ffmpeg_available(self) -> bool:
        return shutil.which("ffmpeg") is not None

    def ffprobe_available(self) -> bool:
        return shutil.which("ffprobe") is not None

    def has_audio_stream(self, video_path: Path) -> bool:
        if not self.ffprobe_available():
            return False
        cmd = [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "a:0",
            "-show_entries",
            "stream=codec_type",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            str(video_path),
        ]
        try:
            result = subprocess.run(cmd, capture_output=True, text=True)
            return result.returncode == 0 and "audio" in (result.stdout or "").lower()
        except Exception:
            return False

    def _concat_to_video(self, clips: List[VideoClip], output_path: Path) -> Tuple[bool, str]:
        if not clips:
            return False, "No clips selected"
        if not self.ffmpeg_available():
            return False, "ffmpeg was not found in PATH"

        output_path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False, encoding="utf-8") as tmp:
            concat_file = Path(tmp.name)
            for clip in clips:
                safe_path = str(clip.path).replace("\\", "/").replace("'", r"'\''")
                tmp.write(f"file '{safe_path}'\n")

        try:
            copy_cmd = [
                "ffmpeg",
                "-y",
                "-f",
                "concat",
                "-safe",
                "0",
                "-i",
                str(concat_file),
                "-c",
                "copy",
                str(output_path),
            ]
            copy_result = subprocess.run(copy_cmd, capture_output=True, text=True)
            if copy_result.returncode == 0:
                return True, str(output_path)

            transcode_cmd = [
                "ffmpeg",
                "-y",
                "-f",
                "concat",
                "-safe",
                "0",
                "-i",
                str(concat_file),
                "-c:v",
                "libx264",
                "-preset",
                "fast",
                "-crf",
                "18",
                "-pix_fmt",
                "yuv420p",
                "-c:a",
                "aac",
                "-b:a",
                "192k",
                str(output_path),
            ]
            transcode_result = subprocess.run(transcode_cmd, capture_output=True, text=True)
            if transcode_result.returncode == 0:
                return True, str(output_path)
            err = transcode_result.stderr or transcode_result.stdout or copy_result.stderr or "Unknown ffmpeg failure"
            return False, err[:2000]
        finally:
            try:
                concat_file.unlink(missing_ok=True)
            except Exception:
                pass

    def add_backing_music(
        self,
        input_video: Path,
        output_path: Path,
        music_path: Path,
        volume: float = 0.12,
    ) -> Tuple[bool, str]:
        if not self.ffmpeg_available():
            return False, "ffmpeg was not found in PATH"
        if not music_path.exists():
            return False, "Selected music file does not exist."

        volume = max(0.0, min(float(volume), 1.0))
        has_audio = self.has_audio_stream(input_video)

        if has_audio:
            cmd = [
                "ffmpeg",
                "-y",
                "-i",
                str(input_video),
                "-stream_loop",
                "-1",
                "-i",
                str(music_path),
                "-filter_complex",
                f"[0:a]aresample=async=1:first_pts=0[a0];[1:a]volume={volume:.3f},aresample=async=1:first_pts=0[a1];[a0][a1]amix=inputs=2:duration=first:dropout_transition=2[aout]",
                "-map",
                "0:v:0",
                "-map",
                "[aout]",
                "-c:v",
                "copy",
                "-c:a",
                "aac",
                "-b:a",
                "192k",
                "-shortest",
                str(output_path),
            ]
        else:
            cmd = [
                "ffmpeg",
                "-y",
                "-i",
                str(input_video),
                "-stream_loop",
                "-1",
                "-i",
                str(music_path),
                "-filter_complex",
                f"[1:a]volume={volume:.3f}[aout]",
                "-map",
                "0:v:0",
                "-map",
                "[aout]",
                "-c:v",
                "copy",
                "-c:a",
                "aac",
                "-b:a",
                "192k",
                "-shortest",
                str(output_path),
            ]

        try:
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode == 0:
                return True, str(output_path)
            err = result.stderr or result.stdout or "Failed while adding backing music."
            return False, err[:2000]
        except Exception as exc:
            return False, str(exc)

    def stitch(
        self,
        clips: List[VideoClip],
        output_path: Path,
        music_path: Optional[Path] = None,
        use_music: bool = False,
        music_volume: float = 0.12,
    ) -> Tuple[bool, str]:
        if not use_music or music_path is None:
            return self._concat_to_video(clips, output_path)

        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
            temp_video = Path(tmp.name)

        try:
            ok, result = self._concat_to_video(clips, temp_video)
            if not ok:
                return False, result
            return self.add_backing_music(temp_video, output_path, music_path, volume=music_volume)
        finally:
            try:
                temp_video.unlink(missing_ok=True)
            except Exception:
                pass

class SequenceBuilder:
    def __init__(self, analyzer: VideoAnalyzer) -> None:
        self.analyzer = analyzer

    def compute_links(self, clips: List[VideoClip]) -> List[LinkScore]:
        links: List[LinkScore] = []
        for src in clips:
            src.best_next_name = ""
            src.best_next_score = 0.0
            src.best_prev_name = ""
            src.best_prev_score = 0.0

        for src in clips:
            if src.last_sig is None:
                continue
            for dst in clips:
                if src.path == dst.path or dst.first_sig is None:
                    continue
                score = self.analyzer.compare(src.last_sig, dst.first_sig)
                links.append(LinkScore(src.name, dst.name, score))
                if score > src.best_next_score:
                    src.best_next_score = score
                    src.best_next_name = dst.name
                if score > dst.best_prev_score:
                    dst.best_prev_score = score
                    dst.best_prev_name = src.name
        return sorted(links, key=lambda x: x.score, reverse=True)

    def assign_batch_groups(
        self,
        clips: List[VideoClip],
        sort_key: str,
        mode: str = "size",
        max_size: int = 4,
        gap_seconds: int = 20,
    ) -> None:
        ordered = sorted(clips, key=lambda c: c.created_ts if sort_key == "created" else c.modified_ts)
        if not ordered:
            return

        if mode == "timegap":
            group_id = 0
            prev_ts: Optional[float] = None
            for clip in ordered:
                ts = clip.created_ts if sort_key == "created" else clip.modified_ts
                if prev_ts is not None and (ts - prev_ts) > gap_seconds:
                    group_id += 1
                clip.batch_group = group_id
                prev_ts = ts
            return

        if mode == "smart":
            group_id = 0
            prev_ts: Optional[float] = None
            group_count = 0
            for clip in ordered:
                ts = clip.created_ts if sort_key == "created" else clip.modified_ts
                start_new = False
                if prev_ts is not None and (ts - prev_ts) > gap_seconds:
                    start_new = True
                if group_count >= max_size:
                    start_new = True
                if start_new:
                    group_id += 1
                    group_count = 0
                clip.batch_group = group_id
                prev_ts = ts
                group_count += 1
            return

        for idx, clip in enumerate(ordered):
            clip.batch_group = idx // max_size

    def guess_start(self, clips: List[VideoClip]) -> Optional[VideoClip]:
        if not clips:
            return None
        ranked = sorted(
            clips,
            key=lambda c: (c.best_next_score - c.best_prev_score, -c.best_prev_score, -(c.duration_sec), c.created_ts),
            reverse=True,
        )
        return ranked[0] if ranked else None

    def top_candidates_for(
        self,
        clips: List[VideoClip],
        src: VideoClip,
        min_score: float = 0.0,
        exclude_used: Optional[set[str]] = None,
        enforce_one_per_batch: bool = False,
        used_batches: Optional[set[int]] = None,
        limit: int = 5,
    ) -> List[Tuple[float, VideoClip]]:
        exclude_used = exclude_used or set()
        used_batches = used_batches or set()
        candidates: List[Tuple[float, VideoClip]] = []
        if src.last_sig is None:
            return candidates

        for other in clips:
            if other.name == src.name:
                continue
            if other.name in exclude_used:
                continue
            if enforce_one_per_batch and other.batch_group in used_batches:
                continue
            if other.first_sig is None:
                continue
            score = self.analyzer.compare(src.last_sig, other.first_sig)
            if score >= min_score:
                candidates.append((score, other))
        candidates.sort(key=lambda x: (x[0], -(x[1].created_ts)), reverse=True)
        return candidates[:limit]

    def build_greedy_sequence(
        self,
        clips: List[VideoClip],
        start_clip: Optional[VideoClip] = None,
        min_score: float = 0.78,
        enforce_one_per_batch: bool = True,
    ) -> List[VideoClip]:
        if not clips:
            return []

        used: set[str] = set()
        used_batches: set[int] = set()
        sequence: List[VideoClip] = []

        def can_use(clip: VideoClip) -> bool:
            if clip.name in used:
                return False
            if enforce_one_per_batch and clip.batch_group in used_batches:
                return False
            return True

        def candidate_starts() -> List[VideoClip]:
            ranked = sorted(
                clips,
                key=lambda c: (c.best_next_score - c.best_prev_score, -c.best_prev_score, c.created_ts),
                reverse=True,
            )
            return [c for c in ranked if can_use(c)]

        current = start_clip if start_clip and can_use(start_clip) else None
        while len(used) < len(clips):
            if current is None:
                starts = candidate_starts()
                if not starts:
                    break
                current = starts[0]

            while current and can_use(current):
                sequence.append(current)
                used.add(current.name)
                if enforce_one_per_batch and current.batch_group >= 0:
                    used_batches.add(current.batch_group)

                candidates = self.top_candidates_for(
                    clips=clips,
                    src=current,
                    min_score=min_score,
                    exclude_used=used,
                    enforce_one_per_batch=enforce_one_per_batch,
                    used_batches=used_batches,
                    limit=10,
                )
                current = candidates[0][1] if candidates else None

            current = None

        return sequence


class VideoCard(ctk.CTkFrame):
    def __init__(self, parent, app: "VideoStitcherApp", clip: VideoClip):
        super().__init__(parent, width=CARD_WIDTH, height=CARD_HEIGHT, corner_radius=12, border_width=1)
        self.grid_propagate(False)
        self.app = app
        self.clip = clip

        title = ctk.CTkLabel(self, text=clip.name, font=("Segoe UI", 14, "bold"), wraplength=CARD_WIDTH - 24)
        title.pack(pady=(10, 4))

        meta = (
            f"Created: {clip.created_dt:%Y-%m-%d %H:%M:%S}\n"
            f"Modified: {clip.modified_dt:%Y-%m-%d %H:%M:%S}\n"
            f"Batch: {clip.batch_group + 1} | {clip.width}x{clip.height} | {clip.duration_sec:.2f}s"
        )
        ctk.CTkLabel(self, text=meta, font=("Segoe UI", 11), justify="left").pack(pady=(0, 8))

        thumb_row = ctk.CTkFrame(self, fg_color="transparent")
        thumb_row.pack(pady=(0, 8))
        self._make_thumb(thumb_row, clip.first_frame_path, "Start")
        self._make_thumb(thumb_row, clip.last_frame_path, "End")

        best_text = (
            f"Next: {clip.best_next_name or '-'} ({clip.best_next_score:.3f})\n"
            f"Prev: {clip.best_prev_name or '-'} ({clip.best_prev_score:.3f})"
        )
        ctk.CTkLabel(self, text=best_text, font=("Consolas", 11), justify="left").pack(pady=(0, 8))

        btn_row1 = ctk.CTkFrame(self, fg_color="transparent")
        btn_row1.pack(pady=(0, 4))
        ctk.CTkButton(btn_row1, text="Add", width=70, command=lambda: self.app.add_to_timeline(self.clip)).pack(side="left", padx=4)
        ctk.CTkButton(
            btn_row1,
            text="Set Start",
            width=90,
            fg_color="#0f766e",
            hover_color="#115e59",
            command=lambda: self.app.set_manual_start(self.clip),
        ).pack(side="left", padx=4)
        ctk.CTkButton(btn_row1, text="Preview", width=80, command=lambda: self.app.preview_clip(self.clip, use_video=True)).pack(side="left", padx=4)

        btn_row2 = ctk.CTkFrame(self, fg_color="transparent")
        btn_row2.pack(pady=(0, 8))
        ctk.CTkButton(btn_row2, text="Compare", width=90, command=lambda: self.app.show_match_candidates(self.clip)).pack(side="left", padx=4)
        ctk.CTkButton(btn_row2, text="Pick Only", width=90, command=lambda: self.app.replace_batch_choice(self.clip)).pack(side="left", padx=4)

    def _make_thumb(self, parent, image_path: Path, label: str) -> None:
        wrap = ctk.CTkFrame(parent, corner_radius=8)
        wrap.pack(side="left", padx=8)
        img = self.app.load_photoimage(image_path, THUMB_SIZE)
        tk_label = ctk.CTkLabel(wrap, text=label, image=img, compound="top")
        tk_label.image = img
        tk_label.pack(padx=4, pady=4)
        tk_label.bind("<Button-1>", lambda _e: self.app.preview_image(image_path, f"{self.clip.name} - {label}"))


class TimelineChip(ctk.CTkFrame):
    def __init__(self, parent, app: "VideoStitcherApp", clip: VideoClip, index: int):
        border_color = "#22c55e" if app.timeline_selected_index == index else None
        super().__init__(parent, corner_radius=8, border_width=2 if app.timeline_selected_index == index else 1, border_color=border_color)
        self.app = app
        self.clip = clip
        self.index = index
        img = app.load_photoimage(clip.first_frame_path, TIMELINE_THUMB)
        lbl = ctk.CTkLabel(
            self,
            text=f"{index + 1}. {clip.name}",
            image=img,
            compound="top",
            wraplength=130,
            cursor="hand2",
        )
        lbl.image = img
        lbl.pack(padx=6, pady=6)

        for widget in (self, lbl):
            widget.bind("<ButtonPress-1>", lambda e, i=index: app.start_timeline_drag(i, e))
            widget.bind("<B1-Motion>", app.drag_timeline_motion)
            widget.bind("<ButtonRelease-1>", app.end_timeline_drag)

        btn_row1 = ctk.CTkFrame(self, fg_color="transparent")
        btn_row1.pack(pady=(0, 4))
        ctk.CTkButton(btn_row1, text="⏮", width=32, command=lambda: app.move_timeline_to(index, 0)).pack(side="left", padx=2)
        ctk.CTkButton(btn_row1, text="↑", width=32, command=lambda: app.move_timeline(index, -1)).pack(side="left", padx=2)
        ctk.CTkButton(btn_row1, text="↓", width=32, command=lambda: app.move_timeline(index, 1)).pack(side="left", padx=2)
        ctk.CTkButton(btn_row1, text="⏭", width=32, command=lambda: app.move_timeline_to(index, len(app.timeline) - 1)).pack(side="left", padx=2)

        btn_row2 = ctk.CTkFrame(self, fg_color="transparent")
        btn_row2.pack(pady=(0, 6))
        ctk.CTkButton(btn_row2, text="Sel", width=42, command=lambda: app.select_timeline_index(index)).pack(side="left", padx=2)
        ctk.CTkButton(btn_row2, text="View", width=48, command=lambda: app.preview_clip(clip, use_video=False)).pack(side="left", padx=2)
        ctk.CTkButton(
            btn_row2,
            text="✕",
            width=36,
            fg_color="#dc2626",
            hover_color="#b91c1c",
            command=lambda: app.remove_from_timeline(index),
        ).pack(side="left", padx=2)


class VideoStitcherApp:
    def __init__(self, root: ctk.CTk):
        self.root = root
        self.root.title(APP_TITLE)
        self.root.geometry("1720x1020")
        self.root.minsize(1360, 840)

        self.analyzer = VideoAnalyzer()
        self.builder = SequenceBuilder(self.analyzer)
        self.stitcher = FFmpegStitcher()

        self.current_dir: Optional[Path] = None
        self.cache_dir: Optional[Path] = None
        self.clips: List[VideoClip] = []
        self.timeline: List[VideoClip] = []
        self.manual_start: Optional[VideoClip] = None
        self.selected_compare_source: Optional[VideoClip] = None
        self.photo_cache: Dict[Tuple[str, Tuple[int, int]], object] = {}
        self.preview_cap = None
        self.preview_job = None
        self.preview_paused = False
        self.current_preview_clip: Optional[VideoClip] = None
        self.timeline_selected_index: Optional[int] = None
        self.drag_timeline_index: Optional[int] = None
        self.drag_target_index: Optional[int] = None
        self.load_queue: "queue.Queue[Tuple[str, object]]" = queue.Queue()
        self.load_thread: Optional[threading.Thread] = None
        self.load_executor: Optional[ThreadPoolExecutor] = None
        self.loading_directory = False
        self.loading_total = 0
        self.loading_done = 0
        self._active_load_id = 0

        self.sort_var = ctk.StringVar(value="created")
        self.min_score_var = ctk.DoubleVar(value=0.82)
        self.one_per_batch_var = ctk.BooleanVar(value=False)
        self.status_var = ctk.StringVar(value="Ready")
        self.load_detail_var = ctk.StringVar(value="Idle")
        self.current_file_var = ctk.StringVar(value="")
        self.load_progress_var = ctk.DoubleVar(value=0.0)
        self.output_var = ctk.StringVar(value="stitched_output.mp4")
        self.music_enabled_var = ctk.BooleanVar(value=False)
        self.music_path_var = ctk.StringVar(value="")
        self.music_volume_var = ctk.DoubleVar(value=0.12)
        self.batch_size_var = ctk.IntVar(value=4)
        self.batch_mode_var = ctk.StringVar(value="smart")
        self.batch_gap_var = ctk.IntVar(value=3)

        self._build_ui()
        self.load_app_state()

    def _build_ui(self) -> None:
        outer = ctk.CTkFrame(self.root, fg_color="transparent")
        outer.pack(fill="both", expand=True, padx=12, pady=12)

        top = ctk.CTkFrame(outer)
        top.pack(fill="x", pady=(0, 10))

        self.dir_entry = ctk.CTkEntry(top, placeholder_text="Choose folder with mp4 clips...")
        self.dir_entry.pack(side="left", fill="x", expand=True, padx=10, pady=10)
        self.browse_btn = ctk.CTkButton(top, text="Browse", command=self.browse_directory)
        self.browse_btn.pack(side="left", padx=6)
        self.rescan_btn = ctk.CTkButton(top, text="Rescan", command=self.start_load_directory)
        self.rescan_btn.pack(side="left", padx=6)
        ctk.CTkButton(top, text="Auto Build", fg_color="#0f766e", hover_color="#115e59", command=self.auto_build_sequence).pack(side="left", padx=6)
        ctk.CTkButton(top, text="Save Project", command=self.save_project).pack(side="left", padx=6)
        ctk.CTkButton(top, text="Load Project", command=self.load_project).pack(side="left", padx=6)
        ctk.CTkButton(top, text="Stitch", fg_color="#7c3aed", hover_color="#6d28d9", command=self.stitch_timeline).pack(side="left", padx=6)

        options = ctk.CTkFrame(outer)
        options.pack(fill="x", pady=(0, 10))
        ctk.CTkLabel(options, text="Sort by:").pack(side="left", padx=(10, 4), pady=10)
        ctk.CTkOptionMenu(options, variable=self.sort_var, values=["created", "modified"], command=lambda _v: self.resort()).pack(side="left", padx=4)

        ctk.CTkLabel(options, text="Min match:").pack(side="left", padx=(16, 4))
        ctk.CTkSlider(
            options,
            from_=0.60,
            to=0.99,
            variable=self.min_score_var,
            number_of_steps=39,
            width=180,
            command=lambda _e: self.update_threshold_label(),
        ).pack(side="left", padx=4)
        self.threshold_label = ctk.CTkLabel(options, text="0.82")
        self.threshold_label.pack(side="left", padx=(4, 12))

        ctk.CTkCheckBox(options, text="One video per batch", variable=self.one_per_batch_var).pack(side="left", padx=6)

        ctk.CTkLabel(options, text="Batch mode:").pack(side="left", padx=(12, 4))
        ctk.CTkOptionMenu(options, variable=self.batch_mode_var, values=["smart", "size", "timegap"], command=lambda _v: self.refresh_batches()).pack(side="left", padx=4)

        ctk.CTkLabel(options, text="Size:").pack(side="left", padx=(12, 4))
        ctk.CTkOptionMenu(options, variable=ctk.StringVar(value="4"), values=["1", "2", "3", "4"], command=self._change_batch_size).pack(side="left", padx=4)

        ctk.CTkLabel(options, text="Gap sec:").pack(side="left", padx=(12, 4))
        ctk.CTkOptionMenu(options, variable=ctk.StringVar(value="3"), values=["1", "2", "3", "5", "10", "20", "30", "60"], command=self._change_batch_gap).pack(side="left", padx=4)

        ctk.CTkLabel(options, text="Output:").pack(side="left", padx=(12, 4))
        ctk.CTkEntry(options, textvariable=self.output_var, width=220).pack(side="left", padx=4)

        music_row = ctk.CTkFrame(outer)
        music_row.pack(fill="x", pady=(0, 10))
        ctk.CTkCheckBox(music_row, text="Add backing music", variable=self.music_enabled_var).pack(side="left", padx=(10, 6), pady=10)
        ctk.CTkLabel(music_row, text="Music file:").pack(side="left", padx=(6, 4))
        self.music_entry = ctk.CTkEntry(music_row, textvariable=self.music_path_var)
        self.music_entry.pack(side="left", fill="x", expand=True, padx=4)
        ctk.CTkButton(music_row, text="Browse Music", command=self.browse_music_file).pack(side="left", padx=6)
        ctk.CTkLabel(music_row, text="Volume:").pack(side="left", padx=(10, 4))
        ctk.CTkSlider(
            music_row,
            from_=0.02,
            to=0.50,
            variable=self.music_volume_var,
            number_of_steps=48,
            width=140,
            command=lambda _e: self.update_music_volume_label(),
        ).pack(side="left", padx=4)
        self.music_volume_label = ctk.CTkLabel(music_row, text="0.12")
        self.music_volume_label.pack(side="left", padx=(4, 10))

        timeline_wrap = ctk.CTkFrame(outer)
        timeline_wrap.pack(fill="x", pady=(0, 10))
        header = ctk.CTkFrame(timeline_wrap, fg_color="transparent")
        header.pack(fill="x", padx=10, pady=(8, 4))
        ctk.CTkLabel(header, text="Timeline", font=("Segoe UI", 16, "bold")).pack(side="left")
        ctk.CTkButton(header, text="Clear", width=72, command=self.clear_timeline).pack(side="right", padx=4)
        ctk.CTkButton(header, text="Sort by time", width=100, command=self.sort_timeline_by_time).pack(side="right", padx=4)
        ctk.CTkButton(header, text="Re-score", width=90, command=self.sort_timeline_by_match).pack(side="right", padx=4)

        self.timeline_canvas = Canvas(timeline_wrap, height=180, bg="#111827", highlightthickness=0)
        self.timeline_canvas.pack(fill="x", padx=10, pady=(0, 4))
        self.timeline_inner = ctk.CTkFrame(self.timeline_canvas, fg_color="transparent")
        self.timeline_canvas.create_window((0, 0), window=self.timeline_inner, anchor="nw")
        self.timeline_scrollbar = ctk.CTkScrollbar(timeline_wrap, orientation="horizontal", command=self.timeline_canvas.xview)
        self.timeline_scrollbar.pack(fill="x", padx=10, pady=(0, 10))
        self.timeline_canvas.configure(xscrollcommand=self.timeline_scrollbar.set)
        self.timeline_inner.bind("<Configure>", lambda _e: self.timeline_canvas.configure(scrollregion=self.timeline_canvas.bbox("all")))
        self.timeline_canvas.bind("<MouseWheel>", self._on_timeline_mousewheel)
        self.timeline_canvas.bind("<Shift-MouseWheel>", self._on_timeline_shift_mousewheel)
        self.timeline_inner.bind("<MouseWheel>", self._on_timeline_mousewheel)
        self.timeline_inner.bind("<Shift-MouseWheel>", self._on_timeline_shift_mousewheel)

        content = ctk.CTkFrame(outer, fg_color="transparent")
        content.pack(fill="both", expand=True)

        left = ctk.CTkFrame(content)
        left.pack(side="left", fill="both", expand=True, padx=(0, 6))
        ctk.CTkLabel(left, text="Available Clips", font=("Segoe UI", 16, "bold")).pack(anchor="w", padx=12, pady=(8, 4))
        self.cards_frame = ctk.CTkScrollableFrame(left)
        self.cards_frame.pack(fill="both", expand=True, padx=10, pady=(0, 10))

        right = ctk.CTkFrame(content, width=590)
        right.pack(side="left", fill="both", padx=(6, 0))
        right.pack_propagate(False)

        ctk.CTkLabel(right, text="Preview", font=("Segoe UI", 16, "bold")).pack(anchor="w", padx=12, pady=(8, 4))
        self.preview_title = ctk.CTkLabel(right, text="Nothing selected", anchor="w")
        self.preview_title.pack(fill="x", padx=12, pady=(0, 6))
        btns = ctk.CTkFrame(right, fg_color="transparent")
        btns.pack(fill="x", padx=12)
        self.pause_btn = ctk.CTkButton(btns, text="Pause", width=80, command=self.pause_preview)
        self.pause_btn.pack(side="right", padx=4)
        self.play_btn = ctk.CTkButton(btns, text="Play", width=80, command=self.play_preview)
        self.play_btn.pack(side="right", padx=4)
        ctk.CTkButton(btns, text="Clear", width=80, fg_color="#dc2626", hover_color="#b91c1c", command=self.clear_preview).pack(side="right", padx=4)
        self.preview_canvas = Canvas(right, bg="#0b1220", highlightthickness=0, height=340)
        self.preview_canvas.pack(fill="x", padx=12, pady=12)

        ctk.CTkLabel(right, text="Match Compare", font=("Segoe UI", 16, "bold")).pack(anchor="w", padx=12, pady=(0, 4))
        compare_frame = ctk.CTkFrame(right)
        compare_frame.pack(fill="x", padx=12, pady=(0, 8))

        self.compare_left_label = ctk.CTkLabel(compare_frame, text="Source end frame", anchor="center")
        self.compare_left_label.grid(row=0, column=0, padx=6, pady=(8, 4))
        self.compare_right_label = ctk.CTkLabel(compare_frame, text="Candidate start frame", anchor="center")
        self.compare_right_label.grid(row=0, column=1, padx=6, pady=(8, 4))

        self.compare_left_canvas = Canvas(compare_frame, bg="#0b1220", width=COMPARE_SIZE[0], height=COMPARE_SIZE[1], highlightthickness=0)
        self.compare_left_canvas.grid(row=1, column=0, padx=6, pady=(0, 8))
        self.compare_right_canvas = Canvas(compare_frame, bg="#0b1220", width=COMPARE_SIZE[0], height=COMPARE_SIZE[1], highlightthickness=0)
        self.compare_right_canvas.grid(row=1, column=1, padx=6, pady=(0, 8))
        compare_frame.grid_columnconfigure(0, weight=1)
        compare_frame.grid_columnconfigure(1, weight=1)

        ctk.CTkLabel(right, text="Top candidate next clips", font=("Segoe UI", 14, "bold")).pack(anchor="w", padx=12, pady=(0, 4))
        self.candidate_frame = ctk.CTkScrollableFrame(right, height=180)
        self.candidate_frame.pack(fill="x", padx=12, pady=(0, 10))

        ctk.CTkLabel(right, text="Batch Inspector", font=("Segoe UI", 14, "bold")).pack(anchor="w", padx=12, pady=(0, 4))
        self.batch_inspector_frame = ctk.CTkScrollableFrame(right, height=170)
        self.batch_inspector_frame.pack(fill="x", padx=12, pady=(0, 10))

        self.sequence_box = ctk.CTkTextbox(right, height=180, font=("Consolas", 12))
        self.sequence_box.pack(fill="both", expand=True, padx=12, pady=(0, 12))

        footer = ctk.CTkFrame(outer)
        footer.pack(fill="x", pady=(6, 0))
        status = ctk.CTkLabel(footer, textvariable=self.status_var, anchor="w")
        status.pack(fill="x", padx=10, pady=(8, 2))
        detail_row = ctk.CTkFrame(footer, fg_color="transparent")
        detail_row.pack(fill="x", padx=10, pady=(0, 8))
        self.progress_bar = ctk.CTkProgressBar(detail_row, variable=self.load_progress_var)
        self.progress_bar.pack(side="left", fill="x", expand=True, padx=(0, 10))
        self.progress_bar.set(0)
        self.load_detail_label = ctk.CTkLabel(detail_row, textvariable=self.load_detail_var, width=180, anchor="w")
        self.load_detail_label.pack(side="left", padx=(0, 10))
        self.current_file_label = ctk.CTkLabel(detail_row, textvariable=self.current_file_var, anchor="w")
        self.current_file_label.pack(side="left", fill="x", expand=True)

        self.update_threshold_label()
        self.update_music_volume_label()
        self.render_timeline()
        self.render_batch_inspector()
        self.clear_compare_panel()

    def _on_timeline_mousewheel(self, event) -> str:
        try:
            if (event.state & 0x0001) != 0:
                return self._on_timeline_shift_mousewheel(event)
        except Exception:
            pass
        delta = getattr(event, "delta", 0)
        if delta:
            units = -1 if delta > 0 else 1
            self.timeline_canvas.xview_scroll(units * 2, "units")
        return "break"

    def _on_timeline_shift_mousewheel(self, event) -> str:
        delta = getattr(event, "delta", 0)
        if delta:
            units = -1 if delta > 0 else 1
            self.timeline_canvas.xview_scroll(units * 4, "units")
        return "break"

    def render_batch_inspector(self) -> None:
        for widget in self.batch_inspector_frame.winfo_children():
            widget.destroy()
        if not self.clips:
            ctk.CTkLabel(self.batch_inspector_frame, text="No batches yet.").pack(anchor="w", padx=8, pady=8)
            return

        ordered = sorted(
            self.clips,
            key=lambda c: (
                c.batch_group,
                c.created_ts if self.sort_var.get() == "created" else c.modified_ts,
                c.name.lower(),
            ),
        )
        groups: Dict[int, List[VideoClip]] = {}
        for clip in ordered:
            groups.setdefault(clip.batch_group, []).append(clip)

        timeline_positions = {clip.name: idx + 1 for idx, clip in enumerate(self.timeline)}
        for group_id, members in groups.items():
            chosen = next((clip for clip in members if clip.name in timeline_positions), None)
            box = ctk.CTkFrame(self.batch_inspector_frame)
            box.pack(fill="x", padx=4, pady=4)

            header = ctk.CTkFrame(box, fg_color="transparent")
            header.pack(fill="x", padx=8, pady=(6, 2))
            ctk.CTkLabel(
                header,
                text=f"Batch {group_id + 1}",
                font=("Segoe UI", 13, "bold"),
            ).pack(side="left")
            summary = f"{len(members)} clip(s)"
            if chosen is not None:
                summary += f" | timeline #{timeline_positions[chosen.name]}"
            ctk.CTkLabel(header, text=summary).pack(side="right")

            for clip in members:
                row = ctk.CTkFrame(box, fg_color="transparent")
                row.pack(fill="x", padx=8, pady=2)
                flags = []
                if clip.name in timeline_positions:
                    flags.append(f"TL {timeline_positions[clip.name]}")
                if self.manual_start is clip:
                    flags.append("START")
                flag_text = " | ".join(flags) if flags else "—"
                ctk.CTkLabel(row, text=clip.name, anchor="w", width=230).pack(side="left", padx=(0, 6))
                ctk.CTkLabel(row, text=flag_text, width=70).pack(side="left", padx=(0, 6))
                ctk.CTkButton(row, text="Cmp", width=44, command=lambda c=clip: self.show_match_candidates(c)).pack(side="right", padx=2)
                ctk.CTkButton(row, text="Add", width=44, command=lambda c=clip: self.add_to_timeline(c)).pack(side="right", padx=2)
                ctk.CTkButton(row, text="Only", width=48, command=lambda c=clip: self.replace_batch_choice(c)).pack(side="right", padx=2)

    def _change_batch_size(self, value: str) -> None:
        self.batch_size_var.set(int(value))
        self.refresh_batches()

    def _change_batch_gap(self, value: str) -> None:
        self.batch_gap_var.set(int(value))
        self.refresh_batches()

    def refresh_batches(self) -> None:
        if not self.clips:
            return
        self.builder.assign_batch_groups(
            self.clips,
            sort_key=self.sort_var.get(),
            mode=self.batch_mode_var.get(),
            max_size=self.batch_size_var.get(),
            gap_seconds=self.batch_gap_var.get(),
        )
        self.builder.compute_links(self.clips)
        self.render_cards()
        self.render_batch_inspector()
        self.sequence_box.delete("1.0", "end")
        self.sequence_box.insert("1.0", self.sequence_report())
        self.set_status("Batch groups refreshed.")

    def update_threshold_label(self) -> None:
        self.threshold_label.configure(text=f"{self.min_score_var.get():.2f}")

    def update_music_volume_label(self) -> None:
        self.music_volume_label.configure(text=f"{self.music_volume_var.get():.2f}")

    def load_app_state(self) -> None:
        try:
            if not APP_STATE_FILE.exists():
                return
            payload = json.loads(APP_STATE_FILE.read_text(encoding="utf-8"))
            last_dir = str(payload.get("last_dir", "")).strip()
            if last_dir:
                self.dir_entry.delete(0, "end")
                self.dir_entry.insert(0, last_dir)
                if Path(last_dir).exists() and Path(last_dir).is_dir():
                    self.root.after(150, self.start_load_directory)
        except Exception:
            pass

    def save_app_state(self) -> None:
        try:
            payload = {"last_dir": str(self.current_dir) if self.current_dir else self.dir_entry.get().strip()}
            APP_STATE_FILE.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        except Exception:
            pass

    def set_status(self, text: str) -> None:
        self.status_var.set(text)

    def set_load_progress(self, done: int = 0, total: int = 0, current_file: str = "") -> None:
        total = max(0, int(total))
        done = max(0, int(done))
        progress = (done / total) if total > 0 else 0.0
        self.load_progress_var.set(progress)
        if total > 0:
            self.load_detail_var.set(f"{done}/{total} processed")
        else:
            self.load_detail_var.set("Idle")
        self.current_file_var.set(current_file or "")

    def browse_directory(self) -> None:
        if self.loading_directory:
            self.set_status("Already scanning a folder. Please wait.")
            return
        start_dir = str(self.current_dir) if self.current_dir else str(Path.home())
        folder = filedialog.askdirectory(initialdir=start_dir)
        if folder:
            self.dir_entry.delete(0, "end")
            self.dir_entry.insert(0, folder)
            self.start_load_directory()


    def browse_music_file(self) -> None:
        start_dir = str(self.current_dir) if self.current_dir else str(Path.home())
        file_path = filedialog.askopenfilename(
            title="Choose backing music",
            initialdir=start_dir,
            filetypes=[
                ("Audio files", "*.mp3 *.wav *.m4a *.aac *.flac *.ogg"),
                ("All files", "*.*"),
            ],
        )
        if file_path:
            self.music_path_var.set(file_path)
            self.music_enabled_var.set(True)
            self.set_status(f"Backing music selected: {Path(file_path).name}")

    def load_directory(self) -> None:
        """Compatibility helper retained for older call sites."""
        self.start_load_directory()

    def start_load_directory(self) -> None:
        folder = self.dir_entry.get().strip()
        if not folder:
            messagebox.showwarning(APP_TITLE, "Choose a folder first.")
            return
        directory = Path(folder)
        if not directory.exists() or not directory.is_dir():
            messagebox.showerror(APP_TITLE, "That folder does not exist.")
            return
        if cv2 is None or Image is None or ImageTk is None:
            messagebox.showerror(APP_TITLE, "This app needs OpenCV and Pillow installed.")
            return

        self._active_load_id += 1
        load_id = self._active_load_id
        self.loading_directory = True
        self.loading_total = 0
        self.loading_done = 0
        self._cards_dirty = False
        self._last_cards_render = 0.0
        self.set_load_progress(0, 0, "")
        self._set_load_controls_state(False)
        self.current_dir = directory
        self.cache_dir = directory / CACHE_DIR_NAME
        self.save_app_state()
        self.stop_preview_video()
        self.timeline.clear()
        self.manual_start = None
        self.selected_compare_source = None
        self.photo_cache.clear()
        self.clips = []
        self.render_cards()
        self.render_timeline()
        self.clear_compare_panel()
        self.clear_preview()
        self.sequence_box.delete("1.0", "end")
        self.sequence_box.insert("1.0", "Scanning folder...")
        self.set_status(f"Scanning: {directory}")

        self.load_thread = threading.Thread(target=self._discover_and_schedule_load, args=(directory, load_id), daemon=True)
        self.load_thread.start()
        self.root.after(40, self._poll_load_queue)

    def _discover_and_schedule_load(self, directory: Path, load_id: int) -> None:
        try:
            cache_dir = directory / CACHE_DIR_NAME
            cache_dir.mkdir(exist_ok=True)
            video_files = []
            for entry in os.scandir(directory):
                if not entry.is_file():
                    continue
                suffix = Path(entry.name).suffix.lower()
                if suffix in {".mp4", ".mov", ".m4v"}:
                    video_files.append(Path(entry.path))
            video_files.sort(key=lambda p: p.name.lower())
            self.load_queue.put(("discovered", {"load_id": load_id, "directory": directory, "cache_dir": cache_dir, "video_files": video_files}))
            if not video_files:
                self.load_queue.put(("finished", {"load_id": load_id}))
                return

            max_workers = max(2, min(4, (os.cpu_count() or 4)))
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = [executor.submit(self.analyzer.extract_clip_data, video_path, cache_dir) for video_path in video_files]
                for future in futures:
                    clip = future.result()
                    self.load_queue.put(("clip", {"load_id": load_id, "clip": clip}))
            self.load_queue.put(("finished", {"load_id": load_id}))
        except Exception as exc:
            self.load_queue.put(("error", {"load_id": load_id, "message": str(exc)}))

    def _poll_load_queue(self) -> None:
        keep_polling = self.loading_directory
        processed_any = False
        processed_count = 0
        max_items_per_tick = 6

        while processed_count < max_items_per_tick:
            try:
                kind, payload = self.load_queue.get_nowait()
            except queue.Empty:
                break

            processed_any = True
            processed_count += 1
            load_id = payload.get("load_id") if isinstance(payload, dict) else None
            if load_id is not None and load_id != self._active_load_id:
                continue

            if kind == "discovered":
                self.current_dir = payload["directory"]
                self.cache_dir = payload["cache_dir"]
                self.loading_total = len(payload["video_files"])
                self.loading_done = 0
                if self.loading_total == 0:
                    self.set_load_progress(0, 0, "")
                    self.set_status("No supported video files found (.mp4/.mov/.m4v).")
                else:
                    self.set_load_progress(0, self.loading_total, "Preparing clip analysis...")
                    self.set_status(f"Found {self.loading_total} clip(s). Extracting previews...")

            elif kind == "clip":
                clip = payload.get("clip")
                self.loading_done += 1
                current_name = ""
                if clip is not None:
                    self.clips.append(clip)
                    current_name = clip.path.name
                    self._cards_dirty = True
                self.set_load_progress(self.loading_done, self.loading_total, current_name)

                if self.loading_done == 1 and self.clips:
                    self.resort(recompute=False)
                    self.render_cards()
                    self._cards_dirty = False
                    self._last_cards_render = time.monotonic()
                    self.preview_clip(self.clips[0], use_video=False)
                    self.show_match_candidates(self.clips[0])
                elif self._cards_dirty and (time.monotonic() - self._last_cards_render) >= 0.25:
                    self.resort(recompute=False)
                    self.render_cards()
                    self._cards_dirty = False
                    self._last_cards_render = time.monotonic()

                if self.loading_done == self.loading_total or self.loading_done % 5 == 0:
                    self.set_status(f"Loading clips... {self.loading_done}/{self.loading_total}")

            elif kind == "finished":
                self.loading_directory = False
                keep_polling = False
                self._set_load_controls_state(True)
                self._finalize_loaded_directory()

            elif kind == "error":
                self.loading_directory = False
                keep_polling = False
                self._set_load_controls_state(True)
                self.set_load_progress(0, 0, "")
                message = payload.get("message", "Unknown error")
                self.set_status(f"Load failed: {message}")
                messagebox.showerror(APP_TITLE, f"Failed to load folder.\n\n{message}")

        if keep_polling:
            if self._cards_dirty and (time.monotonic() - self._last_cards_render) >= 0.35:
                self.resort(recompute=False)
                self.render_cards()
                self._cards_dirty = False
                self._last_cards_render = time.monotonic()
            delay = 15 if processed_any else 45
            self.root.after(delay, self._poll_load_queue)

    def _finalize_loaded_directory(self) -> None:
        if not self.clips:
            self.render_cards()
            self.render_timeline()
            self.clear_compare_panel()
            self.clear_preview()
            self.sequence_box.delete("1.0", "end")
            self.sequence_box.insert("1.0", "No clips loaded.")
            self.set_load_progress(0, 0, "")
            return

        self.resort(recompute=False)
        self.refresh_batches()
        self.render_timeline()
        self.sequence_box.delete("1.0", "end")
        self.sequence_box.insert("1.0", self.sequence_report())
        self.clear_compare_panel()
        self.preview_clip(self.clips[0], use_video=False)
        self.show_match_candidates(self.clips[0])
        self.set_load_progress(len(self.clips), len(self.clips), "Ready")
        if self._pending_project_payload:
            self._apply_pending_project_payload()
        else:
            self.set_status(f"Loaded {len(self.clips)} clip(s).")

    def _set_load_controls_state(self, enabled: bool) -> None:
        state = "normal" if enabled else "disabled"
        try:
            self.browse_btn.configure(state=state)
            self.rescan_btn.configure(state=state)
        except Exception:
            pass

    def resort(self, recompute: bool = True) -> None:
        key = (lambda c: c.created_ts) if self.sort_var.get() == "created" else (lambda c: c.modified_ts)
        self.clips.sort(key=key)
        if recompute and self.clips:
            self.builder.assign_batch_groups(
                self.clips,
                sort_key=self.sort_var.get(),
                mode=self.batch_mode_var.get(),
                max_size=self.batch_size_var.get(),
                gap_seconds=self.batch_gap_var.get(),
            )
            self.builder.compute_links(self.clips)
        self.render_cards()
        self.render_timeline()
        if self.clips:
            self.set_status(f"Resorted by {self.sort_var.get()}.")

    def load_photoimage(self, image_path: Path, size: Tuple[int, int]):
        key = (str(image_path), size)
        if key in self.photo_cache:
            return self.photo_cache[key]
        if Image is None or ImageTk is None:
            return None
        img = Image.open(image_path).convert("RGB")
        img = ImageOps.contain(img, size)
        canvas = Image.new("RGB", size, color=(24, 24, 32))
        x = (size[0] - img.width) // 2
        y = (size[1] - img.height) // 2
        canvas.paste(img, (x, y))
        tk_img = ImageTk.PhotoImage(canvas)
        self.photo_cache[key] = tk_img
        return tk_img

    def render_cards(self) -> None:
        for widget in self.cards_frame.winfo_children():
            widget.destroy()
        for idx, clip in enumerate(self.clips):
            card = VideoCard(self.cards_frame, self, clip)
            card.grid(row=idx // CARDS_PER_ROW, column=idx % CARDS_PER_ROW, padx=10, pady=10, sticky="n")

    def render_timeline(self) -> None:
        for widget in self.timeline_inner.winfo_children():
            widget.destroy()
        if not self.timeline:
            ctk.CTkLabel(self.timeline_inner, text="Timeline empty").pack(padx=12, pady=20)
        else:
            for idx, clip in enumerate(self.timeline):
                TimelineChip(self.timeline_inner, self, clip, idx).pack(side="left", padx=6, pady=8)
            ctk.CTkLabel(
                self.timeline_inner,
                text="Drag a chip by its thumbnail/title area to reorder",
                font=("Segoe UI", 11),
                text_color="#9ca3af",
            ).pack(side="left", padx=12)
        self.timeline_canvas.update_idletasks()
        self.timeline_canvas.configure(scrollregion=self.timeline_canvas.bbox("all"))
        self.render_batch_inspector()
        self.sequence_box.delete("1.0", "end")
        self.sequence_box.insert("1.0", self.sequence_report())

    def sequence_report(self) -> str:
        if not self.clips:
            return "No clips loaded."
        start_guess = self.builder.guess_start(self.clips)
        lines = [
            f"Likely start: {start_guess.name if start_guess else '-'}",
            f"Manual start: {self.manual_start.name if self.manual_start else '-'}",
            f"Batch mode: {self.batch_mode_var.get()} | Batch size: {self.batch_size_var.get()} | Gap: {self.batch_gap_var.get()}s",
            "",
            "Top link suggestions:",
        ]
        links = self.builder.compute_links(self.clips)[:12]
        for link in links:
            lines.append(f"{link.src_name} -> {link.dst_name}    {link.score:.3f}")
        if self.timeline:
            lines.extend(["", "Timeline:"])
            for idx, clip in enumerate(self.timeline, start=1):
                lines.append(f"{idx}. {clip.name}")
        return "\n".join(lines)

    def _show_canvas_image(self, canvas: Canvas, image_path: Path, size: Tuple[int, int], title: Optional[str] = None) -> None:
        img = self.load_photoimage(image_path, size)
        canvas.delete("all")
        w = int(canvas.cget("width")) or size[0]
        h = int(canvas.cget("height")) or size[1]
        canvas.create_image(w // 2, h // 2, image=img, anchor="center")
        canvas.image = img

    def preview_image(self, image_path: Path, title: str) -> None:
        self.stop_preview_video()
        self.preview_title.configure(text=title)
        self._update_preview_buttons()
        img = self.load_photoimage(image_path, PREVIEW_MAX)
        self.preview_canvas.delete("all")
        w = self.preview_canvas.winfo_width() or PREVIEW_MAX[0]
        h = self.preview_canvas.winfo_height() or PREVIEW_MAX[1]
        self.preview_canvas.create_image(w // 2, h // 2, image=img, anchor="center")
        self.preview_canvas.image = img

    def preview_clip(self, clip: VideoClip, use_video: bool = True) -> None:
        self.current_preview_clip = clip
        if not use_video or cv2 is None:
            self.preview_image(clip.first_frame_path, clip.name)
            return
        self._open_preview_video(clip)

    def _open_preview_video(self, clip: VideoClip) -> bool:
        self.stop_preview_video()
        if cv2 is None:
            self.preview_image(clip.first_frame_path, clip.name)
            return False
        cap = cv2.VideoCapture(str(clip.path))
        if not cap.isOpened():
            self.preview_image(clip.first_frame_path, clip.name)
            return False
        self.preview_cap = cap
        self.preview_paused = False
        self.preview_title.configure(text=clip.name)
        self._update_preview_buttons()
        self._preview_step()
        return True

    def _preview_step(self) -> None:
        if self.preview_cap is None or self.preview_paused:
            self.preview_job = None
            return
        ok, frame = self.preview_cap.read()
        if not ok:
            self.preview_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ok, frame = self.preview_cap.read()
        if not ok:
            self.stop_preview_video()
            return
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil = Image.fromarray(rgb)
        pil.thumbnail(PREVIEW_MAX)
        tk_img = ImageTk.PhotoImage(pil)
        self.preview_canvas.delete("all")
        w = self.preview_canvas.winfo_width() or PREVIEW_MAX[0]
        h = self.preview_canvas.winfo_height() or PREVIEW_MAX[1]
        self.preview_canvas.create_image(w // 2, h // 2, image=tk_img, anchor="center")
        self.preview_canvas.image = tk_img
        self.preview_job = self.root.after(33, self._preview_step)

    def _update_preview_buttons(self) -> None:
        has_video = self.preview_cap is not None
        try:
            self.play_btn.configure(state="normal", text="Play")
            self.pause_btn.configure(state="normal" if has_video else "disabled")
        except Exception:
            pass

    def play_preview(self) -> None:
        if self.current_preview_clip is None:
            return
        if self.preview_cap is None:
            self._open_preview_video(self.current_preview_clip)
            return
        if self.preview_paused:
            self.preview_paused = False
            self._update_preview_buttons()
            if self.preview_job is None:
                self._preview_step()

    def pause_preview(self) -> None:
        if self.preview_cap is None:
            return
        self.preview_paused = True
        self._update_preview_buttons()

    def toggle_preview_play(self) -> None:
        if self.preview_paused:
            self.play_preview()
        else:
            self.pause_preview()

    def stop_preview_video(self) -> None:
        if self.preview_job is not None:
            try:
                self.root.after_cancel(self.preview_job)
            except Exception:
                pass
            self.preview_job = None
        if self.preview_cap is not None:
            try:
                self.preview_cap.release()
            except Exception:
                pass
            self.preview_cap = None
        self.preview_paused = False
        self._update_preview_buttons()

    def clear_preview(self) -> None:
        self.stop_preview_video()
        self.preview_canvas.delete("all")
        self.current_preview_clip = None
        self.preview_title.configure(text="Nothing selected")

    def clear_compare_panel(self) -> None:
        self.selected_compare_source = None
        self.compare_left_canvas.delete("all")
        self.compare_right_canvas.delete("all")
        self.compare_left_label.configure(text="Source end frame")
        self.compare_right_label.configure(text="Candidate start frame")
        for widget in self.candidate_frame.winfo_children():
            widget.destroy()

    def show_match_candidates(self, clip: VideoClip) -> None:
        self.selected_compare_source = clip
        self._show_canvas_image(self.compare_left_canvas, clip.last_frame_path, COMPARE_SIZE)
        self.compare_left_label.configure(text=f"{clip.name} - END")

        self.compare_right_canvas.delete("all")
        self.compare_right_label.configure(text="Select a candidate")
        for widget in self.candidate_frame.winfo_children():
            widget.destroy()

        candidates = self.builder.top_candidates_for(
            clips=self.clips,
            src=clip,
            min_score=0.0,
            exclude_used=set(),
            enforce_one_per_batch=False,
            used_batches=set(),
            limit=6,
        )

        if not candidates:
            ctk.CTkLabel(self.candidate_frame, text="No candidates found.").pack(anchor="w", padx=8, pady=8)
            return

        for score, candidate in candidates:
            row = ctk.CTkFrame(self.candidate_frame)
            row.pack(fill="x", padx=4, pady=4)

            thumb = self.load_photoimage(candidate.first_frame_path, (96, 54))
            thumb_label = ctk.CTkLabel(row, text="", image=thumb)
            thumb_label.image = thumb
            thumb_label.pack(side="left", padx=6, pady=6)

            text = (
                f"{candidate.name}\n"
                f"score={score:.3f} | batch={candidate.batch_group + 1} | "
                f"{candidate.created_dt:%H:%M:%S}"
            )
            ctk.CTkLabel(row, text=text, justify="left", anchor="w").pack(side="left", fill="x", expand=True, padx=6)

            btns = ctk.CTkFrame(row, fg_color="transparent")
            btns.pack(side="right", padx=6)
            ctk.CTkButton(btns, text="View", width=58, command=lambda c=candidate, s=score: self.preview_candidate_match(c, s)).pack(side="left", padx=2)
            ctk.CTkButton(btns, text="Add", width=58, command=lambda c=candidate: self.add_to_timeline(c)).pack(side="left", padx=2)
            ctk.CTkButton(btns, text="Use Next", width=76, command=lambda c=candidate: self.append_candidate_after_source(c)).pack(side="left", padx=2)

        self.preview_candidate_match(candidates[0][1], candidates[0][0])

    def preview_candidate_match(self, candidate: VideoClip, score: float) -> None:
        self._show_canvas_image(self.compare_right_canvas, candidate.first_frame_path, COMPARE_SIZE)
        self.compare_right_label.configure(text=f"{candidate.name} - START | score={score:.3f}")

    def append_candidate_after_source(self, candidate: VideoClip) -> None:
        src = self.selected_compare_source
        if src is None:
            self.add_to_timeline(candidate)
            return

        if self.one_per_batch_var.get():
            conflict_index = next((i for i, existing in enumerate(self.timeline) if existing.batch_group == candidate.batch_group), None)
            if conflict_index is not None:
                self.timeline[conflict_index] = candidate
                self.timeline_selected_index = conflict_index
                self.render_timeline()
                self.set_status(f"Replaced batch {candidate.batch_group + 1} choice with {candidate.name}.")
                return

        if src in self.timeline:
            idx = self.timeline.index(src)
            if candidate in self.timeline:
                self.timeline.remove(candidate)
            self.timeline.insert(idx + 1, candidate)
            self.timeline_selected_index = idx + 1
        else:
            if candidate not in self.timeline:
                self.timeline.append(candidate)
                self.timeline_selected_index = len(self.timeline) - 1
        self.render_timeline()
        self.set_status(f"Placed {candidate.name} after {src.name}.")

    def add_to_timeline(self, clip: VideoClip) -> None:
        if clip in self.timeline:
            self.timeline_selected_index = self.timeline.index(clip)
            self.render_timeline()
            self.set_status(f"{clip.name} is already in the timeline.")
            return

        insert_at = len(self.timeline)

        if self.one_per_batch_var.get():
            existing_index = next((i for i, existing in enumerate(self.timeline) if existing.batch_group == clip.batch_group), None)
            if existing_index is not None:
                self.timeline[existing_index] = clip
                self.timeline_selected_index = existing_index
                self.render_timeline()
                self.set_status(f"Replaced existing choice in batch {clip.batch_group + 1} with {clip.name}.")
                return

        self.timeline.insert(insert_at, clip)
        self.timeline_selected_index = insert_at
        self.render_timeline()
        self.set_status(f"Added {clip.name} at position {insert_at + 1}.")

    def replace_batch_choice(self, clip: VideoClip) -> None:
        existing_index = next((i for i, existing in enumerate(self.timeline) if existing.batch_group == clip.batch_group), None)
        if existing_index is None:
            self.timeline.append(clip)
        else:
            self.timeline[existing_index] = clip
        self.render_timeline()
        self.set_status(f"Batch {clip.batch_group + 1} now uses {clip.name}.")

    def set_manual_start(self, clip: VideoClip) -> None:
        self.manual_start = clip
        self.set_status(f"Manual start set to {clip.name}.")
        self.sequence_box.delete("1.0", "end")
        self.sequence_box.insert("1.0", self.sequence_report())
        self.show_match_candidates(clip)

    def move_timeline(self, index: int, delta: int) -> None:
        new_index = index + delta
        if new_index < 0 or new_index >= len(self.timeline):
            return
        self.timeline[index], self.timeline[new_index] = self.timeline[new_index], self.timeline[index]
        self.render_timeline()

    def move_timeline_to(self, index: int, new_index: int) -> None:
        if index < 0 or index >= len(self.timeline):
            return
        clip = self.timeline.pop(index)
        new_index = max(0, min(new_index, len(self.timeline)))
        self.timeline.insert(new_index, clip)
        self.render_timeline()

    def select_timeline_index(self, index: int) -> None:
        if 0 <= index < len(self.timeline):
            self.timeline_selected_index = index
            self.render_timeline()
            self.set_status(f"Selected timeline position {index + 1}: {self.timeline[index].name}")


    def start_timeline_drag(self, index: int, _event=None) -> None:
        if 0 <= index < len(self.timeline):
            self.drag_timeline_index = index
            self.drag_target_index = index
            self.timeline_selected_index = index
            self.set_status(f"Dragging timeline clip {index + 1}: {self.timeline[index].name}")

    def drag_timeline_motion(self, event) -> None:
        if self.drag_timeline_index is None or not self.timeline:
            return
        rel_x = (event.x_root - self.timeline_inner.winfo_rootx()) + self.timeline_canvas.canvasx(0)
        target = int(max(0, min(len(self.timeline) - 1, rel_x // TIMELINE_SLOT_WIDTH)))
        if target != self.drag_target_index:
            self.drag_target_index = target
            self.set_status(
                f"Dragging {self.timeline[self.drag_timeline_index].name} -> drop at position {target + 1}"
            )

    def end_timeline_drag(self, _event=None) -> None:
        if self.drag_timeline_index is None:
            return
        source = self.drag_timeline_index
        target = self.drag_target_index if self.drag_target_index is not None else source
        self.drag_timeline_index = None
        self.drag_target_index = None
        if source != target:
            self.move_timeline_to(source, target)
            self.timeline_selected_index = target
            self.set_status(f"Moved clip to position {target + 1}.")
        else:
            self.timeline_selected_index = source
            self.render_timeline()

    def remove_from_timeline(self, index: int) -> None:
        if 0 <= index < len(self.timeline):
            removed = self.timeline.pop(index)
            if self.timeline_selected_index is not None:
                if not self.timeline:
                    self.timeline_selected_index = None
                elif index <= self.timeline_selected_index:
                    self.timeline_selected_index = max(0, min(self.timeline_selected_index - 1, len(self.timeline) - 1))
            self.render_timeline()
            self.set_status(f"Removed {removed.name}.")

    def clear_timeline(self) -> None:
        self.timeline.clear()
        self.timeline_selected_index = None
        self.render_timeline()
        self.set_status("Timeline cleared.")

    def sort_timeline_by_time(self) -> None:
        key = (lambda c: c.created_ts) if self.sort_var.get() == "created" else (lambda c: c.modified_ts)
        self.timeline.sort(key=key)
        if self.timeline_selected_index is not None and self.timeline:
            selected_clip = self.timeline[self.timeline_selected_index if self.timeline_selected_index < len(self.timeline) else -1]
            self.timeline.sort(key=key)
            self.timeline_selected_index = self.timeline.index(selected_clip)
        else:
            self.timeline.sort(key=key)
        self.render_timeline()
        self.set_status("Timeline sorted by time.")

    def sort_timeline_by_match(self) -> None:
        if len(self.timeline) < 2:
            return
        selected_clip = None
        if self.timeline_selected_index is not None and 0 <= self.timeline_selected_index < len(self.timeline):
            selected_clip = self.timeline[self.timeline_selected_index]
        first = self.timeline[0]
        rest = self.timeline[1:]
        scored = []
        current = first
        while rest:
            candidates = [(self.analyzer.compare(current.last_sig, clip.first_sig), clip) for clip in rest if current.last_sig and clip.first_sig]
            if not candidates:
                scored.extend(rest)
                break
            candidates.sort(key=lambda x: x[0], reverse=True)
            best = candidates[0][1]
            scored.append(best)
            rest.remove(best)
            current = best
        self.timeline = [first] + scored
        if selected_clip in self.timeline:
            self.timeline_selected_index = self.timeline.index(selected_clip)
        self.render_timeline()
        self.set_status("Timeline re-ordered by frame match from the first clip.")

    def auto_build_sequence(self) -> None:
        if not self.clips:
            self.set_status("Load clips first.")
            return
        self.refresh_batches()
        sequence = self.builder.build_greedy_sequence(
            self.clips,
            start_clip=self.manual_start,
            min_score=self.min_score_var.get(),
            enforce_one_per_batch=self.one_per_batch_var.get(),
        )
        self.timeline = sequence
        self.timeline_selected_index = 0 if sequence else None
        self.render_timeline()
        if sequence:
            extra = ""
            if self.one_per_batch_var.get():
                batch_count = len({clip.batch_group for clip in self.clips})
                extra = f" | batches used: {len(sequence)}/{batch_count}"
            self.set_status(f"Built sequence with {len(sequence)} clip(s). Start: {sequence[0].name}{extra}")
            self.preview_clip(sequence[0], use_video=False)
            self.show_match_candidates(sequence[0])
        else:
            self.set_status("Could not build a sequence with the current threshold.")

    def _project_payload(self) -> dict:
        return {
            "current_dir": str(self.current_dir) if self.current_dir else "",
            "sort": self.sort_var.get(),
            "min_score": float(self.min_score_var.get()),
            "one_per_batch": bool(self.one_per_batch_var.get()),
            "output_name": self.output_var.get(),
            "music_enabled": bool(self.music_enabled_var.get()),
            "music_path": self.music_path_var.get(),
            "music_volume": float(self.music_volume_var.get()),
            "batch_size": int(self.batch_size_var.get()),
            "batch_mode": self.batch_mode_var.get(),
            "batch_gap_seconds": int(self.batch_gap_var.get()),
            "manual_start": self.manual_start.name if self.manual_start else "",
            "timeline": [clip.name for clip in self.timeline],
        }

    def save_project(self) -> None:
        if self.current_dir is None:
            messagebox.showwarning(APP_TITLE, "Load a folder before saving a project.")
            return
        default_path = self.current_dir / PROJECT_FILE_NAME
        save_path = filedialog.asksaveasfilename(
            title="Save project",
            initialdir=str(self.current_dir),
            initialfile=default_path.name,
            defaultextension=".json",
            filetypes=[("JSON", "*.json")],
        )
        if not save_path:
            return
        Path(save_path).write_text(json.dumps(self._project_payload(), indent=2), encoding="utf-8")
        self.set_status(f"Saved project to {Path(save_path).name}")

    def load_project(self) -> None:
        open_path = filedialog.askopenfilename(title="Load project", filetypes=[("JSON", "*.json")])
        if not open_path:
            return
        try:
            payload = json.loads(Path(open_path).read_text(encoding="utf-8"))
        except Exception as exc:
            messagebox.showerror(APP_TITLE, f"Could not read project file.\n\n{exc}")
            return

        folder = payload.get("current_dir", "").strip()
        if not folder:
            messagebox.showerror(APP_TITLE, "Project file does not include a working folder.")
            return

        self._pending_project_payload = (payload, Path(open_path).name)
        self.dir_entry.delete(0, "end")
        self.dir_entry.insert(0, folder)
        self.save_app_state()
        self.start_load_directory()

    def _apply_pending_project_payload(self) -> None:
        if not self._pending_project_payload or not self.clips:
            return
        payload, project_name = self._pending_project_payload
        self._pending_project_payload = None

        self.sort_var.set(payload.get("sort", "created"))
        self.min_score_var.set(float(payload.get("min_score", 0.82)))
        self.one_per_batch_var.set(bool(payload.get("one_per_batch", False)))
        self.output_var.set(payload.get("output_name", "stitched_output.mp4"))
        self.music_enabled_var.set(bool(payload.get("music_enabled", False)))
        self.music_path_var.set(payload.get("music_path", ""))
        self.music_volume_var.set(float(payload.get("music_volume", 0.12)))
        self.batch_size_var.set(int(payload.get("batch_size", 4)))
        self.batch_mode_var.set(payload.get("batch_mode", "smart"))
        self.batch_gap_var.set(int(payload.get("batch_gap_seconds", 3)))
        self.update_threshold_label()
        self.update_music_volume_label()
        self.refresh_batches()

        by_name = {clip.name: clip for clip in self.clips}
        manual_name = payload.get("manual_start", "")
        self.manual_start = by_name.get(manual_name)

        self.timeline = [by_name[name] for name in payload.get("timeline", []) if name in by_name]
        self.render_timeline()
        self.save_app_state()
        if self.manual_start is not None:
            self.show_match_candidates(self.manual_start)
        elif self.clips:
            self.preview_clip(self.clips[0], use_video=True)
            self.show_match_candidates(self.clips[0])
        self.set_status(f"Loaded project from {project_name}")

    def stitch_timeline(self) -> None:
        if not self.timeline:
            self.set_status("Nothing in timeline to stitch.")
            return
        if self.current_dir is None:
            self.set_status("No working directory.")
            return

        output_name = self.output_var.get().strip() or "stitched_output.mp4"
        if not output_name.lower().endswith(".mp4"):
            output_name += ".mp4"
        output_path = self.current_dir / output_name

        use_music = bool(self.music_enabled_var.get())
        music_path: Optional[Path] = None
        if use_music:
            music_text = self.music_path_var.get().strip()
            if not music_text:
                messagebox.showwarning(APP_TITLE, "Choose a music file or untick 'Add backing music'.")
                return
            music_path = Path(music_text)
            if not music_path.exists():
                messagebox.showerror(APP_TITLE, "The selected music file does not exist.")
                return

        status_text = f"Stitching {len(self.timeline)} clips"
        if use_music and music_path is not None:
            status_text += f" + music ({music_path.name})"
        self.set_status(status_text + "...")

        ok, result = self.stitcher.stitch(
            self.timeline,
            output_path,
            music_path=music_path,
            use_music=use_music,
            music_volume=float(self.music_volume_var.get()),
        )
        if ok:
            if use_music:
                self.set_status(f"Created {output_path.name} with backing music")
            else:
                self.set_status(f"Created {output_path.name}")
            try:
                temp_clip = self.analyzer.extract_clip_data(output_path, self.cache_dir or output_path.parent / CACHE_DIR_NAME)
                if temp_clip is not None:
                    self.preview_clip(temp_clip, use_video=True)
            except Exception:
                pass
        else:
            self.set_status(f"Stitch failed: {result}")
            messagebox.showerror(APP_TITLE, result)

    def on_close(self) -> None:
        self.stop_preview_video()
        self.root.destroy()


def main() -> None:
    root = ctk.CTk()
    app = VideoStitcherApp(root)
    root.protocol("WM_DELETE_WINDOW", app.on_close)
    root.mainloop()


if __name__ == "__main__":
    main()

