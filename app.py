from __future__ import annotations

import argparse
import csv
import json
import os
import queue
import random
import re
import shutil
import statistics
import subprocess
import sys
import threading
import time
import uuid
from datetime import datetime, timedelta
from pathlib import Path
from tkinter import END, BooleanVar, DoubleVar, IntVar, StringVar, filedialog, messagebox, simpledialog
import tkinter as tk
from tkinter import ttk

from tools.workspace import initialize_workspace

import av
import psutil
from PIL import Image, ImageTk


ROOT = Path(__file__).resolve().parent
PYTHON = ROOT / ".venv" / "Scripts" / "python.exe"
ACCELERATE = ROOT / ".venv" / "Scripts" / "accelerate.exe"
ENGINE = ROOT / "trainer-engine" / "src" / "musubi_tuner"
DATASET_DIR = ROOT / "dataset" / "videos"
CACHE_DIR = ROOT / "cache"
DATASET_CONFIG = ROOT / "dataset_config.toml"
MODELS_DIR = ROOT / "models"
OUTPUT_DIR = ROOT / "output"
SAMPLES_DIR = ROOT / "samples"
LOGS_DIR = ROOT / "logs"
SETTINGS_PATH = ROOT / "app_settings.json"
STORYBOARD_PATH = ROOT / "storyboard.json"
DIT_PATH = MODELS_DIR / "wan2.1_t2v_1.3B_bf16.safetensors"
VAE_PATH = MODELS_DIR / "wan_2.1_vae.safetensors"
T5_PATH = MODELS_DIR / "models_t5_umt5-xxl-enc-bf16.pth"
VIDEO_EXTENSIONS = {".mp4", ".mov", ".mkv", ".webm", ".m4v", ".avi"}


def ensure_runtime_workspace() -> None:
    """Create an empty, machine-local workspace without shipping user data."""
    initialize_workspace(ROOT)


class VideoLoraTrainerApp(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("LoRA Video Trainer")
        self.geometry("1260x820")
        self.minsize(1050, 700)
        self.protocol("WM_DELETE_WINDOW", self._on_close)

        self.events: queue.Queue[tuple[str, object]] = queue.Queue()
        self.active_process: subprocess.Popen[str] | None = None
        self.active_thread: threading.Thread | None = None
        self.monitor_stop = threading.Event()
        self.monitor_thread: threading.Thread | None = None
        self.monitor_progress_lock = threading.Lock()
        self.monitor_step = 0
        self.monitor_total_steps = 0
        self.monitor_step_seconds = 0.0
        self.cancel_requested = threading.Event()
        self.settings = self._load_settings()
        self.storyboard_scenes = self._load_storyboard()
        self.pending_storyboard_scene_id: str | None = None
        self.pending_storyboard_samples: set[str] = set()
        self.settings_save_after_id: str | None = None
        self.job_started_at: float | None = None
        self.job_label = ""
        raw_history = self.settings.get("job_history", {})
        self.job_history: dict[str, list[float]] = raw_history if isinstance(raw_history, dict) else {}
        self.preview_image: ImageTk.PhotoImage | None = None
        self.preview_frames: list[ImageTk.PhotoImage] = []
        self.preview_frame_index = 0
        self.preview_after_id: str | None = None
        self.selected_video: Path | None = None
        self.generation_preview_image: ImageTk.PhotoImage | None = None
        self.last_auto_prompt_path: Path | None = None

        self._build_variables()
        self._configure_theme()
        self._build_ui()
        self._restore_last_tab()
        self._refresh_dataset()
        self._refresh_checkpoints()
        self._refresh_status()
        self._enable_settings_autosave()
        self.after(100, self._poll_events)

    def _build_variables(self) -> None:
        self.trigger_var = StringVar(value=self.settings.get("trigger", "subject_token"))
        self.epochs_var = IntVar(value=int(self.settings.get("epochs", 12)))
        self.learning_rate_var = StringVar(value=self.settings.get("learning_rate", "0.0001"))
        self.rank_var = IntVar(value=int(self.settings.get("rank", 16)))
        self.alpha_var = IntVar(value=int(self.settings.get("alpha", 16)))
        self.blocks_var = IntVar(value=int(self.settings.get("blocks", 20)))
        self.run_name_var = StringVar(value=self.settings.get("run_name", "my_video_lora"))
        self.lora_continuation_var = StringVar(value=self.settings.get("lora_continuation", ""))
        self.status_var = StringVar(value="Ready")
        self.dataset_summary_var = StringVar(value="No clips loaded")
        self.dataset_selection_var = StringVar(value="No clips selected")
        self.caption_scope_var = StringVar(value="Select one or more clips to edit captions")
        self.model_status_var = StringVar(value="Checking models...")
        self.workflow_status_var = StringVar(value="Idle")
        self.generation_status_var = StringVar(value="Idle")
        self.settings_status_var = StringVar(value="Settings restore automatically")
        self.job_eta_var = StringVar(value="No job running")
        self.checkpoint_var = StringVar(value=self.settings.get("generation_checkpoint", ""))
        self.prompt_var = StringVar(value=self.settings.get(
            "generation_prompt",
            "subject_token speaking directly to the camera, medium close-up, static camera",
        ))
        self.lora_strength_var = DoubleVar(value=float(self.settings.get("generation_strength", 0.8)))
        # Wan 2.1 T2V accepts these 480p size buckets for generation.
        self.width_var = IntVar(value=832)
        self.height_var = IntVar(value=480)
        self.resolution_var = StringVar(value="Landscape — 832 x 480")
        saved_resolution = str(self.settings.get("generation_resolution", "Landscape — 832 x 480"))
        self.resolution_var.set(saved_resolution)
        if saved_resolution.startswith("Portrait"):
            self.width_var.set(480)
            self.height_var.set(832)
        self.fps_var = IntVar(value=int(self.settings.get("generation_fps", 12)))
        saved_duration = float(self.settings.get("generation_duration", 2.0))
        self.duration_var = DoubleVar(value=saved_duration)
        self.frames_var = IntVar(value=self._frames_for_seconds(saved_duration, self.fps_var.get()))
        self.duration_label_var = StringVar()
        self.steps_var = IntVar(value=int(self.settings.get("generation_steps", 20)))
        # 29 is maximum safety; 24 is a conservative first speed/VRAM test for a 12 GB RTX 3060.
        self.generation_blocks_var = IntVar(value=int(self.settings.get("generation_blocks", 24)))
        self.experimental_speed_var = BooleanVar(value=bool(self.settings.get("experimental_speed", True)))
        self.seed_var = IntVar(value=int(self.settings.get("generation_seed", 1701)))
        self.randomize_seed_var = BooleanVar(value=bool(self.settings.get("generation_randomize_seed", False)))
        self.auto_prompt_var = BooleanVar(value=bool(self.settings.get("generation_auto_prompt", False)))
        self.auto_scroll_var = BooleanVar(value=bool(self.settings.get("auto_scroll", True)))
        self._update_duration_label()

    def _configure_theme(self) -> None:
        self.configure(bg="#151922")
        style = ttk.Style(self)
        style.theme_use("clam")
        style.configure(".", background="#151922", foreground="#e7eaf0", fieldbackground="#242a36")
        style.configure("TFrame", background="#151922")
        style.configure("Card.TFrame", background="#1c212c")
        style.configure("TLabel", background="#151922", foreground="#e7eaf0")
        style.configure("Card.TLabel", background="#1c212c", foreground="#e7eaf0")
        style.configure("Muted.TLabel", background="#151922", foreground="#9aa4b5")
        style.configure("Title.TLabel", font=("Segoe UI Semibold", 18), foreground="#f4f6fb")
        style.configure("Section.TLabel", font=("Segoe UI Semibold", 11), foreground="#f4f6fb")
        style.configure("TButton", padding=(10, 7), background="#30394a", foreground="#f4f6fb")
        style.map("TButton", background=[("active", "#3c475c"), ("disabled", "#252a34")])
        style.configure("Accent.TButton", background="#5568e8", foreground="white")
        style.map("Accent.TButton", background=[("active", "#687af0"), ("disabled", "#343b65")])
        style.configure("Danger.TButton", background="#873f4b", foreground="white")
        style.map("Danger.TButton", background=[("active", "#a14c5a")])
        style.configure("TLabelframe", background="#1c212c", foreground="#e7eaf0", padding=9)
        style.configure("TLabelframe.Label", background="#1c212c", foreground="#e7eaf0", font=("Segoe UI Semibold", 10))
        style.configure("TNotebook", background="#151922", borderwidth=0)
        style.configure("TNotebook.Tab", padding=(18, 9), background="#242a36", foreground="#c7ced9")
        style.map("TNotebook.Tab", background=[("selected", "#3b4660")], foreground=[("selected", "white")])
        style.configure("Treeview", background="#202631", foreground="#e7eaf0", fieldbackground="#202631", rowheight=27)
        style.configure("Treeview.Heading", background="#30394a", foreground="#f4f6fb")
        style.map("Treeview", background=[("selected", "#465887")])
        style.configure("TEntry", fieldbackground="#242a36", foreground="#f4f6fb", insertcolor="white")
        style.configure("TCombobox", fieldbackground="#242a36", foreground="#f4f6fb")
        style.configure("Horizontal.TProgressbar", troughcolor="#242a36", background="#6478ed")

    def _build_ui(self) -> None:
        self.columnconfigure(0, weight=1)
        self.rowconfigure(1, weight=1)

        header = ttk.Frame(self, padding=(16, 12))
        header.grid(row=0, column=0, sticky="ew")
        header.columnconfigure(1, weight=1)
        ttk.Label(header, text="LoRA Video Trainer", style="Title.TLabel").grid(row=0, column=0, sticky="w")
        ttk.Label(header, textvariable=self.model_status_var, style="Muted.TLabel").grid(row=0, column=1, sticky="e")

        self.notebook = ttk.Notebook(self)
        self.notebook.grid(row=1, column=0, sticky="nsew", padx=12, pady=(0, 8))
        self.dataset_tab = ttk.Frame(self.notebook, padding=12)
        self.training_tab = ttk.Frame(self.notebook, padding=12)
        self.generate_tab = ttk.Frame(self.notebook, padding=12)
        self.storyboard_tab = ttk.Frame(self.notebook, padding=12)
        self.notebook.add(self.dataset_tab, text="Dataset")
        self.notebook.add(self.training_tab, text="Training")
        self.notebook.add(self.generate_tab, text="Generate")
        self.notebook.add(self.storyboard_tab, text="Storyboard")
        self.notebook.bind("<<NotebookTabChanged>>", self._on_tab_changed)

        self._build_dataset_tab()
        self._build_training_tab()
        self._build_generate_tab()
        self._build_storyboard_tab()

        footer = ttk.Frame(self, padding=(14, 4, 14, 10))
        footer.grid(row=2, column=0, sticky="ew")
        footer.columnconfigure(0, weight=1)
        ttk.Label(footer, textvariable=self.status_var, style="Muted.TLabel").grid(row=0, column=0, sticky="w")
        ttk.Label(footer, textvariable=self.settings_status_var, style="Muted.TLabel").grid(row=0, column=1, padx=12)
        ttk.Button(footer, text="Open Project Folder", command=lambda: self._open_folder(ROOT)).grid(row=0, column=2)

    def _build_dataset_tab(self) -> None:
        tab = self.dataset_tab
        tab.columnconfigure(0, weight=3)
        tab.columnconfigure(1, weight=2)
        tab.rowconfigure(2, weight=1)

        actions = ttk.LabelFrame(tab, text="Dataset Actions")
        actions.grid(row=0, column=0, columnspan=2, sticky="ew")
        ttk.Button(actions, text="Add Video Clips", style="Accent.TButton", command=self._add_videos).pack(side="left", padx=4)
        ttk.Button(actions, text="Split Video(s)", command=self._split_long_video).pack(side="left", padx=4)
        ttk.Button(actions, text="Open Dataset Folder", command=lambda: self._open_folder(DATASET_DIR)).pack(side="left", padx=4)
        ttk.Button(actions, text="Validate Dataset", command=self._validate_dataset).pack(side="left", padx=4)
        ttk.Button(actions, text="Refresh", command=self._refresh_dataset).pack(side="left", padx=4)
        ttk.Button(actions, text="Remove Selected", style="Danger.TButton", command=self._remove_selected).pack(side="right", padx=4)
        ttk.Button(actions, text="Clear Dataset", style="Danger.TButton", command=self._clear_dataset).pack(side="right", padx=4)

        summary = ttk.Frame(tab, padding=(2, 9))
        summary.grid(row=1, column=0, columnspan=2, sticky="ew")
        summary.columnconfigure(1, weight=1)
        ttk.Label(summary, textvariable=self.dataset_summary_var, style="Section.TLabel").grid(row=0, column=0, sticky="w")
        ttk.Label(summary, text="Trigger token:").grid(row=0, column=2, padx=(12, 4))
        trigger = ttk.Entry(summary, textvariable=self.trigger_var, width=18)
        trigger.grid(row=0, column=3)

        list_frame = ttk.LabelFrame(tab, text="Clips")
        list_frame.grid(row=2, column=0, sticky="nsew", padx=(0, 6))
        list_frame.columnconfigure(0, weight=1)
        list_frame.rowconfigure(1, weight=1)
        selection_actions = ttk.Frame(list_frame)
        selection_actions.grid(row=0, column=0, columnspan=2, sticky="ew", pady=(0, 6))
        ttk.Button(selection_actions, text="Select All", command=self._select_all_videos).pack(side="left")
        ttk.Button(selection_actions, text="Clear Selection", command=self._clear_video_selection).pack(side="left", padx=6)
        ttk.Label(selection_actions, textvariable=self.dataset_selection_var, style="Card.TLabel").pack(side="right")
        columns = ("duration", "size", "caption")
        self.video_tree = ttk.Treeview(list_frame, columns=columns, show="tree headings", selectmode="extended")
        self.video_tree.heading("#0", text="Video")
        self.video_tree.heading("duration", text="Duration")
        self.video_tree.heading("size", text="Resolution")
        self.video_tree.heading("caption", text="Caption")
        self.video_tree.column("#0", width=250)
        self.video_tree.column("duration", width=85, anchor="center")
        self.video_tree.column("size", width=100, anchor="center")
        self.video_tree.column("caption", width=260)
        tree_scroll = ttk.Scrollbar(list_frame, orient="vertical", command=self.video_tree.yview)
        self.video_tree.configure(yscrollcommand=tree_scroll.set)
        self.video_tree.grid(row=1, column=0, sticky="nsew")
        tree_scroll.grid(row=1, column=1, sticky="ns")
        self.video_tree.bind("<<TreeviewSelect>>", self._on_video_selected)

        editor = ttk.LabelFrame(tab, text="Preview and Caption")
        editor.grid(row=2, column=1, sticky="nsew", padx=(6, 0))
        editor.columnconfigure(0, weight=1)
        editor.rowconfigure(0, weight=3)
        editor.rowconfigure(2, weight=2)
        self.preview_label = ttk.Label(editor, text="Select a clip", anchor="center")
        self.preview_label.grid(row=0, column=0, sticky="nsew", pady=(0, 8))
        ttk.Label(editor, textvariable=self.caption_scope_var, wraplength=430).grid(row=1, column=0, sticky="w")
        self.caption_text = tk.Text(editor, height=8, wrap="word", bg="#202631", fg="#f1f3f7", insertbackground="white", relief="flat", padx=8, pady=8)
        self.caption_text.grid(row=2, column=0, sticky="nsew", pady=6)
        caption_actions = ttk.Frame(editor)
        caption_actions.grid(row=3, column=0, sticky="ew")
        ttk.Button(caption_actions, text="Add Trigger to Selected", command=self._insert_trigger).pack(side="left")
        ttk.Button(caption_actions, text="Save Caption & Next", style="Accent.TButton", command=self._save_caption).pack(side="right")
        self.caption_text.bind("<Control-Return>", self._save_caption_and_consume_event)

    def _build_training_tab(self) -> None:
        tab = self.training_tab
        tab.columnconfigure(0, weight=1)
        tab.columnconfigure(1, weight=2)
        tab.rowconfigure(1, weight=1)

        settings = ttk.LabelFrame(tab, text="RTX 3060 Training Settings")
        settings.grid(row=0, column=0, sticky="nsew", padx=(0, 6))
        settings.columnconfigure(1, weight=1)
        rows = [
            ("Run name", self.run_name_var),
            ("Epochs", self.epochs_var),
            ("Learning rate", self.learning_rate_var),
            ("LoRA rank", self.rank_var),
            ("LoRA alpha", self.alpha_var),
            ("Blocks to swap", self.blocks_var),
        ]
        for row, (label, variable) in enumerate(rows):
            ttk.Label(settings, text=label).grid(row=row, column=0, sticky="w", pady=5, padx=(0, 8))
            if isinstance(variable, IntVar):
                limits = (1, 100) if "Epoch" in label else (1, 29 if "Blocks" in label else 128)
                widget = ttk.Spinbox(settings, from_=limits[0], to=limits[1], textvariable=variable)
            else:
                widget = ttk.Entry(settings, textvariable=variable)
            widget.grid(row=row, column=1, sticky="ew", pady=5)
        continuation_row = len(rows)
        ttk.Label(settings, text="LoRA Continuation").grid(row=continuation_row, column=0, sticky="w", pady=5, padx=(0, 8))
        continuation = ttk.Frame(settings)
        continuation.grid(row=continuation_row, column=1, sticky="ew", pady=5)
        continuation.columnconfigure(0, weight=1)
        ttk.Entry(continuation, textvariable=self.lora_continuation_var).grid(row=0, column=0, sticky="ew")
        ttk.Button(continuation, text="Browse...", command=self._browse_lora_continuation).grid(row=0, column=1, padx=(5, 0))
        ttk.Button(continuation, text="Clear", command=lambda: self.lora_continuation_var.set("")).grid(row=0, column=2, padx=(5, 0))
        ttk.Label(
            settings,
            text="Optional: start from an existing LoRA's weights and train them further on this dataset.",
            wraplength=500,
            style="Card.TLabel",
        ).grid(row=continuation_row + 1, column=0, columnspan=2, sticky="ew", pady=(2, 4))
        ttk.Label(settings, text="Preset: 448x256, 25/49 frames, batch 1, fp8 base, bf16, gradient checkpointing", wraplength=500, style="Card.TLabel").grid(row=continuation_row + 2, column=0, columnspan=2, sticky="ew", pady=(6, 2))

        workflow = ttk.LabelFrame(tab, text="Workflow")
        workflow.grid(row=0, column=1, sticky="nsew", padx=(6, 0))
        for index in range(3):
            workflow.columnconfigure(index, weight=1)
        ttk.Button(workflow, text="1. Cache Video", command=lambda: self._run_scripts("Cache video", [ROOT / "scripts" / "01_cache_video.ps1"])).grid(row=0, column=0, sticky="ew", padx=4, pady=4)
        ttk.Button(workflow, text="2. Cache Captions", command=lambda: self._run_scripts("Cache captions", [ROOT / "scripts" / "02_cache_text.ps1"])).grid(row=0, column=1, sticky="ew", padx=4, pady=4)
        ttk.Button(workflow, text="3. Train LoRA", command=self._start_training).grid(row=0, column=2, sticky="ew", padx=4, pady=4)
        ttk.Button(workflow, text="Run Full Training Pipeline", style="Accent.TButton", command=self._run_full_pipeline).grid(row=1, column=0, columnspan=2, sticky="ew", padx=4, pady=8)
        self.stop_button = ttk.Button(workflow, text="Stop Current Job", style="Danger.TButton", command=self._stop_job, state="disabled")
        self.stop_button.grid(row=1, column=2, sticky="ew", padx=4, pady=8)
        ttk.Button(workflow, text="Open Output Folder", command=lambda: self._open_folder(OUTPUT_DIR)).grid(row=2, column=0, sticky="ew", padx=4, pady=4)
        ttk.Button(workflow, text="Open TensorBoard", command=self._open_tensorboard).grid(row=2, column=1, sticky="ew", padx=4, pady=4)
        ttk.Button(workflow, text="Check Setup", command=self._verify_setup).grid(row=2, column=2, sticky="ew", padx=4, pady=4)
        ttk.Label(workflow, textvariable=self.workflow_status_var, style="Card.TLabel").grid(row=3, column=0, columnspan=3, sticky="w", padx=4, pady=(8, 2))

        log_frame = ttk.LabelFrame(tab, text="Live Output")
        log_frame.grid(row=1, column=0, columnspan=2, sticky="nsew", pady=(12, 0))
        log_frame.columnconfigure(0, weight=1)
        log_frame.rowconfigure(1, weight=1)
        top = ttk.Frame(log_frame)
        top.grid(row=0, column=0, sticky="ew", pady=(0, 5))
        top.columnconfigure(1, weight=1)
        ttk.Checkbutton(top, text="Auto-scroll", variable=self.auto_scroll_var).grid(row=0, column=0, sticky="w")
        self.job_progress = ttk.Progressbar(top, mode="indeterminate")
        self.job_progress.grid(row=0, column=1, sticky="ew", padx=10)
        ttk.Label(top, textvariable=self.job_eta_var, style="Muted.TLabel").grid(row=0, column=2, padx=(0, 10))
        ttk.Button(top, text="Clear", command=lambda: self.training_log.delete("1.0", END)).grid(row=0, column=3)
        self.training_log = tk.Text(log_frame, wrap="word", bg="#10141b", fg="#dce3ee", insertbackground="white", relief="flat")
        log_scroll = ttk.Scrollbar(log_frame, orient="vertical", command=self.training_log.yview)
        self.training_log.configure(yscrollcommand=log_scroll.set)
        self.training_log.grid(row=1, column=0, sticky="nsew")
        log_scroll.grid(row=1, column=1, sticky="ns")

    def _build_generate_tab(self) -> None:
        tab = self.generate_tab
        tab.columnconfigure(0, weight=1)
        tab.rowconfigure(2, weight=1)

        controls = ttk.LabelFrame(tab, text="Generate a Test Video")
        controls.grid(row=0, column=0, sticky="ew")
        controls.columnconfigure(1, weight=1)
        ttk.Label(controls, text="LoRA checkpoint").grid(row=0, column=0, sticky="w", pady=5)
        self.checkpoint_combo = ttk.Combobox(controls, textvariable=self.checkpoint_var, state="readonly")
        self.checkpoint_combo.grid(row=0, column=1, columnspan=5, sticky="ew", padx=6, pady=5)
        ttk.Button(controls, text="Refresh", command=self._refresh_checkpoints).grid(row=0, column=6, padx=4)
        ttk.Label(controls, text="Prompt").grid(row=1, column=0, sticky="w", pady=5)
        self.prompt_entry = ttk.Entry(controls, textvariable=self.prompt_var)
        self.prompt_entry.grid(row=1, column=1, columnspan=4, sticky="ew", padx=6, pady=5)
        ttk.Checkbutton(
            controls, text="AutoPrompt", variable=self.auto_prompt_var,
            command=self._on_auto_prompt_toggled,
        ).grid(row=1, column=5, sticky="w", padx=4)
        self.auto_prompt_pick_button = ttk.Button(controls, text="Pick another", command=self._pick_auto_prompt)
        self.auto_prompt_pick_button.grid(row=1, column=6, padx=4)
        self._sync_auto_prompt_controls()

        fields = [
            ("Strength", self.lora_strength_var, 0.0, 2.0),
            ("Steps", self.steps_var, 1, 100),
            ("Blocks to swap", self.generation_blocks_var, 0, 29),
            ("Seed", self.seed_var, 0, 2147483647),
            ("Playback FPS", self.fps_var, 4, 60),
        ]
        for index, (label, variable, low, high) in enumerate(fields):
            ttk.Label(controls, text=label).grid(row=2, column=index, sticky="w", padx=4)
            spinbox = ttk.Spinbox(controls, from_=low, to=high, textvariable=variable, width=11)
            spinbox.grid(row=3, column=index, sticky="ew", padx=4, pady=(2, 7))
            if variable is self.fps_var:
                spinbox.configure(command=self._on_fps_changed)
                spinbox.bind("<FocusOut>", self._on_fps_changed)
                spinbox.bind("<Return>", self._on_fps_changed)

        ttk.Checkbutton(
            controls, text="Randomize each generation", variable=self.randomize_seed_var,
        ).grid(row=2, column=6, sticky="w", padx=4)

        ttk.Label(controls, text="Wan format").grid(row=2, column=5, sticky="w", padx=4)
        resolution_combo = ttk.Combobox(
            controls,
            textvariable=self.resolution_var,
            values=("Landscape — 832 x 480", "Portrait — 480 x 832"),
            state="readonly",
            width=25,
        )
        resolution_combo.grid(row=3, column=5, columnspan=2, sticky="ew", padx=4, pady=(2, 7))
        resolution_combo.bind("<<ComboboxSelected>>", self._on_resolution_selected)

        ttk.Label(controls, text="Video length").grid(row=4, column=0, sticky="w", padx=4, pady=(4, 0))
        duration_scale = tk.Scale(
            controls,
            from_=2.0,
            to=15.0,
            resolution=0.5,
            orient=tk.HORIZONTAL,
            variable=self.duration_var,
            command=self._on_duration_changed,
            showvalue=False,
            length=500,
            bg="#1c212c",
            fg="#e7eaf0",
            troughcolor="#30394a",
            highlightthickness=0,
            activebackground="#687af0",
        )
        duration_scale.grid(row=4, column=1, columnspan=4, sticky="ew", padx=6, pady=(2, 0))
        ttk.Label(controls, textvariable=self.duration_label_var, style="Card.TLabel").grid(row=4, column=5, columnspan=2, sticky="w", padx=6, pady=(4, 0))
        ttk.Checkbutton(
            controls,
            text="Experimental speed mode (TF32; compiled mode disabled after compatibility failure)",
            variable=self.experimental_speed_var,
        ).grid(row=5, column=0, columnspan=7, sticky="w", padx=4, pady=(5, 2))
        ttk.Label(
            controls,
            text="Lower Blocks to swap uses more VRAM and may reduce step time. Use 0-2 for speed if it fits; raise it if generation runs out of memory.",
            style="Card.TLabel",
            wraplength=860,
        ).grid(row=6, column=0, columnspan=7, sticky="w", padx=4, pady=(0, 4))

        actions = ttk.Frame(tab, padding=(0, 10))
        actions.grid(row=1, column=0, sticky="ew")
        ttk.Button(actions, text="Generate Video", style="Accent.TButton", command=self._generate_video).pack(side="left")
        ttk.Button(actions, text="Fast Preview Preset", command=self._set_fast_preview).pack(side="left", padx=7)
        self.generation_stop_button = ttk.Button(actions, text="Stop", style="Danger.TButton", command=self._stop_job, state="disabled")
        self.generation_stop_button.pack(side="left", padx=7)
        ttk.Button(actions, text="Open Samples Folder", command=lambda: self._open_folder(SAMPLES_DIR)).pack(side="left")
        ttk.Label(actions, textvariable=self.generation_status_var).pack(side="right")

        outputs = ttk.LabelFrame(tab, text="Generated Videos (double-click to open)")
        outputs.grid(row=2, column=0, sticky="nsew")
        outputs.columnconfigure(0, weight=1)
        outputs.rowconfigure(0, weight=1)
        self.sample_list = tk.Listbox(outputs, bg="#202631", fg="#eef1f6", selectbackground="#465887", relief="flat")
        sample_scroll = ttk.Scrollbar(outputs, orient="vertical", command=self.sample_list.yview)
        self.sample_list.configure(yscrollcommand=sample_scroll.set)
        self.sample_list.grid(row=0, column=0, sticky="nsew")
        sample_scroll.grid(row=0, column=1, sticky="ns")
        self.sample_list.bind("<Double-1>", self._open_selected_sample)
        self.sample_list.bind("<<ListboxSelect>>", self._show_selected_generation_preview)
        preview_panel = ttk.Frame(outputs, style="Card.TFrame", padding=8)
        preview_panel.grid(row=0, column=2, sticky="nsew", padx=(10, 0))
        ttk.Label(preview_panel, text="First-frame preview", style="Section.TLabel").pack(anchor="w")
        self.generation_preview_label = ttk.Label(
            preview_panel,
            text="A finished generation's first frame will appear here.\n\n"
                 "Wan decodes viewable pixels after denoising completes.",
            style="Card.TLabel",
            anchor="center",
            justify="center",
            width=42,
        )
        self.generation_preview_label.pack(expand=True, fill="both", pady=(8, 0))
        self._refresh_samples()

    def _build_storyboard_tab(self) -> None:
        tab = self.storyboard_tab
        tab.columnconfigure(0, weight=3)
        tab.columnconfigure(1, weight=2)
        tab.rowconfigure(1, weight=1)

        actions = ttk.LabelFrame(tab, text="Storyboard Actions")
        actions.grid(row=0, column=0, columnspan=2, sticky="ew", pady=(0, 10))
        ttk.Button(actions, text="Add Scene", style="Accent.TButton", command=self._add_storyboard_scene).pack(side="left", padx=4)
        ttk.Button(actions, text="Edit Scene", command=self._edit_storyboard_scene).pack(side="left", padx=4)
        ttk.Button(actions, text="Delete Scene", style="Danger.TButton", command=self._delete_storyboard_scene).pack(side="left", padx=4)
        ttk.Button(actions, text="Move Up", command=lambda: self._move_storyboard_scene(-1)).pack(side="left", padx=(18, 4))
        ttk.Button(actions, text="Move Down", command=lambda: self._move_storyboard_scene(1)).pack(side="left", padx=4)
        ttk.Button(actions, text="Generate Selected Scene", command=self._generate_storyboard_scene).pack(side="right", padx=4)

        scenes = ttk.LabelFrame(tab, text="Scenes (double-click to edit)")
        scenes.grid(row=1, column=0, sticky="nsew", padx=(0, 6))
        scenes.columnconfigure(0, weight=1)
        scenes.rowconfigure(0, weight=1)
        columns = ("duration", "status", "prompt")
        self.storyboard_tree = ttk.Treeview(scenes, columns=columns, show="tree headings", selectmode="browse")
        self.storyboard_tree.heading("#0", text="Scene")
        self.storyboard_tree.heading("duration", text="Length")
        self.storyboard_tree.heading("status", text="Status")
        self.storyboard_tree.heading("prompt", text="Prompt")
        self.storyboard_tree.column("#0", width=210)
        self.storyboard_tree.column("duration", width=75, anchor="center")
        self.storyboard_tree.column("status", width=110, anchor="center")
        self.storyboard_tree.column("prompt", width=480)
        scene_scroll = ttk.Scrollbar(scenes, orient="vertical", command=self.storyboard_tree.yview)
        self.storyboard_tree.configure(yscrollcommand=scene_scroll.set)
        self.storyboard_tree.grid(row=0, column=0, sticky="nsew")
        scene_scroll.grid(row=0, column=1, sticky="ns")
        self.storyboard_tree.bind("<<TreeviewSelect>>", self._on_storyboard_scene_selected)
        self.storyboard_tree.bind("<Double-1>", self._edit_storyboard_scene)

        takes = ttk.LabelFrame(tab, text="Generated Takes")
        takes.grid(row=1, column=1, sticky="nsew", padx=(6, 0))
        takes.columnconfigure(0, weight=1)
        takes.rowconfigure(1, weight=1)
        self.storyboard_scene_info_var = StringVar(value="Select a scene to see its takes.")
        ttk.Label(takes, textvariable=self.storyboard_scene_info_var, style="Card.TLabel", wraplength=410).grid(
            row=0, column=0, columnspan=2, sticky="ew", pady=(0, 7)
        )
        self.storyboard_take_list = tk.Listbox(
            takes, bg="#202631", fg="#eef1f6", selectbackground="#465887", relief="flat"
        )
        take_scroll = ttk.Scrollbar(takes, orient="vertical", command=self.storyboard_take_list.yview)
        self.storyboard_take_list.configure(yscrollcommand=take_scroll.set)
        self.storyboard_take_list.grid(row=1, column=0, sticky="nsew")
        take_scroll.grid(row=1, column=1, sticky="ns")
        take_actions = ttk.Frame(takes)
        take_actions.grid(row=2, column=0, columnspan=2, sticky="ew", pady=(8, 0))
        ttk.Button(take_actions, text="Preview Take", command=self._preview_storyboard_take).pack(side="left")
        ttk.Button(take_actions, text="Use This Take", style="Accent.TButton", command=self._accept_storyboard_take).pack(side="left", padx=6)
        ttk.Button(take_actions, text="Open Take", command=self._open_storyboard_take).pack(side="left")
        ttk.Button(take_actions, text="Remove From Scene", command=self._remove_storyboard_take).pack(side="right")
        self._refresh_storyboard()

    def _load_settings(self) -> dict:
        try:
            loaded = json.loads(SETTINGS_PATH.read_text(encoding="utf-8"))
            return loaded if isinstance(loaded, dict) else {}
        except (OSError, ValueError):
            return {}

    def _load_storyboard(self) -> list[dict]:
        try:
            loaded = json.loads(STORYBOARD_PATH.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return []
        if not isinstance(loaded, list):
            return []
        scenes: list[dict] = []
        for item in loaded:
            if not isinstance(item, dict) or not str(item.get("prompt", "")).strip():
                continue
            scenes.append({
                "id": str(item.get("id") or uuid.uuid4().hex),
                "title": str(item.get("title") or f"Scene {len(scenes) + 1}"),
                "prompt": str(item.get("prompt", "")).strip(),
                "duration": float(item.get("duration", 5.0)),
                "takes": [str(path) for path in item.get("takes", []) if isinstance(path, str)],
                "accepted_take": str(item.get("accepted_take", "")),
            })
        return scenes

    def _save_storyboard(self) -> bool:
        try:
            temporary_path = STORYBOARD_PATH.with_suffix(".json.tmp")
            temporary_path.write_text(json.dumps(self.storyboard_scenes, indent=2), encoding="utf-8")
            temporary_path.replace(STORYBOARD_PATH)
            return True
        except OSError as exc:
            messagebox.showerror("Could Not Save Storyboard", str(exc))
            return False

    def _enable_settings_autosave(self) -> None:
        """Save user-editable fields shortly after they change without interrupting typing."""
        variables = (
            self.trigger_var, self.epochs_var, self.learning_rate_var, self.rank_var,
            self.alpha_var, self.blocks_var, self.run_name_var, self.lora_continuation_var, self.checkpoint_var,
            self.prompt_var, self.lora_strength_var, self.resolution_var, self.fps_var,
            self.duration_var, self.steps_var, self.generation_blocks_var,
            self.experimental_speed_var, self.seed_var, self.auto_scroll_var,
            self.randomize_seed_var, self.auto_prompt_var,
        )
        for variable in variables:
            variable.trace_add("write", self._schedule_settings_save)

    def _on_tab_changed(self, _event=None) -> None:
        self._save_settings()

    def _restore_last_tab(self) -> None:
        wanted = self.settings.get("last_tab", "Dataset")
        for tab_id in self.notebook.tabs():
            if self.notebook.tab(tab_id, "text") == wanted:
                self.notebook.select(tab_id)
                break

    def _schedule_settings_save(self, *_args) -> None:
        self.settings_status_var.set("Saving changes…")
        if self.settings_save_after_id is not None:
            self.after_cancel(self.settings_save_after_id)
        self.settings_save_after_id = self.after(450, self._save_settings)

    @staticmethod
    def _variable_value(variable, fallback):
        try:
            return variable.get()
        except tk.TclError:
            return fallback

    def _save_settings(self) -> bool:
        previous = self.settings
        payload = {
            "settings_version": 3,
            "trigger": str(self._variable_value(self.trigger_var, previous.get("trigger", "subject_token"))).strip(),
            "epochs": self._variable_value(self.epochs_var, previous.get("epochs", 12)),
            "learning_rate": str(self._variable_value(self.learning_rate_var, previous.get("learning_rate", "0.0001"))).strip(),
            "rank": self._variable_value(self.rank_var, previous.get("rank", 16)),
            "alpha": self._variable_value(self.alpha_var, previous.get("alpha", 16)),
            "blocks": self._variable_value(self.blocks_var, previous.get("blocks", 20)),
            "run_name": str(self._variable_value(self.run_name_var, previous.get("run_name", ""))).strip(),
            "lora_continuation": str(self._variable_value(self.lora_continuation_var, previous.get("lora_continuation", ""))).strip(),
            "generation_checkpoint": self._variable_value(self.checkpoint_var, previous.get("generation_checkpoint", "")),
            "generation_prompt": self._variable_value(self.prompt_var, previous.get("generation_prompt", "")),
            "generation_strength": self._variable_value(self.lora_strength_var, previous.get("generation_strength", 0.8)),
            "generation_resolution": self._variable_value(self.resolution_var, previous.get("generation_resolution", "Landscape — 832 x 480")),
            "generation_duration": self._variable_value(self.duration_var, previous.get("generation_duration", 2.0)),
            "generation_steps": self._variable_value(self.steps_var, previous.get("generation_steps", 20)),
            "generation_blocks": self._variable_value(self.generation_blocks_var, previous.get("generation_blocks", 24)),
            "generation_fps": self._variable_value(self.fps_var, previous.get("generation_fps", 12)),
            "generation_seed": self._variable_value(self.seed_var, previous.get("generation_seed", 1701)),
            "generation_randomize_seed": self._variable_value(self.randomize_seed_var, previous.get("generation_randomize_seed", False)),
            "generation_auto_prompt": self._variable_value(self.auto_prompt_var, previous.get("generation_auto_prompt", False)),
            "experimental_speed": self._variable_value(self.experimental_speed_var, previous.get("experimental_speed", True)),
            "auto_scroll": self._variable_value(self.auto_scroll_var, previous.get("auto_scroll", True)),
            "last_tab": self.notebook.tab(self.notebook.select(), "text"),
            "job_history": self.job_history,
        }
        try:
            temporary_path = SETTINGS_PATH.with_suffix(".json.tmp")
            temporary_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
            temporary_path.replace(SETTINGS_PATH)
        except OSError as exc:
            self.settings_status_var.set("Could not save settings")
            self.status_var.set(f"Could not save settings: {exc}")
            return False
        self.settings = payload
        self.settings_save_after_id = None
        self.settings_status_var.set(f"Settings saved automatically at {datetime.now():%I:%M:%S %p}")
        return True

    def _refresh_status(self) -> None:
        found = sum(path.exists() for path in (DIT_PATH, VAE_PATH, T5_PATH))
        self.model_status_var.set(f"12 GB NVIDIA preset | Models: {found}/3 | Local workspace")

    def _video_info(self, path: Path) -> tuple[str, str]:
        try:
            with av.open(str(path)) as container:
                stream = container.streams.video[0]
                duration = 0.0
                if stream.duration is not None and stream.time_base is not None:
                    duration = float(stream.duration * stream.time_base)
                elif container.duration:
                    duration = float(container.duration / av.time_base)
                return f"{duration:.1f}s", f"{stream.width}x{stream.height}"
        except Exception:
            return "Unreadable", "-"

    def _refresh_dataset(self) -> None:
        DATASET_DIR.mkdir(parents=True, exist_ok=True)
        previous_selection = set(self.video_tree.selection())
        if not previous_selection and self.selected_video:
            previous_selection.add(str(self.selected_video))
        for item in self.video_tree.get_children():
            self.video_tree.delete(item)
        videos = sorted(path for path in DATASET_DIR.iterdir() if path.suffix.lower() in VIDEO_EXTENSIONS)
        caption_count = 0
        for video in videos:
            caption_path = video.with_suffix(".txt")
            caption = caption_path.read_text(encoding="utf-8-sig", errors="replace").strip() if caption_path.exists() else "[missing]"
            if caption_path.exists() and caption:
                caption_count += 1
            duration, size = self._video_info(video)
            item = self.video_tree.insert("", END, iid=str(video), text=video.name, values=(duration, size, caption[:80]))
            if str(video) in previous_selection:
                self.video_tree.selection_add(item)
        self.dataset_summary_var.set(f"{len(videos)} clips | {caption_count} captions | {len(videos) - caption_count} missing")
        selected_count = len(self.video_tree.selection())
        self.dataset_selection_var.set(f"{selected_count} clip{'s' if selected_count != 1 else ''} selected" if selected_count else "No clips selected")
        self.status_var.set("Dataset refreshed")

    def _selected_videos(self) -> list[Path]:
        return [
            Path(item) for item in self.video_tree.selection()
            if Path(item).exists() and Path(item).parent.resolve() == DATASET_DIR.resolve()
        ]

    def _select_all_videos(self) -> None:
        items = self.video_tree.get_children()
        if items:
            self.video_tree.selection_set(items)
            self.video_tree.focus(items[0])
            self._on_video_selected()

    def _clear_video_selection(self) -> None:
        self.video_tree.selection_remove(self.video_tree.selection())
        self.selected_video = None
        self.dataset_selection_var.set("No clips selected")
        self.caption_scope_var.set("Select one or more clips to edit captions")
        self._stop_preview()
        self.preview_label.configure(image="", text="Select a clip")
        self.caption_text.delete("1.0", END)

    def _add_videos(self) -> None:
        paths = filedialog.askopenfilenames(title="Select short video clips", filetypes=[("Video files", "*.mp4 *.mov *.mkv *.webm *.m4v *.avi"), ("All files", "*.*")])
        if not paths:
            return
        copied = 0
        for raw in paths:
            source = Path(raw)
            destination = DATASET_DIR / source.name
            if destination.exists():
                replace = messagebox.askyesno("Clip Already Exists", f"Replace {destination.name}?")
                if not replace:
                    continue
            shutil.copy2(source, destination)
            copied += 1
        self._refresh_dataset()
        self.status_var.set(f"Added {copied} clip(s)")

    def _split_long_video(self) -> None:
        sources = self._selected_videos()
        if not sources:
            source_paths = filedialog.askopenfilenames(
                title="Select one or more source videos to split",
                initialdir=str(DATASET_DIR),
                filetypes=[("Video files", "*.mp4 *.mov *.mkv *.webm *.m4v *.avi"), ("All files", "*.*")],
            )
            sources = [Path(path) for path in source_paths]
        if not sources:
            return
        count = simpledialog.askinteger(
            "Clips Per Source Video",
            f"How many evenly spaced 3-second clips should be created from each of the {len(sources)} selected video(s)?\n\n"
            "The same count is used for every source. You can remove unsuitable clips afterward.",
            initialvalue=32,
            minvalue=1,
            maxvalue=100,
            parent=self,
        )
        if count is None:
            return
        total_clips = len(sources) * count
        if len(sources) > 1 and not messagebox.askyesno(
            "Confirm Batch Split",
            f"Split {len(sources)} source videos into up to {total_clips} new 3-second clips?\n\n"
            "Each source's existing caption will be copied to its own new clips.",
        ):
            return
        active_sources = [source for source in sources if source.parent.resolve() == DATASET_DIR.resolve()]
        archive = False
        if active_sources:
            archive = messagebox.askyesno(
                "Archive Source Videos",
                f"After splitting, move {len(active_sources)} selected source video(s) from the active dataset and their old cache files into safe archive folders?\n\n"
                "Choose Yes so only the new short clips are used during the next training run. The original is preserved and can be restored.",
            )
            if not archive:
                messagebox.showwarning(
                    "Source Videos Will Remain Active",
                    "The selected sources will remain in the training dataset. Move them out before caching if you want only the new short clips to train.",
                )
        commands = []
        for source in sources:
            caption_file = source.with_suffix(".txt")
            command = [
                str(PYTHON), str(ROOT / "tools" / "split_video.py"), str(source),
                "--output-dir", str(DATASET_DIR), "--count", str(count), "--trigger", self.trigger_var.get().strip(),
                "--cache-dir", str(ROOT / "cache"),
            ]
            if caption_file.exists():
                command.extend(["--caption-file", str(caption_file)])
            if archive and source in active_sources:
                command.append("--archive-source")
            commands.append(command)
        self.notebook.select(self.training_tab)
        self._run_commands(f"Split {len(sources)} video(s)", commands)

    def _on_video_selected(self, _event=None) -> None:
        self._stop_preview()
        selected = self._selected_videos()
        selected_count = len(selected)
        self.dataset_selection_var.set(
            f"{selected_count} clip{'s' if selected_count != 1 else ''} selected" if selected_count else "No clips selected"
        )
        if not selected:
            self.selected_video = None
            return
        path = selected[0]
        self.selected_video = path
        captions = []
        for selected_path in selected:
            caption_path = selected_path.with_suffix(".txt")
            captions.append(caption_path.read_text(encoding="utf-8-sig", errors="replace").strip() if caption_path.exists() else "")
        self.caption_text.delete("1.0", END)
        if selected_count == 1:
            self.caption_scope_var.set("One visual caption for this short clip")
            self.caption_text.insert("1.0", captions[0])
        elif len(set(captions)) == 1:
            self.caption_scope_var.set(f"Editing {selected_count} clips with the same caption — Save applies to all selected clips")
            self.caption_text.insert("1.0", captions[0])
        else:
            self.caption_scope_var.set(
                f"{selected_count} clips have different captions — enter a replacement only if you want to overwrite all selected captions"
            )
            self.preview_label.configure(image="", text=f"{selected_count} clips selected\nPreview is available for a single clip")
            return
        try:
            self._load_looping_preview(path)
        except Exception as exc:
            self.preview_label.configure(image="", text=f"Preview unavailable\n{exc}")

    def _stop_preview(self) -> None:
        if self.preview_after_id is not None:
            self.after_cancel(self.preview_after_id)
            self.preview_after_id = None
        self.preview_frames = []
        self.preview_frame_index = 0
        self.preview_image = None

    def _load_looping_preview(self, path: Path) -> None:
        """Decode a small, looping visual preview without opening another player."""
        frames: list[ImageTk.PhotoImage] = []
        last_timestamp = -1.0
        with av.open(str(path)) as container:
            stream = container.streams.video[0]
            for frame in container.decode(stream):
                timestamp = float(frame.time) if frame.time is not None else last_timestamp + 0.1
                # Ten preview frames per second is smooth enough for captioning and avoids excess memory use.
                if last_timestamp >= 0 and timestamp - last_timestamp < 0.095:
                    continue
                image = Image.fromarray(frame.to_ndarray(format="rgb24"))
                image.thumbnail((430, 300), Image.Resampling.LANCZOS)
                frames.append(ImageTk.PhotoImage(image))
                last_timestamp = timestamp
                if len(frames) >= 120:
                    break
        if not frames:
            raise RuntimeError("The clip contains no decodable video frames.")
        self.preview_frames = frames
        self.preview_frame_index = 0
        self._show_next_preview_frame()

    def _show_next_preview_frame(self) -> None:
        if not self.preview_frames:
            return
        self.preview_image = self.preview_frames[self.preview_frame_index]
        self.preview_label.configure(image=self.preview_image, text="")
        self.preview_frame_index = (self.preview_frame_index + 1) % len(self.preview_frames)
        self.preview_after_id = self.after(100, self._show_next_preview_frame)

    def _insert_trigger(self) -> None:
        trigger = self.trigger_var.get().strip() or "subject_token"
        selected = self._selected_videos()
        if len(selected) > 1:
            updated = 0
            skipped_empty = 0
            for path in selected:
                caption_path = path.with_suffix(".txt")
                content = caption_path.read_text(encoding="utf-8-sig", errors="replace").strip() if caption_path.exists() else ""
                if not content:
                    skipped_empty += 1
                    continue
                if not self._caption_has_trigger(content, trigger):
                    caption_path.write_text(f"{trigger}, {content}\n", encoding="utf-8")
                    updated += 1
            self._refresh_dataset()
            self.status_var.set(
                f"Added trigger to {updated} caption(s)" + (f"; {skipped_empty} empty caption(s) still need text" if skipped_empty else "")
            )
            return
        content = self.caption_text.get("1.0", END).strip()
        if not self._caption_has_trigger(content, trigger):
            self.caption_text.delete("1.0", END)
            self.caption_text.insert("1.0", f"{trigger}, {content}" if content else f"{trigger}, ")

    def _save_caption(self) -> None:
        selected = self._selected_videos()
        if not selected:
            messagebox.showwarning("No Clips Selected", "Select one or more video clips first.")
            return
        caption = self.caption_text.get("1.0", END).strip()
        trigger = self.trigger_var.get().strip()
        if not caption:
            messagebox.showwarning("Empty Caption", "Enter a visual caption for this clip.")
            return
        if not trigger or not self._caption_has_trigger(caption, trigger):
            messagebox.showwarning("Missing Trigger", f"The caption must contain the exact trigger token '{trigger or 'subject_token'}'.")
            return
        if len(selected) > 1 and not messagebox.askyesno(
            "Replace Multiple Captions",
            f"Apply this exact caption to all {len(selected)} selected clips?\n\n"
            "This replaces their current captions. The video files are not changed.",
            icon="warning",
        ):
            return
        for path in selected:
            path.with_suffix(".txt").write_text(caption + "\n", encoding="utf-8")
        self._save_settings()
        next_item = None
        if len(selected) == 1:
            items = self.video_tree.get_children()
            current_item = str(selected[0])
            if current_item in items:
                current_index = items.index(current_item)
                if current_index + 1 < len(items):
                    next_item = items[current_index + 1]
        self._refresh_dataset()
        if next_item and self.video_tree.exists(next_item):
            self.video_tree.selection_set(next_item)
            self.video_tree.focus(next_item)
            self.video_tree.see(next_item)
            self._on_video_selected()
            self.caption_text.focus_set()
            self.status_var.set("Saved caption and moved to the next clip")
        else:
            self.status_var.set(f"Saved caption to {len(selected)} selected clip(s)")

    def _save_caption_and_consume_event(self, _event=None) -> str:
        self._save_caption()
        return "break"

    @staticmethod
    def _caption_has_trigger(caption: str, trigger: str) -> bool:
        if not trigger:
            return False
        return re.search(rf"(?<![\w]){re.escape(trigger)}(?![\w])", caption) is not None

    def _remove_selected(self) -> None:
        selected = self._selected_videos()
        if not selected:
            messagebox.showwarning("No Clips Selected", "Select one or more video clips first.")
            return
        names = selected[0].name if len(selected) == 1 else f"{len(selected)} selected clips"
        if not messagebox.askyesno("Remove Selected Clips", f"Remove {names} and their captions from this dataset?\n\nThe original source files outside this folder are not affected."):
            return
        for video in selected:
            caption = video.with_suffix(".txt")
            video.unlink()
            if caption.exists():
                caption.unlink()
        self.selected_video = None
        self._stop_preview()
        self.preview_label.configure(image="", text="Select a clip")
        self.caption_text.delete("1.0", END)
        self._refresh_dataset()

    def _clear_dataset(self) -> None:
        """Remove all imported training data and caches, keeping models and results."""
        if self.active_thread and self.active_thread.is_alive():
            messagebox.showwarning(
                "Job Still Running",
                "Stop or wait for the current job before clearing its dataset.",
            )
            return

        video_count = sum(
            1 for path in DATASET_DIR.rglob("*")
            if path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS
        ) if DATASET_DIR.exists() else 0
        caption_count = sum(1 for _ in DATASET_DIR.rglob("*.txt")) if DATASET_DIR.exists() else 0
        cache_count = sum(1 for path in CACHE_DIR.rglob("*") if path.is_file()) if CACHE_DIR.exists() else 0
        if not video_count and not caption_count and not cache_count:
            messagebox.showinfo("Dataset Already Empty", "There are no dataset clips, captions, or cached training files to clear.")
            return

        confirmed = messagebox.askyesno(
            "Clear Entire Dataset",
            f"Remove {video_count} video clip(s), {caption_count} caption file(s), and {cache_count} cached training file(s)?\n\n"
            "This permanently clears the current dataset, including archived source clips. "
            "Your LoRA files in Output, generated videos in Samples, downloaded models, and app settings will not be removed.",
            icon="warning",
        )
        if not confirmed:
            return

        # Cache filenames are derived from the dataset clips, so clear both folders together.
        shutil.rmtree(DATASET_DIR, ignore_errors=True)
        shutil.rmtree(CACHE_DIR, ignore_errors=True)
        DATASET_DIR.mkdir(parents=True, exist_ok=True)
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        self.selected_video = None
        self._stop_preview()
        self.preview_label.configure(image="", text="Select a clip")
        self.caption_text.delete("1.0", END)
        self._refresh_dataset()
        self.status_var.set("Dataset and training cache cleared. Ready for a new project.")

    def _validate_dataset(self) -> None:
        self._run_commands("Dataset validation", [[str(PYTHON), str(ROOT / "tools" / "validate_dataset.py"), "--trigger", self.trigger_var.get().strip()]])

    def _verify_setup(self) -> None:
        self._run_commands("Setup check", [[str(PYTHON), str(ROOT / "tools" / "verify_setup.py")]])

    def _run_scripts(self, label: str, scripts: list[Path]) -> None:
        commands = [["powershell.exe", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", str(script)] for script in scripts]
        self._run_commands(label, commands)

    def _browse_lora_continuation(self) -> None:
        initial = self.lora_continuation_var.get().strip()
        initial_dir = Path(initial).parent if initial else OUTPUT_DIR
        if not initial_dir.is_dir():
            initial_dir = OUTPUT_DIR
        selected = filedialog.askopenfilename(
            title="Choose a LoRA to continue training",
            initialdir=str(initial_dir),
            filetypes=[("SafeTensors LoRA", "*.safetensors"), ("All files", "*.*")],
        )
        if selected:
            self.lora_continuation_var.set(selected)

    def _training_command(self) -> list[str]:
        name = self.run_name_var.get().strip() or "my_video_lora"
        command = [
            str(ACCELERATE), "launch", "--num_cpu_threads_per_process", "1", "--mixed_precision", "bf16",
            str(ENGINE / "wan_train_network.py"), "--task", "t2v-1.3B", "--dit", str(DIT_PATH),
            "--dataset_config", str(DATASET_CONFIG), "--sdpa", "--mixed_precision", "bf16", "--save_precision", "bf16",
            "--fp8_base", "--optimizer_type", "adamw8bit", "--learning_rate", self.learning_rate_var.get().strip(),
            "--gradient_checkpointing", "--blocks_to_swap", str(self.blocks_var.get()), "--block_swap_h2d_only",
            "--block_swap_ring_size", "1", "--max_data_loader_n_workers", "1", "--network_module", "networks.lora_wan",
            "--network_dim", str(self.rank_var.get()), "--network_alpha", str(self.alpha_var.get()), "--timestep_sampling", "shift",
            "--discrete_flow_shift", "3.0", "--max_train_epochs", str(self.epochs_var.get()), "--save_every_n_epochs", "2",
            "--seed", "42", "--output_dir", str(OUTPUT_DIR), "--output_name", name, "--log_with", "tensorboard",
            "--logging_dir", str(LOGS_DIR),
        ]
        continuation = self.lora_continuation_var.get().strip()
        if continuation:
            command.extend(["--network_weights", continuation, "--dim_from_weights", continuation])
        return command

    def _request_unique_training_name(self) -> bool:
        """Require an intentional, filesystem-safe name that cannot overwrite an existing LoRA."""
        while True:
            entered = simpledialog.askstring(
                "Name This LoRA Model",
                "Enter a unique name for this training run.\n\n"
                "Example: my_character_wan21 or product_demo_wan21\n"
                "An existing model name cannot be reused.",
                parent=self,
                initialvalue=self.run_name_var.get().strip(),
            )
            if entered is None:
                return False
            name = re.sub(r"[^A-Za-z0-9._-]+", "_", entered.strip()).strip("._-")
            if not name:
                messagebox.showwarning("Model Name Required", "Enter a descriptive model name before training.")
                continue
            existing = [
                path for path in OUTPUT_DIR.glob(f"{name}*.safetensors")
                if path.stem == name or path.stem.startswith(f"{name}-")
            ]
            if existing:
                messagebox.showwarning(
                    "Model Name Already Exists",
                    f"'{name}' already has {len(existing)} checkpoint file(s) in Output.\n\n"
                    "Choose a different name so the existing model cannot be overwritten.",
                )
                continue
            clip_count = sum(
                1 for path in DATASET_DIR.rglob("*")
                if path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS
            ) if DATASET_DIR.exists() else 0
            if not messagebox.askyesno(
                "Confirm Training Identity",
                f"Train a new LoRA named:\n{name}\n\n"
                f"Dataset clips: {clip_count}\n"
                f"Trigger word: {self.trigger_var.get().strip() or '(none)'}\n"
                f"Epochs: {self.epochs_var.get()}\n"
                f"Starting weights: {Path(self.lora_continuation_var.get()).name if self.lora_continuation_var.get().strip() else 'New LoRA'}\n\n"
                "Is this the correct model and dataset?",
            ):
                continue
            self.run_name_var.set(name)
            self._save_settings()
            return True

    def _start_training(self) -> None:
        if not self._validate_training_settings():
            return
        if not self._request_unique_training_name():
            return
        self._run_commands("LoRA training", [self._training_command()])

    def _run_full_pipeline(self) -> None:
        if not self._validate_training_settings():
            return
        if not self._request_unique_training_name():
            return
        commands = [
            [str(PYTHON), str(ROOT / "tools" / "validate_dataset.py"), "--trigger", self.trigger_var.get().strip()],
            ["powershell.exe", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", str(ROOT / "scripts" / "01_cache_video.ps1")],
            ["powershell.exe", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", str(ROOT / "scripts" / "02_cache_text.ps1")],
            self._training_command(),
        ]
        self._run_commands("Full training pipeline", commands)

    def _validate_training_settings(self) -> bool:
        clip_count = sum(
            1 for path in DATASET_DIR.rglob("*")
            if path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS
        ) if DATASET_DIR.exists() else 0
        if clip_count == 0:
            messagebox.showwarning("Dataset Is Empty", "Add or split at least one video clip on the Dataset tab before training.")
            self.notebook.select(self.dataset_tab)
            return False
        try:
            epochs = self.epochs_var.get()
            rank = self.rank_var.get()
            alpha = self.alpha_var.get()
            blocks = self.blocks_var.get()
            learning_rate = float(self.learning_rate_var.get())
        except (tk.TclError, ValueError):
            messagebox.showwarning("Invalid Training Settings", "Check that every training number is filled in correctly.")
            return False
        if not (1 <= epochs <= 100 and 1 <= rank <= 128 and 1 <= alpha <= 128 and 1 <= blocks <= 29 and learning_rate > 0):
            messagebox.showwarning("Invalid Training Settings", "Epochs, rank, alpha, block swap, or learning rate is outside its supported range.")
            return False
        continuation = self.lora_continuation_var.get().strip()
        if continuation:
            continuation_path = Path(continuation)
            if not continuation_path.is_file() or continuation_path.suffix.lower() != ".safetensors":
                messagebox.showwarning(
                    "Invalid LoRA Continuation",
                    "Choose an existing .safetensors LoRA file, or clear LoRA Continuation to train a new one.",
                )
                return False
        return True

    def _run_commands(
        self,
        label: str,
        commands: list[list[str]],
        fallback_commands: list[list[str]] | None = None,
        speed_mode: bool = False,
    ) -> None:
        if self.active_thread and self.active_thread.is_alive():
            messagebox.showwarning("Job Already Running", "Stop or wait for the current job before starting another.")
            return
        self.cancel_requested.clear()
        self._save_settings()
        self.job_started_at = time.monotonic()
        self.job_label = label
        self.workflow_status_var.set(f"Running: {label}")
        self.generation_status_var.set(f"Running: {label}" if label == "Video generation" else self.generation_status_var.get())
        self.status_var.set(f"Running: {label}")
        self.stop_button.configure(state="normal")
        self.generation_stop_button.configure(state="normal")
        self.job_progress.configure(mode="indeterminate", maximum=100, value=0)
        self.job_progress.start(12)
        history = []
        for value in self.job_history.get(label, []):
            try:
                numeric_value = float(value)
            except (TypeError, ValueError):
                continue
            if numeric_value > 0:
                history.append(numeric_value)
        if history:
            expected = statistics.fmean(history[-5:])
            finish = datetime.now() + timedelta(seconds=expected)
            self.job_eta_var.set(f"Estimated finish {finish:%I:%M %p} (about {self._format_duration(expected)})")
        else:
            self.job_eta_var.set("Estimating after the first progress update…")
        self._append_log(f"\n=== {label} ===\n")
        self.active_thread = threading.Thread(
            target=self._command_worker,
            args=(label, commands, fallback_commands, speed_mode),
            daemon=True,
        )
        self.active_thread.start()

    def _command_worker(
        self,
        label: str,
        commands: list[list[str]],
        fallback_commands: list[list[str]] | None = None,
        speed_mode: bool = False,
    ) -> None:
        env = os.environ.copy()
        env["PYTHONUTF8"] = "1"
        env["PYTHONUNBUFFERED"] = "1"
        if speed_mode:
            env["TORCH_ALLOW_TF32_CUBLAS_OVERRIDE"] = "1"
        for index, command in enumerate(commands, start=1):
            if self.cancel_requested.is_set():
                self.events.put(("finished", (label, -1, True)))
                return
            self.events.put(("stage", (label, index, len(commands), Path(command[0]).name)))
            self.events.put(("log", f"\nStage {index}/{len(commands)}: {Path(command[0]).name}\n"))
            try:
                code = self._execute_command(label, command, env)
                if code != 0:
                    can_fallback = (
                        fallback_commands is not None
                        and index <= len(fallback_commands)
                        and not self.cancel_requested.is_set()
                    )
                    if can_fallback:
                        self.events.put(("log", "\nExperimental compilation failed; retrying automatically in stable mode.\n"))
                        fallback_env = env.copy()
                        fallback_env.pop("TORCH_ALLOW_TF32_CUBLAS_OVERRIDE", None)
                        code = self._execute_command(label, fallback_commands[index - 1], fallback_env)
                    if code != 0:
                        self.events.put(("finished", (label, code, self.cancel_requested.is_set())))
                        return
            except Exception as exc:
                self.events.put(("log", f"ERROR: {exc}\n"))
                self.events.put(("finished", (label, 1, False)))
                return
        self.events.put(("finished", (label, 0, False)))

    def _execute_command(self, label: str, command: list[str], env: dict[str, str]) -> int:
        creationflags = subprocess.CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0
        try:
            self.active_process = subprocess.Popen(
                command,
                cwd=str(ROOT),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
                bufsize=1,
                env=env,
                creationflags=creationflags,
            )
            if label == "Video generation":
                self._start_generation_monitor(self.active_process.pid)
            assert self.active_process.stdout is not None
            for line in self.active_process.stdout:
                self._record_job_progress(label, line)
                if label == "Video generation":
                    self._record_generation_progress(line)
                self.events.put(("log", line))
            return self.active_process.wait()
        finally:
            if label == "Video generation":
                self._stop_generation_monitor()
            self.active_process = None

    @staticmethod
    def _format_duration(seconds: float) -> str:
        seconds = max(0, int(round(seconds)))
        if seconds < 60:
            return f"{seconds}s"
        hours, remainder = divmod(seconds, 3600)
        minutes, secs = divmod(remainder, 60)
        if hours:
            return f"{hours}h {minutes}m"
        return f"{minutes}m {secs}s"

    def _record_job_progress(self, label: str, line: str) -> None:
        """Extract tqdm-style progress and send a thread-safe live ETA to the UI."""
        cleaned = re.sub(r"\x1b\[[0-9;?]*[A-Za-z]", "", line)
        count_match = re.search(r"(?<!\d)(\d+)\s*/\s*(\d+)(?!\d)", cleaned)
        if not count_match:
            return
        current, total = int(count_match.group(1)), int(count_match.group(2))
        if total <= 0 or current < 0 or current > total:
            return
        seconds_per_step = 0.0
        rate_match = re.search(r"([\d.]+)\s*s/it", cleaned)
        inverse_match = re.search(r"([\d.]+)\s*it/s", cleaned)
        if rate_match:
            seconds_per_step = float(rate_match.group(1))
        elif inverse_match and float(inverse_match.group(1)) > 0:
            seconds_per_step = 1.0 / float(inverse_match.group(1))
        if seconds_per_step <= 0 and self.job_started_at and current > 0:
            seconds_per_step = (time.monotonic() - self.job_started_at) / current
        remaining = max(0.0, (total - current) * seconds_per_step)
        self.events.put(("progress", (label, current, total, remaining)))

    def _start_generation_monitor(self, process_pid: int) -> None:
        """Sample system/GPU health during inference and save it for diagnosis."""
        self.monitor_stop.clear()
        with self.monitor_progress_lock:
            self.monitor_step = 0
            self.monitor_total_steps = 0
            self.monitor_step_seconds = 0.0
        LOGS_DIR.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        report_path = LOGS_DIR / f"generation-monitor-{stamp}.csv"
        self.events.put(("log", f"\nPerformance monitor: recording to {report_path}\n"))
        self.monitor_thread = threading.Thread(
            target=self._generation_monitor_worker,
            args=(process_pid, report_path),
            daemon=True,
        )
        self.monitor_thread.start()

    def _record_generation_progress(self, line: str) -> None:
        match = re.search(r"(\d+)\s*/\s*(\d+).*?([\d.]+)s/it", line)
        if not match:
            return
        with self.monitor_progress_lock:
            self.monitor_step = int(match.group(1))
            self.monitor_total_steps = int(match.group(2))
            self.monitor_step_seconds = float(match.group(3))

    def _stop_generation_monitor(self) -> None:
        self.monitor_stop.set()
        if self.monitor_thread and self.monitor_thread is not threading.current_thread():
            self.monitor_thread.join(timeout=5)
        self.monitor_thread = None

    def _generation_monitor_worker(self, process_pid: int, report_path: Path) -> None:
        fields = [
            "timestamp", "elapsed_s", "gpu_util_pct", "vram_used_mib", "vram_total_mib",
            "gpu_temp_c", "gpu_power_w", "gpu_power_limit_w", "gpu_clock_mhz",
            "memory_clock_mhz", "pcie_gen", "pcie_width", "cpu_total_pct",
            "app_cpu_pct", "ram_used_pct", "ram_available_gib", "app_ram_gib",
            "current_step", "total_steps", "reported_seconds_per_step",
        ]
        gpu_query = (
            "utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw,power.limit,"
            "clocks.current.graphics,clocks.current.memory,pcie.link.gen.current,pcie.link.width.current"
        )
        samples: list[dict[str, float | str]] = []
        started = time.monotonic()
        try:
            app_process = psutil.Process(process_pid)
            app_process.cpu_percent(None)
            psutil.cpu_percent(None)
            with report_path.open("w", newline="", encoding="utf-8") as report:
                writer = csv.DictWriter(report, fieldnames=fields)
                writer.writeheader()
                while not self.monitor_stop.wait(1.0):
                    row: dict[str, float | str] = {
                        "timestamp": datetime.now().isoformat(timespec="seconds"),
                        "elapsed_s": round(time.monotonic() - started, 1),
                    }
                    try:
                        result = subprocess.run(
                            ["nvidia-smi", f"--query-gpu={gpu_query}", "--format=csv,noheader,nounits"],
                            capture_output=True, text=True, timeout=3, check=True,
                            creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
                        )
                        values = [value.strip() for value in result.stdout.splitlines()[0].split(",")]
                        gpu_fields = fields[2:12]
                        row.update({name: float(value) for name, value in zip(gpu_fields, values)})
                    except (OSError, subprocess.SubprocessError, ValueError, IndexError):
                        pass
                    try:
                        memory = psutil.virtual_memory()
                        descendants = app_process.children(recursive=True)
                        processes = [app_process, *descendants]
                        row["cpu_total_pct"] = round(psutil.cpu_percent(None), 1)
                        row["app_cpu_pct"] = round(sum(p.cpu_percent(None) for p in processes if p.is_running()), 1)
                        row["ram_used_pct"] = round(memory.percent, 1)
                        row["ram_available_gib"] = round(memory.available / (1024 ** 3), 2)
                        row["app_ram_gib"] = round(
                            sum(p.memory_info().rss for p in processes if p.is_running()) / (1024 ** 3), 2
                        )
                    except (psutil.Error, OSError):
                        pass
                    with self.monitor_progress_lock:
                        row["current_step"] = self.monitor_step
                        row["total_steps"] = self.monitor_total_steps
                        row["reported_seconds_per_step"] = self.monitor_step_seconds
                    writer.writerow(row)
                    report.flush()
                    samples.append(row)
        except OSError as exc:
            self.events.put(("log", f"Performance monitor could not write its report: {exc}\n"))
            return

        if not samples:
            self.events.put(("log", "Performance monitor collected no samples.\n"))
            return

        def numbers(name: str) -> list[float]:
            return [float(row[name]) for row in samples if name in row and row[name] != ""]

        def avg(name: str) -> float:
            values = numbers(name)
            return statistics.fmean(values) if values else 0.0

        gpu_util = numbers("gpu_util_pct")
        vram = numbers("vram_used_mib")
        temp = numbers("gpu_temp_c")
        power = numbers("gpu_power_w")
        power_limit = numbers("gpu_power_limit_w")
        low_gpu_share = 100 * sum(value < 70 for value in gpu_util) / len(gpu_util) if gpu_util else 0
        peak_power_share = (
            100 * max(power) / max(power_limit) if power and power_limit and max(power_limit) else 0
        )
        step_timings: dict[int, float] = {}
        for row in samples:
            step = int(float(row.get("current_step", 0)))
            seconds = float(row.get("reported_seconds_per_step", 0))
            if step and seconds:
                step_timings[step] = seconds
        summary = (
            "\n=== Generation performance summary ===\n"
            f"Report: {report_path}\n"
            f"Samples: {len(samples)} over {samples[-1]['elapsed_s']} seconds\n"
            f"GPU utilization: {avg('gpu_util_pct'):.0f}% average, {low_gpu_share:.0f}% of samples below 70%\n"
            f"VRAM: {avg('vram_used_mib') / 1024:.1f} GiB average, {max(vram, default=0) / 1024:.1f} GiB peak\n"
            f"GPU temperature: {avg('gpu_temp_c'):.0f} C average, {max(temp, default=0):.0f} C peak\n"
            f"GPU power: {avg('gpu_power_w'):.0f} W average, {peak_power_share:.0f}% of power limit at peak\n"
            f"CPU: {avg('cpu_total_pct'):.0f}% average | RAM: {avg('ram_used_pct'):.0f}% used average\n"
        )
        if step_timings:
            summary += (
                f"Denoising steps: {statistics.fmean(step_timings.values()):.1f} seconds/step average "
                f"({min(step_timings.values()):.1f}-{max(step_timings.values()):.1f})\n"
            )
        if low_gpu_share >= 35:
            summary += "Finding: GPU was frequently under-fed; CPU/RAM block swapping or another wait is likely limiting speed.\n"
        elif peak_power_share >= 95:
            summary += "Finding: GPU reached its power limit; the workload appears GPU-compute limited.\n"
        elif max(temp, default=0) >= 83:
            summary += "Finding: GPU temperature is high enough that thermal throttling may be contributing.\n"
        else:
            summary += "Finding: no obvious utilization, power, or thermal bottleneck; compare this report with step timing.\n"
        self.events.put(("log", summary))

    def _stop_job(self) -> None:
        process = self.active_process
        if not process or process.poll() is not None:
            return
        if not messagebox.askyesno("Stop Current Job", "Stop the current caching, training, or generation job?"):
            return
        self.cancel_requested.set()
        self.workflow_status_var.set("Stopping...")
        self.status_var.set("Stopping current job...")
        try:
            if os.name == "nt":
                subprocess.run(["taskkill", "/PID", str(process.pid), "/T", "/F"], capture_output=True, check=False)
            else:
                process.terminate()
        except OSError as exc:
            self._append_log(f"Could not stop process cleanly: {exc}\n")

    def _append_log(self, text: str) -> None:
        self.training_log.insert(END, text)
        if self.auto_scroll_var.get():
            self.training_log.see(END)

    def _poll_events(self) -> None:
        try:
            while True:
                event, payload = self.events.get_nowait()
                if event == "log":
                    self._append_log(str(payload))
                elif event == "stage":
                    label, index, total, executable = payload  # type: ignore[misc]
                    stage_text = f"Running: {label} — stage {index}/{total} ({executable})"
                    self.workflow_status_var.set(stage_text)
                    self.status_var.set(stage_text)
                elif event == "progress":
                    label, current, total, remaining = payload  # type: ignore[misc]
                    self.job_progress.stop()
                    self.job_progress.configure(mode="determinate", maximum=total, value=current)
                    if current >= total:
                        eta_text = f"{current}/{total} • finishing this stage…"
                    else:
                        finish = datetime.now() + timedelta(seconds=remaining)
                        eta_text = (
                            f"{current}/{total} • {self._format_duration(remaining)} left • "
                            f"finish about {finish:%I:%M %p}"
                        )
                    self.job_eta_var.set(eta_text)
                    self.workflow_status_var.set(f"Running: {label} — {eta_text}")
                    if label == "Video generation":
                        self.generation_status_var.set(eta_text)
                        self.generation_preview_label.configure(
                            image="",
                            text=f"Denoising step {current}/{total}\n\n"
                                 "An accurate frame becomes available when Wan loads its decoder.",
                        )
                        self.generation_preview_image = None
                elif event == "finished":
                    label, code, cancelled = payload  # type: ignore[misc]
                    self.job_progress.stop()
                    self.job_progress.configure(mode="determinate", value=0)
                    self.stop_button.configure(state="disabled")
                    self.generation_stop_button.configure(state="disabled")
                    if cancelled:
                        message = f"{label} stopped."
                    elif code == 0:
                        message = f"{label} completed successfully."
                    else:
                        message = f"{label} failed with exit code {code}. See Live Output."
                    self.workflow_status_var.set(message)
                    if label == "Video generation":
                        self.generation_status_var.set(message)
                    self.status_var.set(message)
                    if code == 0 and not cancelled and self.job_started_at is not None:
                        duration = time.monotonic() - self.job_started_at
                        values = self.job_history.setdefault(label, [])
                        values.append(round(duration, 1))
                        del values[:-5]
                        self._save_settings()
                        self.job_eta_var.set(f"Finished in {self._format_duration(duration)} at {datetime.now():%I:%M %p}")
                    elif cancelled:
                        self.job_eta_var.set("Job stopped")
                    else:
                        self.job_eta_var.set("Job ended before an estimate was available")
                    self.job_started_at = None
                    self.job_label = ""
                    self._append_log(f"\n=== {message} ===\n")
                    self._refresh_checkpoints()
                    self._refresh_samples()
                    if label == "Video generation":
                        if code == 0 and not cancelled:
                            self._record_storyboard_take()
                        else:
                            self.pending_storyboard_scene_id = None
                            self.pending_storyboard_samples.clear()
                    self._refresh_dataset()
        except queue.Empty:
            pass
        self.after(100, self._poll_events)

    def _refresh_checkpoints(self) -> None:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        checkpoints = sorted(OUTPUT_DIR.glob("*.safetensors"), key=lambda path: path.stat().st_mtime, reverse=True)
        values = [str(path) for path in checkpoints]
        self.checkpoint_combo.configure(values=values)
        if values and self.checkpoint_var.get() not in values:
            self.checkpoint_var.set(values[0])

    @staticmethod
    def _frames_for_seconds(seconds: float, fps: int) -> int:
        """Return the closest Wan-compatible frame count (4N+1) for a playback rate."""
        return max(5, int(round((seconds * fps - 1) / 4)) * 4 + 1)

    def _on_resolution_selected(self, _event=None) -> None:
        if self.resolution_var.get().startswith("Portrait"):
            self.width_var.set(480)
            self.height_var.set(832)
        else:
            self.width_var.set(832)
            self.height_var.set(480)

    def _set_fast_preview(self) -> None:
        """Use the quickest supported local-preview configuration."""
        self.resolution_var.set("Landscape — 832 x 480")
        self._on_resolution_selected()
        self.duration_var.set(2.0)
        self.frames_var.set(self._frames_for_seconds(2.0, self.fps_var.get()))
        self.steps_var.set(8)
        self._update_duration_label()
        self.generation_status_var.set(
            f"Fast preview set: {self.frames_var.get() / self.fps_var.get():.1f} seconds, "
            f"{self.frames_var.get()} frames at {self.fps_var.get()} FPS, 8 steps."
        )

    def _on_duration_changed(self, value: str) -> None:
        seconds = round(float(value) * 2) / 2
        if self.duration_var.get() != seconds:
            self.duration_var.set(seconds)
        self.frames_var.set(self._frames_for_seconds(seconds, self.fps_var.get()))
        self._update_duration_label()

    def _on_fps_changed(self, _event=None) -> None:
        try:
            fps = self.fps_var.get()
        except tk.TclError:
            return
        if 4 <= fps <= 60:
            self.frames_var.set(self._frames_for_seconds(self.duration_var.get(), fps))
            self._update_duration_label()

    def _update_duration_label(self) -> None:
        frames = self.frames_var.get()
        fps = max(1, self.fps_var.get())
        actual_seconds = frames / fps
        note = f"{actual_seconds:.1f}s / {frames} frames / {fps} FPS"
        if actual_seconds > 5:
            note += " — long runs can be very slow"
        self.duration_label_var.set(note)

    def _dataset_prompts(self) -> list[tuple[Path, str]]:
        """Return non-empty captions belonging to videos in the active dataset."""
        prompts: list[tuple[Path, str]] = []
        if not DATASET_DIR.exists():
            return prompts
        for video in sorted(DATASET_DIR.iterdir()):
            if video.suffix.lower() not in VIDEO_EXTENSIONS:
                continue
            caption_path = video.with_suffix(".txt")
            if not caption_path.exists():
                continue
            caption = caption_path.read_text(encoding="utf-8-sig", errors="replace").strip()
            if caption:
                prompts.append((caption_path, caption))
        return prompts

    def _sync_auto_prompt_controls(self) -> None:
        enabled = self.auto_prompt_var.get()
        self.prompt_entry.configure(state="readonly" if enabled else "normal")
        self.auto_prompt_pick_button.configure(state="normal" if enabled else "disabled")

    def _on_auto_prompt_toggled(self) -> None:
        self._sync_auto_prompt_controls()
        if self.auto_prompt_var.get():
            self._pick_auto_prompt()
        else:
            self.generation_status_var.set("AutoPrompt off; using the prompt shown above.")

    def _pick_auto_prompt(self, *, warn_if_empty: bool = True) -> bool:
        prompts = self._dataset_prompts()
        if not prompts:
            if warn_if_empty:
                messagebox.showwarning(
                    "No Dataset Captions",
                    "AutoPrompt needs at least one non-empty .txt caption beside a video in Dataset > videos.",
                )
            return False
        choices = [item for item in prompts if item[0] != self.last_auto_prompt_path] or prompts
        caption_path, prompt = random.SystemRandom().choice(choices)
        self.last_auto_prompt_path = caption_path
        self.prompt_var.set(prompt)
        self.generation_status_var.set(f"AutoPrompt: {caption_path.name}")
        return True

    def _selected_storyboard_scene(self) -> dict | None:
        selection = self.storyboard_tree.selection()
        if not selection:
            return None
        scene_id = selection[0]
        return next((scene for scene in self.storyboard_scenes if scene["id"] == scene_id), None)

    def _refresh_storyboard(self, select_id: str | None = None) -> None:
        if not hasattr(self, "storyboard_tree"):
            return
        previous = select_id or (self.storyboard_tree.selection()[0] if self.storyboard_tree.selection() else None)
        for item in self.storyboard_tree.get_children():
            self.storyboard_tree.delete(item)
        for index, scene in enumerate(self.storyboard_scenes, start=1):
            takes = [path for path in scene.get("takes", []) if Path(path).exists()]
            accepted = scene.get("accepted_take", "")
            if accepted and Path(accepted).exists():
                status = "Accepted"
            elif takes:
                status = f"{len(takes)} take{'s' if len(takes) != 1 else ''}"
            else:
                status = "Not generated"
            title = str(scene.get("title", "")).strip() or f"Scene {index}"
            self.storyboard_tree.insert(
                "", END, iid=scene["id"], text=f"{index}. {title}",
                values=(f"{float(scene.get('duration', 5.0)):.1f}s", status, scene.get("prompt", "")),
            )
        if previous and self.storyboard_tree.exists(previous):
            self.storyboard_tree.selection_set(previous)
            self.storyboard_tree.focus(previous)
            self.storyboard_tree.see(previous)
        elif self.storyboard_scenes:
            first = self.storyboard_scenes[0]["id"]
            self.storyboard_tree.selection_set(first)
            self.storyboard_tree.focus(first)
        self._on_storyboard_scene_selected()

    def _scene_editor(self, scene: dict | None = None) -> dict | None:
        window = tk.Toplevel(self)
        window.title("Edit Storyboard Scene" if scene else "Add Storyboard Scene")
        window.geometry("720x390")
        window.transient(self)
        window.grab_set()
        window.columnconfigure(1, weight=1)
        window.rowconfigure(1, weight=1)
        title_var = StringVar(value=str(scene.get("title", "")) if scene else f"Scene {len(self.storyboard_scenes) + 1}")
        duration_var = DoubleVar(value=float(scene.get("duration", 5.0)) if scene else 5.0)
        ttk.Label(window, text="Scene title").grid(row=0, column=0, sticky="nw", padx=12, pady=12)
        title_entry = ttk.Entry(window, textvariable=title_var)
        title_entry.grid(row=0, column=1, sticky="ew", padx=(0, 12), pady=12)
        ttk.Label(window, text="Prompt").grid(row=1, column=0, sticky="nw", padx=12, pady=6)
        prompt_text = tk.Text(window, wrap="word", bg="#202631", fg="#f1f3f7", insertbackground="white", padx=8, pady=8)
        prompt_text.grid(row=1, column=1, sticky="nsew", padx=(0, 12), pady=6)
        if scene:
            prompt_text.insert("1.0", str(scene.get("prompt", "")))
        ttk.Label(window, text="Duration").grid(row=2, column=0, sticky="w", padx=12, pady=10)
        ttk.Spinbox(window, from_=2.0, to=15.0, increment=0.5, textvariable=duration_var, width=10).grid(
            row=2, column=1, sticky="w", pady=10
        )
        result: dict[str, object] = {}

        def save() -> None:
            prompt = prompt_text.get("1.0", END).strip()
            try:
                duration = float(duration_var.get())
            except (tk.TclError, ValueError):
                duration = 0
            if not prompt:
                messagebox.showwarning("Prompt Is Empty", "Give this scene a generation prompt.", parent=window)
                return
            if not 2.0 <= duration <= 15.0:
                messagebox.showwarning("Invalid Duration", "Scene duration must be between 2 and 15 seconds.", parent=window)
                return
            result.update(title=title_var.get().strip() or "Untitled scene", prompt=prompt, duration=duration)
            window.destroy()

        buttons = ttk.Frame(window, padding=12)
        buttons.grid(row=3, column=0, columnspan=2, sticky="e")
        ttk.Button(buttons, text="Cancel", command=window.destroy).pack(side="right")
        ttk.Button(buttons, text="Save Scene", style="Accent.TButton", command=save).pack(side="right", padx=7)
        window.bind("<Control-Return>", lambda _event: save())
        title_entry.focus_set()
        self.wait_window(window)
        return result or None

    def _add_storyboard_scene(self) -> None:
        values = self._scene_editor()
        if not values:
            return
        scene = {"id": uuid.uuid4().hex, **values, "takes": [], "accepted_take": ""}
        self.storyboard_scenes.append(scene)
        self._save_storyboard()
        self._refresh_storyboard(scene["id"])

    def _edit_storyboard_scene(self, _event=None) -> None:
        scene = self._selected_storyboard_scene()
        if not scene:
            messagebox.showinfo("Select a Scene", "Select the scene you want to edit.")
            return
        values = self._scene_editor(scene)
        if not values:
            return
        scene.update(values)
        self._save_storyboard()
        self._refresh_storyboard(scene["id"])

    def _delete_storyboard_scene(self) -> None:
        scene = self._selected_storyboard_scene()
        if not scene:
            return
        if not messagebox.askyesno("Delete Scene", f"Remove '{scene['title']}' from the storyboard?\n\nGenerated video files will be kept."):
            return
        self.storyboard_scenes.remove(scene)
        self._save_storyboard()
        self._refresh_storyboard()

    def _move_storyboard_scene(self, direction: int) -> None:
        scene = self._selected_storyboard_scene()
        if not scene:
            return
        index = self.storyboard_scenes.index(scene)
        target = index + direction
        if not 0 <= target < len(self.storyboard_scenes):
            return
        self.storyboard_scenes[index], self.storyboard_scenes[target] = self.storyboard_scenes[target], self.storyboard_scenes[index]
        self._save_storyboard()
        self._refresh_storyboard(scene["id"])

    def _on_storyboard_scene_selected(self, _event=None) -> None:
        if not hasattr(self, "storyboard_take_list"):
            return
        self.storyboard_take_list.delete(0, END)
        scene = self._selected_storyboard_scene()
        self.storyboard_take_paths: list[str] = []
        if not scene:
            self.storyboard_scene_info_var.set("Select a scene to see its takes.")
            return
        accepted = scene.get("accepted_take", "")
        for path_string in scene.get("takes", []):
            path = Path(path_string)
            marker = "✓ " if path_string == accepted else ""
            missing = " [missing]" if not path.exists() else ""
            self.storyboard_take_list.insert(END, f"{marker}{path.name}{missing}")
            self.storyboard_take_paths.append(path_string)
        self.storyboard_scene_info_var.set(
            f"{scene['title']} — {float(scene['duration']):.1f}s\n{scene['prompt']}"
        )

    def _selected_storyboard_take(self) -> str | None:
        selection = self.storyboard_take_list.curselection()
        if not selection or selection[0] >= len(getattr(self, "storyboard_take_paths", [])):
            return None
        return self.storyboard_take_paths[selection[0]]

    def _preview_storyboard_take(self) -> None:
        take = self._selected_storyboard_take()
        if not take:
            return
        for index in range(self.sample_list.size()):
            if self.sample_list.get(index) == take:
                self.sample_list.selection_clear(0, END)
                self.sample_list.selection_set(index)
                self.sample_list.see(index)
                self._show_selected_generation_preview()
                self.notebook.select(self.generate_tab)
                return

    def _open_storyboard_take(self) -> None:
        take = self._selected_storyboard_take()
        if take and Path(take).exists():
            os.startfile(str(Path(take)))

    def _accept_storyboard_take(self) -> None:
        scene = self._selected_storyboard_scene()
        take = self._selected_storyboard_take()
        if not scene or not take or not Path(take).exists():
            return
        scene["accepted_take"] = take
        self._save_storyboard()
        self._refresh_storyboard(scene["id"])

    def _remove_storyboard_take(self) -> None:
        scene = self._selected_storyboard_scene()
        take = self._selected_storyboard_take()
        if not scene or not take:
            return
        scene["takes"] = [path for path in scene.get("takes", []) if path != take]
        if scene.get("accepted_take") == take:
            scene["accepted_take"] = ""
        self._save_storyboard()
        self._refresh_storyboard(scene["id"])

    def _generate_storyboard_scene(self) -> None:
        scene = self._selected_storyboard_scene()
        if not scene:
            messagebox.showinfo("Select a Scene", "Select a storyboard scene to generate.")
            return
        if self.active_thread and self.active_thread.is_alive():
            messagebox.showwarning("Job Already Running", "Stop or wait for the current job before generating a scene.")
            return
        self.prompt_var.set(scene["prompt"])
        self.duration_var.set(float(scene["duration"]))
        self.frames_var.set(self._frames_for_seconds(float(scene["duration"]), self.fps_var.get()))
        self._update_duration_label()
        self.notebook.select(self.generate_tab)
        self._generate_video(storyboard_scene_id=scene["id"])

    def _record_storyboard_take(self) -> None:
        scene_id = self.pending_storyboard_scene_id
        if not scene_id:
            return
        current = sorted(SAMPLES_DIR.glob("*.mp4"), key=lambda path: path.stat().st_mtime, reverse=True)
        new_takes = [str(path) for path in current if str(path) not in self.pending_storyboard_samples]
        scene = next((item for item in self.storyboard_scenes if item["id"] == scene_id), None)
        if scene and new_takes:
            scene.setdefault("takes", [])
            for take in reversed(new_takes):
                if take not in scene["takes"]:
                    scene["takes"].append(take)
            self._save_storyboard()
            self._refresh_storyboard(scene_id)
        self.pending_storyboard_scene_id = None
        self.pending_storyboard_samples.clear()

    def _generate_video(self, storyboard_scene_id: str | None = None) -> None:
        checkpoint = Path(self.checkpoint_var.get())
        if not checkpoint.exists():
            messagebox.showwarning("No LoRA Checkpoint", "Train a LoRA or select an existing checkpoint first.")
            return
        if storyboard_scene_id is None and self.auto_prompt_var.get() and not self._pick_auto_prompt():
            return
        if not self.prompt_var.get().strip():
            messagebox.showwarning("Prompt Is Empty", "Enter what you want the generated video to show.")
            return
        try:
            fps = self.fps_var.get()
            duration = self.duration_var.get()
            steps = self.steps_var.get()
            seed = self.seed_var.get()
            strength = self.lora_strength_var.get()
            generation_blocks = self.generation_blocks_var.get()
        except tk.TclError:
            messagebox.showwarning("Invalid Generation Settings", "Check that every generation number is filled in correctly.")
            return
        if self.randomize_seed_var.get():
            seed = random.SystemRandom().randrange(0, 2147483648)
            self.seed_var.set(seed)
        if not 4 <= fps <= 60:
            messagebox.showwarning("Invalid FPS", "Playback FPS must be between 4 and 60.")
            return
        if not 2.0 <= duration <= 15.0 or not 1 <= steps <= 100 or not 0 <= seed <= 2147483647 or not 0.0 <= strength <= 2.0:
            messagebox.showwarning("Invalid Generation Settings", "Duration, steps, seed, or LoRA strength is outside its supported range.")
            return
        frames = self._frames_for_seconds(duration, fps)
        self.frames_var.set(frames)
        self._update_duration_label()
        if (frames - 1) % 4 != 0:
            messagebox.showwarning("Invalid Frame Count", "Wan frame counts must follow 4N+1, such as 25, 49, or 81.")
            return
        size = (self.width_var.get(), self.height_var.get())
        if size not in {(832, 480), (480, 832)}:
            messagebox.showwarning(
                "Unsupported Wan Video Size",
                "Wan 2.1 T2V only accepts 832 x 480 (landscape) or 480 x 832 (portrait).\n\n"
                "The earlier 640 x 360 request caused the attention assertion. No retraining is needed—set one of the supported sizes and generate again.",
            )
            return
        if not 0 <= generation_blocks <= 29:
            messagebox.showwarning("Invalid Block Swap", "Blocks to swap must be between 0 and 29 for Wan 2.1 1.3B.")
            return
        SAMPLES_DIR.mkdir(parents=True, exist_ok=True)
        if storyboard_scene_id is not None:
            self.pending_storyboard_scene_id = storyboard_scene_id
            self.pending_storyboard_samples = {str(path) for path in SAMPLES_DIR.glob("*.mp4")}
        prompt = self.prompt_var.get().strip()
        source_note = (
            f" from {self.last_auto_prompt_path.name}"
            if storyboard_scene_id is None and self.auto_prompt_var.get() and self.last_auto_prompt_path else ""
        )
        self._append_log(f"Generation prompt{source_note}: {prompt}\nSeed: {seed}{' (randomized)' if self.randomize_seed_var.get() else ''}\n")
        command = [
            str(PYTHON), str(ENGINE / "wan_generate_video.py"), "--fp8", "--task", "t2v-1.3B",
            "--video_size", str(self.width_var.get()), str(self.height_var.get()), "--video_length", str(frames),
            "--fps", str(fps),
            "--infer_steps", str(steps), "--prompt", prompt, "--seed", str(seed),
            "--save_path", str(SAMPLES_DIR), "--output_type", "video", "--dit", str(DIT_PATH), "--vae", str(VAE_PATH),
            "--t5", str(T5_PATH), "--attn_mode", "torch", "--blocks_to_swap", str(generation_blocks),
            "--lora_weight", str(checkpoint), "--lora_multiplier", str(strength),
        ]
        speed_mode = self.experimental_speed_var.get()
        if speed_mode:
            self._append_log(
                f"Generation configuration: {generation_blocks} blocks swapped, experimental TF32 enabled. "
                "Compiled mode is disabled on this system after its compatibility failure.\n"
            )
            self._run_commands(
                "Video generation",
                [command],
                speed_mode=True,
            )
        else:
            self._append_log(f"Generation configuration: {generation_blocks} blocks swapped, stable mode.\n")
            self._run_commands("Video generation", [command])

    def _refresh_samples(self) -> None:
        SAMPLES_DIR.mkdir(parents=True, exist_ok=True)
        videos = sorted(SAMPLES_DIR.glob("*.mp4"), key=lambda path: path.stat().st_mtime, reverse=True)
        self.sample_list.delete(0, END)
        for video in videos:
            self.sample_list.insert(END, str(video))
        if videos:
            self.sample_list.selection_set(0)
            self.after_idle(self._show_selected_generation_preview)

    def _show_selected_generation_preview(self, _event=None) -> None:
        selection = self.sample_list.curselection()
        if not selection:
            return
        path = Path(self.sample_list.get(selection[0]))
        if not path.exists():
            return
        try:
            with av.open(str(path)) as container:
                frame = next(container.decode(video=0)).to_image()
            frame.thumbnail((390, 320), Image.Resampling.LANCZOS)
            self.generation_preview_image = ImageTk.PhotoImage(frame)
            self.generation_preview_label.configure(image=self.generation_preview_image, text="")
        except Exception as exc:
            self.generation_preview_image = None
            self.generation_preview_label.configure(image="", text=f"Preview unavailable\n{exc}")

    def _open_selected_sample(self, _event=None) -> None:
        selection = self.sample_list.curselection()
        if selection:
            path = Path(self.sample_list.get(selection[0]))
            if path.exists():
                os.startfile(path)

    def _open_tensorboard(self) -> None:
        executable = ROOT / ".venv" / "Scripts" / "tensorboard.exe"
        if not executable.exists():
            messagebox.showwarning("TensorBoard Missing", "TensorBoard is not installed in the trainer environment.")
            return
        subprocess.Popen([str(executable), "--logdir", str(LOGS_DIR)], cwd=str(ROOT), creationflags=subprocess.CREATE_NEW_PROCESS_GROUP)
        messagebox.showinfo("TensorBoard Started", "TensorBoard is running at http://localhost:6006")

    @staticmethod
    def _open_folder(path: Path) -> None:
        path.mkdir(parents=True, exist_ok=True)
        os.startfile(path)

    def _on_close(self) -> None:
        if self.active_process and self.active_process.poll() is None:
            if not messagebox.askyesno("Job Running", "A job is still running. Close the app and stop it?"):
                return
            self.cancel_requested.set()
            subprocess.run(["taskkill", "/PID", str(self.active_process.pid), "/T", "/F"], capture_output=True, check=False)
        self._stop_preview()
        self._save_settings()
        self.destroy()


def self_test() -> int:
    required = [PYTHON, ACCELERATE, DATASET_CONFIG, DIT_PATH, VAE_PATH, T5_PATH, ENGINE / "wan_train_network.py"]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        print("Missing required files:")
        for path in missing:
            print(f"- {path}")
        return 1
    print("LoRA Video Trainer self-test passed.")
    return 0


def main() -> int:
    ensure_runtime_workspace()
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--initialize", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--smoke-test-ui", action="store_true")
    args, _ = parser.parse_known_args()
    if args.initialize:
        print("Empty runtime workspace initialized.")
        return 0
    if args.self_test:
        return self_test()
    app = VideoLoraTrainerApp()
    if args.smoke_test_ui:
        app.withdraw()
        app.update_idletasks()
        app.update()
        app.destroy()
        print("Tkinter UI smoke test passed.")
        return 0
    app.mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
