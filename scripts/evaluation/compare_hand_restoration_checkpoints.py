from __future__ import annotations

import json
import shutil
import sys
import traceback
from pathlib import Path

import cv2
import numpy as np
from PySide6.QtCore import QObject, QThread, QTimer, QUrl, Signal, Slot, Qt
from PySide6.QtGui import QDesktopServices, QFont, QImage, QPixmap
from PySide6.QtWidgets import (
    QApplication,
    QComboBox,
    QFileDialog,
    QFormLayout,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from hand_restoration.config import load_json_config
from hand_restoration.inference import (
    InferenceResult,
    build_restorer,
    checkpoint_label,
    create_experiment_directory,
    discover_checkpoints,
    load_sample,
    run_inference,
    save_checkpoint_result,
    save_experiment_inputs,
)
from hand_restoration.visualize import rgb_float_to_u8


def _pixmap(image: np.ndarray) -> QPixmap:
    rgb = np.ascontiguousarray(rgb_float_to_u8(image))
    height, width, channels = rgb.shape
    qimage = QImage(rgb.data, width, height, width * channels, QImage.Format_RGB888).copy()
    return QPixmap.fromImage(qimage)


def _comparison_panel(image: np.ndarray, title: str, subtitle: str = "") -> np.ndarray:
    rgb = rgb_float_to_u8(image)
    header_height = 64 if subtitle else 44
    header = np.full((header_height, rgb.shape[1], 3), 24, dtype=np.uint8)
    cv2.putText(header, title, (12, 27), cv2.FONT_HERSHEY_SIMPLEX, 0.68, (245, 245, 245), 2, cv2.LINE_AA)
    if subtitle:
        cv2.putText(header, subtitle, (12, 51), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (165, 210, 255), 1, cv2.LINE_AA)
    return np.concatenate((header, rgb), axis=0)


def save_comparison_images(
    output: Path,
    target: np.ndarray,
    results: list[tuple[Path, InferenceResult]],
) -> None:
    for field in ("generated", "restored"):
        panels = [_comparison_panel(target, "Target")]
        for checkpoint, result in results:
            if field == "generated":
                full, masked = result.generated_psnr_full, result.generated_psnr_masked
            else:
                full, masked = result.psnr_full, result.psnr_masked
            subtitle = f"PSNR full {full:.3f} dB | masked {masked:.3f} dB"
            panels.append(_comparison_panel(getattr(result, field), checkpoint_label(checkpoint), subtitle))
        comparison = np.concatenate(panels, axis=0)
        path = output / f"comparison_{field}.png"
        if not cv2.imwrite(str(path), cv2.cvtColor(comparison, cv2.COLOR_RGB2BGR)):
            raise OSError(f"Failed to save comparison image: {path}")


class ComparisonWorker(QObject):
    input_ready = Signal(object, object, object, int, str)
    result_ready = Signal(str, str, object, object, float, float, float, float)
    progress = Signal(int, int, str)
    completed = Signal(str)
    failed = Signal(str)

    def __init__(self, config_path: Path, frame_id: int, checkpoint_dir: Path, output_root: Path, steps: int, seed: int) -> None:
        super().__init__()
        self.config_path = config_path
        self.frame_id = frame_id
        self.checkpoint_dir = checkpoint_dir
        self.output_root = output_root
        self.steps = steps
        self.seed = seed
        self.cancel_requested = False

    @Slot()
    def cancel(self) -> None:
        self.cancel_requested = True

    @Slot()
    def run(self) -> None:
        try:
            config = load_json_config(self.config_path)
            checkpoints = discover_checkpoints(self.checkpoint_dir)
            output = create_experiment_directory(self.output_root, self.frame_id)
            self.progress.emit(0, len(checkpoints), "Loading frame and rendering condition...")
            sample = load_sample(config, frame_id=self.frame_id)
            save_experiment_inputs(sample, output)
            self.input_ready.emit(
                sample["target_rgb_np"],
                sample["condition_rgb_np"],
                sample["metadata"],
                len(checkpoints),
                str(output),
            )

            self.progress.emit(0, len(checkpoints), "Loading Stable Diffusion and ControlNet...")
            restorer = build_restorer(config)
            results: list[tuple[Path, InferenceResult]] = []
            for index, checkpoint in enumerate(checkpoints, start=1):
                if self.cancel_requested:
                    self.progress.emit(index - 1, len(checkpoints), "Cancelled between checkpoints.")
                    break
                label = checkpoint_label(checkpoint)
                self.progress.emit(index - 1, len(checkpoints), f"Generating {label}...")
                restorer.load_controlnet(checkpoint)
                result = run_inference(restorer, sample, config, steps=self.steps, seed=self.seed)
                save_checkpoint_result(output, checkpoint, result)
                results.append((checkpoint, result))
                self.result_ready.emit(
                    label,
                    checkpoint.name,
                    result.generated,
                    result.restored,
                    result.generated_psnr_full,
                    result.generated_psnr_masked,
                    result.psnr_full,
                    result.psnr_masked,
                )
                self.progress.emit(index, len(checkpoints), f"Finished {label}.")

            if results:
                save_comparison_images(output, sample["target_rgb_np"], results)
            manifest = {
                "config": str(self.config_path.resolve()),
                "frame_id": self.frame_id,
                "steps": self.steps,
                "seed": self.seed,
                "checkpoint_directory": str(self.checkpoint_dir.resolve()),
                "checkpoints_completed": [path.name for path, _ in results],
                "sample_metadata": sample["metadata"],
            }
            (output / "experiment.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
            self.completed.emit(str(output))
        except Exception:
            self.failed.emit(traceback.format_exc())


class HorizontalScrollArea(QScrollArea):
    """Map vertical wheel/trackpad input to the horizontal checkpoint strip."""

    def wheelEvent(self, event) -> None:
        bar = self.horizontalScrollBar()
        pixel_delta = event.pixelDelta()
        angle_delta = event.angleDelta()
        if not pixel_delta.isNull():
            delta = pixel_delta.x() if pixel_delta.x() else pixel_delta.y()
            bar.setValue(bar.value() - delta)
            event.accept()
            return
        if not angle_delta.isNull():
            delta = angle_delta.x() if angle_delta.x() else angle_delta.y()
            bar.setValue(bar.value() - int(delta / 120 * 120))
            event.accept()
            return
        super().wheelEvent(event)


class ImageCard(QFrame):
    def __init__(self, title: str, subtitle: str = "", image_size: int = 512) -> None:
        super().__init__()
        self.setObjectName("imageCard")
        self.image_size = image_size
        self.setMaximumWidth(image_size + 34)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 14, 16, 16)
        title_label = QLabel(title)
        title_label.setObjectName("cardTitle")
        self.subtitle_label = QLabel(subtitle)
        self.subtitle_label.setObjectName("cardSubtitle")
        self.subtitle_label.setWordWrap(True)
        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setFixedSize(image_size, image_size)
        layout.addWidget(title_label)
        layout.addWidget(self.subtitle_label)
        layout.addWidget(self.image_label)

    def set_image(self, image: np.ndarray) -> None:
        pixmap = _pixmap(image)
        self.image_label.setText("")
        self.image_label.setPixmap(
            pixmap.scaled(self.image_size, self.image_size, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        )

    def clear_image(self, message: str = "No frame loaded") -> None:
        self.image_label.clear()
        self.image_label.setText(message)

    def set_subtitle(self, text: str) -> None:
        self.subtitle_label.setText(text)


class CheckpointComparisonWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Hand Restoration Checkpoint Comparator")
        self.resize(1280, 900)
        self.thread: QThread | None = None
        self.worker: ComparisonWorker | None = None
        self.results: list[dict] = []
        self.target: np.ndarray | None = None
        self.experiment_dir: Path | None = None

        central = QWidget()
        self.setCentralWidget(central)
        outer = QHBoxLayout(central)
        outer.setContentsMargins(12, 12, 12, 12)
        outer.setSpacing(12)

        left_panel = QFrame()
        left_panel.setObjectName("leftPanel")
        left_panel.setFixedWidth(390)
        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(12, 12, 12, 12)
        left_layout.setSpacing(8)

        heading = QLabel("Checkpoint Comparison")
        heading.setObjectName("heading")
        description = QLabel("Run one exact HOT3D frame through every checkpoint with identical sampling settings.")
        description.setObjectName("description")
        description.setWordWrap(True)
        left_layout.addWidget(heading)
        left_layout.addWidget(description)

        controls = QFrame()
        controls.setObjectName("controls")
        form = QFormLayout(controls)
        form.setContentsMargins(10, 10, 10, 10)
        form.setVerticalSpacing(6)
        form.setRowWrapPolicy(QFormLayout.WrapAllRows)
        form.setFieldGrowthPolicy(QFormLayout.AllNonFixedFieldsGrow)

        self.config_edit = QLineEdit("configs/hand_restoration/inference.json")
        form.addRow("Inference config", self._path_row(self.config_edit, self._browse_config))
        self.checkpoint_edit = QLineEdit("outputs/hand_restoration/tiny_overfit_frames10_20")
        form.addRow("Training output", self._path_row(self.checkpoint_edit, self._browse_checkpoint_dir))
        self.output_edit = QLineEdit("outputs/hand_restoration/checkpoint_comparisons")
        form.addRow("Experiment root", self._path_row(self.output_edit, self._browse_output_dir))

        self.frame_spin = QSpinBox()
        self.frame_spin.setRange(0, 999999)
        self.frame_spin.setValue(15)
        self.steps_spin = QSpinBox()
        self.steps_spin.setRange(1, 200)
        self.steps_spin.setValue(30)
        self.seed_spin = QSpinBox()
        self.seed_spin.setRange(0, 2147483647)
        self.seed_spin.setValue(7)
        form.addRow("Frame", self.frame_spin)
        form.addRow("Inference steps", self.steps_spin)
        form.addRow("Seed", self.seed_spin)
        left_layout.addWidget(controls)

        action_row = QHBoxLayout()
        self.run_button = QPushButton("Run all checkpoints")
        self.run_button.setObjectName("primaryButton")
        self.run_button.clicked.connect(self._run)
        self.cancel_button = QPushButton("Cancel after current")
        self.cancel_button.setEnabled(False)
        self.cancel_button.clicked.connect(self._cancel)
        self.save_button = QPushButton("Save comparison as...")
        self.save_button.setEnabled(False)
        self.save_button.clicked.connect(self._save_comparison_as)
        self.open_button = QPushButton("Open experiment folder")
        self.open_button.setEnabled(False)
        self.open_button.clicked.connect(self._open_experiment)
        action_row.addWidget(self.run_button)
        action_row.addWidget(self.cancel_button)
        left_layout.addLayout(action_row)

        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 1)
        self.status_label = QLabel("Ready.")
        self.status_label.setObjectName("status")
        self.status_label.setWordWrap(True)
        left_layout.addWidget(self.progress_bar)
        left_layout.addWidget(self.status_label)
        left_layout.addStretch(1)

        output_actions = QHBoxLayout()
        output_actions.addWidget(self.save_button)
        output_actions.addWidget(self.open_button)
        left_layout.addLayout(output_actions)

        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(12)

        self.target_card = ImageCard("Target", "Ground-truth HOT3D frame", image_size=280)
        self.target_card.clear_image("Choose a frame and run inference")
        right_layout.addWidget(self.target_card, 0, Qt.AlignHCenter)

        comparison_toolbar = QHBoxLayout()
        comparison_title = QLabel("Checkpoint outputs")
        comparison_title.setObjectName("sectionTitle")
        self.display_combo = QComboBox()
        self.display_combo.addItem("Restored (hard mask)", "restored")
        self.display_combo.addItem("Generated (raw model output)", "generated")
        self.display_combo.currentIndexChanged.connect(self._rebuild_cards)
        comparison_toolbar.addWidget(comparison_title)
        comparison_toolbar.addStretch(1)
        comparison_toolbar.addWidget(QLabel("Display"))
        comparison_toolbar.addWidget(self.display_combo)
        right_layout.addLayout(comparison_toolbar)

        self.scroll = HorizontalScrollArea()
        self.scroll.setWidgetResizable(True)
        self.scroll.setFrameShape(QFrame.NoFrame)
        self.scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.cards_widget = QWidget()
        self.cards_layout = QHBoxLayout(self.cards_widget)
        self.cards_layout.setContentsMargins(0, 0, 0, 0)
        self.cards_layout.setAlignment(Qt.AlignLeft | Qt.AlignTop)
        self.cards_layout.setSpacing(14)
        self.scroll.setWidget(self.cards_widget)
        right_layout.addWidget(self.scroll, 1)

        outer.addWidget(left_panel)
        outer.addWidget(right_panel, 1)
        self._apply_style()

    def _path_row(self, edit: QLineEdit, callback) -> QWidget:
        row = QWidget()
        layout = QHBoxLayout(row)
        layout.setContentsMargins(0, 0, 0, 0)
        button = QPushButton("...")
        button.setToolTip("Browse")
        button.setFixedWidth(34)
        button.clicked.connect(callback)
        layout.addWidget(edit, 1)
        layout.addWidget(button)
        return row

    def _browse_config(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Select inference config", self.config_edit.text(), "JSON (*.json)")
        if path:
            self.config_edit.setText(path)
            try:
                config = load_json_config(path)
                self.frame_spin.setValue(int(config.get("data", {}).get("frame_start", 15)))
                self.steps_spin.setValue(int(config.get("inference", {}).get("steps", 30)))
                self.seed_spin.setValue(int(config.get("seed", 7)))
            except Exception as exc:
                QMessageBox.warning(self, "Config warning", str(exc))

    def _browse_checkpoint_dir(self) -> None:
        path = QFileDialog.getExistingDirectory(self, "Select training output", self.checkpoint_edit.text())
        if path:
            self.checkpoint_edit.setText(path)

    def _browse_output_dir(self) -> None:
        path = QFileDialog.getExistingDirectory(self, "Select experiment root", self.output_edit.text())
        if path:
            self.output_edit.setText(path)

    def _run(self) -> None:
        config_path = Path(self.config_edit.text()).expanduser()
        checkpoint_dir = Path(self.checkpoint_edit.text()).expanduser()
        output_root = Path(self.output_edit.text()).expanduser()
        try:
            if not config_path.is_file():
                raise FileNotFoundError(f"Config does not exist: {config_path}")
            checkpoints = discover_checkpoints(checkpoint_dir)
            output_root.mkdir(parents=True, exist_ok=True)
        except Exception as exc:
            QMessageBox.critical(self, "Invalid input", str(exc))
            return

        self.target = None
        self.results.clear()
        self.experiment_dir = None
        self._clear_cards()
        self.target_card.set_subtitle("Ground-truth HOT3D frame")
        self.target_card.clear_image("Loading selected frame...")
        self.progress_bar.setRange(0, len(checkpoints))
        self.progress_bar.setValue(0)
        self.status_label.setText(f"Queued {len(checkpoints)} checkpoints.")
        self.run_button.setEnabled(False)
        self.cancel_button.setEnabled(True)
        self.save_button.setEnabled(False)
        self.open_button.setEnabled(False)

        self.thread = QThread(self)
        self.worker = ComparisonWorker(
            config_path,
            self.frame_spin.value(),
            checkpoint_dir,
            output_root,
            self.steps_spin.value(),
            self.seed_spin.value(),
        )
        self.worker.moveToThread(self.thread)
        self.thread.started.connect(self.worker.run)
        self.worker.input_ready.connect(self._on_input_ready)
        self.worker.result_ready.connect(self._on_result_ready)
        self.worker.progress.connect(self._on_progress)
        self.worker.completed.connect(self._on_completed)
        self.worker.failed.connect(self._on_failed)
        self.worker.completed.connect(self.thread.quit)
        self.worker.failed.connect(self.thread.quit)
        self.thread.finished.connect(self.worker.deleteLater)
        self.thread.finished.connect(self.thread.deleteLater)
        self.thread.finished.connect(self._thread_finished)
        self.thread.start()

    def _cancel(self) -> None:
        if self.worker is not None:
            self.worker.cancel_requested = True
            self.status_label.setText("Cancellation requested; finishing the current checkpoint...")
            self.cancel_button.setEnabled(False)

    @Slot(object, object, object, int, str)
    def _on_input_ready(self, target: np.ndarray, condition: np.ndarray, metadata: dict, count: int, output: str) -> None:
        self.target = target
        self.experiment_dir = Path(output)
        self.target_card.set_subtitle(
            f"{metadata['sequence_id']}  |  frame {metadata['frame_id']}  |  camera {metadata['camera_id']}"
        )
        self.target_card.set_image(target)
        self.status_label.setText(
            f"Loaded {metadata['sequence_id']} frame {metadata['frame_id']} from camera {metadata['camera_id']}; {count} checkpoints."
        )
        self._rebuild_cards()

    @Slot(str, str, object, object, float, float, float, float)
    def _on_result_ready(
        self,
        label: str,
        filename: str,
        generated: np.ndarray,
        restored: np.ndarray,
        generated_full: float,
        generated_masked: float,
        restored_full: float,
        restored_masked: float,
    ) -> None:
        self.results.append(
            {
                "label": label,
                "filename": filename,
                "generated": generated,
                "restored": restored,
                "generated_full": generated_full,
                "generated_masked": generated_masked,
                "restored_full": restored_full,
                "restored_masked": restored_masked,
            }
        )
        self._rebuild_cards()

    @Slot(int, int, str)
    def _on_progress(self, current: int, total: int, message: str) -> None:
        self.progress_bar.setRange(0, total)
        self.progress_bar.setValue(current)
        self.status_label.setText(message)

    @Slot(str)
    def _on_completed(self, output: str) -> None:
        self.experiment_dir = Path(output)
        self.status_label.setText(f"Completed {len(self.results)} checkpoints. Saved to {output}")
        self._finish_run()

    @Slot(str)
    def _on_failed(self, details: str) -> None:
        self.status_label.setText("Comparison failed. See error dialog.")
        self._finish_run()
        QMessageBox.critical(self, "Inference failed", details)

    def _finish_run(self) -> None:
        self.run_button.setEnabled(True)
        self.cancel_button.setEnabled(False)
        self.save_button.setEnabled(self.experiment_dir is not None and bool(self.results))
        self.open_button.setEnabled(self.experiment_dir is not None)

    @Slot()
    def _thread_finished(self) -> None:
        self.worker = None
        self.thread = None

    def _clear_cards(self) -> None:
        while self.cards_layout.count():
            item = self.cards_layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()

    @Slot()
    def _rebuild_cards(self) -> None:
        self._clear_cards()
        field = self.display_combo.currentData()
        for result in self.results:
            full = result[f"{field}_full"]
            masked = result[f"{field}_masked"]
            subtitle = (
                f"{result['filename']}  |  PSNR full {full:.3f} dB  |  "
                f"masked {masked:.3f} dB"
            )
            card = ImageCard(result["label"], subtitle, image_size=320)
            card.set_image(result[field])
            self.cards_layout.addWidget(card)
        QTimer.singleShot(0, self._scroll_to_latest)

    def _scroll_to_latest(self) -> None:
        bar = self.scroll.horizontalScrollBar()
        bar.setValue(bar.maximum())

    def _save_comparison_as(self) -> None:
        if self.experiment_dir is None:
            return
        field = self.display_combo.currentData()
        source = self.experiment_dir / f"comparison_{field}.png"
        destination, _ = QFileDialog.getSaveFileName(self, "Save comparison", source.name, "PNG (*.png)")
        if destination:
            shutil.copy2(source, destination)

    def _open_experiment(self) -> None:
        if self.experiment_dir is not None:
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(self.experiment_dir.resolve())))

    def closeEvent(self, event) -> None:
        if self.thread is not None and self.thread.isRunning():
            QMessageBox.information(
                self,
                "Inference is running",
                "Cancel the run and wait for the current checkpoint to finish before closing the window.",
            )
            event.ignore()
            return
        super().closeEvent(event)

    def _apply_style(self) -> None:
        self.setStyleSheet(
            """
            QMainWindow, QWidget { background: #111315; color: #e7e7e2; font-size: 10pt; }
            QLabel#heading { font-size: 20pt; font-weight: 700; color: #f4f0e6; }
            QLabel#description, QLabel#status, QLabel#cardSubtitle { color: #9ea7aa; }
            QFrame#leftPanel, QFrame#controls, QFrame#imageCard { background: #1a1e20; border: 1px solid #303638; border-radius: 8px; }
            QLabel#cardTitle { font-size: 14pt; font-weight: 650; color: #f4f0e6; }
            QLabel#sectionTitle { font-size: 13pt; font-weight: 650; color: #f4f0e6; }
            QLineEdit, QComboBox { background: #0e1011; border: 1px solid #3b4345; border-radius: 5px; padding: 5px; }
            QSpinBox { background: #0e1011; border: 1px solid #3b4345; border-radius: 5px; padding: 5px; padding-right: 24px; }
            QPushButton { background: #2a3032; border: 1px solid #424b4e; border-radius: 5px; padding: 6px 8px; }
            QPushButton:hover { background: #343c3f; }
            QPushButton:disabled { color: #666d70; background: #202426; }
            QPushButton#primaryButton { background: #d56b3f; color: #15100d; border: none; font-weight: 700; }
            QPushButton#primaryButton:hover { background: #e47a4d; }
            QProgressBar { background: #0e1011; border: 1px solid #303638; border-radius: 4px; height: 10px; text-align: center; }
            QProgressBar::chunk { background: #d56b3f; border-radius: 3px; }
            QScrollArea { background: transparent; }
            """
        )


def main() -> None:
    app = QApplication(sys.argv)
    app.setFont(QFont("Segoe UI", 10))
    window = CheckpointComparisonWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
