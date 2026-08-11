import argparse
from pathlib import Path
import tkinter as tk

import numpy as np
from PIL import Image, ImageDraw, ImageTk


ROOT = Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Open a lightweight interactive viewer that shows glove vertices and "
            "their matched MANO vertices side by side with synchronized rotation."
        )
    )
    parser.add_argument(
        "--rig-npz",
        type=Path,
        default=ROOT
        / "model"
        / "canonical_left_glove"
        / "canonical_left_glove_left_rig_k20_posefit.npz",
    )
    parser.add_argument("--width", type=int, default=1600)
    parser.add_argument("--height", type=int, default=900)
    parser.add_argument(
        "--point-size",
        type=int,
        default=2,
        help="Base point size for both point clouds.",
    )
    return parser.parse_args()


def normalize_colors(vertices: np.ndarray) -> np.ndarray:
    vmin = vertices.min(axis=0)
    vmax = vertices.max(axis=0)
    denom = np.maximum(vmax - vmin, 1e-8)
    colors = (vertices - vmin[None, :]) / denom[None, :]
    return np.clip(colors * 255.0, 0.0, 255.0).astype(np.uint8)


def average_mano_colors(
    glove_colors: np.ndarray, knn_indices: np.ndarray, mano_vertex_count: int
) -> np.ndarray:
    accum = np.zeros((mano_vertex_count, 3), dtype=np.float64)
    counts = np.zeros(mano_vertex_count, dtype=np.float64)
    nearest = knn_indices[:, 0]
    for glove_idx, mano_idx in enumerate(nearest):
        accum[mano_idx] += glove_colors[glove_idx]
        counts[mano_idx] += 1.0

    out = np.full((mano_vertex_count, 3), 120, dtype=np.uint8)
    valid = counts > 0
    out[valid] = np.clip(
        accum[valid] / counts[valid, None], 0.0, 255.0
    ).astype(np.uint8)
    return out


def rotation_matrix(yaw: float, pitch: float) -> np.ndarray:
    cy, sy = np.cos(yaw), np.sin(yaw)
    cx, sx = np.cos(pitch), np.sin(pitch)
    ry = np.array(
        [[cy, 0.0, sy], [0.0, 1.0, 0.0], [-sy, 0.0, cy]],
        dtype=np.float64,
    )
    rx = np.array(
        [[1.0, 0.0, 0.0], [0.0, cx, -sx], [0.0, sx, cx]],
        dtype=np.float64,
    )
    return rx @ ry


def project_points(
    vertices: np.ndarray,
    width: int,
    height: int,
    yaw: float,
    pitch: float,
    zoom: float,
) -> tuple[np.ndarray, np.ndarray]:
    rot = rotation_matrix(yaw, pitch)
    pts = vertices @ rot.T

    z = pts[:, 2]
    cam_dist = 3.0
    denom = np.maximum(cam_dist - z, 0.2)
    persp = zoom / denom

    x = pts[:, 0] * persp * width * 0.42 + width * 0.5
    y = -pts[:, 1] * persp * width * 0.42 + height * 0.54
    screen = np.stack([x, y], axis=1)
    return screen, z


def prepare_vertices(vertices: np.ndarray) -> np.ndarray:
    centered = vertices - vertices.mean(axis=0, keepdims=True)
    scale = np.max(np.linalg.norm(centered, axis=1))
    return centered / max(scale, 1e-8)


class CorrespondenceViewer:
    def __init__(
        self,
        glove_vertices: np.ndarray,
        glove_colors: np.ndarray,
        mano_vertices: np.ndarray,
        mano_colors: np.ndarray,
        width: int,
        height: int,
        point_size: int,
        meta_text: str,
    ) -> None:
        self.glove_vertices = prepare_vertices(glove_vertices)
        self.glove_colors = glove_colors
        self.mano_vertices = prepare_vertices(mano_vertices)
        self.mano_colors = mano_colors
        self.width = width
        self.height = height
        self.point_size = point_size
        self.meta_text = meta_text

        self.root = tk.Tk()
        self.root.title("Glove / MANO Correspondence Viewer")
        self.canvas = tk.Canvas(
            self.root, width=self.width, height=self.height, bg="#111111", highlightthickness=0
        )
        self.canvas.pack(fill="both", expand=True)

        self.yaw = 0.35
        self.pitch = -0.35
        self.zoom = 1.05
        self.last_mouse = None
        self.photo = None

        self.canvas.bind("<ButtonPress-1>", self.on_mouse_down)
        self.canvas.bind("<B1-Motion>", self.on_mouse_drag)
        self.canvas.bind("<MouseWheel>", self.on_mouse_wheel)
        self.root.bind("r", self.on_reset)
        self.root.bind("<Escape>", lambda _e: self.root.destroy())

    def on_mouse_down(self, event) -> None:
        self.last_mouse = (event.x, event.y)

    def on_mouse_drag(self, event) -> None:
        if self.last_mouse is None:
            self.last_mouse = (event.x, event.y)
            return
        dx = event.x - self.last_mouse[0]
        dy = event.y - self.last_mouse[1]
        self.last_mouse = (event.x, event.y)
        self.yaw += dx * 0.01
        self.pitch += dy * 0.01
        self.pitch = float(np.clip(self.pitch, -1.4, 1.4))
        self.render()

    def on_mouse_wheel(self, event) -> None:
        step = 1.08 if event.delta > 0 else 1.0 / 1.08
        self.zoom = float(np.clip(self.zoom * step, 0.3, 3.5))
        self.render()

    def on_reset(self, _event) -> None:
        self.yaw = 0.35
        self.pitch = -0.35
        self.zoom = 1.05
        self.render()

    def _draw_points(
        self,
        draw: ImageDraw.ImageDraw,
        vertices: np.ndarray,
        colors: np.ndarray,
        viewport_left: int,
        viewport_width: int,
        label: str,
    ) -> None:
        screen, z = project_points(
            vertices=vertices,
            width=viewport_width,
            height=self.height,
            yaw=self.yaw,
            pitch=self.pitch,
            zoom=self.zoom,
        )
        order = np.argsort(z)

        for idx in order:
            x = int(screen[idx, 0] + viewport_left)
            y = int(screen[idx, 1])
            if x < 0 or x >= self.width or y < 0 or y >= self.height:
                continue
            c = tuple(int(v) for v in colors[idx])
            r = self.point_size
            draw.ellipse((x - r, y - r, x + r, y + r), fill=c, outline=None)

        draw.text((viewport_left + 18, 16), label, fill=(240, 240, 240))

    def render(self) -> None:
        img = Image.new("RGB", (self.width, self.height), (17, 17, 17))
        draw = ImageDraw.Draw(img)

        half = self.width // 2
        draw.rectangle((0, 0, half - 1, self.height - 1), fill=(24, 24, 24))
        draw.rectangle((half, 0, self.width - 1, self.height - 1), fill=(24, 24, 24))
        draw.line((half, 0, half, self.height), fill=(60, 60, 60), width=1)

        self._draw_points(
            draw,
            self.glove_vertices,
            self.glove_colors,
            viewport_left=0,
            viewport_width=half,
            label="Glove vertices",
        )
        self._draw_points(
            draw,
            self.mano_vertices,
            self.mano_colors,
            viewport_left=half,
            viewport_width=self.width - half,
            label="Matched MANO vertices",
        )

        draw.text(
            (18, self.height - 40),
            "Drag: rotate | Wheel: zoom | R: reset | Esc: close",
            fill=(220, 220, 220),
        )
        draw.text((18, self.height - 20), self.meta_text, fill=(180, 180, 180))

        self.photo = ImageTk.PhotoImage(img)
        self.canvas.create_image(0, 0, image=self.photo, anchor="nw")

    def run(self) -> None:
        self.render()
        self.root.mainloop()


def main() -> None:
    args = parse_args()
    rig = np.load(args.rig_npz, allow_pickle=True)

    glove_vertices = rig["glove_template"].astype(np.float64)
    knn_indices = rig["knn_indices"].astype(np.int64)
    knn_distances = rig["knn_distances"].astype(np.float64)

    reference_verts_path = Path(str(rig["reference_verts_npy"]))
    if not reference_verts_path.is_absolute():
        reference_verts_path = ROOT / reference_verts_path
    mano_vertices = np.load(reference_verts_path).astype(np.float64)

    glove_colors = normalize_colors(glove_vertices)
    mano_colors = average_mano_colors(
        glove_colors=glove_colors,
        knn_indices=knn_indices,
        mano_vertex_count=mano_vertices.shape[0],
    )

    meta_text = (
        f"glove verts={glove_vertices.shape[0]} | mano verts={mano_vertices.shape[0]} | "
        f"k={int(rig['k'])} | mean nn dist={float(knn_distances[:, 0].mean()):.6f}"
    )

    viewer = CorrespondenceViewer(
        glove_vertices=glove_vertices,
        glove_colors=glove_colors,
        mano_vertices=mano_vertices,
        mano_colors=mano_colors,
        width=args.width,
        height=args.height,
        point_size=args.point_size,
        meta_text=meta_text,
    )
    viewer.run()


if __name__ == "__main__":
    main()
