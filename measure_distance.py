"""
Record3D USB: click on RGB to measure distance in meters.

Controls:
  Left click  — measure at that pixel
  q / Esc     — quit
"""

from threading import Event

import cv2
import numpy as np
from record3d import Record3DStream

DEVICE_TYPE__TRUEDEPTH = 0
DEVICE_TYPE__LIDAR = 1
WINDOW_NAME = "Record3D Distance"
PATCH = 5  # odd size; median over neighborhood
MIN_CONF = 1  # 0=low, 1=med, 2=high


class DistanceApp:
    def __init__(self):
        self.event = Event()
        self.session = None
        self.click = None  # (x, y) in displayed RGB coords
        self.last = None  # last measurement text
        self.rgb_disp = None
        self.depth_aligned = None
        self.conf_aligned = None
        self.K = None

    def on_new_frame(self):
        self.event.set()

    def on_stream_stopped(self):
        print("Stream stopped")

    def connect(self, dev_idx=0):
        print("Searching for devices...")
        while True:
            devs = Record3DStream.get_connected_devices()
            print(f"{len(devs)} device(s) found")
            for i, dev in enumerate(devs):
                print(f"  [{i}] ID={dev.product_id}  UDID={dev.udid}")
            if len(devs) > dev_idx:
                break
            print("No device yet. Enable USB Streaming in Record3D, press Record, wait...")
            self.event.wait(2.0)
            self.event.clear()

        self.session = Record3DStream()
        self.session.on_new_frame = self.on_new_frame
        self.session.on_stream_stopped = self.on_stream_stopped
        self.session.connect(devs[dev_idx])
        print("Connected. Click on the image to measure distance.")

    def on_mouse(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.click = (x, y)

    def _align_depth_to_rgb(self, depth, conf, rgb_shape):
        h, w = rgb_shape[:2]
        depth_a = cv2.resize(depth, (w, h), interpolation=cv2.INTER_NEAREST)
        if conf is not None and conf.size > 0:
            conf_a = cv2.resize(conf, (w, h), interpolation=cv2.INTER_NEAREST)
        else:
            conf_a = np.full((h, w), 2, dtype=np.uint8)
        return depth_a, conf_a

    def measure_at(self, x, y):
        depth = self.depth_aligned
        conf = self.conf_aligned
        K = self.K
        if depth is None or K is None:
            return None

        h, w = depth.shape
        if not (0 <= x < w and 0 <= y < h):
            return None

        r = PATCH // 2
        y0, y1 = max(0, y - r), min(h, y + r + 1)
        x0, x1 = max(0, x - r), min(w, x + r + 1)
        patch_d = depth[y0:y1, x0:x1]
        patch_c = conf[y0:y1, x0:x1]

        mask = (patch_d > 0.05) & (patch_d < 20.0) & (patch_c >= MIN_CONF)
        if not np.any(mask):
            return {"ok": False, "reason": "no valid depth (try another spot / better lighting)"}

        z = float(np.median(patch_d[mask]))
        fx, fy, cx, cy = float(K.fx), float(K.fy), float(K.tx), float(K.ty)
        X = (x - cx) * z / fx
        Y = (y - cy) * z / fy
        dist = float(np.sqrt(X * X + Y * Y + z * z))
        c_med = int(np.median(patch_c[mask]))

        return {
            "ok": True,
            "z": z,
            "dist": dist,
            "conf": c_med,
            "xyz": (X, Y, z),
            "uv": (x, y),
        }

    def run(self):
        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(WINDOW_NAME, self.on_mouse)

        while True:
            self.event.wait()
            depth = self.session.get_depth_frame()
            rgb = self.session.get_rgb_frame()
            conf = self.session.get_confidence_frame()
            self.K = self.session.get_intrinsic_mat()

            if self.session.get_device_type() == DEVICE_TYPE__TRUEDEPTH:
                depth = cv2.flip(depth, 1)
                rgb = cv2.flip(rgb, 1)
                if conf is not None and conf.size > 0:
                    conf = cv2.flip(conf, 1)

            rgb_bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            self.depth_aligned, self.conf_aligned = self._align_depth_to_rgb(
                depth, conf, rgb_bgr.shape
            )
            self.rgb_disp = rgb_bgr.copy()

            if self.click is not None:
                m = self.measure_at(*self.click)
                self.click = None
                if m is None:
                    self.last = "no frame"
                elif not m["ok"]:
                    self.last = m["reason"]
                    print(self.last)
                else:
                    self.last = (
                        f"depth Z={m['z']:.3f} m | 3D dist={m['dist']:.3f} m "
                        f"| conf={m['conf']} | xyz=({m['xyz'][0]:.2f},{m['xyz'][1]:.2f},{m['xyz'][2]:.2f})"
                    )
                    print(self.last)
                    cv2.circle(self.rgb_disp, m["uv"], 6, (0, 255, 0), 2)
                    cv2.circle(self.rgb_disp, m["uv"], 2, (0, 255, 0), -1)

            # depth colormap overlay (subtle)
            d_vis = np.clip(self.depth_aligned, 0, 5.0) / 5.0
            d_color = cv2.applyColorMap((d_vis * 255).astype(np.uint8), cv2.COLORMAP_TURBO)
            vis = cv2.addWeighted(self.rgb_disp, 0.72, d_color, 0.28, 0)

            if self.last:
                cv2.rectangle(vis, (0, 0), (vis.shape[1], 36), (0, 0, 0), -1)
                cv2.putText(
                    vis,
                    self.last,
                    (10, 24),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (0, 255, 180),
                    1,
                    cv2.LINE_AA,
                )
            else:
                cv2.putText(
                    vis,
                    "Click to measure distance (meters). q=quit",
                    (10, 24),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 255),
                    1,
                    cv2.LINE_AA,
                )

            cv2.imshow(WINDOW_NAME, vis)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break

            self.event.clear()

        cv2.destroyAllWindows()


if __name__ == "__main__":
    app = DistanceApp()
    app.connect(dev_idx=0)
    app.run()
