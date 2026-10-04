"""Native Record3D reader: wire parsing and latest-frame-wins behaviour."""

import ctypes
import socket
import struct
import threading

import cv2
import numpy as np
import pytest

from assist.capture import native


def _has_libcompression() -> bool:
    # System dylibs live in the dyld shared cache on macOS 11+, so the path
    # does not exist on disk even though it loads.
    try:
        native._lzfse()
        return True
    except OSError:
        return False


pytestmark = pytest.mark.skipif(
    not _has_libcompression(), reason="LZFSE decode uses macOS libcompression"
)


def _lzfse_encode(data: bytes) -> bytes:
    lib = ctypes.CDLL("/usr/lib/libcompression.dylib")
    fn = lib.compression_encode_buffer
    fn.restype = ctypes.c_size_t
    fn.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_char_p,
                   ctypes.c_size_t, ctypes.c_void_p, ctypes.c_int]
    dst = ctypes.create_string_buffer(len(data) + 4096)
    n = fn(dst, len(dst), data, len(data), None, 0x801)
    assert n > 0
    return dst.raw[:n]


def _message(rgb_bgr, depth, conf, device_type=1, K=(500.0, 501.0, 96.0, 128.0)):
    jpg = cv2.imencode(".jpg", rgb_bgr, [cv2.IMWRITE_JPEG_QUALITY, 95])[1].tobytes()
    d = _lzfse_encode(depth.astype(np.float32).tobytes())
    c = _lzfse_encode(conf.astype(np.uint8).tobytes())
    h, w = rgb_bgr.shape[:2]
    dh, dw = depth.shape
    header = struct.pack("<11I", w, h, dw, dh, dw, dh, len(jpg), len(d), len(c), 0, device_type)
    body = header + struct.pack("<4f", *K) + struct.pack("<7f", *([0.0] * 7)) + jpg + d + c
    return body


def _frame_inputs():
    rgb = np.zeros((128, 96, 3), np.uint8)
    rgb[:, :48] = (255, 0, 0)  # left half blue in BGR
    depth = np.linspace(0.2, 4.0, 32 * 24, dtype=np.float32).reshape(32, 24)
    conf = (np.arange(32 * 24) % 3).astype(np.uint8).reshape(32, 24)
    return rgb, depth, conf


def test_decode_frame_roundtrip():
    rgb, depth, conf = _frame_inputs()
    out = native.decode_frame(bytearray(_message(rgb, depth, conf)))
    assert out is not None
    assert out.rgb_bgr.shape == rgb.shape
    # BGR channel order preserved (JPEG is lossy, so compare loosely).
    assert out.rgb_bgr[64, 10, 0] > 200 and out.rgb_bgr[64, 10, 2] < 50
    np.testing.assert_array_equal(out.depth, depth)
    np.testing.assert_array_equal(out.conf, conf)
    assert out.K.fx == pytest.approx(500.0) and out.K.ty == pytest.approx(128.0)
    assert out.device_type == 1


def test_reader_keeps_only_newest_frame(monkeypatch):
    rgb, depth, conf = _frame_inputs()
    body = _message(rgb, depth, conf)
    phone, mac = socket.socketpair()
    handed = []

    def fake_connect(_device_id, port=native.RECORD3D_PORT):
        if handed:
            raise ConnectionRefusedError("gone")
        handed.append(mac)
        return mac

    monkeypatch.setattr(native, "connect_device", fake_connect)
    got_five = threading.Event()
    seen = []

    def on_new():
        seen.append(1)
        if len(seen) == 5:
            got_five.set()

    reader = native.LatestFrameReader(device_id=1, on_new_frame=on_new)
    reader.start()
    try:
        for _ in range(5):
            phone.sendall(struct.pack(">IIII", 1, 101, 0, len(body)) + body)
        assert got_five.wait(5.0)
        frame = reader.latest()
        assert frame.seq == 5
        stats = reader.take_stats()
        # Four frames were never consumed: dropped, not queued.
        assert stats["rx_frames"] == 5 and stats["dropped"] == 4
        assert native.decode_frame(frame.body) is not None
    finally:
        phone.close()
        reader.stop()
