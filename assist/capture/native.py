"""Record3D USB stream reader that never falls behind the phone.

The record3d C++ library decodes *every* frame on one thread (stb JPEG +
LZFSE) and then calls back into Python. It never drops a frame. When decoding
a 1440x1920 JPEG plus taking the GIL costs more than the phone's frame
interval, unread frames pile up between the phone and the Mac and the picture
falls further behind every second. Measured on an M2: ~50 fps decode ceiling
when idle, ~37 fps with YOLO + render running — so minutes of use meant a
30–60 s lag.

This reader speaks the same wire protocol (usbmuxd tunnel to port 1337,
PeerTalk framing) but only *receives* in its thread: frames are kept as raw
bytes and each new one replaces the previous. Decoding happens on demand, for
the newest frame only, so stale frames are dropped instead of queued.
"""

from __future__ import annotations

import ctypes
import plistlib
import socket
import struct
import threading
import time
import traceback
from dataclasses import dataclass
from typing import List, Optional

import cv2
import numpy as np

from ..debuglog import log, log_exc

USBMUXD_SOCKET = "/var/run/usbmuxd"
RECORD3D_PORT = 1337
RECONNECT_SEC = 1.0
# The phone sends 60 fps even when nothing moves, so this long without a byte
# means the stream is dead while the socket still looks open (Record3D paused,
# phone locked, or another client took the stream). Reconnect instead of
# freezing on the last frame forever.
STALL_RECONNECT_SEC = 3.0
# A 1440x1920 frame is ~0.2–0.5 MB. Anything near this cap means the byte
# stream lost framing (we would otherwise try to allocate garbage sizes).
MAX_MESSAGE_BYTES = 64 * 1024 * 1024

# usbmuxd framing: little-endian {length, version=1 (plist), message=8 (plist), tag}
_MUX_HEADER = struct.Struct("<IIII")
# PeerTalk framing: big-endian {version, type, tag, payload size}
_PT_HEADER = struct.Struct(">IIII")
# Record3D body: header, intrinsics (fx, fy, tx, ty), pose (qx..qw, tx..tz)
_R3D_HEADER = struct.Struct("<11I")
_INTRINSICS = struct.Struct("<4f")
_POSE = struct.Struct("<7f")

_COMPRESSION_LZFSE = 0x801


@dataclass
class Intrinsics:
    fx: float
    fy: float
    tx: float
    ty: float


@dataclass
class RawFrame:
    """One undecoded Record3D message plus its arrival order."""

    seq: int
    body: bytearray


@dataclass
class DecodedFrame:
    rgb_bgr: np.ndarray
    depth: np.ndarray
    conf: Optional[np.ndarray]
    K: Intrinsics
    device_type: int


def _recv_exact(sock: socket.socket, n: int) -> bytearray:
    buf = bytearray(n)
    _recv_into_exact(sock, memoryview(buf))
    return buf


def _recv_into_exact(sock: socket.socket, view: memoryview) -> None:
    # MSG_WAITALL keeps a whole frame in one blocking syscall with the GIL
    # released, instead of one GIL round-trip per ~64 KB chunk.
    got = 0
    total = len(view)
    while got < total:
        n = sock.recv_into(view[got:], total - got, socket.MSG_WAITALL)
        if n == 0:
            raise ConnectionError("Record3D stream closed")
        got += n


def _set_recv_timeout(sock: socket.socket, seconds: float) -> None:
    """Kernel-level receive timeout; keeps MSG_WAITALL single-syscall reads."""
    sec = int(seconds)
    usec = int((seconds - sec) * 1_000_000)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVTIMEO, struct.pack("ll", sec, usec))


def _mux_request(sock: socket.socket, payload: dict, tag: int = 1) -> dict:
    payload = {"ClientVersionString": "nekit", "ProgName": "nekit", **payload}
    body = plistlib.dumps(payload)
    sock.sendall(_MUX_HEADER.pack(_MUX_HEADER.size + len(body), 1, 8, tag) + body)
    length, _version, _message, _tag = _MUX_HEADER.unpack(
        _recv_exact(sock, _MUX_HEADER.size)
    )
    return plistlib.loads(bytes(_recv_exact(sock, length - _MUX_HEADER.size)))


def _mux_socket() -> socket.socket:
    sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    sock.connect(USBMUXD_SOCKET)
    return sock


@dataclass
class UsbDevice:
    device_id: int
    product_id: int
    udid: str


def list_usb_devices() -> List[UsbDevice]:
    """USB-attached iOS devices, in usbmuxd order (same as record3d's list)."""
    with _mux_socket() as sock:
        reply = _mux_request(sock, {"MessageType": "ListDevices"})
    out: List[UsbDevice] = []
    for entry in reply.get("DeviceList", []):
        props = entry.get("Properties", {})
        if props.get("ConnectionType") != "USB":
            continue
        out.append(
            UsbDevice(
                device_id=int(entry.get("DeviceID", props.get("DeviceID", 0))),
                product_id=int(props.get("ProductID", 0)),
                udid=str(props.get("SerialNumber", "")),
            )
        )
    return out


def connect_device(device_id: int, port: int = RECORD3D_PORT) -> socket.socket:
    """Open a usbmuxd tunnel to `port` on the device; raises if refused."""
    sock = _mux_socket()
    try:
        reply = _mux_request(
            sock,
            {
                "MessageType": "Connect",
                "DeviceID": device_id,
                "PortNumber": socket.htons(port),
            },
        )
    except Exception:
        sock.close()
        raise
    code = reply.get("Number", -1)
    if code != 0:
        sock.close()
        # 3 = connection refused: Record3D is not in USB Streaming mode.
        raise ConnectionRefusedError(f"usbmuxd connect failed (code {code})")
    return sock


_lzfse_fn = None


def _lzfse():
    global _lzfse_fn
    if _lzfse_fn is None:
        lib = ctypes.CDLL("/usr/lib/libcompression.dylib")
        fn = lib.compression_decode_buffer
        fn.restype = ctypes.c_size_t
        fn.argtypes = [
            ctypes.c_void_p,
            ctypes.c_size_t,
            ctypes.c_void_p,
            ctypes.c_size_t,
            ctypes.c_void_p,
            ctypes.c_int,
        ]
        _lzfse_fn = fn
    return _lzfse_fn


def lzfse_decode(body: bytearray, offset: int, size: int, out_size: int) -> Optional[np.ndarray]:
    """Decode an LZFSE block of `body` into `out_size` bytes (None on mismatch)."""
    if size <= 0 or out_size <= 0:
        return None
    src = (ctypes.c_char * size).from_buffer(body, offset)
    dst = np.empty(out_size, dtype=np.uint8)
    n = _lzfse()(dst.ctypes.data, out_size, ctypes.addressof(src), size, None, _COMPRESSION_LZFSE)
    if n != out_size:
        return None
    return dst


def decode_frame(body: bytearray) -> Optional[DecodedFrame]:
    """Parse one Record3D message (layout from record3d's Record3DStream.cpp)."""
    (
        _rgb_w,
        _rgb_h,
        depth_w,
        depth_h,
        conf_w,
        conf_h,
        rgb_size,
        depth_size,
        conf_size,
        _misc_size,
        device_type,
    ) = _R3D_HEADER.unpack_from(body, 0)
    off = _R3D_HEADER.size
    K = Intrinsics(*_INTRINSICS.unpack_from(body, off))
    off += _INTRINSICS.size + _POSE.size

    jpg = np.frombuffer(body, dtype=np.uint8, count=rgb_size, offset=off)
    rgb_bgr = cv2.imdecode(jpg, cv2.IMREAD_COLOR)
    off += rgb_size
    if rgb_bgr is None:
        return None

    raw_depth = lzfse_decode(body, off, depth_size, depth_w * depth_h * 4)
    off += depth_size
    if raw_depth is None:
        return None
    depth = raw_depth.view(np.float32).reshape(depth_h, depth_w)

    conf = None
    raw_conf = lzfse_decode(body, off, conf_size, conf_w * conf_h)
    if raw_conf is not None:
        conf = raw_conf.reshape(conf_h, conf_w)

    return DecodedFrame(
        rgb_bgr=rgb_bgr, depth=depth, conf=conf, K=K, device_type=int(device_type)
    )


class LatestFrameReader:
    """Background receiver that keeps only the newest raw Record3D message."""

    def __init__(self, device_id: int, on_new_frame=None, on_stream_stopped=None) -> None:
        self.device_id = device_id
        self.on_new_frame = on_new_frame
        self.on_stream_stopped = on_stream_stopped
        self._lock = threading.Lock()
        self._latest: Optional[RawFrame] = None
        self._seq = 0
        self._taken_seq = 0
        self._stop = threading.Event()
        self._sock: Optional[socket.socket] = None
        self._thread: Optional[threading.Thread] = None
        # Counters for FrameLogger; reset by take_stats().
        self._rx_frames = 0
        self._rx_bytes = 0
        self._dropped = 0

    def start(self) -> None:
        if self._thread is not None:
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name="record3d-rx", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        sock = self._sock
        if sock is not None:
            try:
                sock.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None

    def latest(self) -> Optional[RawFrame]:
        """Newest message; frames that arrived since the last call are dropped."""
        with self._lock:
            frame = self._latest
            if frame is not None and frame.seq > self._taken_seq:
                self._dropped += frame.seq - self._taken_seq - 1
                self._taken_seq = frame.seq
            return frame

    def take_stats(self) -> dict:
        with self._lock:
            stats = {
                "rx_frames": self._rx_frames,
                "rx_bytes": self._rx_bytes,
                "dropped": self._dropped,
            }
            self._rx_frames = self._rx_bytes = self._dropped = 0
        return stats

    def _run(self) -> None:
        refused_logged = False
        while not self._stop.is_set():
            try:
                sock = connect_device(self.device_id)
            except (OSError, ValueError) as exc:
                if not refused_logged:
                    log("frame", f"Record3D not streaming yet ({exc}); retrying")
                    print("Waiting for Record3D: USB Streaming → Record on the phone...")
                    refused_logged = True
                self._stop.wait(RECONNECT_SEC)
                continue
            refused_logged = False
            self._sock = sock
            log("frame", "Record3D stream connected", device_id=self.device_id)
            try:
                _set_recv_timeout(sock, STALL_RECONNECT_SEC)
                self._pump(sock)
            except (BlockingIOError, TimeoutError):
                if not self._stop.is_set():
                    log(
                        "frame",
                        "Record3D sent nothing — reconnecting",
                        silent_s=STALL_RECONNECT_SEC,
                    )
            except (OSError, ConnectionError) as exc:
                if not self._stop.is_set():
                    log("frame", f"Record3D stream lost: {type(exc).__name__}: {exc}")
            except Exception as exc:
                # Never let the receiver thread die silently: that would freeze
                # the picture on the last frame with nothing in the log.
                log_exc("frame", "Record3D receiver error — reconnecting", exc)
                print(traceback.format_exc().rstrip(), flush=True)
            finally:
                self._sock = None
                sock.close()
            if self._stop.is_set():
                return
            if self.on_stream_stopped is not None:
                self.on_stream_stopped()
            self._stop.wait(RECONNECT_SEC)

    def _pump(self, sock: socket.socket) -> None:
        header = bytearray(_PT_HEADER.size)
        header_view = memoryview(header)
        while not self._stop.is_set():
            _recv_into_exact(sock, header_view)
            size = _PT_HEADER.unpack(header)[3]
            if size <= _R3D_HEADER.size or size > MAX_MESSAGE_BYTES:
                raise ConnectionError(f"bad Record3D message size {size} — stream out of sync")
            body = bytearray(size)
            _recv_into_exact(sock, memoryview(body))
            with self._lock:
                self._seq += 1
                self._latest = RawFrame(seq=self._seq, body=body)
                self._rx_frames += 1
                self._rx_bytes += size + _PT_HEADER.size
            if self.on_new_frame is not None:
                self.on_new_frame()


def wait_for_device(dev_idx: int) -> UsbDevice:
    """Block until a USB iOS device exists at `dev_idx` (mirrors old prints)."""
    print("Searching for devices...")
    while True:
        try:
            devs = list_usb_devices()
        except OSError as exc:
            log("frame", f"usbmuxd unavailable: {exc}")
            devs = []
        print(f"{len(devs)} device(s) found")
        for i, dev in enumerate(devs):
            print(f"  [{i}] ID={dev.product_id}  UDID={dev.udid}")
        if len(devs) > dev_idx:
            return devs[dev_idx]
        print("No device yet. Enable USB Streaming in Record3D, press Record, wait...")
        time.sleep(2.0)
