"""
S25 Pro FLOW (wifi_uav family) → PC video streamer.

Protocol distilled from turbodrone (backend/protocols/wifi_uav_video_protocol.py
and backend/utils/wifi_uav_*.py).

Wire-up:
    1. Connect PC to drone Wi-Fi (FLOW_99401C). Drone is 192.168.169.1.
    2. python stream_drone.py
    3. q in the OpenCV window quits.

Deps: pip install opencv-python numpy
"""

from __future__ import annotations

import socket
import threading
import time
from collections import OrderedDict

import cv2
import numpy as np

DRONE_IP = "192.168.169.1"
CTRL_PORT = 8800           # drone listens here for start + frame-request
JPEG_W, JPEG_H = 640, 360

START_STREAM = b"\xef\x00\x04\x00"

# Two "next-frame" packets that must be sent for every frame, with a
# little-endian 16-bit frame counter patched into specific offsets.
REQUEST_A = bytearray(
    b"\xef\x02\x58\x00\x02\x02"
    b"\x00\x01\x00\x00\x00\x00\x05\x00\x00\x00\x14\x00\x66\x14\x80\x80"
    b"\x80\x80\x00\x02\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x02\x99"
    b"\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00"
    b"\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00"
    b"\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x32\x4b\x14\x2d"
    b"\x00\x00"
)
REQUEST_B = bytearray(
    b"\xef\x02\x6c\x00\x02\x02"
    b"\x00\x01\x02\x00\x00\x00\x09\x00\x00\x00\x14\x00\x66\x14\x80\x80"
    b"\x80\x80\x00\x02\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x02\x99"
    b"\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00"
    b"\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00"
    b"\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x32\x4b\x14\x2d"
    b"\x00\x00\x08\x00\x00\x00\x00\x00\x00\x00\x01\x00\x00\x00\x14\x00"
    b"\x00\x00\xff\xff\xff\xff\x09\x00\x00\x00\x00\x00\x00\x00\x03\x00"
    b"\x00\x00\x10\x00\x00\x00"
)

# ─────────────────────────── JPEG header builder ───────────────────────────
SOI = b"\xff\xd8"
EOI = b"\xff\xd9"

# Annex-K-ish luminance / chrominance tables. The drone strips SOI/DQT/SOF/SOS
# from every frame; we re-attach this prefix (and append EOI) before decoding.
_LUMA_QT = bytes([
    16, 11, 10, 16, 24,  40,  51,  61,
    12, 12, 14, 19, 26,  58,  60,  55,
    14, 13, 16, 24, 40,  57,  69,  56,
    14, 17, 22, 29, 51,  87,  80,  62,
    18, 22, 37, 56, 68, 109, 103,  77,
    24, 35, 55, 64, 81, 104, 113,  92,
    49, 64, 78, 87,103, 121, 120, 101,
    72, 92, 95, 98,112, 100, 103,  99,
])
_CHROMA_QT = bytes([
    17, 18, 24, 47, 99, 99, 99, 99,
    18, 21, 26, 66, 99, 99, 99, 99,
    24, 26, 56, 99, 99, 99, 99, 99,
    47, 66, 99, 99, 99, 99, 99, 99,
    99, 99, 99, 99, 99, 99, 99, 99,
    99, 99, 99, 99, 99, 99, 99, 99,
    99, 99, 99, 99, 99, 99, 99, 99,
    99, 99, 99, 99, 99, 99, 99, 99,
])


def _dqt(table_id: int, table: bytes) -> bytes:
    payload = bytes([table_id]) + table          # precision=0 → high nibble 0
    return b"\xff\xdb" + (len(payload) + 2).to_bytes(2, "big") + payload


def _sof0(width: int, height: int) -> bytes:
    # 4:4:4 sampling, 3 components, qt 0 for Y and 1 for Cb/Cr
    comps = bytes([
        1, 0x11, 0,
        2, 0x11, 1,
        3, 0x11, 1,
    ])
    payload = bytes([8]) + height.to_bytes(2, "big") + width.to_bytes(2, "big") + bytes([3]) + comps
    return b"\xff\xc0" + (len(payload) + 2).to_bytes(2, "big") + payload


def _sos() -> bytes:
    payload = bytes([
        3,
        1, 0x00,
        2, 0x11,
        3, 0x11,
        0, 63, 0,
    ])
    return b"\xff\xda" + (len(payload) + 2).to_bytes(2, "big") + payload


def build_jpeg_prefix(width: int, height: int) -> bytes:
    return SOI + _dqt(0, _LUMA_QT) + _dqt(1, _CHROMA_QT) + _sof0(width, height) + _sos()


# ─────────────────────────── streamer ───────────────────────────
class DroneStreamer:
    FRAME_TIMEOUT = 0.08      # 80 ms without progress → resend frame request
    MAX_RETRIES = 3
    WATCHDOG_TICK = 0.05

    def __init__(self, ip: str = DRONE_IP, port: int = CTRL_PORT) -> None:
        self.ip = ip
        self.port = port
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.sock.bind(("", 0))
        self.sock.settimeout(1.0)

        self.jpeg_prefix = build_jpeg_prefix(JPEG_W, JPEG_H)

        # Frame assembly state
        self.current_fid = 1                       # drone is more reliable when starting at 1
        self.fragments: "OrderedDict[int, bytes]" = OrderedDict()
        self.last_rx = time.time()
        self.last_req = time.time()
        self.retries = 0

        # Most-recent decoded frame for the display thread
        self._frame_lock = threading.Lock()
        self._latest_frame: np.ndarray | None = None

        self.frames_ok = 0
        self.frames_dropped = 0
        self.pkts_rx = 0
        self.pkts_video = 0
        self.decode_fail = 0

        self.running = True
        self._first_frame_seen = False

    # ───────── tx helpers ─────────
    def send_start(self) -> None:
        self.sock.sendto(START_STREAM, (self.ip, self.port))

    def send_frame_request(self, fid: int) -> None:
        lo, hi = fid & 0xFF, (fid >> 8) & 0xFF
        a = bytearray(REQUEST_A)
        a[12], a[13] = lo, hi
        b = bytearray(REQUEST_B)
        for base in (12, 88, 107):
            b[base], b[base + 1] = lo, hi
        self.sock.sendto(a, (self.ip, self.port))
        self.sock.sendto(b, (self.ip, self.port))
        self.last_req = time.time()

    # ───────── rx + assembly ─────────
    def handle_packet(self, pkt: bytes) -> None:
        self.pkts_rx += 1
        if self.pkts_rx <= 5:
            print(f"[rx #{self.pkts_rx}] len={len(pkt)}  head={pkt[:8].hex()}")
        # Video packets: byte 1 == 0x01, ≥56 byte header.
        if len(pkt) < 56 or pkt[1] != 0x01:
            return
        self.pkts_video += 1

        self.last_rx = time.time()
        self.retries = 0

        frame_id = int.from_bytes(pkt[16:18], "little")
        frag_id = int.from_bytes(pkt[32:34], "little")

        if frame_id != self.current_fid:
            # Drone skipped ahead — resync.
            self.frames_dropped += 1
            self.fragments.clear()
            self.current_fid = frame_id

        self.fragments.setdefault(frag_id, pkt[56:])

        # byte 2 == 0x38 means "more fragments coming"
        if pkt[2] == 0x38:
            return

        # Last fragment — assemble.
        ordered = [self.fragments[i] for i in sorted(self.fragments)]
        jpeg = self.jpeg_prefix + b"".join(ordered) + EOI
        img = cv2.imdecode(np.frombuffer(jpeg, dtype=np.uint8), cv2.IMREAD_COLOR)
        if img is not None:
            with self._frame_lock:
                self._latest_frame = img
            self.frames_ok += 1
            self._first_frame_seen = True
        else:
            self.decode_fail += 1
            self.frames_dropped += 1

        self.fragments.clear()
        # Ask the drone for the next frame.
        self.send_frame_request(frame_id)
        self.current_fid = (frame_id + 1) & 0xFFFF

    # ───────── threads ─────────
    def rx_loop(self) -> None:
        while self.running:
            try:
                pkt, _ = self.sock.recvfrom(4096)
            except socket.timeout:
                continue
            except OSError:
                break
            self.handle_packet(pkt)

    def warmup_loop(self) -> None:
        # Until the first frame arrives, keep resending start + frame-request.
        while self.running and not self._first_frame_seen:
            try:
                self.send_start()
                self.send_frame_request((self.current_fid - 1) & 0xFFFF)
            except OSError:
                break
            time.sleep(0.2)

    def watchdog_loop(self) -> None:
        # If a frame stalls partway through, resend the request for it.
        while self.running:
            time.sleep(self.WATCHDOG_TICK)
            if time.time() - self.last_req < self.FRAME_TIMEOUT:
                continue
            if self.retries < self.MAX_RETRIES:
                self.retries += 1
                try:
                    self.send_frame_request((self.current_fid - 1) & 0xFFFF)
                except OSError:
                    return
            else:
                self.frames_dropped += 1
                self.fragments.clear()
                self.retries = 0
                self.current_fid = (self.current_fid + 1) & 0xFFFF
                try:
                    self.send_frame_request((self.current_fid - 1) & 0xFFFF)
                except OSError:
                    return

    def get_latest(self) -> np.ndarray | None:
        with self._frame_lock:
            return self._latest_frame

    def start(self) -> None:
        self.send_start()
        self.send_frame_request(0)
        for target, name in [
            (self.rx_loop, "rx"),
            (self.warmup_loop, "warmup"),
            (self.watchdog_loop, "watchdog"),
        ]:
            t = threading.Thread(target=target, name=name, daemon=True)
            t.start()

    def stop(self) -> None:
        self.running = False
        try:
            self.sock.close()
        except OSError:
            pass


# ─────────────────────────── main ───────────────────────────
def main() -> None:
    streamer = DroneStreamer()
    streamer.start()

    win = "S25 Pro FLOW"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win, JPEG_W * 2, JPEG_H * 2)

    placeholder = np.zeros((JPEG_H, JPEG_W, 3), dtype=np.uint8)
    cv2.putText(placeholder, "Waiting for drone...", (40, JPEG_H // 2),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
    cv2.imshow(win, placeholder)
    cv2.waitKey(1)

    last_stats = time.time()
    last_ok = 0
    try:
        while True:
            frame = streamer.get_latest()
            cv2.imshow(win, frame if frame is not None else placeholder)
            if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                break
            now = time.time()
            if now - last_stats >= 2.0:
                fps = (streamer.frames_ok - last_ok) / (now - last_stats)
                last_ok = streamer.frames_ok
                last_stats = now
                print(
                    f"[stats] fps={fps:5.1f}  ok={streamer.frames_ok}  "
                    f"drop={streamer.frames_dropped}  "
                    f"rx={streamer.pkts_rx}  video={streamer.pkts_video}  "
                    f"decode_fail={streamer.decode_fail}"
                )
    finally:
        streamer.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
