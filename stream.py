"""S25 Pro FLOW video streamer.

Connects to a "WiFi UAV" style drone at 192.168.169.1:8800, requests JPEG
frames over a duplex UDP socket, reassembles fragments, prepends a
generated JPEG header (the drone strips it to save bandwidth) and decodes
with OpenCV.

Protocol distilled from marshallrichards/turbodrone (backend/protocols/
wifi_uav_video_protocol.py + utils/wifi_uav_*).

Usage:
    pip install opencv-python numpy
    # Connect PC to the FLOW_99401C wifi
    python stream.py
    # press q in the window to quit

Optional:
    --debug           verbose per-fragment logging
    --dump-frames     write every reassembled JPEG to ./frames/
    --width / --height for non-default resolution
"""

import argparse
import os
import queue
import socket
import threading
import time
from typing import Dict, Optional

import cv2
import numpy as np


DRONE_IP = "192.168.169.1"
PORT = 8800

START_STREAM = b"\xef\x00\x04\x00"

REQUEST_A = (
    b"\xef\x02\x58\x00\x02\x02"
    b"\x00\x01\x00\x00\x00\x00\x05\x00\x00\x00\x14\x00\x66\x14\x80\x80"
    b"\x80\x80\x00\x02\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x02\x99"
    b"\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00"
    b"\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00"
    b"\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x32\x4b\x14\x2d"
    b"\x00\x00"
)

REQUEST_B = (
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

SOI = b"\xff\xd8"
EOI = b"\xff\xd9"

STD_LUMINANCE_QT = [
    16, 11, 10, 16, 24, 40, 51, 61,
    12, 12, 14, 19, 26, 58, 60, 55,
    14, 13, 16, 24, 40, 57, 69, 56,
    14, 17, 22, 29, 51, 87, 80, 62,
    18, 22, 37, 56, 68, 109, 103, 77,
    24, 35, 55, 64, 81, 104, 113, 92,
    49, 64, 78, 87, 103, 121, 120, 101,
    72, 92, 95, 98, 112, 100, 103, 99,
]

STD_CHROMINANCE_QT = [
    17, 18, 24, 47, 99, 99, 99, 99,
    18, 21, 26, 66, 99, 99, 99, 99,
    24, 26, 56, 99, 99, 99, 99, 99,
    47, 66, 99, 99, 99, 99, 99, 99,
    99, 99, 99, 99, 99, 99, 99, 99,
    99, 99, 99, 99, 99, 99, 99, 99,
    99, 99, 99, 99, 99, 99, 99, 99,
    99, 99, 99, 99, 99, 99, 99, 99,
]


def _dqt(table_id: int, table) -> bytes:
    payload = bytes([table_id]) + bytes(table)
    return b"\xff\xdb" + (len(payload) + 2).to_bytes(2, "big") + payload


def _sof0(width: int, height: int) -> bytes:
    comps = b""
    for cid, qt in ((1, 0), (2, 1), (3, 1)):
        comps += bytes([cid, 0x11, qt])
    body = b"\x08" + height.to_bytes(2, "big") + width.to_bytes(2, "big") + b"\x03" + comps
    return b"\xff\xc0" + (len(body) + 2).to_bytes(2, "big") + body


def _sos() -> bytes:
    body = b"\x03" + bytes([1, 0x00, 2, 0x11, 3, 0x11]) + b"\x00\x3f\x00"
    return b"\xff\xda" + (len(body) + 2).to_bytes(2, "big") + body


def build_jpeg_header(width: int, height: int) -> bytes:
    return SOI + _dqt(0, STD_LUMINANCE_QT) + _dqt(1, STD_CHROMINANCE_QT) + _sof0(width, height) + _sos()


class FlowVideo:
    FRAME_TIMEOUT = 0.08
    MAX_RETRIES = 3
    WATCHDOG_SLEEP = 0.05

    def __init__(self, drone_ip: str, port: int, width: int, height: int, debug: bool = False):
        self.drone_ip = drone_ip
        self.port = port
        self.debug = debug
        self._dbg = print if debug else (lambda *a, **k: None)

        self._jpeg_header = build_jpeg_header(width, height)
        self._frag: Dict[int, bytes] = {}
        self._cur_fid = 1
        self._last_req_ts = time.time()
        self._retry_cnt = 0
        self._first_frame = True

        self.frames_ok = 0
        self.frames_dropped = 0

        self.frame_q: "queue.Queue[bytes]" = queue.Queue(maxsize=2)

        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._sock.bind(("", 0))
        self._sock.settimeout(1.0)
        self._dbg(f"[init] local UDP *:{self._sock.getsockname()[1]} -> {drone_ip}:{port}")

        self._running = True
        self._send_start()
        self._send_frame_request(0)

        threading.Thread(target=self._rx_loop, daemon=True, name="rx").start()
        threading.Thread(target=self._watchdog, daemon=True, name="wd").start()
        threading.Thread(target=self._warmup, daemon=True, name="warm").start()

    def _send_start(self):
        self._sock.sendto(START_STREAM, (self.drone_ip, self.port))

    def _send_frame_request(self, fid: int):
        lo, hi = fid & 0xFF, (fid >> 8) & 0xFF
        a = bytearray(REQUEST_A)
        a[12], a[13] = lo, hi
        b = bytearray(REQUEST_B)
        for base in (12, 88, 107):
            b[base] = lo
            b[base + 1] = hi
        self._sock.sendto(a, (self.drone_ip, self.port))
        self._sock.sendto(b, (self.drone_ip, self.port))
        self._last_req_ts = time.time()

    def _warmup(self):
        while self._first_frame and self._running:
            self._send_start()
            self._send_frame_request((self._cur_fid - 1) & 0xFFFF)
            time.sleep(0.2)

    def _rx_loop(self):
        while self._running:
            try:
                pkt, _ = self._sock.recvfrom(4096)
            except socket.timeout:
                continue
            except OSError:
                break
            self._handle(pkt)

    def _handle(self, pkt: bytes):
        if len(pkt) < 56 or pkt[1] != 0x01:
            return

        fid = int.from_bytes(pkt[16:18], "little")
        if fid != self._cur_fid:
            self.frames_dropped += 1
            self._frag.clear()
            self._cur_fid = fid

        frag_id = int.from_bytes(pkt[32:34], "little")
        self._frag.setdefault(frag_id, pkt[56:])

        if pkt[2] == 0x38:
            return

        ordered = [self._frag[i] for i in sorted(self._frag)]
        jpeg = self._jpeg_header + b"".join(ordered) + EOI
        self.frames_ok += 1
        self._first_frame = False

        try:
            self.frame_q.put_nowait(jpeg)
        except queue.Full:
            try:
                self.frame_q.get_nowait()
                self.frame_q.put_nowait(jpeg)
            except queue.Empty:
                pass

        self._dbg(f"frame {fid:04x} ok ({len(self._frag)} frags, {len(jpeg)} B)")

        self._frag.clear()
        self._retry_cnt = 0
        self._send_frame_request(fid)
        self._cur_fid = (fid + 1) & 0xFFFF

    def _watchdog(self):
        while self._running:
            time.sleep(self.WATCHDOG_SLEEP)
            if time.time() - self._last_req_ts < self.FRAME_TIMEOUT:
                continue
            if self._retry_cnt < self.MAX_RETRIES:
                self._retry_cnt += 1
                self._send_frame_request((self._cur_fid - 1) & 0xFFFF)
            else:
                self.frames_dropped += 1
                self._frag.clear()
                self._retry_cnt = 0
                self._cur_fid = (self._cur_fid + 1) & 0xFFFF
                self._send_frame_request((self._cur_fid - 1) & 0xFFFF)

    def stop(self):
        self._running = False
        try:
            self._sock.close()
        except Exception:
            pass


def display(stream: FlowVideo, dump_dir: Optional[str]):
    cv2.namedWindow("S25 Pro FLOW", cv2.WINDOW_NORMAL)
    placeholder = np.zeros((360, 640, 3), np.uint8)
    cv2.putText(placeholder, "waiting for frames...", (130, 180),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    last_fps = time.time()
    n = 0
    while True:
        try:
            jpeg = stream.frame_q.get(timeout=1.0)
            arr = np.frombuffer(jpeg, dtype=np.uint8)
            img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if img is None:
                img = placeholder
            else:
                n += 1
                if dump_dir:
                    with open(os.path.join(dump_dir, f"f_{int(time.time()*1000)}.jpg"), "wb") as f:
                        f.write(jpeg)
        except queue.Empty:
            img = placeholder

        cv2.imshow("S25 Pro FLOW", img)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

        if time.time() - last_fps > 2.0:
            print(f"[stats] ok:{stream.frames_ok}  dropped:{stream.frames_dropped}  ~{n/2:.1f} fps")
            last_fps = time.time()
            n = 0

    cv2.destroyAllWindows()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--drone-ip", default=DRONE_IP)
    p.add_argument("--port", type=int, default=PORT)
    p.add_argument("--width", type=int, default=640)
    p.add_argument("--height", type=int, default=360)
    p.add_argument("--debug", action="store_true")
    p.add_argument("--dump-frames", action="store_true")
    args = p.parse_args()

    dump_dir = None
    if args.dump_frames:
        dump_dir = "frames"
        os.makedirs(dump_dir, exist_ok=True)

    stream = FlowVideo(args.drone_ip, args.port, args.width, args.height, args.debug)
    try:
        display(stream, dump_dir)
    finally:
        stream.stop()
        print(f"[done] ok:{stream.frames_ok}  dropped:{stream.frames_dropped}")


if __name__ == "__main__":
    main()
