"""Host-side receiver/logger for the tactile hand.

Decodes both the legacy frame (0xAAAA | 440 x uint16 LE | 0x5555, 884 B,
firmware as built) and the proposed packed frame (tactile_frame.c, 665 B),
and logs timestamped frames to an .npz file for the characterisation
experiments (characterize.py).

  python tactile_logger.py --port /dev/ttyUSB0 --baud 115200 --legacy -o run1.npz
"""
from __future__ import annotations
import argparse, time, struct
import numpy as np

CELLS, ROWS, COLS = 440, 44, 10
LEGACY_LEN = (CELLS + 2) * 2
PACKED_LEN = 2 + 1 + CELLS * 3 // 2 + 2


def crc16_ccitt(data: bytes) -> int:
    crc = 0xFFFF
    for b in data:
        crc ^= b << 8
        for _ in range(8):
            crc = ((crc << 1) ^ 0x1021) & 0xFFFF if crc & 0x8000 else (crc << 1) & 0xFFFF
    return crc


def decode_packed(buf: bytes):
    """Return (seq, cells[440]) or None if sync/CRC fails."""
    if len(buf) < PACKED_LEN or buf[0] != 0xA5 or buf[1] != 0x5A:
        return None
    body = buf[2:PACKED_LEN - 2]
    if crc16_ccitt(body) != (buf[PACKED_LEN - 2] << 8 | buf[PACKED_LEN - 1]):
        return None
    b = np.frombuffer(body[1:], np.uint8).reshape(-1, 3).astype(np.uint16)
    a = (b[:, 0] << 4) | (b[:, 1] >> 4)
    c = ((b[:, 1] & 0x0F) << 8) | b[:, 2]
    return body[0], np.stack([a, c], 1).ravel()


def decode_legacy(buf: bytes):
    if len(buf) < LEGACY_LEN:
        return None
    w = struct.unpack("<%dH" % (CELLS + 2), buf[:LEGACY_LEN])
    if w[0] != 0xAAAA or w[-1] != 0x5555:
        return None
    return None, np.array(w[1:-1], np.uint16)


def to_pads(cells: np.ndarray) -> dict:
    """Split a 440-vector into the 7 pads (layout of firmware MyArray split)."""
    m = cells.reshape(ROWS, COLS)
    pads = {}
    for f in range(3):
        pads[f"distal_{f+1}"] = m[8*f:8*f+8, 0:5]
        pads[f"proximal_{f+1}"] = m[8*f:8*f+8, 5:10]
    pads["palm"] = m[24:44, :]
    return pads


class FrameReader:
    def __init__(self, ser, legacy: bool):
        self.ser, self.legacy, self.buf = ser, legacy, bytearray()
        self.sync = b"\xaa\xaa" if legacy else b"\xa5\x5a"
        self.n = LEGACY_LEN if legacy else PACKED_LEN
        self.bad = 0

    def read(self):
        while True:
            self.buf += self.ser.read(max(1, self.ser.in_waiting))
            i = self.buf.find(self.sync)
            if i < 0:
                del self.buf[:-1]; continue
            if len(self.buf) - i < self.n:
                continue
            frame = bytes(self.buf[i:i + self.n])
            res = decode_legacy(frame) if self.legacy else decode_packed(frame)
            if res is None:
                self.bad += 1; del self.buf[:i + 1]; continue
            del self.buf[:i + self.n]
            return time.perf_counter(), res[1]


def main():
    import serial
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", required=True); ap.add_argument("--baud", type=int, default=115200)
    ap.add_argument("--legacy", action="store_true"); ap.add_argument("--seconds", type=float, default=30)
    ap.add_argument("-o", "--out", default="tactile_log.npz")
    a = ap.parse_args()
    ser = serial.Serial(a.port, a.baud, timeout=0.05)
    rd = FrameReader(ser, a.legacy)
    ts, frames = [], []
    t_end = time.perf_counter() + a.seconds
    while time.perf_counter() < t_end:
        t, f = rd.read(); ts.append(t); frames.append(f)
    ts = np.array(ts) - ts[0]
    np.savez_compressed(a.out, t=ts, frames=np.array(frames))
    print(f"{len(ts)} frames, {len(ts)/ts[-1]:.1f} Hz, {rd.bad} rejected -> {a.out}")


if __name__ == "__main__":
    main()
