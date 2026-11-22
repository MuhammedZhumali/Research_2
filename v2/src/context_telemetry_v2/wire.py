"""Actually serialized research frames. Security bytes are accounted placeholders.

This is NOT an implementation of CCSDS, AEAD, or a deployable secure link.
Only public observables enter serialization. Raw measurements omit command epoch.
"""
from __future__ import annotations
from dataclasses import dataclass
import struct
import zlib
from .model import CommandEvent, Measurement

HEADER = struct.Struct("!2sBBIId")
CONTEXT = struct.Struct("!d?I")
BASE_RECORD = struct.Struct("!dfB")
ACK_BYTES = HEADER.size + 4 + 12 + 16 + 4

def encoded_wire_size(measurements=(), contexts=(), *, include_epoch=False):
    """Exact length of the fixed binary schema; final frames are still serialized."""
    data_size = sum(BASE_RECORD.size + (4 if include_epoch else 0)
                    + (4 if m.current_a is not None else 0) for m in measurements)
    return HEADER.size + 12 + 4 + len(contexts) * CONTEXT.size + data_size + 16 + 4 + ACK_BYTES

@dataclass(frozen=True)
class Frame:
    blob: bytes
    measurements: tuple[Measurement, ...]
    contexts: tuple[CommandEvent, ...]
    data_bytes: int
    context_bytes: int
    sequence: int
    @property
    def wire_bytes(self):
        return len(self.blob) + ACK_BYTES

def encode(sequence, created_s, measurements=(), contexts=(), *, include_epoch=False):
    payload = bytearray(struct.pack("!HH", len(contexts), len(measurements)))
    context_size = 0
    for c in contexts:
        item = CONTEXT.pack(c.time_s, c.heater_on, c.epoch)
        payload.extend(item)
        context_size += len(item)
    data_size = 0
    for m in measurements:
        flags = int(m.current_a is not None) | (2 if include_epoch else 0)
        record = bytearray(BASE_RECORD.pack(m.time_s, m.temperature_c, flags))
        if include_epoch:
            record.extend(struct.pack("!I", m.command_epoch))
        if m.current_a is not None:
            record.extend(struct.pack("!f", m.current_a))
        payload.extend(record)
        data_size += len(record)
    header = HEADER.pack(b"CT", 2, int(include_epoch), sequence, len(payload), created_s)
    # 96-bit nonce + 128-bit authentication tag: cost model, not cryptography.
    body = header + bytes(12) + bytes(payload) + bytes(16)
    blob = body + struct.pack("!I", zlib.crc32(body))
    assert len(blob) + ACK_BYTES == encoded_wire_size(measurements, contexts, include_epoch=include_epoch)
    decoded_m, decoded_c = decode(blob)
    return Frame(blob, tuple(decoded_m), tuple(decoded_c), data_size, context_size, sequence)

def decode(blob):
    if zlib.crc32(blob[:-4]) != struct.unpack("!I", blob[-4:])[0]:
        raise ValueError("Research frame CRC mismatch")
    magic, version, _, _, size, _ = HEADER.unpack_from(blob)
    if magic != b"CT" or version != 2:
        raise ValueError("Unknown frame")
    offset = HEADER.size + 12
    stop = offset + size
    n_context, n_measure = struct.unpack_from("!HH", blob, offset)
    offset += 4
    contexts, measurements = [], []
    for _ in range(n_context):
        t, mode, epoch = CONTEXT.unpack_from(blob, offset)
        offset += CONTEXT.size
        contexts.append(CommandEvent(t, mode, epoch))
    for _ in range(n_measure):
        t, temp, flags = BASE_RECORD.unpack_from(blob, offset)
        offset += BASE_RECORD.size
        epoch, current = -1, None
        if flags & 2:
            epoch = struct.unpack_from("!I", blob, offset)[0]
            offset += 4
        if flags & 1:
            current = struct.unpack_from("!f", blob, offset)[0]
            offset += 4
        measurements.append(Measurement(t, temp, current, epoch))
    if offset != stop or len(blob) != HEADER.size + 12 + size + 16 + 4:
        raise ValueError("Frame length mismatch")
    return measurements, contexts

def encode_ack(sequence, received_s):
    payload = struct.pack("!I", sequence)
    head = HEADER.pack(b"CA", 2, 0, sequence, len(payload), received_s)
    body = head + bytes(12) + payload + bytes(16)
    return body + struct.pack("!I", zlib.crc32(body))
