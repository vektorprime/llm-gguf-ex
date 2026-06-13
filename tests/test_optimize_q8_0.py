from __future__ import annotations

import math
import struct
import tempfile
import unittest
from pathlib import Path

from gguf_explorer.gguf import BinaryWriter, DEFAULT_ALIGNMENT, GGUF_MAGIC, GgufFile, align_to, encode_bf16
from gguf_explorer.optimize_q8_0 import OptimizationSettings, optimize_q8_0_file


def encode_q4_0_block(scale: float, qs: list[int]) -> bytes:
    assert len(qs) == 32
    packed = bytearray(16)
    for j in range(16):
        q0 = max(0, min(15, qs[j] + 8))
        q1 = max(0, min(15, qs[j + 16] + 8))
        packed[j] = q0 | (q1 << 4)
    return struct.pack("<e", scale) + bytes(packed)


class Q8OptimizerTests(unittest.TestCase):
    def test_optimizes_q8_0_tensor_against_bf16_reference(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            target = root / "target.gguf"
            reference = root / "reference.gguf"
            output = root / "target.optimized.gguf"
            name = "blk.0.weight"
            values = [(i - 63.5) / 11.0 for i in range(128)]
            write_one_tensor(reference, name, [128], 30, b"".join(encode_bf16(value) for value in values))
            bad_block = struct.pack("<e", 0.25) + struct.pack("<32b", *([0] * 32))
            write_one_tensor(target, name, [128], 8, bad_block * 4)

            progress = []
            before_sse = sampled_sse(target, reference, name, count=128)
            result = optimize_q8_0_file(
                target,
                reference,
                output,
                settings=OptimizationSettings(passes=8, workers=2, chunk_blocks=1),
                progress_callback=lambda item: progress.append(item.to_json()),
            )
            after_sse = sampled_sse(output, reference, name, count=128)

            self.assertEqual(result.compatible_tensors, 1)
            self.assertEqual(result.total_blocks, 4)
            self.assertEqual(result.processed_blocks, 4)
            self.assertEqual(result.settings.parallelism, "process")
            self.assertEqual(progress[-1]["status"], "complete")
            self.assertGreater(result.changed_blocks, 0)
            self.assertLess(after_sse, before_sse)
            self.assertAlmostEqual(after_sse, result.optimized_sse, places=7)


def sampled_sse(path: Path, reference: Path, name: str, count: int = 32) -> float:
    reader = GgufFile(path)
    ref_reader = GgufFile(reference)
    try:
        sample = reader.sample_tensor(name, start=0, count=count, reference=ref_reader)
        return sum(float(row["diff"]) ** 2 for row in sample["rows"])
    finally:
        reader.close()
        ref_reader.close()


def write_one_tensor(path: Path, name: str, dimensions: list[int], type_id: int, data: bytes) -> None:
    metadata = [
        ("general.architecture", 8, "test"),
        ("general.name", 8, path.stem),
        ("general.alignment", 4, DEFAULT_ALIGNMENT),
    ]
    with path.open("wb") as file_obj:
        writer = BinaryWriter(file_obj)
        writer.write(GGUF_MAGIC)
        writer.u32(3)
        writer.u64(1)
        writer.u64(len(metadata))
        for key, value_type, value in metadata:
            writer.string(key)
            writer.u32(value_type)
            if value_type == 8:
                writer.string(value)
            elif value_type == 4:
                writer.u32(value)
            else:
                raise AssertionError(value_type)
        writer.string(name)
        writer.u32(len(dimensions))
        for dimension in dimensions:
            writer.u64(dimension)
        writer.u32(type_id)
        writer.u64(0)
        current = file_obj.tell()
        writer.write(b"\x00" * (align_to(current, DEFAULT_ALIGNMENT) - current))
        writer.write(data)


class Q4OptimizerTests(unittest.TestCase):
    def test_optimizes_q4_0_tensor_against_bf16_reference(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            target = root / "target.gguf"
            reference = root / "reference.gguf"
            output = root / "target.optimized.gguf"
            name = "blk.0.weight"
            values = [(i - 63.5) / 11.0 for i in range(128)]
            write_one_tensor(reference, name, [128], 30, b"".join(encode_bf16(value) for value in values))
            bad_block = encode_q4_0_block(0.25, [0] * 32)
            write_one_tensor(target, name, [128], 2, bad_block * 4)

            progress = []
            before_sse = sampled_sse(target, reference, name, count=128)
            result = optimize_q8_0_file(
                target,
                reference,
                output,
                settings=OptimizationSettings(passes=8, workers=2, chunk_blocks=1),
                progress_callback=lambda item: progress.append(item.to_json()),
            )
            after_sse = sampled_sse(output, reference, name, count=128)

            self.assertEqual(result.q4_0_tensors, 1)
            self.assertEqual(result.q8_0_tensors, 0)
            self.assertEqual(result.compatible_tensors, 1)
            self.assertEqual(result.total_blocks, 4)
            self.assertEqual(result.processed_blocks, 4)
            self.assertEqual(result.settings.parallelism, "process")
            self.assertEqual(progress[-1]["status"], "complete")
            self.assertGreater(result.changed_blocks, 0)
            self.assertLess(after_sse, before_sse)
            self.assertAlmostEqual(after_sse, result.optimized_sse, places=5)

    def test_q4_0_value_sampling(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path = root / "test.gguf"
            name = "blk.0.weight"
            scale = 0.5
            qs = [i % 16 - 8 for i in range(32)]
            block = encode_q4_0_block(scale, qs)
            write_one_tensor(path, name, [32], 2, block)

            reader = GgufFile(path)
            try:
                sample = reader.sample_tensor(name, start=0, count=32, mode="dequantized")
                for i, row in enumerate(sample["rows"]):
                    expected = scale * (i % 16 - 8)
                    self.assertAlmostEqual(row["decoded"], expected, places=5)
                    self.assertEqual(row["raw"], i % 16 - 8)
            finally:
                reader.close()

    def test_mixed_q8_q4_optimization(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            target = root / "target.gguf"
            reference = root / "reference.gguf"
            output = root / "target.optimized.gguf"
            values_q8 = [(i - 15.5) / 5.0 for i in range(32)]
            values_q4 = [(i - 15.5) / 5.0 for i in range(32)]
            ref_data_q8 = b"".join(encode_bf16(v) for v in values_q8)
            ref_data_q4 = b"".join(encode_bf16(v) for v in values_q4)
            q8_block = struct.pack("<e", 0.1) + struct.pack("<32b", *([0] * 32))
            q4_block = encode_q4_0_block(0.1, [0] * 32)
            write_two_tensors(reference, [
                ("blk.0.attn_q.weight", [32], 30, ref_data_q8),
                ("blk.0.ffn_down.weight", [32], 30, ref_data_q4),
            ])
            write_two_tensors(target, [
                ("blk.0.attn_q.weight", [32], 8, q8_block),
                ("blk.0.ffn_down.weight", [32], 2, q4_block),
            ])

            result = optimize_q8_0_file(target, reference, output, settings=OptimizationSettings(passes=8, workers=1))
            self.assertEqual(result.q8_0_tensors, 1)
            self.assertEqual(result.q4_0_tensors, 1)
            self.assertEqual(result.compatible_tensors, 2)
            self.assertEqual(result.total_blocks, 2)
            self.assertGreater(result.changed_blocks, 0)
            self.assertGreater(result.improvement, 0)


def write_two_tensors(path: Path, tensors: list[tuple[str, list[int], int, bytes]]) -> None:
    metadata = [
        ("general.architecture", 8, "test"),
        ("general.name", 8, path.stem),
        ("general.alignment", 4, DEFAULT_ALIGNMENT),
    ]
    offset = 0
    tensor_specs = []
    for name, dims, type_id, data in tensors:
        tensor_specs.append((name, dims, type_id, offset, data))
        offset = align_to(offset + len(data), DEFAULT_ALIGNMENT)

    with path.open("wb") as file_obj:
        writer = BinaryWriter(file_obj)
        writer.write(GGUF_MAGIC)
        writer.u32(3)
        writer.u64(len(tensor_specs))
        writer.u64(len(metadata))
        for key, value_type, value in metadata:
            writer.string(key)
            writer.u32(value_type)
            if value_type == 8:
                writer.string(value)
            elif value_type == 4:
                writer.u32(value)
            else:
                raise AssertionError(value_type)
        for name, dims, type_id, off, data in tensor_specs:
            writer.string(name)
            writer.u32(len(dims))
            for d in dims:
                writer.u64(d)
            writer.u32(type_id)
            writer.u64(off)
        current = file_obj.tell()
        writer.write(b"\x00" * (align_to(current, DEFAULT_ALIGNMENT) - current))
        data_start = file_obj.tell()
        for name, dims, type_id, off, data in tensor_specs:
            target_off = data_start + off
            cur = file_obj.tell()
            if cur < target_off:
                writer.write(b"\x00" * (target_off - cur))
            writer.write(data)


if __name__ == "__main__":
    unittest.main()
