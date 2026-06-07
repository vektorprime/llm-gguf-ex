from __future__ import annotations

import math
import os
import struct
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any


class GgufError(Exception):
    """Raised when a GGUF file cannot be parsed or sampled."""


GGUF_MAGIC = b"GGUF"
DEFAULT_ALIGNMENT = 32
MAX_METADATA_ARRAY_PREVIEW = 12
MAX_METADATA_STRING_PREVIEW = 320
MAX_VALUE_SAMPLE_COUNT = 1024
SAMPLEABLE_KINDS = {"f32", "f16", "bf16", "f64", "i8", "i16", "i32", "i64", "q8_0"}


GGUF_VALUE_TYPES: dict[int, tuple[str, int | None, str | None]] = {
    0: ("UINT8", 1, "<B"),
    1: ("INT8", 1, "<b"),
    2: ("UINT16", 2, "<H"),
    3: ("INT16", 2, "<h"),
    4: ("UINT32", 4, "<I"),
    5: ("INT32", 4, "<i"),
    6: ("FLOAT32", 4, "<f"),
    7: ("BOOL", 1, "<?"),
    8: ("STRING", None, None),
    9: ("ARRAY", None, None),
    10: ("UINT64", 8, "<Q"),
    11: ("INT64", 8, "<q"),
    12: ("FLOAT64", 8, "<d"),
}


@dataclass(frozen=True)
class GgmlTypeInfo:
    id: int
    name: str
    block_size: int
    type_size: int
    kind: str

    @property
    def is_quantized(self) -> bool:
        return self.block_size > 1


# Sizes match the GGML on-disk type sizes. Value sampling is intentionally
# implemented only for the scalar formats below and Q8_0.
GGML_TYPES: dict[int, GgmlTypeInfo] = {
    0: GgmlTypeInfo(0, "F32", 1, 4, "f32"),
    1: GgmlTypeInfo(1, "F16", 1, 2, "f16"),
    2: GgmlTypeInfo(2, "Q4_0", 32, 18, "unsupported_quant"),
    3: GgmlTypeInfo(3, "Q4_1", 32, 20, "unsupported_quant"),
    6: GgmlTypeInfo(6, "Q5_0", 32, 22, "unsupported_quant"),
    7: GgmlTypeInfo(7, "Q5_1", 32, 24, "unsupported_quant"),
    8: GgmlTypeInfo(8, "Q8_0", 32, 34, "q8_0"),
    9: GgmlTypeInfo(9, "Q8_1", 32, 36, "unsupported_quant"),
    10: GgmlTypeInfo(10, "Q2_K", 256, 84, "unsupported_quant"),
    11: GgmlTypeInfo(11, "Q3_K", 256, 110, "unsupported_quant"),
    12: GgmlTypeInfo(12, "Q4_K", 256, 144, "unsupported_quant"),
    13: GgmlTypeInfo(13, "Q5_K", 256, 176, "unsupported_quant"),
    14: GgmlTypeInfo(14, "Q6_K", 256, 210, "unsupported_quant"),
    15: GgmlTypeInfo(15, "Q8_K", 256, 292, "unsupported_quant"),
    16: GgmlTypeInfo(16, "IQ2_XXS", 256, 66, "unsupported_quant"),
    17: GgmlTypeInfo(17, "IQ2_XS", 256, 74, "unsupported_quant"),
    18: GgmlTypeInfo(18, "IQ3_XXS", 256, 98, "unsupported_quant"),
    19: GgmlTypeInfo(19, "IQ1_S", 256, 50, "unsupported_quant"),
    20: GgmlTypeInfo(20, "IQ4_NL", 32, 18, "unsupported_quant"),
    21: GgmlTypeInfo(21, "IQ3_S", 256, 110, "unsupported_quant"),
    22: GgmlTypeInfo(22, "IQ2_S", 256, 70, "unsupported_quant"),
    23: GgmlTypeInfo(23, "IQ4_XS", 256, 136, "unsupported_quant"),
    24: GgmlTypeInfo(24, "I8", 1, 1, "i8"),
    25: GgmlTypeInfo(25, "I16", 1, 2, "i16"),
    26: GgmlTypeInfo(26, "I32", 1, 4, "i32"),
    27: GgmlTypeInfo(27, "I64", 1, 8, "i64"),
    28: GgmlTypeInfo(28, "F64", 1, 8, "f64"),
    29: GgmlTypeInfo(29, "IQ1_M", 256, 56, "unsupported_quant"),
    30: GgmlTypeInfo(30, "BF16", 1, 2, "bf16"),
    34: GgmlTypeInfo(34, "TQ1_0", 256, 54, "unsupported_quant"),
    35: GgmlTypeInfo(35, "TQ2_0", 256, 66, "unsupported_quant"),
}


@dataclass
class MetadataEntry:
    key: str
    value_type: str
    value: Any


@dataclass
class TensorInfo:
    index: int
    name: str
    dimensions: list[int]
    type_id: int
    offset: int
    absolute_offset: int = 0
    byte_size: int | None = None

    @property
    def type_info(self) -> GgmlTypeInfo | None:
        return GGML_TYPES.get(self.type_id)

    @property
    def type_name(self) -> str:
        info = self.type_info
        return info.name if info else f"UNKNOWN_{self.type_id}"

    @property
    def element_count(self) -> int:
        return math.prod(self.dimensions) if self.dimensions else 1


class BinaryReader:
    def __init__(self, file_obj):
        self.file = file_obj

    @property
    def pos(self) -> int:
        return self.file.tell()

    def read(self, size: int) -> bytes:
        data = self.file.read(size)
        if len(data) != size:
            raise GgufError("Unexpected end of file while reading GGUF data")
        return data

    def seek(self, offset: int, whence: int = os.SEEK_SET) -> None:
        self.file.seek(offset, whence)

    def skip(self, size: int) -> None:
        if size:
            self.file.seek(size, os.SEEK_CUR)

    def unpack(self, fmt: str) -> Any:
        return struct.unpack(fmt, self.read(struct.calcsize(fmt)))[0]

    def u32(self) -> int:
        return self.unpack("<I")

    def u64(self) -> int:
        return self.unpack("<Q")

    def read_string(self) -> str:
        size = self.u64()
        raw = self.read(size)
        return raw.decode("utf-8", errors="replace")

    def read_string_preview(self, max_chars: int = MAX_METADATA_STRING_PREVIEW) -> dict[str, Any]:
        size = self.u64()
        read_size = min(size, max_chars * 4)
        raw = self.read(read_size)
        if size > read_size:
            self.skip(size - read_size)
        text = raw.decode("utf-8", errors="replace")
        if len(text) > max_chars:
            text = text[:max_chars]
        return {
            "text": text,
            "length": size,
            "truncated": size > len(raw) or len(text) >= max_chars,
        }


class GgufFile:
    def __init__(self, path: str | os.PathLike[str]):
        self.path = Path(path).expanduser().resolve()
        if not self.path.exists():
            raise GgufError(f"File does not exist: {self.path}")
        if not self.path.is_file():
            raise GgufError(f"Path is not a file: {self.path}")

        self._file = self.path.open("rb")
        self._lock = threading.Lock()
        self.version = 0
        self.tensor_count = 0
        self.metadata_count = 0
        self.metadata: list[MetadataEntry] = []
        self.metadata_by_key: dict[str, Any] = {}
        self.tensors: list[TensorInfo] = []
        self.tensors_by_name: dict[str, TensorInfo] = {}
        self.alignment = DEFAULT_ALIGNMENT
        self.data_start = 0
        self._tree: dict[str, Any] | None = None
        self._parse()

    def close(self) -> None:
        self._file.close()

    def _parse(self) -> None:
        with self._lock:
            reader = BinaryReader(self._file)
            reader.seek(0)
            magic = reader.read(4)
            if magic != GGUF_MAGIC:
                raise GgufError("Not a GGUF file: missing GGUF magic")

            self.version = reader.u32()
            if self.version not in (2, 3):
                raise GgufError(f"Unsupported GGUF version {self.version}; this reader supports v2/v3")

            self.tensor_count = reader.u64()
            self.metadata_count = reader.u64()
            self.metadata = []
            self.metadata_by_key = {}

            for _ in range(self.metadata_count):
                key = reader.read_string()
                value_type_id = reader.u32()
                type_name = self._metadata_type_name(value_type_id)
                value = self._read_metadata_value(reader, value_type_id)
                self.metadata.append(MetadataEntry(key=key, value_type=type_name, value=value))
                self.metadata_by_key[key] = value

            alignment = self.metadata_by_key.get("general.alignment")
            if isinstance(alignment, int) and alignment > 0:
                self.alignment = alignment

            tensors: list[TensorInfo] = []
            for index in range(self.tensor_count):
                name = reader.read_string()
                dimension_count = reader.u32()
                dimensions = [reader.u64() for _ in range(dimension_count)]
                type_id = reader.u32()
                offset = reader.u64()
                tensor = TensorInfo(
                    index=index,
                    name=name,
                    dimensions=dimensions,
                    type_id=type_id,
                    offset=offset,
                )
                tensor.byte_size = self._tensor_byte_size(tensor)
                tensors.append(tensor)

            self.data_start = align_to(reader.pos, self.alignment)
            for tensor in tensors:
                tensor.absolute_offset = self.data_start + tensor.offset

            self.tensors = tensors
            self.tensors_by_name = {tensor.name: tensor for tensor in tensors}

    def _metadata_type_name(self, value_type_id: int) -> str:
        info = GGUF_VALUE_TYPES.get(value_type_id)
        return info[0] if info else f"UNKNOWN_{value_type_id}"

    def _read_metadata_value(self, reader: BinaryReader, value_type_id: int) -> Any:
        if value_type_id == 8:
            preview = reader.read_string_preview()
            return preview["text"] if not preview["truncated"] else preview

        if value_type_id == 9:
            element_type = reader.u32()
            element_type_name = self._metadata_type_name(element_type)
            length = reader.u64()
            preview = []
            preview_count = min(length, MAX_METADATA_ARRAY_PREVIEW)
            for _ in range(preview_count):
                preview.append(self._read_metadata_value(reader, element_type))
            remaining = length - preview_count
            self._skip_metadata_values(reader, element_type, remaining)
            return {
                "kind": "array",
                "element_type": element_type_name,
                "length": length,
                "preview": preview,
                "truncated": remaining > 0,
            }

        info = GGUF_VALUE_TYPES.get(value_type_id)
        if not info or not info[2]:
            raise GgufError(f"Unsupported GGUF metadata value type {value_type_id}")
        value = reader.unpack(info[2])
        return bool(value) if value_type_id == 7 else value

    def _skip_metadata_values(self, reader: BinaryReader, value_type_id: int, count: int) -> None:
        if count <= 0:
            return
        if value_type_id == 8:
            for _ in range(count):
                size = reader.u64()
                reader.skip(size)
            return
        if value_type_id == 9:
            for _ in range(count):
                element_type = reader.u32()
                length = reader.u64()
                self._skip_metadata_values(reader, element_type, length)
            return
        info = GGUF_VALUE_TYPES.get(value_type_id)
        if not info or info[1] is None:
            raise GgufError(f"Cannot skip GGUF metadata value type {value_type_id}")
        reader.skip(info[1] * count)

    def _tensor_byte_size(self, tensor: TensorInfo) -> int | None:
        type_info = tensor.type_info
        if not type_info:
            return None
        elements = tensor.element_count
        if elements % type_info.block_size != 0:
            return math.ceil(elements / type_info.block_size) * type_info.type_size
        return (elements // type_info.block_size) * type_info.type_size

    def summary(self) -> dict[str, Any]:
        type_counts: dict[str, int] = {}
        type_bytes: dict[str, int] = {}
        for tensor in self.tensors:
            type_counts[tensor.type_name] = type_counts.get(tensor.type_name, 0) + 1
            if tensor.byte_size is not None:
                type_bytes[tensor.type_name] = type_bytes.get(tensor.type_name, 0) + tensor.byte_size

        return {
            "path": str(self.path),
            "name": self.path.name,
            "model_name": metadata_text(self.metadata_by_key.get("general.name")),
            "architecture": metadata_text(self.metadata_by_key.get("general.architecture")),
            "size_bytes": self.path.stat().st_size,
            "version": self.version,
            "tensor_count": self.tensor_count,
            "metadata_count": self.metadata_count,
            "alignment": self.alignment,
            "data_start": self.data_start,
            "type_counts": type_counts,
            "type_bytes": type_bytes,
        }

    def metadata_json(self) -> list[dict[str, Any]]:
        return [
            {
                "key": entry.key,
                "value_type": entry.value_type,
                "value": entry.value,
            }
            for entry in self.metadata
        ]

    def tensor_detail(self, name: str) -> dict[str, Any]:
        tensor = self._get_tensor(name)
        return {
            "index": tensor.index,
            "name": tensor.name,
            "dimensions": tensor.dimensions,
            "element_count": tensor.element_count,
            "type_id": tensor.type_id,
            "type_name": tensor.type_name,
            "type": type_info_json(tensor.type_info),
            "offset": tensor.offset,
            "absolute_offset": tensor.absolute_offset,
            "byte_size": tensor.byte_size,
            "supports_values": tensor.type_info is not None and tensor.type_info.kind in SAMPLEABLE_KINDS,
        }

    def tree(self) -> dict[str, Any]:
        if self._tree is None:
            self._tree = self._build_tree()
        return self._tree

    def _build_tree(self) -> dict[str, Any]:
        root: dict[str, Any] = {
            "kind": "group",
            "name": "",
            "path": "",
            "tensor_count": len(self.tensors),
            "children": {},
        }
        for tensor in self.tensors:
            parts = tensor.name.split(".")
            current = root
            path_parts: list[str] = []
            for part in parts[:-1]:
                path_parts.append(part)
                children = current["children"]
                child = children.get(part)
                if child is None:
                    child = {
                        "kind": "group",
                        "name": part,
                        "path": ".".join(path_parts),
                        "tensor_count": 0,
                        "children": {},
                    }
                    children[part] = child
                child["tensor_count"] += 1
                current = child

            leaf_name = parts[-1]
            current["children"][leaf_name] = {
                "kind": "tensor",
                "name": leaf_name,
                "path": tensor.name,
                "tensor_name": tensor.name,
                "tensor_type": tensor.type_name,
                "dimensions": tensor.dimensions,
                "element_count": tensor.element_count,
                "byte_size": tensor.byte_size,
            }

        return normalize_tree(root)

    def sample_tensor(
        self,
        name: str,
        start: int = 0,
        count: int = 64,
        mode: str = "dequantized",
        reference: GgufFile | None = None,
    ) -> dict[str, Any]:
        tensor = self._get_tensor(name)
        if start < 0:
            raise GgufError("Sample start must be non-negative")
        if count <= 0:
            raise GgufError("Sample count must be positive")
        if count > MAX_VALUE_SAMPLE_COUNT:
            raise GgufError(f"Sample count is capped at {MAX_VALUE_SAMPLE_COUNT}")
        if start >= tensor.element_count:
            raise GgufError("Sample start is outside the tensor")
        count = min(count, tensor.element_count - start)

        type_info = tensor.type_info
        if not type_info:
            raise GgufError(f"Cannot sample unsupported tensor type id {tensor.type_id}")

        if type_info.kind == "q8_0":
            rows = self._sample_q8_0(tensor, start, count, mode)
        elif type_info.kind in {"f32", "f16", "bf16", "f64", "i8", "i16", "i32", "i64"}:
            rows = self._sample_scalar(tensor, start, count, mode)
        else:
            raise GgufError(f"Value sampling for {type_info.name} is not implemented yet")

        payload = {
            "name": tensor.name,
            "mode": mode,
            "start": start,
            "count": len(rows),
            "rows": rows,
            "reference": {"open": False, "compatible": False, "message": "No reference GGUF is loaded."},
        }
        if reference is not None:
            payload["reference"] = self._attach_reference_sample(reference, tensor, rows, start, len(rows))
        return payload

    def _attach_reference_sample(
        self,
        reference: GgufFile,
        tensor: TensorInfo,
        rows: list[dict[str, Any]],
        start: int,
        count: int,
    ) -> dict[str, Any]:
        ref_tensor = reference.tensors_by_name.get(tensor.name)
        if ref_tensor is None:
            return {
                "open": True,
                "compatible": False,
                "message": f"Reference tensor not found: {tensor.name}",
                "file": reference.summary(),
            }
        if ref_tensor.dimensions != tensor.dimensions:
            return {
                "open": True,
                "compatible": False,
                "message": "Reference tensor dimensions do not match.",
                "file": reference.summary(),
                "type_name": ref_tensor.type_name,
                "dimensions": ref_tensor.dimensions,
            }
        if ref_tensor.type_info is None or ref_tensor.type_info.kind not in SAMPLEABLE_KINDS:
            return {
                "open": True,
                "compatible": False,
                "message": f"Reference tensor type {ref_tensor.type_name} cannot be sampled.",
                "file": reference.summary(),
                "type_name": ref_tensor.type_name,
                "dimensions": ref_tensor.dimensions,
            }

        try:
            ref_sample = reference.sample_tensor(tensor.name, start=start, count=count, mode="dequantized")
        except GgufError as exc:
            return {
                "open": True,
                "compatible": False,
                "message": str(exc),
                "file": reference.summary(),
                "type_name": ref_tensor.type_name,
                "dimensions": ref_tensor.dimensions,
            }

        ref_rows = {row["index"]: row for row in ref_sample["rows"]}
        matched = 0
        for row in rows:
            ref_row = ref_rows.get(row["index"])
            if not ref_row:
                continue
            reference_value = ref_row.get("decoded")
            row["reference_raw"] = ref_row.get("raw")
            row["reference_value"] = reference_value
            diff = numeric_diff(row.get("decoded"), reference_value)
            if diff is not None:
                row["diff"] = diff
                matched += 1

        return {
            "open": True,
            "compatible": matched == len(rows),
            "matched": matched,
            "message": f"Compared with {ref_tensor.type_name} reference values.",
            "file": reference.summary(),
            "type_name": ref_tensor.type_name,
            "dimensions": ref_tensor.dimensions,
        }

    def _sample_scalar(self, tensor: TensorInfo, start: int, count: int, mode: str) -> list[dict[str, Any]]:
        type_info = tensor.type_info
        assert type_info is not None
        size = type_info.type_size
        with self._lock:
            self._file.seek(tensor.absolute_offset + start * size)
            raw = self._file.read(count * size)
        if len(raw) != count * size:
            raise GgufError("Could not read the requested tensor value range")

        rows = []
        for i in range(count):
            value_raw = raw[i * size : (i + 1) * size]
            index = start + i
            raw_hex = "0x" + value_raw.hex()
            numeric_value = decode_scalar(value_raw, type_info.kind)
            rows.append(
                {
                    "index": index,
                    "coords": flat_index_to_coords(index, tensor.dimensions),
                    "raw": raw_hex,
                    "value": raw_hex if mode == "static" else numeric_value,
                    "decoded": numeric_value,
                }
            )
        return rows

    def _sample_q8_0(self, tensor: TensorInfo, start: int, count: int, mode: str) -> list[dict[str, Any]]:
        type_info = tensor.type_info
        assert type_info is not None
        block_size = type_info.block_size
        block_bytes = type_info.type_size
        start_block = start // block_size
        end_block = (start + count - 1) // block_size
        block_count = end_block - start_block + 1
        with self._lock:
            self._file.seek(tensor.absolute_offset + start_block * block_bytes)
            raw = self._file.read(block_count * block_bytes)
        if len(raw) != block_count * block_bytes:
            raise GgufError("Could not read the requested Q8_0 block range")

        rows = []
        for index in range(start, start + count):
            block_index = index // block_size
            in_block = index % block_size
            local_block = block_index - start_block
            block = raw[local_block * block_bytes : (local_block + 1) * block_bytes]
            scale = struct.unpack("<e", block[:2])[0]
            quantized = struct.unpack("<b", block[2 + in_block : 3 + in_block])[0]
            dequantized = scale * quantized
            rows.append(
                {
                    "index": index,
                    "coords": flat_index_to_coords(index, tensor.dimensions),
                    "block": block_index,
                    "in_block": in_block,
                    "raw": quantized,
                    "scale": scale,
                    "value": quantized if mode == "static" else dequantized,
                    "decoded": dequantized,
                }
            )
        return rows

    def _get_tensor(self, name: str) -> TensorInfo:
        tensor = self.tensors_by_name.get(name)
        if tensor is None:
            raise GgufError(f"Tensor not found: {name}")
        return tensor


def align_to(value: int, alignment: int) -> int:
    remainder = value % alignment
    return value if remainder == 0 else value + alignment - remainder


def type_info_json(type_info: GgmlTypeInfo | None) -> dict[str, Any] | None:
    if type_info is None:
        return None
    return {
        "id": type_info.id,
        "name": type_info.name,
        "block_size": type_info.block_size,
        "type_size": type_info.type_size,
        "kind": type_info.kind,
        "is_quantized": type_info.is_quantized,
    }


def normalize_tree(node: dict[str, Any]) -> dict[str, Any]:
    children = node.get("children", {})
    if isinstance(children, dict):
        normalized_children = [normalize_tree(child) for child in children.values()]
        normalized_children.sort(key=tree_sort_key)
        node["children"] = normalized_children
    return node


def tree_sort_key(node: dict[str, Any]) -> tuple[int, tuple[Any, ...]]:
    kind_rank = 0 if node.get("kind") == "group" else 1
    name = str(node.get("name", ""))
    if name.isdigit():
        name_key: tuple[Any, ...] = (0, int(name))
    else:
        name_key = (1, name.lower())
    return kind_rank, name_key


def decode_scalar(raw: bytes, kind: str) -> int | float:
    if kind == "f32":
        return struct.unpack("<f", raw)[0]
    if kind == "f16":
        return struct.unpack("<e", raw)[0]
    if kind == "bf16":
        bits = struct.unpack("<H", raw)[0] << 16
        return struct.unpack("<f", struct.pack("<I", bits))[0]
    if kind == "f64":
        return struct.unpack("<d", raw)[0]
    if kind == "i8":
        return struct.unpack("<b", raw)[0]
    if kind == "i16":
        return struct.unpack("<h", raw)[0]
    if kind == "i32":
        return struct.unpack("<i", raw)[0]
    if kind == "i64":
        return struct.unpack("<q", raw)[0]
    raise GgufError(f"Cannot decode scalar kind {kind}")


def encode_bf16(value: float) -> bytes:
    bits = struct.unpack("<I", struct.pack("<f", float(value)))[0]
    return struct.pack("<H", bits >> 16)


def flat_index_to_coords(index: int, dimensions: list[int]) -> list[int]:
    coords = []
    remaining = index
    for dimension in dimensions:
        if dimension <= 0:
            coords.append(0)
        else:
            coords.append(remaining % dimension)
            remaining //= dimension
    return coords


def numeric_diff(left: Any, right: Any) -> float | None:
    try:
        left_value = float(left)
        right_value = float(right)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(left_value) or not math.isfinite(right_value):
        return None
    return left_value - right_value


def metadata_text(value: Any) -> str | None:
    if isinstance(value, str):
        return value
    if isinstance(value, dict) and isinstance(value.get("text"), str):
        return value["text"]
    return None


def write_sample_gguf(path: str | os.PathLike[str]) -> Path:
    """Write a tiny v3 GGUF file with BF16 and Q8_0 tensors for local testing."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)

    metadata = [
        ("general.architecture", 8, "sample"),
        ("general.name", 8, "Tiny BF16/Q8_0 sample"),
        ("general.alignment", 4, 32),
        ("sample.block_count", 4, 1),
    ]

    bf16_values = [0.0, 1.0, -2.5, 3.25, 4.5, -8.0, 0.125, 16.0]
    bf16_data = b"".join(encode_bf16(value) for value in bf16_values)
    q8_scale = struct.pack("<e", 0.25)
    q8_values = list(range(-16, 16))
    q8_data = q8_scale + struct.pack("<32b", *q8_values)

    tensors = [
        {
            "name": "blk.0.attn_q.weight",
            "dimensions": [4, 2],
            "type_id": 30,
            "data": bf16_data,
        },
        {
            "name": "blk.0.ffn_down.weight",
            "dimensions": [32],
            "type_id": 8,
            "data": q8_data,
        },
    ]

    offset = 0
    for tensor in tensors:
        tensor["offset"] = offset
        offset = align_to(offset + len(tensor["data"]), DEFAULT_ALIGNMENT)

    with out.open("wb") as file_obj:
        writer = BinaryWriter(file_obj)
        writer.write(GGUF_MAGIC)
        writer.u32(3)
        writer.u64(len(tensors))
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
        for tensor in tensors:
            writer.string(tensor["name"])
            writer.u32(len(tensor["dimensions"]))
            for dimension in tensor["dimensions"]:
                writer.u64(dimension)
            writer.u32(tensor["type_id"])
            writer.u64(tensor["offset"])

        current = file_obj.tell()
        padding = align_to(current, DEFAULT_ALIGNMENT) - current
        writer.write(b"\x00" * padding)
        data_start = file_obj.tell()
        for tensor in tensors:
            target = data_start + tensor["offset"]
            current = file_obj.tell()
            if current < target:
                writer.write(b"\x00" * (target - current))
            writer.write(tensor["data"])

    return out


class BinaryWriter:
    def __init__(self, file_obj):
        self.file = file_obj

    def write(self, data: bytes) -> None:
        self.file.write(data)

    def u32(self, value: int) -> None:
        self.write(struct.pack("<I", value))

    def u64(self, value: int) -> None:
        self.write(struct.pack("<Q", value))

    def string(self, value: str) -> None:
        raw = value.encode("utf-8")
        self.u64(len(raw))
        self.write(raw)
