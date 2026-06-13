from __future__ import annotations

import math
import multiprocessing as mp
import os
import shutil
import struct
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Literal

from .gguf import GgufError, GgufFile, TensorInfo


Q8_0_BLOCK_SIZE = 32
Q8_0_TYPE_SIZE = 34
Q4_0_BLOCK_SIZE = 32
Q4_0_TYPE_SIZE = 18
DEFAULT_PASSES = 8
DEFAULT_CHUNK_BLOCKS = 8192
FP16_MAX = 65504.0
SUPPORTED_REFERENCE_KINDS = {"f32", "f16", "bf16", "f64", "i8", "i16", "i32", "i64", "q8_0", "q4_0"}
Parallelism = Literal["process", "thread", "none"]
ProgressCallback = Callable[["OptimizationProgress"], None]


@dataclass(frozen=True)
class OptimizationSettings:
    passes: int = DEFAULT_PASSES
    qmin: int = -128
    qmax: int = 127
    workers: int | None = None
    chunk_blocks: int = DEFAULT_CHUNK_BLOCKS
    preserve_when_worse: bool = True
    parallelism: Parallelism = "process"

    def normalized(self) -> "OptimizationSettings":
        passes = max(1, min(int(self.passes), 64))
        qmin = max(-128, min(int(self.qmin), 127))
        qmax = max(qmin, min(int(self.qmax), 127))
        workers = None if self.workers is None or self.workers <= 0 else int(self.workers)
        chunk_blocks = max(1, min(int(self.chunk_blocks), 65536))
        parallelism = self.parallelism if self.parallelism in {"process", "thread", "none"} else "process"
        return OptimizationSettings(
            passes=passes,
            qmin=qmin,
            qmax=qmax,
            workers=workers,
            chunk_blocks=chunk_blocks,
            preserve_when_worse=bool(self.preserve_when_worse),
            parallelism=parallelism,
        )


@dataclass
class TensorOptimizationPlan:
    target: TensorInfo
    reference: TensorInfo
    block_count: int
    element_count: int
    kind: str = "q8_0"


@dataclass
class ChunkResult:
    tensor_name: str
    target_offset: int = 0
    blocks: int = 0
    changed_blocks: int = 0
    values: int = 0
    previous_sse: float = 0.0
    optimized_sse: float = 0.0
    bytes_written: int = 0
    raw_blocks: bytes = field(default=b"", repr=False)


@dataclass
class OptimizationResult:
    source_path: str
    reference_path: str
    output_path: str
    settings: OptimizationSettings
    q8_0_tensors: int = 0
    q4_0_tensors: int = 0
    compatible_tensors: int = 0
    skipped_tensors: list[dict[str, str]] = field(default_factory=list)
    total_blocks: int = 0
    processed_blocks: int = 0
    changed_blocks: int = 0
    total_values: int = 0
    previous_sse: float = 0.0
    optimized_sse: float = 0.0
    bytes_written: int = 0

    @property
    def improvement(self) -> float:
        return self.previous_sse - self.optimized_sse

    @property
    def improvement_percent(self) -> float | None:
        if self.previous_sse <= 0:
            return None
        return (self.improvement / self.previous_sse) * 100.0

    @property
    def progress_percent(self) -> float:
        if self.total_blocks <= 0:
            return 0.0
        return min(100.0, (self.processed_blocks / self.total_blocks) * 100.0)

    def to_json(self) -> dict[str, Any]:
        improvement_percent = self.improvement_percent
        return {
            "source_path": self.source_path,
            "reference_path": self.reference_path,
            "output_path": self.output_path,
            "passes": self.settings.passes,
            "qmin": self.settings.qmin,
            "qmax": self.settings.qmax,
            "workers": self.settings.workers,
            "chunk_blocks": self.settings.chunk_blocks,
            "parallelism": self.settings.parallelism,
            "q8_0_tensors": self.q8_0_tensors,
            "q4_0_tensors": self.q4_0_tensors,
            "compatible_tensors": self.compatible_tensors,
            "skipped_tensors": self.skipped_tensors,
            "total_blocks": self.total_blocks,
            "processed_blocks": self.processed_blocks,
            "progress_percent": self.progress_percent,
            "changed_blocks": self.changed_blocks,
            "total_values": self.total_values,
            "previous_sse": self.previous_sse,
            "optimized_sse": self.optimized_sse,
            "improvement": self.improvement,
            "improvement_percent": improvement_percent,
            "bytes_written": self.bytes_written,
        }


@dataclass(frozen=True)
class OptimizationProgress:
    status: str
    message: str
    total_blocks: int
    processed_blocks: int
    changed_blocks: int
    previous_sse: float
    optimized_sse: float
    workers: int
    parallelism: str
    passes: int
    chunk_blocks: int
    output_path: str
    current_tensor: str | None = None

    @property
    def progress_percent(self) -> float:
        if self.total_blocks <= 0:
            return 0.0
        return min(100.0, (self.processed_blocks / self.total_blocks) * 100.0)

    @property
    def improvement(self) -> float:
        return self.previous_sse - self.optimized_sse

    def to_json(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "message": self.message,
            "total_blocks": self.total_blocks,
            "processed_blocks": self.processed_blocks,
            "progress_percent": self.progress_percent,
            "changed_blocks": self.changed_blocks,
            "previous_sse": self.previous_sse,
            "optimized_sse": self.optimized_sse,
            "improvement": self.improvement,
            "workers": self.workers,
            "parallelism": self.parallelism,
            "passes": self.passes,
            "chunk_blocks": self.chunk_blocks,
            "output_path": self.output_path,
            "current_tensor": self.current_tensor,
        }


def optimize_q8_0_file(
    target_path: str | os.PathLike[str],
    reference_path: str | os.PathLike[str],
    output_path: str | os.PathLike[str] | None = None,
    *,
    settings: OptimizationSettings | None = None,
    tensor_names: Iterable[str] | None = None,
    progress_callback: ProgressCallback | None = None,
) -> OptimizationResult:
    """Copy a Q8_0 GGUF and rewrite matching Q8_0 blocks against a reference GGUF.

    The output file keeps the original GGUF metadata and tensor layout. Only the
    Q8_0 block payloads for compatible tensors are rewritten. Each block is
    optimized with alternating least-squares passes:

    * for a fixed scale, choose nearest clamped int8 values;
    * for fixed int8 values, choose the least-squares scale;
    * round the scale to the float16 value actually stored by Q8_0.

    CPU-heavy chunks run in a process pool by default. The previous thread-pool
    implementation did not effectively use multiple cores because this hot loop
    is pure Python and is therefore limited by the GIL.
    """

    normalized = (settings or OptimizationSettings()).normalized()
    source = Path(target_path).expanduser().resolve()
    reference = Path(reference_path).expanduser().resolve()
    if not source.exists() or not source.is_file():
        raise GgufError(f"Target GGUF does not exist: {source}")
    if not reference.exists() or not reference.is_file():
        raise GgufError(f"Reference GGUF does not exist: {reference}")

    output = _default_output_path(source) if output_path is None else Path(output_path).expanduser().resolve()
    if source == output:
        raise GgufError("Refusing to overwrite the open source GGUF. Choose a separate output path.")
    if reference == output:
        raise GgufError("Refusing to overwrite the reference GGUF. Choose a separate output path.")
    if output.exists():
        raise GgufError(f"Output GGUF already exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)

    target_reader = GgufFile(source)
    reference_reader = GgufFile(reference)
    try:
        plans, skipped, q8_count, q4_count = _build_plans(target_reader, reference_reader, tensor_names)
        if not plans:
            details = "; ".join(f"{item['name']}: {item['reason']}" for item in skipped[:5])
            suffix = f" ({details})" if details else ""
            raise GgufError(f"No compatible quantized tensors were found to optimize{suffix}")

        tasks = list(_chunk_tasks(plans, normalized.chunk_blocks))
        worker_count = _effective_worker_count(normalized, len(tasks))
        parallelism = "none" if worker_count == 1 else normalized.parallelism
        effective_settings = OptimizationSettings(
            passes=normalized.passes,
            qmin=normalized.qmin,
            qmax=normalized.qmax,
            workers=worker_count,
            chunk_blocks=normalized.chunk_blocks,
            preserve_when_worse=normalized.preserve_when_worse,
            parallelism=parallelism,
        )

        result = OptimizationResult(
            source_path=str(source),
            reference_path=str(reference),
            output_path=str(output),
            settings=effective_settings,
            q8_0_tensors=q8_count,
            q4_0_tensors=q4_count,
            compatible_tensors=len(plans),
            skipped_tensors=skipped,
            total_blocks=sum(plan.block_count for plan in plans),
            total_values=sum(plan.element_count for plan in plans),
        )
        _emit_progress(progress_callback, result, "preparing", "Copying source GGUF before optimization")

        # Work on a copy so failed optimization never corrupts the loaded source.
        shutil.copyfile(source, output)
        _emit_progress(progress_callback, result, "running", "Optimizing quantized tensor blocks")

        with output.open("r+b") as output_file:
            if worker_count == 1:
                for task in tasks:
                    chunk = _optimize_chunk(str(output), str(reference), *task, effective_settings)
                    _write_chunk_result(output_file, chunk, result)
                    _emit_progress(progress_callback, result, "running", "Optimizing quantized tensor blocks", chunk.tensor_name)
            else:
                executor_cls = ProcessPoolExecutor if effective_settings.parallelism == "process" else ThreadPoolExecutor
                with _make_executor(executor_cls, worker_count) as executor:
                    futures = [
                        executor.submit(
                            _optimize_chunk,
                            str(output),
                            str(reference),
                            plan,
                            start_block,
                            block_count,
                            effective_settings,
                        )
                        for plan, start_block, block_count in tasks
                    ]
                    for future in as_completed(futures):
                        chunk = future.result()
                        _write_chunk_result(output_file, chunk, result)
                        _emit_progress(progress_callback, result, "running", "Optimizing quantized tensor blocks", chunk.tensor_name)

        _emit_progress(progress_callback, result, "complete", "Optimization complete")
        return result
    finally:
        target_reader.close()
        reference_reader.close()



def _make_executor(executor_cls, worker_count: int):
    if executor_cls is ProcessPoolExecutor:
        return ProcessPoolExecutor(max_workers=worker_count, mp_context=_process_context())
    return executor_cls(max_workers=worker_count)


def _process_context():
    # The server starts optimization from a worker thread. Avoid raw fork from a
    # multithreaded process because it can deadlock extension/module locks.
    for method in ("forkserver", "spawn"):
        try:
            return mp.get_context(method)
        except ValueError:
            continue
    return mp.get_context()

def _effective_worker_count(settings: OptimizationSettings, task_count: int) -> int:
    if task_count <= 0 or settings.parallelism == "none":
        return 1
    requested = settings.workers or (os.cpu_count() or 1)
    return max(1, min(int(requested), task_count))


def _emit_progress(
    callback: ProgressCallback | None,
    result: OptimizationResult,
    status: str,
    message: str,
    current_tensor: str | None = None,
) -> None:
    if callback is None:
        return
    callback(
        OptimizationProgress(
            status=status,
            message=message,
            total_blocks=result.total_blocks,
            processed_blocks=result.processed_blocks,
            changed_blocks=result.changed_blocks,
            previous_sse=result.previous_sse,
            optimized_sse=result.optimized_sse,
            workers=result.settings.workers or 1,
            parallelism=result.settings.parallelism,
            passes=result.settings.passes,
            chunk_blocks=result.settings.chunk_blocks,
            output_path=result.output_path,
            current_tensor=current_tensor,
        )
    )


def _write_chunk_result(output_file, chunk: ChunkResult, result: OptimizationResult) -> None:
    if chunk.raw_blocks:
        output_file.seek(chunk.target_offset)
        output_file.write(chunk.raw_blocks)
    result.processed_blocks += chunk.blocks
    result.changed_blocks += chunk.changed_blocks
    result.previous_sse += chunk.previous_sse
    result.optimized_sse += chunk.optimized_sse
    result.bytes_written += chunk.bytes_written


def _default_output_path(source: Path) -> Path:
    candidate = source.with_name(f"{source.stem}.optimized{source.suffix}")
    if not candidate.exists():
        return candidate
    for index in range(2, 10000):
        candidate = source.with_name(f"{source.stem}.optimized-{index}{source.suffix}")
        if not candidate.exists():
            return candidate
    raise GgufError("Could not choose a unique optimized GGUF output path")


def _build_plans(
    target_reader: GgufFile,
    reference_reader: GgufFile,
    tensor_names: Iterable[str] | None,
) -> tuple[list[TensorOptimizationPlan], list[dict[str, str]], int, int]:
    requested = set(tensor_names or [])
    plans: list[TensorOptimizationPlan] = []
    skipped: list[dict[str, str]] = []
    q8_count = 0
    q4_count = 0
    supported_target_kinds = {"q8_0", "q4_0"}
    for tensor in target_reader.tensors:
        type_info = tensor.type_info
        if type_info is None or type_info.kind not in supported_target_kinds:
            continue
        if type_info.kind == "q8_0":
            q8_count += 1
        elif type_info.kind == "q4_0":
            q4_count += 1
        if requested and tensor.name not in requested:
            continue
        reference_tensor = reference_reader.tensors_by_name.get(tensor.name)
        if reference_tensor is None:
            skipped.append({"name": tensor.name, "reason": "reference tensor not found"})
            continue
        if reference_tensor.dimensions != tensor.dimensions:
            skipped.append({"name": tensor.name, "reason": "reference dimensions do not match"})
            continue
        reference_type = reference_tensor.type_info
        if reference_type is None or reference_type.kind not in SUPPORTED_REFERENCE_KINDS:
            skipped.append({"name": tensor.name, "reason": f"reference type {reference_tensor.type_name} is not supported"})
            continue
        block_size = type_info.block_size
        block_count = math.ceil(tensor.element_count / block_size)
        plans.append(
            TensorOptimizationPlan(
                target=tensor,
                reference=reference_tensor,
                block_count=block_count,
                element_count=tensor.element_count,
                kind=type_info.kind,
            )
        )
    return plans, skipped, q8_count, q4_count


def _chunk_tasks(
    plans: list[TensorOptimizationPlan],
    chunk_blocks: int,
) -> Iterable[tuple[TensorOptimizationPlan, int, int]]:
    for plan in plans:
        start = 0
        while start < plan.block_count:
            count = min(chunk_blocks, plan.block_count - start)
            yield plan, start, count
            start += count


def _optimize_chunk(
    output_path: str,
    reference_path: str,
    plan: TensorOptimizationPlan,
    start_block: int,
    block_count: int,
    settings: OptimizationSettings,
) -> ChunkResult:
    target = plan.target
    kind = plan.kind
    if kind == "q4_0":
        block_size = Q4_0_BLOCK_SIZE
        type_size = Q4_0_TYPE_SIZE
    else:
        block_size = Q8_0_BLOCK_SIZE
        type_size = Q8_0_TYPE_SIZE
    type_name = "Q4_0" if kind == "q4_0" else "Q8_0"
    first_element = start_block * block_size
    requested_values = min(block_count * block_size, target.element_count - first_element)
    if requested_values <= 0:
        return ChunkResult(tensor_name=target.name)

    target_offset = target.absolute_offset + start_block * type_size
    raw_size = block_count * type_size
    with open(output_path, "rb") as output_file, open(reference_path, "rb") as reference_file:
        output_file.seek(target_offset)
        raw_blocks = bytearray(output_file.read(raw_size))
        if len(raw_blocks) != raw_size:
            raise GgufError(f"Could not read {type_name} blocks for tensor {target.name}")

        reference_values = _read_reference_values(reference_file, plan.reference, first_element, requested_values)
        if len(reference_values) != requested_values:
            raise GgufError(f"Could not read reference values for tensor {target.name}")

    changed = 0
    previous_sse = 0.0
    optimized_sse = 0.0
    for local_block in range(block_count):
        block_start_value = local_block * block_size
        active = min(block_size, requested_values - block_start_value)
        if active <= 0:
            break
        block_offset = local_block * type_size
        block = raw_blocks[block_offset : block_offset + type_size]
        values = reference_values[block_start_value : block_start_value + active]
        optimized = _optimize_block_bytes(bytes(block), values, settings, kind)
        previous_sse += optimized.previous_sse
        optimized_sse += optimized.optimized_sse
        if optimized.block != block:
            raw_blocks[block_offset : block_offset + type_size] = optimized.block
            changed += 1

    return ChunkResult(
        tensor_name=target.name,
        target_offset=target_offset,
        blocks=block_count,
        changed_blocks=changed,
        values=requested_values,
        previous_sse=previous_sse,
        optimized_sse=optimized_sse,
        bytes_written=len(raw_blocks),
        raw_blocks=bytes(raw_blocks),
    )


def _read_reference_values(file_obj, tensor: TensorInfo, start: int, count: int) -> list[float]:
    type_info = tensor.type_info
    if type_info is None:
        raise GgufError(f"Unsupported reference tensor type for {tensor.name}")
    if type_info.kind == "q8_0":
        return _read_q8_0_values(file_obj, tensor, start, count)
    if type_info.kind == "q4_0":
        return _read_q4_0_values(file_obj, tensor, start, count)
    if type_info.kind not in SUPPORTED_REFERENCE_KINDS:
        raise GgufError(f"Unsupported reference tensor type {type_info.name} for {tensor.name}")
    size = type_info.type_size
    file_obj.seek(tensor.absolute_offset + start * size)
    raw = file_obj.read(count * size)
    if len(raw) != count * size:
        raise GgufError(f"Could not read scalar reference values for {tensor.name}")
    return _decode_scalar_values(raw, type_info.kind)


def _read_q8_0_values(file_obj, tensor: TensorInfo, start: int, count: int) -> list[float]:
    start_block = start // Q8_0_BLOCK_SIZE
    end_block = (start + count - 1) // Q8_0_BLOCK_SIZE
    block_count = end_block - start_block + 1
    file_obj.seek(tensor.absolute_offset + start_block * Q8_0_TYPE_SIZE)
    raw = file_obj.read(block_count * Q8_0_TYPE_SIZE)
    if len(raw) != block_count * Q8_0_TYPE_SIZE:
        raise GgufError(f"Could not read Q8_0 reference values for {tensor.name}")

    values: list[float] = []
    for index in range(start, start + count):
        block_index = index // Q8_0_BLOCK_SIZE
        in_block = index % Q8_0_BLOCK_SIZE
        local_block = block_index - start_block
        block_offset = local_block * Q8_0_TYPE_SIZE
        scale = struct.unpack("<e", raw[block_offset : block_offset + 2])[0]
        quantized = struct.unpack("<b", raw[block_offset + 2 + in_block : block_offset + 3 + in_block])[0]
        values.append(float(scale) * float(quantized))
    return values


def _read_q4_0_values(file_obj, tensor: TensorInfo, start: int, count: int) -> list[float]:
    start_block = start // Q4_0_BLOCK_SIZE
    end_block = (start + count - 1) // Q4_0_BLOCK_SIZE
    block_count = end_block - start_block + 1
    file_obj.seek(tensor.absolute_offset + start_block * Q4_0_TYPE_SIZE)
    raw = file_obj.read(block_count * Q4_0_TYPE_SIZE)
    if len(raw) != block_count * Q4_0_TYPE_SIZE:
        raise GgufError(f"Could not read Q4_0 reference values for {tensor.name}")

    values: list[float] = []
    for index in range(start, start + count):
        block_index = index // Q4_0_BLOCK_SIZE
        in_block = index % Q4_0_BLOCK_SIZE
        local_block = block_index - start_block
        block_offset = local_block * Q4_0_TYPE_SIZE
        scale = struct.unpack("<e", raw[block_offset : block_offset + 2])[0]
        qs = raw[block_offset + 2:]
        if in_block < Q4_0_BLOCK_SIZE // 2:
            quantized = (qs[in_block] & 0x0F) - 8
        else:
            quantized = (qs[in_block - Q4_0_BLOCK_SIZE // 2] >> 4) - 8
        values.append(float(scale) * float(quantized))
    return values


def _decode_scalar_values(raw: bytes, kind: str) -> list[float]:
    if kind == "f32":
        return [float(item[0]) for item in struct.iter_unpack("<f", raw)]
    if kind == "f16":
        return [float(item[0]) for item in struct.iter_unpack("<e", raw)]
    if kind == "bf16":
        return [_bf16_to_float(item[0]) for item in struct.iter_unpack("<H", raw)]
    if kind == "f64":
        return [float(item[0]) for item in struct.iter_unpack("<d", raw)]
    if kind == "i8":
        return [float(item[0]) for item in struct.iter_unpack("<b", raw)]
    if kind == "i16":
        return [float(item[0]) for item in struct.iter_unpack("<h", raw)]
    if kind == "i32":
        return [float(item[0]) for item in struct.iter_unpack("<i", raw)]
    if kind == "i64":
        return [float(item[0]) for item in struct.iter_unpack("<q", raw)]
    raise GgufError(f"Cannot decode reference kind {kind}")


def _bf16_to_float(bits16: int) -> float:
    bits = int(bits16) << 16
    return struct.unpack("<f", struct.pack("<I", bits))[0]


@dataclass(frozen=True)
class BlockOptimization:
    block: bytes
    previous_sse: float
    optimized_sse: float


def _optimize_block_bytes(block: bytes, values: list[float], settings: OptimizationSettings, kind: str = "q8_0") -> BlockOptimization:
    active = len(values)
    existing_scale = float(struct.unpack("<e", block[:2])[0])

    if kind == "q4_0":
        if len(block) != Q4_0_TYPE_SIZE:
            raise GgufError("Invalid Q4_0 block size")
        existing_qs = _decode_q4_0_qs(block[2:], Q4_0_BLOCK_SIZE)
        qmin = -8
        qmax = 7
    else:
        if len(block) != Q8_0_TYPE_SIZE:
            raise GgufError("Invalid Q8_0 block size")
        existing_qs = list(struct.unpack("<32b", block[2:]))
        qmin = settings.qmin
        qmax = settings.qmax

    previous_sse = _sse(values, existing_scale, existing_qs)

    if active == 0 or not all(math.isfinite(value) for value in values):
        return BlockOptimization(block=block, previous_sse=previous_sse, optimized_sse=previous_sse)

    max_abs = max(abs(value) for value in values)
    if max_abs == 0.0:
        new_qs = existing_qs[:]
        for index in range(active):
            new_qs[index] = 0
        candidate = _encode_block(0.0, new_qs, kind)
        return BlockOptimization(block=candidate, previous_sse=previous_sse, optimized_sse=0.0)

    effective_settings = OptimizationSettings(
        passes=settings.passes,
        qmin=qmin,
        qmax=qmax,
        workers=settings.workers,
        chunk_blocks=settings.chunk_blocks,
        preserve_when_worse=settings.preserve_when_worse,
        parallelism=settings.parallelism,
    )

    candidates: list[tuple[float, list[int], float]] = [(abs(existing_scale), existing_qs[:active], previous_sse)]
    scale_seed = abs(existing_scale) if math.isfinite(existing_scale) else 0.0
    if scale_seed > 0:
        candidates.append(_optimize_from_scale(values, scale_seed, effective_settings))
    candidates.append(_optimize_from_scale(values, max_abs / max(1, qmax), effective_settings))

    # A seed based on the current integer pattern can improve files already close
    # to convergence but with a suboptimal stored half-precision scale.
    current_qs = existing_qs[:active]
    denom = sum(q * q for q in current_qs)
    if denom:
        scale = _to_storable_f16(sum(q * w for q, w in zip(current_qs, values)) / denom)
        if scale > 0:
            candidates.append(_optimize_from_scale(values, scale, effective_settings))

    best_scale, best_qs, best_sse = min(candidates, key=lambda item: item[2])
    if settings.preserve_when_worse:
        tolerance = max(1e-18, abs(previous_sse) * 1e-12)
        if best_sse >= previous_sse - tolerance:
            return BlockOptimization(block=block, previous_sse=previous_sse, optimized_sse=previous_sse)

    new_qs = existing_qs[:]
    for index, value in enumerate(best_qs):
        new_qs[index] = value
    new_block = _encode_block(best_scale, new_qs, kind)
    return BlockOptimization(block=new_block, previous_sse=previous_sse, optimized_sse=best_sse)


def _optimize_from_scale(values: list[float], initial_scale: float, settings: OptimizationSettings) -> tuple[float, list[int], float]:
    scale = _to_storable_f16(initial_scale)
    if scale <= 0.0 or not math.isfinite(scale):
        return 0.0, [0 for _ in values], _sse(values, 0.0, [0 for _ in values])

    previous_qs: list[int] | None = None
    qs: list[int] = []
    for _ in range(settings.passes):
        qs = [_clamp_int(_round_nearest(value / scale), settings.qmin, settings.qmax) for value in values]
        denom = sum(q * q for q in qs)
        if denom == 0:
            scale = 0.0
            break
        numerator = sum(q * value for q, value in zip(qs, values))
        next_scale = _to_storable_f16(numerator / denom)
        if next_scale <= 0.0 or not math.isfinite(next_scale):
            break
        if previous_qs == qs and next_scale == scale:
            scale = next_scale
            break
        previous_qs = qs
        scale = next_scale

    if scale <= 0.0 or not qs:
        qs = [0 for _ in values]
    return scale, qs, _sse(values, scale, qs)


def _sse(values: list[float], scale: float, qs: list[int]) -> float:
    total = 0.0
    for value, quantized in zip(values, qs):
        error = float(scale) * float(quantized) - float(value)
        total += error * error
    return total


def _encode_q8_0_block(scale: float, qs: list[int]) -> bytes:
    if len(qs) != Q8_0_BLOCK_SIZE:
        raise GgufError("Q8_0 blocks must contain 32 quantized values")
    return struct.pack("<e", _to_storable_f16(scale)) + struct.pack("<32b", *qs)


def _encode_block(scale: float, qs: list[int], kind: str) -> bytes:
    if kind == "q4_0":
        return _encode_q4_0_block(scale, qs, Q4_0_BLOCK_SIZE)
    return _encode_q8_0_block(scale, qs)


def _decode_q4_0_qs(qs_raw: bytes, block_size: int) -> list[int]:
    result: list[int] = []
    half = block_size // 2
    for j in range(half):
        result.append((qs_raw[j] & 0x0F) - 8)
    for j in range(half):
        result.append((qs_raw[j] >> 4) - 8)
    return result


def _encode_q4_0_block(scale: float, qs: list[int], block_size: int = Q4_0_BLOCK_SIZE) -> bytes:
    if len(qs) != block_size:
        raise GgufError(f"Q4_0 blocks must contain {block_size} quantized values")
    half = block_size // 2
    packed = bytearray(half)
    for j in range(half):
        q0 = _clamp_int(qs[j] + 8, 0, 15)
        q1 = _clamp_int(qs[j + half] + 8, 0, 15)
        packed[j] = q0 | (q1 << 4)
    return struct.pack("<e", _to_storable_f16(scale)) + bytes(packed)


def _to_storable_f16(value: float) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return 0.0
    if not math.isfinite(numeric) or numeric <= 0.0:
        return 0.0
    numeric = min(numeric, FP16_MAX)
    try:
        return float(struct.unpack("<e", struct.pack("<e", numeric))[0])
    except OverflowError:
        return FP16_MAX


def _round_nearest(value: float) -> int:
    if value >= 0:
        return int(math.floor(value + 0.5))
    return int(math.ceil(value - 0.5))


def _clamp_int(value: int, low: int, high: int) -> int:
    if value < low:
        return low
    if value > high:
        return high
    return value
