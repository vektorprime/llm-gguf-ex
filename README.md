# GGUF Explorer

A local web GUI for inspecting GGUF model files without loading the whole model into memory.

## Run

```powershell
python server.py --port 8765
python server.py --port 8765 --open .\Qwen3.5-2B-Q8_0.gguf --reference .\Qwen3.5-2B-BF16.gguf
```

Open `http://127.0.0.1:8765`. The app scans the working folder for `.gguf` files, shows them in the top model list, and lets you load one as the main model or as the comparison reference. You can also paste a GGUF file path or a folder path in the top bar and scan/load from there.

The layer browser can be resized by dragging the vertical divider. Value table column headers show explanations on hover, and the columns can be rearranged by dragging headers left or right.
Quantized value tables include `Reference` and `Diff` columns. Load a BF16/native GGUF in the sidebar reference slot, then browse the Q8_0 GGUF as the main file; `Diff` shows `Q8_0 final - reference decoded value`, with a stronger reddish tint as the absolute difference grows.

## Qwen3.5-2B GGUF files

The first target repo is `unsloth/Qwen3.5-2B-GGUF` on Hugging Face. The relevant files are:

- `Qwen3.5-2B-BF16.gguf`
- `Qwen3.5-2B-Q8_0.gguf`

One download route, if `huggingface-cli` is available:

```powershell
huggingface-cli download unsloth/Qwen3.5-2B-GGUF Qwen3.5-2B-BF16.gguf --local-dir models
huggingface-cli download unsloth/Qwen3.5-2B-GGUF Qwen3.5-2B-Q8_0.gguf --local-dir models
```

Then load the full local path in the GUI.

## Current support

- GGUF v2/v3 headers, metadata previews, tensor descriptors, and hierarchy.
- BF16, F16, F32, F64, I8, I16, I32, I64 value sampling.
- Q8_0 value sampling in two modes:
  - `Static`: the on-disk int8 value inside its 32-value block.
  - `Final`: the dequantized `scale * int8` value.

Other quantized tensor types are listed with dimensions and byte sizes, but value decoding is left for follow-up passes.

## Verify locally

```powershell
python -m unittest discover -s tests -p "test_*.py"
python -m py_compile server.py gguf_explorer\gguf.py tools\make_sample_gguf.py
```

The UI verifier uses Playwright and can open a real GGUF path:

```powershell
$env:GGUF_EXPLORER_OPEN_PATH='E:\llm-gguf-ex\Qwen3.5-2B-Q8_0.gguf'
$env:GGUF_EXPLORER_REFERENCE_PATH='E:\llm-gguf-ex\Qwen3.5-2B-BF16.gguf'
$env:GGUF_EXPLORER_DRILL_PATH='token_embd'
$env:GGUF_EXPLORER_EXPECT_FINAL='-0.029269218'
$env:GGUF_EXPLORER_EXPECT_STATIC='-47'
node tools\verify_ui.cjs
```
