# Reproduction notes

## Saved data

Each `results/<id>-native-001/` directory contains:

- `attention.f32`: little-endian float32, shape `{12, 12, 197, 197}`, in `{layer, head, query, key}` order.
- `input.f32`: the normalized input to patch embedding, shape `{3, 224, 224}`.
- `model-crop.png`: that input converted back to RGB.
- `logits.json`: the 1,000 class logits.
- `analysis.json`: head maps, rollout maps, and layer metrics.
- `result.json`: prediction, tensor hashes, and numerical checks.

Token 1 is CLS. The other 196 tokens form a 14 × 14 patch grid. CLS stays in the matrix products and is removed only when extracting a spatial map. The four attention tensors occupy about 89 MB in total.

`Reproduce.wl` checks the tensor hash, reruns the Wolfram calculation, and compares the maps with the recorded results.

## Model and normalization

All four examples use Wolfram's ViT Base/16: twelve blocks, twelve heads per block, and a 224 × 224 model crop. Model download information and SHA256 are in `experiments/model.json`.

The downloaded wrapper stores the upstream standard deviations in `VarianceImage`. The inference script applies:

```wolfram
net = NetReplacePart[net,
  {"Input", "VarianceImage"} -> {0.229, 0.224, 0.225}^2];
```

The weights are unchanged. This makes the encoder divide by the upstream standard deviations. The saved tensors and predictions all use this adjustment. The original PyTorch photograph also has a saved reference check against the corresponding Torchvision weights in `results/samoyed-native-001/reference.json`.

## Figures

From the repository folder:

```sh
python -m pip install -r requirements.txt
python scripts/render_figures.py --image samoyed
python scripts/render_figures.py --image samoyed-option-4
python scripts/render_figures.py --image samoyed-option-5
python scripts/render_figures.py --image samoyed-option-1
```

The renderer recomputes the analysis in NumPy and checks maps and metrics against Wolfram before saving figures. It uses the saved model crop, bilinear interpolation of the 14 × 14 map, the inferno color map, and 45% overlay opacity. Each panel is scaled between its own minimum and maximum. Colors therefore show relative variation within a panel, not equal attention magnitudes across panels. The layer plot uses normalized spatial entropy; 1 is uniform.

## Tests

`run_wolfram.py` uses `WOLFRAM_KERNEL` if set. Otherwise, its default is the Wolfram 15 Windows installation path. On another installation, pass `--kernel /path/to/WolframKernel`.

```sh
python scripts/run_wolfram.py tests/test_rollout.wl --output generated/core-tests
python scripts/run_wolfram.py tests/check_cached_examples.wl --output generated/cached-checks
python scripts/run_wolfram.py tests/check_notebook.wl --output generated/notebook-check
```

Use a new output directory when repeating a run. The core tests include noncommuting matrices to check multiplication order, residual mixing, head averaging, CLS exclusion, and invalid inputs. The notebook check evaluates the saved rollout definition and toy example, then compares the definition's output with the cached maps.

## Run inference again

This is optional. Map reproduction above uses the saved tensors.

```sh
python scripts/download_inputs.py --model
python scripts/run_wolfram.py scripts/extract_attention.wl --output generated/samoyed-inference --timeout 600
```

The downloader verifies the source photos and the 346 MB model against their recorded hashes. These files go into the ignored `.cache/` directory. To select another example in PowerShell:

```powershell
$env:ATTENTION_ROLLOUT_IMAGE = "samoyed-option-5"
python scripts/run_wolfram.py scripts/extract_attention.wl --output generated/winter-inference --timeout 600
```

The default example is `samoyed` when that variable is unset. New runs write to the chosen output directory and do not replace the committed results.
