# Histopathology pipeline

This directory contains the histopathology code used for the BMVC 2026 TWT experiments. Follow these steps:

1. Tile raw WSIs and retain foreground tissue tiles in one `.tar` archive per WSI/scan.
2. Compute transformer-block similarity on PANDA training tiles.
3. Find the contiguous merge plan.
4. Audition one surrogate layer per discovered block-group and distil the pruned TWT student.
5. Extract frozen tile embeddings from the pruned model.
6. Train ABMIL or TransMIL heads on the frozen WSI bags.

## Raw WSI preprocessing

The repository includes the preprocessing used to produce the tile archives consumed by the TWT pipeline. The implementation uses **OpenSlide only**. Preprocessing uses 20x magnification and stores 256 x 256 JPEG tiles in one `.tar` archive per WSI. The regular grid is non-overlapping. But if more than 50 pixels remain at the right or bottom boundary, an additional edge-aligned tile is added; this edge tile intentionally overlaps the preceding grid tile.

The preprocessing scripts require the OpenSlide system library plus the Python packages `openslide-python`, `numpy`, `Pillow`, `opencv-python`, and `scikit-image`.

### PANDA and TCGA-PRAD

PANDA and TCGA-PRAD use two steps. First create unfiltered 20x tile archives:

```bash
python histopathology/preprocessing/tile_wsi.py \
  --input-dir /data/PANDA/raw \
  --output-dir /data/PANDA/tiles_unfiltered \
  --extensions tiff \
  --target-magnification 20

python histopathology/preprocessing/tile_wsi.py \
  --input-dir /data/TCGA_PRAD/raw \
  --output-dir /data/TCGA_PRAD/tiles_unfiltered \
  --extensions svs \
  --target-magnification 20
```

Then apply the foreground-pixel filter used in the experiments:

```bash
python histopathology/preprocessing/filter_tiles.py \
  --src-dir /data/PANDA/tiles_unfiltered \
  --dst-dir /data/PANDA/tiles

python histopathology/preprocessing/filter_tiles.py \
  --src-dir /data/TCGA_PRAD/tiles_unfiltered \
  --dst-dir /data/TCGA_PRAD/tiles
```

A pixel is counted as foreground when its grayscale intensity satisfies `3 < gray < 230`. A tile is retained when at least 60% of its pixels are foreground. The filtered tar files are the inputs to the rest of the pipeline.

### CAMELYON16 and CAMELYON17

CAMELYON preprocessing applies tissue filtering while tiling. The pipeline constructs a low-resolution HSV tissue mask using Otsu thresholds, morphological closing/opening, and retains tiles with at least 80% tissue-mask coverage. The default mask level is 5 (or the deepest available level if a slide has fewer pyramid levels).

```bash
python histopathology/preprocessing/tile_wsi.py \
  --input-dir /data/camelyon17/raw \
  --output-dir /data/camelyon17/tar_tiles_20x_256_otsuTH80 \
  --extensions tif \
  --target-magnification 20 \
  --tissue-filter otsu \
  --mask-level 5 \
  --min-tissue-fraction 0.8

python histopathology/preprocessing/tile_wsi.py \
  --input-dir /data/camelyon16/raw \
  --output-dir /data/camelyon16/tar_tiles_20x_256_otsuTH80 \
  --extensions tif \
  --target-magnification 20 \
  --tissue-filter otsu \
  --mask-level 5 \
  --min-tissue-fraction 0.8
```

## Prepared tile archives (tar files)

After preprocessing, each WSI/scan is represented by one **foreground-only** `.tar` archive. Every image member in that archive is treated as a valid tissue (foreground) tile by the downstream code. You can also use `zip` (or any other format) but then please modify the code accordingly.

```text
data/
├── PANDA/
│   ├── tiles/
│   │   ├── <slide_id>.tar
│   │   └── ...
│   └── splits/
│       ├── train.csv
│       ├── val.csv
│       └── test.csv
├── TCGA_PRAD/
│   ├── tiles/
│   │   ├── <scan_id>.tar
│   │   └── ...
│   └── TCGA_PRAD_clinical_info.csv
├── camelyon17/
│   ├── tar_tiles_20x_256_otsuTH80/
│   │   ├── <slide_id>.tar
│   │   └── ...
│   └── slides_split.csv
└── camelyon16/
    ├── tar_tiles_20x_256_otsuTH80/
    │   ├── <slide_id>.tar
    │   └── ...
    └── camelyon+16_allAreTest.csv
```

The raw datasets and processed tiles are not redistributed by this repository. Obtain the WSIs from their original sources and follow the corresponding licences/terms.

### PANDA split files

The paper uses the PANDA train/validation/test partition released with the PANTHER codebase by Song et al.:

- PANTHER repository: https://github.com/mahmoodlab/PANTHER
- split tree: https://github.com/mahmoodlab/PANTHER/tree/main/src/splits/classification

Use the PANDA `train.csv`, `val.csv`, and `test.csv` from that classification split.

Each split must contain the columns used by the released PANDA loaders:

```text
FILENAME,isup_grade
<slide_id>,<0-5>
```

`FILENAME` must resolve to `<slide_id>.tar` in the prepared PANDA tile directory.

### TCGA-PRAD clinical metadata

The external TCGA-PRAD evaluation expects:

```text
scan_name_aperio,gleason_1,gleason_2
```

### CAMELYON labels and split files

For CAMELYON17/CAMELYON16 we use the corrected slide labels released by Ling et al. with the CAMELYON+ benchmark. Their public benchmark repository documents the cleaned dataset and links the associated release:

- CAMELYON+ benchmark: https://github.com/lingxitong/CAMELYON-PLUS-BENCHMARK
- accompanying paper: https://doi.org/10.1038/s41597-025-05586-5

For the binary metastasis task used here, prepare split CSVs with:

```text
slide_id,label,split
<slide_id>,<0-or-1>,<train|val|test>
```

Use the corrected CAMELYON+ labels rather than the original uncorrected slide labels.

## Foundation models

Model weights are loaded from local paths.

For H0-mini, provide the files `pytorch_model.bin` and `config.json` in a directory. For Hibou-B, provide the local Hugging Face model directory containing the model/configuration and image-processor files. You can obtain these files from HuggingFace:
- H0-mini: https://huggingface.co/bioptimus/H0-mini
- Hibou-B: https://huggingface.co/histai/hibou-b

## 1. Compute block similarity

Histopathology phase discovery uses a uniform sample of 10,000 tiles from the PANDA **training** split and averages pairwise cosine similarity over patch-token outputs.

H0-mini:

```bash
python -m histopathology.compute_block_similarity \
  --model-name H0-mini \
  --data-root /data/PANDA/tiles \
  --train-csv /data/PANDA/splits/train.csv \
  --h0-model /models/H0-mini/pytorch_model.bin \
  --num-samples 10000 \
  --batch-size 128 \
  --device cuda:0
```

Hibou-B:

```bash
python -m histopathology.compute_block_similarity \
  --model-name hibou \
  --data-root /data/PANDA/tiles \
  --train-csv /data/PANDA/splits/train.csv \
  --hibou-model-dir /models/Hibou-B \
  --num-samples 10000 \
  --batch-size 128 \
  --device cuda:0
```

The output used by phase discovery is:

```text
<model>_block_similarities/<model>_avgcos_patches.npy
```

## 2. Discover a merge plan

`find_merge_plan.py` implements the max-min dynamic programme used by TWT: first minimise the number of contiguous blocks satisfying the similarity threshold, then maximise the weakest selected block-boundary similarity among tied partitions.

```bash
python -m histopathology.find_merge_plan \
  H0-mini_block_similarities/H0-mini_avgcos_patches.npy \
  --threshold 0.4
```

`reduce_depth.py` contains the merge plans for the four TWT configurations reported in the histopathology tables:

- H0-mini: depth 5 and depth 4;
- Hibou-B: depth 4 and depth 3.

## 3. Audition surrogates and distil TWT

The histopathology path does **not** use LayerScale/weight-scaling experiments, staggered block phase-in, or a separate local block-training stage. Candidate surrogate layers are auditioned locally; the selected blocks are assembled into one student and optimised jointly with boundary and final-feature MSE supervision.

H0-mini, depth 5:

```bash
python -m histopathology.reduce_depth \
  --teacher H0-mini \
  --target-depth 5 \
  --root /data/PANDA/tiles \
  --train-csv /data/PANDA/splits/train.csv \
  --h0-model /models/H0-mini/pytorch_model.bin \
  --output-dir ./checkpoints/h0_depth5 \
  --epochs 2 \
  --num-workers 12
```

Hibou-B, depth 4:

```bash
python -m histopathology.reduce_depth \
  --teacher hibou \
  --target-depth 4 \
  --root /data/PANDA/tiles \
  --train-csv /data/PANDA/splits/train.csv \
  --hibou-model-dir /models/Hibou-B \
  --output-dir ./checkpoints/hibou_depth4 \
  --epochs 2 \
  --num-workers 12
```

`--batch-size` controls the GPU micro-batch. If omitted, a conservative value is chosen from available CUDA memory while the effective gradient-accumulation target remains 1024 images.

## 4. Extract frozen tile embeddings

Each saved file contains:

```python
{
    "tile_names": ["tile_...jpg", ...],
    "features": Tensor[N_tiles, D],
}
```

For the DINO-style 261-token outputs used in these experiments, the MIL representation is the class token concatenated with the mean patch-token representation; register tokens are excluded.

PANDA:

```bash
python -m histopathology.extract_features \
  --run-panda \
  --panda-root /data/PANDA/tiles \
  --panda-splits-dir /data/PANDA/splits \
  --base-model H0-mini \
  --h0-model /models/H0-mini/pytorch_model.bin \
  --method twt \
  --depth 5 \
  --checkpoint ./checkpoints/h0_depth5/H0-mini_depth5.pth \
  --out-dir ./features
```

TCGA-PRAD:

```bash
python -m histopathology.extract_features \
  --run-tcga \
  --tcga-root /data/TCGA_PRAD/tiles \
  --base-model H0-mini \
  --h0-model /models/H0-mini/pytorch_model.bin \
  --method twt \
  --depth 5 \
  --checkpoint ./checkpoints/h0_depth5/H0-mini_depth5.pth \
  --out-dir ./features
```

CAMELYON17:

```bash
python -m histopathology.extract_features \
  --run-camelyon camelyon17 \
  --camelyon-root /data/camelyon17 \
  --base-model H0-mini \
  --h0-model /models/H0-mini/pytorch_model.bin \
  --method twt \
  --depth 5 \
  --checkpoint ./checkpoints/h0_depth5/H0-mini_depth5.pth \
  --out-dir ./features
```

Run the corresponding command with `camelyon16` for the external breast cohort.

Outputs are written under:

```text
features/
├── panda_extracted_features/<model_name>/<slide_id>.pt
├── tcga_prad_extracted_features/<model_name>/<scan_id>.pt
├── camelyon17_extracted_features/<model_name>/<slide_id>.pt
└── camelyon16_extracted_features/<model_name>/<slide_id>.pt
```

## 5. Train MIL heads

### PANDA -> PANDA / TCGA-PRAD

```bash
python -m histopathology.train_mil_heads.train_panda_on_features \
  --method abmil \
  --panda-features ./features/panda_extracted_features/H0-mini_custom_depth_model \
  --tcga-features ./features/tcga_prad_extracted_features/H0-mini_custom_depth_model \
  --train-csv /data/PANDA/splits/train.csv \
  --val-csv /data/PANDA/splits/val.csv \
  --test-csv /data/PANDA/splits/test.csv \
  --tcga-clinical-csv /data/TCGA_PRAD/TCGA_PRAD_clinical_info.csv \
  --epochs 20 \
  --lr 1e-4 \
  --grad-accumulation-count 32
```

Use `--method transmil` for TransMIL. The selected epoch is the one with the highest PANDA validation QWK; the corresponding PANDA test and TCGA-PRAD QWK are reported.

### CAMELYON17 -> CAMELYON17 / CAMELYON16

```bash
python -m histopathology.train_mil_heads.train_camelyon_on_features \
  --method abmil \
  --camelyon17-features ./features/camelyon17_extracted_features/H0-mini_custom_depth_model \
  --camelyon17-splits-csv /data/camelyon17/slides_split.csv \
  --camelyon16-features ./features/camelyon16_extracted_features/H0-mini_custom_depth_model \
  --camelyon16-splits-csv /data/camelyon16/camelyon+16_allAreTest.csv \
  --epochs 100 \
  --lr 1e-4
```

Checkpoint selection follows the highest trailing five-epoch moving average of CAMELYON17 validation accuracy.
