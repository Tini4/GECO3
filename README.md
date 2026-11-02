# GeCo2 - Generalized-Scale Object Counting with Gradual Query Aggregation

> Official repository of **GeCo2**
> 🏆 **Accepted to AAAI 2026**
> 📄 Paper: [GeCo2 PDF](https://arxiv.org/pdf/2511.08048)

---

[![Papers with Code: SOTA on FSCD-147](https://paperswithcode.co/api/v1/papers/2511.08048/leaderboard-badge.svg?eval=10566&live=1)](https://paperswithcode.co/api/v1/papers/2511.08048/leaderboard-badge-link?eval=10566)
[![Papers with Code: #2 on MCAC](https://paperswithcode.co/api/v1/papers/2511.08048/leaderboard-badge.svg?eval=10696&live=1)](https://paperswithcode.co/api/v1/papers/2511.08048/leaderboard-badge-link?eval=10696)

## Abstract

Few-shot detection-based counters estimate the number of category instances in an image using only a few test-time exemplars.
Existing methods often rely on ad-hoc image upscaling and tiling to detect small, densely packed objects, and they struggle when object sizes vary widely within a single image.
**GeCo2** introduces a generalized-scale dense query map that is gradually aggregated across multiple backbone resolutions.
Scale-specific query encoders interact with exemplar appearance and shape prototypes at each feature level and then fuse them into a high-resolution query map for detection.
This avoids heuristic upscaling/tiling, improves counting and detection accuracy, and reduces memory and runtime.
A lightweight SAM2-based mask refinement further polishes box quality.
On standard few-shot counting/detection benchmarks, GeCo2 achieves strong gains in MAE/RMSE and AP/AP50, while running ~3× faster with a smaller GPU footprint.

https://github.com/user-attachments/assets/8b5f3f06-45f8-439f-9333-b7a747db28a5

## Live Demo

Try the interactive demo on Hugging Face:
👉 [DEMO HERE](https://huggingface.co/spaces/jerpelhan/GECO2-demo)

## Highlights

<img width="2575" height="912" alt="GECO2_first_image_motivation_neurips-1" src="https://github.com/user-attachments/assets/adf4dcfd-aa17-4cff-9113-8b8a0e37de31" />

- 🔁 **Gradual cross-scale query aggregation** → one high-res dense query map without tiling.
- 🧩 **Per-scale exemplar interaction** with **appearance** + **shape** prototypes.
- ⚡ **Fast & memory-efficient** inference.
- 📈 Strong results on **FSCD147**, **FSCD-LVIS**, and **MCAC** (few-shot & multi-class).

<img width="2097" height="587" alt="Geco2_architevture-1" src="https://github.com/user-attachments/assets/88d27ee8-e84e-409a-a87d-095ca24e8a89" />

## Installation

`install.sh` creates a conda environment called `test_geco3` (Python 3.10, PyTorch 2.7.1 + CUDA 12.6), builds the Deformable-DETR attention ops, and installs the remaining dependencies. It expects [Miniforge](https://github.com/conda-forge/miniforge) installed in `~/miniforge3`.
```bash
bash install.sh
```

All scripts (`train.sh`, `run_inference.sh`, `demo_gradio.sh`) activate `test_geco3` themselves. To run Python files directly, activate it first with `conda activate test_geco3`.

**Detectron2** is needed only for the detection metrics (AP / AP50) in `eval_bboxes.py`, and `install.sh` does not install it. A CPU-only build is enough, so no CUDA compiler is needed:
```bash
cd /tmp && ~/miniforge3/envs/test_geco3/bin/python -m pip install --no-build-isolation 'git+https://github.com/facebookresearch/detectron2.git'
```
Run it from outside the repo, so the repo's `utils`/`models` packages can't shadow anything during the build. `--no-build-isolation` makes it compile against the environment's PyTorch.

## Demo

#### Download Weights

Download the model weights from:

👉 [CNTQG_multitrain_ca44.pth](https://huggingface.co/datasets/jerpelhan/geco2-assets/resolve/main/weights/CNTQG_multitrain_ca44.pth?download=true)

and place the file in the **project root directory**.

#### Launch the Demo

Then run:
```bash
bash demo_gradio.sh
```

---
<img width="1832" height="2661" alt="GeCoV2Qualitative_segmentation-1" src="https://github.com/user-attachments/assets/8797ada0-e8a7-4e4c-8967-4ebbb365f63f" />
---

## Reproducing Results on FSCD147

This section describes how to reproduce the reported counting and detection results on the **FSCD147** benchmark (FSC147 images with box annotations).

### Dataset

The scripts expect this layout under `./data/fsc147` (the default `--data_path`):
```
data/fsc147/
├── images_384_VarV2/                          # FSC147 images
├── gt_density_map_adaptive_512_512_object_VarV2/  # 512x512 density maps (.npy)
├── annotation_FSC147_384.json -> annotations/annotation_FSC147_384.json   # symlink, needed by eval_bboxes.py
└── annotations/
    ├── annotation_FSC147_384.json             # points + exemplar boxes per image
    ├── Train_Test_Val_FSC_147.json            # split lists
    ├── ImageClasses_FSC147.txt
    └── instances_{train,val,test}.json        # FSCD147 GT boxes (COCO format)
```

1. **Images:** download `FSC147_384_V2.zip` from the [LearningToCountEverything](https://github.com/cvlab-stonybrook/LearningToCountEverything) README ([Google Drive](https://drive.google.com/file/d/1ymDYrGs9DSRicfZbSCDiOu0ikGDh5k6S/view?usp=sharing)) and unzip it into `data/fsc147/`. Only `images_384_VarV2/` is used; its 384-px density maps are not.
2. **FSC147 annotations:** download `annotation_FSC147_384.json`, `Train_Test_Val_FSC_147.json` and `ImageClasses_FSC147.txt` from the [`data/` folder of LearningToCountEverything](https://github.com/cvlab-stonybrook/LearningToCountEverything/tree/master/data) into `data/fsc147/annotations/`.
3. **FSCD147 boxes:** `instances_train.json`, `instances_val.json` and `instances_test.json` go into `data/fsc147/annotations/`. FSCD147 comes from [Counting-DETR](https://github.com/VinAIResearch/Counting-DETR); the COCO-format files used here were provided by the GeCo2 authors.
4. **Symlink:** `utils/data.py` reads the annotation file from `annotations/`, but `eval_bboxes.py` reads it from the dataset root:
   ```bash
   ln -s annotations/annotation_FSC147_384.json data/fsc147/annotation_FSC147_384.json
   ```
5. **Density maps:** generate them with [LOCA](https://github.com/djukicn/loca)'s script, which reads `annotation_FSC147_384.json` from the dataset root (hence the symlink above) and writes `gt_density_map_adaptive_512_512_object_VarV2/`:
   ```bash
   git clone https://github.com/djukicn/loca && cd loca
   python utils/data.py --data_path /path/to/GECO3/data/fsc147 --image_size 512
   ```

GeCo2 never uses the density maps in the loss; it only takes their **sums** as the ground-truth counts for MAE/RMSE. Each sum equals the number of annotated points in `annotation_FSC147_384.json`. Counts must come from the points, not from the boxes: the FSCD147 boxes do not cover every object (about 30% of val/test images have fewer boxes than points), so counting boxes gives wrong MAE/RMSE.

---

### Option 1: Inference with Official Weights

1. Download the official [weights for FSCD GECO2](https://drive.google.com/file/d/1OG6-tTP2Egvx-O5oK6lxPzvQO1LWgHbb/view?usp=sharing) and save them as `checkpoints/GECO2FSCD.pth`.

2. Run the model on val and test. This prints MAE/RMSE and writes the predicted boxes to `geco2_val.json` and `geco2_test.json` in the current directory:
```bash
bash run_inference.sh
```
`run_inference.sh` uses batch size 1 and `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`, so it fits on an 8 GB GPU (about 25 minutes on an RTX 3050). Larger GPUs can raise `--batch_size`; it does not change the results.

3. Compute the detection metrics from those files (run from the same directory; needs Detectron2):
```bash
conda activate test_geco3
python eval_bboxes.py --data_path ./data/fsc147
```

Expected results (paper Table 1 / this repo with the official weights):

| Split | MAE | RMSE | AP | AP50 |
|---|---|---|---|---|
| val | 9.40 / 9.39 | 33.28 / 33.26 | 34.08 / 34.08 | 63.21 / 63.21 |
| test | 7.64 / 7.64 | 39.39 / 39.33 | 46.81 / 46.87 | 75.89 / 75.89 |

---

### Option 2: Retraining on FSCD147

1. Train the model:
```bash
bash train.sh
```
Whenever the validation error improves, the model is saved to `checkpoints/<model_name>.pth` (`checkpoints/GECO3.pth` with the current `train.sh`). After each epoch it prints the losses, train MAE, and val/test MAE and RMSE. These in-training numbers are computed without SAM box refinement, so they differ from the final evaluation.

**Note:** `train.sh` is currently set up as a single-GPU smoke test (`--epochs 2 --batch_size 1`). The paper trains for 200 epochs with batch size 8 (4 per GPU × 2 GPUs) and weight decay 5e-5. The `#SBATCH` header is for a SLURM cluster. Training at 1024 px does not fit on an 8 GB GPU. Results on FSCD147 vary between runs, so reaching a low test RMSE may take several training runs.

2. Evaluate the trained model as in Option 1, pointing `run_inference.sh` at the new checkpoint with `--model_name GECO3`.

## Citation

If you find this work useful, please cite:

```bibtex
@inproceedings{pelhan2026generalized,
  title={Generalized-Scale Object Counting with Gradual Query Aggregation},
  author={Pelhan, Jer and Luke{\v{z}}i{\v{c}}, Alan and Kristan, Matej},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  volume={40},
  number={10},
  pages={8314--8321},
  year={2026}
}
```

