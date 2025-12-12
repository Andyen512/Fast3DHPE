### 2D Pose Extraction

Please use the provided script `infer_mmpose_2d.py` to extract the 2D keypoints for your video. We utilize [MMPose](https://github.com/open-mmlab/mmpose) with **RTMDet** for detection and the **HRNet** model trained on the **COCO dataset** (17 keypoints).

**Note:** This script is designed for **single-person inference**. It automatically filters and tracks the target person with the highest detection confidence in the video frame.

```bash
python infer_mmpose_2d.py \
  --video <your_video.mp4> \
  --out_dir <output_path>
```

### 3D Pose Extraction

Please use the provided script `infer_3d.py` to reconstruct 3D human poses from the extracted 2D keypoints. This script automatically handles the conversion of 2D keypoints from the **COCO format** to the **Human3.6M format** and utilizes various 3D pose estimation models (such as **MixSTE**) to generate the final 3D predictions.

**Note:** You can specify different 3D backbones using the `--model` argument.

```bash
python infer_3d.py \
  --model MixSTE \
  --json_2d <your_json_2d.json> \
  --video_path <your_video.mp4> \
  --output <output_path>
```

### 3D Pose Extraction (Human3.6M)

Please use the provided script `compare_h36m.py` to perform 3D pose inference and comparison on the Human3.6M dataset. This script supports running multiple models (such as **MixSTE, DDHPose, D3DP, KTPFormer**) simultaneously for side-by-side evaluation.

**Note:** You must specify the target action category (e.g., "Walking", "Sitting", "Directions") using the `--action` argument to select the corresponding sequences from the dataset.

```bash
python compare_h36m.py \
  --models MixSTE DDHPose D3DP KTPFormer \
  --action "Walking" \
  --output <your_output_path4>
```