POL-ISAR NEXT EXPERIMENTS: RUN ORDER

1. POLISAR_YOLO_OBB_Augmentation_Backbone_Study.ipynb
   - Primary experiment for the next meeting.
   - Rebuilds and asserts the clean 1314/162/162 split.
   - Runs YOLO11m-OBB with augmentation off and OBB-safe augmentation on.
   - Uses 100 epochs, per-epoch Drive checkpoints, automatic resume, and saves
     train loss, validation loss, LR, precision, recall, F1 and mAP plots.
   - Keep RUN_OPTIONAL_YOLO26=False until both YOLO11m runs finish.

2. POLISAR_DINOv3_ResNet18_Backbone_Study.ipynb
   - Classification backbone experiment.
   - Uses VV magnitude and training-only statistics.
   - Compares frozen ResNet18 and frozen DINOv3 ViT-S/16 linear probes.
   - DINOv3 requires accepting Meta's model license and setting HF_TOKEN in
     Colab Secrets. No official DINOv4 model is claimed.
   - Saves features, metrics, confusion matrices, and a patch-feature overlay.

3. POLISAR_PaperMethods_MMRotate_Isolated.ipynb
   - Paper-method replication notebook.
   - Uses an isolated Python 3.10 environment to protect the Colab runtime.
   - Defaults to O-RTMDet and O-Faster R-CNN for the practical two-day run.
   - O-FCOS, O-ATSS, and S2A-Net are selectable after the first two runs.
   - PSC is retained as a paper reference only because there is no official
     MMRotate PSC configuration in the checked repository.

PPT MATERIAL

- PPT_VERIFIED_RESULTS_AND_NEXT_EXPERIMENTS_2026-09-14.txt contains all verified
  numerical results, caveats, and slide changes.
- ppt_assets_verified contains original executed-notebook plots plus newly
  generated K-fold, confusion matrix, TFPSSTD, paper-method, and capacity plots.
- Existing YOLO results based on 174 test labels must be labelled provisional.
  Replace them after the clean 162-image evaluation completes.

Recommended immediate sequence:

1. Run the augmentation notebook through both YOLO11m experiments.
2. Start DINOv3 feature extraction while the detector checkpoints are safe on Drive.
3. Run MMRotate O-RTMDet only after its environment/version smoke test passes.
