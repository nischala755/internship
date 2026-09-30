"""Generate the reconstructed DRDO internship diary in the university format."""

from datetime import date, timedelta
from pathlib import Path


OUTPUT = Path(__file__).resolve().parents[2] / "DRDO_Internship_Diary_01Jul_to_01Nov2026.txt"

HOLIDAYS = {
    date(2026, 8, 26): "Onam / Id-Milad",
    date(2026, 8, 28): "Varamahalakshmi Vratha / Rakshabandhan",
    date(2026, 9, 4): "Krishna Janmashtami",
    date(2026, 9, 14): "Ganesh Chaturthi",
    date(2026, 10, 2): "Mahatma Gandhi Jayanti",
    date(2026, 10, 20): "Vijayadashami / Dussehra",
}

COMMON_REFS = "Project folder: MyDrive/drdo; Progress presentation and experiment notebooks"


def e(summary, learning, blocker, skills, refs=COMMON_REFS, hours=7.0):
    return {
        "summary": summary,
        "learning": learning,
        "blocker": blocker,
        "skills": skills,
        "refs": refs,
        "hours": hours,
    }


JULY = [
    e("Attended project orientation and discussed the internship objective: polarimetric ISAR-based fine recognition and component detection for space targets. Broke the work into literature study, dataset understanding, classification, oriented detection and evaluation stages.", "Understood the distinction between target-level classification and component-level localization. Established a staged experimental workflow instead of directly training a large model.", "The exact expected deliverables and evaluation protocol required clarification during the initial discussion.", "Technical planning, radar imaging fundamentals, research methodology", hours=6.0),
    e("Studied radar, SAR and ISAR imaging principles, including range and cross-range formation, coherent processing and the effect of target rotation. Prepared notes connecting ISAR signatures with satellite geometry.", "Learned why ISAR can form two-dimensional target structure from rotational motion and why scattering centres are important for recognition.", "Several radar terms were unfamiliar and required cross-checking across introductory references.", "ISAR fundamentals, literature review, technical note-taking", hours=6.5),
    e("Reviewed polarimetric radar concepts and the HH, HV, VH and VV channels. Compared co-polarized and cross-polarized returns and examined why polarization diversity may reveal different target components.", "Recognized that each polarization channel contains complementary scattering information, motivating later magnitude, phase and fusion experiments.", "The physical interpretation of cross-polarized scattering was initially difficult to connect with the image data.", "Polarimetric radar, electromagnetic scattering, comparative analysis", hours=6.5),
    e("Read the available Pol-ISARSpaceTarget-1.0 dataset documentation and recorded its purpose, target classes and annotation formats. Created a checklist for verifying the downloaded folder structure.", "Identified the dataset as an electromagnetic-simulation benchmark containing six satellite categories and full-polarimetric complex data.", "The source paper and dataset documentation used terminology that required careful translation and interpretation.", "Dataset study, literature review, requirements analysis", refs="Pol-ISARSpaceTarget-1.0 dataset paper; project study notes", hours=7.0),
    e("Inspected the dataset directories, including complex DAT data, amplitude images, Pauli images, DOTA labels and XML annotations. Verified representative filenames and polarization suffixes.", "Learned how filenames encode satellite class, elevation, azimuth and polarization, which later enabled automatic manifest construction.", "Duplicate downloaded files and multiple representations made initial file counting confusing.", "Python file handling, dataset auditing, pattern recognition", refs="Pol_ISARSpaceTarget1.0 dataset folders", hours=7.0),
    e("Implemented an initial Python reader for complex DAT files and verified the expected interleaved real/imaginary layout. Reconstructed 256 x 256 complex arrays for sample channels.", "Confirmed that preserving real and imaginary parts is necessary to retain phase; magnitude-only conversion discards part of the original signal.", "Incorrect shape or datatype assumptions can silently corrupt complex radar data, so strict size checks were added.", "Python, NumPy, binary data parsing, complex-valued signals", hours=7.0),
    e("Generated sample magnitude and phase images from complex channels using absolute value and atan2. Compared visual responses across HH, HV, VH and VV for the same target view.", "Observed that magnitude emphasizes scattering intensity while phase contains structural information that is visually less intuitive but potentially discriminative.", "Phase wrapping near -pi and +pi required care when plotting and preprocessing.", "NumPy, Matplotlib, signal visualization, polarimetric analysis", hours=7.0),
    e("Prepared a preprocessing plan covering channel loading, resizing, normalization and label extraction. Defined separate paths for conventional magnitude CNNs and complex-valued models.", "A common manifest and reproducible split are essential for comparing different representations fairly.", "Balancing computational cost against preservation of complex information required design decisions.", "Experimental design, preprocessing, reproducibility", hours=6.5),
    e("Studied complex-valued neural network operations. Derived the real and imaginary outputs of complex convolution using paired real-valued convolutions and reviewed complex activations and normalization.", "Understood how complex convolution preserves interactions between real and imaginary components instead of treating them as unrelated channels.", "Complex batch normalization is more involved than ordinary per-channel normalization.", "Complex neural networks, PyTorch, mathematical modelling", hours=7.0),
    e("Outlined the first ComplexCNN architecture with complex convolution blocks, residual connections, pooling and a six-class classifier. Defined the expected tensor shapes through the network.", "Architecture diagrams must match the implemented tensor dimensions and classification objective exactly.", "An early design mixed classification and detection responsibilities and needed simplification.", "Deep-learning architecture, tensor-shape analysis, PyTorch", hours=7.0),
    e("Implemented core ComplexCNN layers and checked forward propagation on a small batch. Added deterministic seeds and basic runtime diagnostics for Colab.", "Small smoke tests catch channel and shape errors before expensive full training runs.", "Colab environment differences and GPU availability affected repeatability.", "PyTorch, unit testing, GPU computing, reproducibility", refs="POLISAR_ComplexCNN_Classification_CLEAN.ipynb", hours=7.0),
    e("Built the initial classification data loader for six satellite classes and ensured all four polarization channels from a frame were kept together. Checked class counts and sample labels.", "Sample-level integrity is critical: mixing polarization channels from different views would create invalid training examples.", "The directory structure had to be interpreted carefully to avoid duplicate or incomplete samples.", "DataLoader design, pandas, dataset validation", hours=6.5),
    e("Ran a short ComplexCNN training experiment and monitored training/validation loss, accuracy and macro-F1. Saved the model history for later plotting.", "Macro-F1 is important alongside accuracy because it gives equal weight to all target classes.", "The first run needed conservative batch size and memory monitoring in Colab.", "Model training, metrics, experiment tracking", hours=7.0),
    e("Reviewed the initial classification results and produced a confusion matrix. Investigated which satellite classes were confused and checked whether the split was balanced.", "A high aggregate score is insufficient without class-wise inspection and split verification.", "Unexpectedly high performance raised concern about correlated views and possible leakage.", "Confusion-matrix analysis, classification metrics, critical evaluation", hours=7.0),
    e("Removed the unnecessary detection/objectness branch from the clean ComplexCNN classification notebook. Retained the classifier and feature-extraction output required for later analysis.", "Model components must correspond directly to the task; an unused detector head complicates interpretation and training.", "Cleaning inherited multitask code required tracing every output and loss term.", "Code review, PyTorch refactoring, architecture validation", refs="POLISAR_ComplexCNN_Classification_CLEAN.ipynb", hours=7.0),
    e("Implemented magnitude-only, phase-only, magnitude-plus-phase, real/imaginary and complex-only feature representations. Standardized the feature extraction interface for controlled comparison.", "Representation choice can have as much influence as classifier choice in polarimetric ISAR recognition.", "Flattened high-dimensional features increased memory and computation requirements.", "Feature engineering, NumPy, experimental controls", hours=7.0),
    e("Constructed feature matrices by resizing each polarization channel to 64 x 64 and concatenating selected representations. Verified dimensions, finite values and label alignment.", "The magnitude-plus-phase representation produced 32,768 values per frame when eight 64 x 64 maps were concatenated.", "Large dense matrices had to be processed carefully to avoid unnecessary RAM copies.", "Array programming, dimensionality checks, memory optimization", hours=6.5),
    e("Measured magnitude and phase ranges before normalization. Recorded the wide magnitude dynamic range and phase bounds close to -pi and +pi.", "Magnitude and phase should not be normalized blindly with identical assumptions because their numerical distributions differ strongly.", "Outliers in scattering magnitude can dominate simple scaling methods.", "Exploratory data analysis, descriptive statistics, signal preprocessing", hours=7.0),
    e("Designed a normalization study comparing raw features, z-score, min-max [0,1], min-max [-1,1], robust-IQR and max-absolute scaling. Enforced train-only fitting of every scaler.", "Normalization parameters are learned quantities; using validation or test statistics would constitute data leakage.", "The earlier workflow needed revision to guarantee that no test information influenced scaling.", "Data leakage prevention, normalization, machine-learning methodology", refs="POLISAR_Feature_Normalization_ResNet18_Study.ipynb", hours=7.0),
    e("Implemented and tested the six normalization variants with feature classifiers. Saved validation and test tables for direct comparison.", "Most representations were highly separable under the official split, while max-absolute scaling was less stable than the other methods.", "Near-perfect scores required cautious interpretation rather than immediate claims of generalization.", "scikit-learn, normalization, comparative evaluation", hours=7.0),
    e("Added ANOVA-F, classifier-weight, ExtraTrees-importance and mutual-information feature-selection experiments. Compared selected subsets using macro-F1.", "Feature selection was more useful for interpretability and dimensionality reduction than for improving an already easy official split.", "Ranking tens of thousands of features was computationally expensive.", "Feature selection, statistical learning, model interpretation", hours=6.5),
    e("Downloaded and inspected the TFPSSTD dataset from IEEE DataPort. Verified Satellite I, II and III ZIP archives and counted 393 MAT files after extraction.", "TFPSSTD provides a second data source with a different organization and full-polarimetric reconstruction outputs.", "The download was large and required credentialed S3 access plus integrity checks.", "AWS CLI, IEEE DataPort, dataset verification, file management", refs="TFPSSTD dataset in MyDrive/drdo; TFPSSTD_New_Dataset_Colab_Pipeline_RAM_SAFE.ipynb", hours=7.0),
    e("Investigated Colab RAM crashes while reading TFPSSTD MAT files. Reworked the pipeline to load one file at a time, use complex64 and resize inputs before batching.", "Streaming and memory-mapped processing are essential when individual scientific files are tens of megabytes and the full archive is about 14.6 GB.", "Attempting to retain multiple complex128 arrays exhausted the Colab runtime memory.", "Memory profiling, SciPy MAT loading, RAM-safe pipeline design", refs="TFPSSTD_New_Dataset_Colab_Pipeline_RAM_SAFE_REVIEWED.ipynb", hours=7.0),
]

AUGUST = [
    e("Completed a RAM-safe TFPSSTD manifest and inspected MAT keys, shapes and metadata fields such as range, cross-range, centre frequency and bandwidth. Documented usable image arrays.", "Scientific MAT files may contain several algorithm outputs; model input must be selected explicitly instead of assuming the first array is correct.", "MAT key naming varied and required defensive inspection logic.", "SciPy, HDF5/MAT inspection, metadata analysis", refs="TFPSSTD_New_Dataset_Colab_Pipeline_RAM_SAFE_REVIEWED.ipynb", hours=7.0),
    e("Prepared TFPSSTD classification experiments using view-disjoint AccAngle/StaAngle splits, handcrafted RCS-like features and train-only normalization.", "View-disjoint evaluation is stronger than random image splitting because adjacent radar views can be highly correlated.", "Class and angle balance had to be preserved simultaneously.", "Dataset partitioning, radar features, leakage-aware evaluation", refs="TFPSSTD_Classification_RAM_SAFE.ipynb", hours=7.0),
    e("Added SGD logistic regression, linear SVM and Random Forest baselines for TFPSSTD feature vectors. Defined accuracy, macro-precision, macro-recall and macro-F1 reporting.", "Classical baselines provide a useful reference before investing in deeper networks.", "High-dimensional inputs required scaling and restrained model settings for Colab memory.", "scikit-learn, classification, model benchmarking", hours=6.5),
    e("Added magnitude-only ResNet18 and an eight-channel real/imaginary CNN to the TFPSSTD notebook. Kept dataset-specific training separate from Pol-ISAR weights.", "Cross-dataset weight reuse is not automatically valid when target types, simulation settings and image-generation pipelines differ.", "End-to-end image training remained expensive on the larger 601 x 601 source arrays.", "ResNet18, complex inputs, transfer-learning assessment", hours=7.0),
    e("Reviewed TFPSSTD outputs, documented RAM-safe execution steps and prepared a technical update for the guide. Identified cross-dataset testing as future robustness work.", "Operational reliability and reproducible loading are research contributions when datasets are too large for naive workflows.", "Some experiments required longer uninterrupted GPU sessions than the free runtime allowed.", "Technical reporting, experiment review, Colab optimization", hours=6.0),
    e("Studied DOTA oriented bounding-box annotations for satellite components. Parsed eight polygon coordinates, class names and difficulty flags and visualized sample labels.", "Oriented boxes are better suited than horizontal boxes for long rotated structures such as solar arrays and antennas.", "Polygon vertex ordering and coordinate normalization required careful handling.", "Computer vision, DOTA format, annotation visualization", hours=7.0),
    e("Converted DOTA polygons to Ultralytics YOLO-OBB normalized coordinates and validated labels against image dimensions. Added checks for malformed and out-of-range vertices.", "Annotation conversion must preserve all four corners and target orientation; ordinary xywh conversion would lose rotation.", "A few path and duplicate-file inconsistencies initially caused missing-image errors.", "YOLO-OBB, coordinate geometry, data validation", refs="POLISAR_OBB_Detection_PauliFusion_YOLO.ipynb", hours=7.0),
    e("Built a single-VV amplitude image baseline for oriented component detection. Created train, validation and test directories without loading the entire dataset into RAM.", "A strong single-channel baseline is necessary before claiming benefits from polarimetric fusion.", "One amplitude TIFF was missing under an expected duplicate filename, so the loader needed DAT fallback logic.", "Image preprocessing, YOLO dataset preparation, robust file handling", hours=7.0),
    e("Implemented fixed Pauli pseudo-colour fusion from polarimetric channels and prepared the corresponding OBB dataset. Compared qualitative contrast with single VV images.", "Fixed physical fusion can expose different scattering mechanisms but does not guarantee better detection metrics.", "Pauli conventions and dynamic-range scaling needed consistent implementation.", "Polarimetric fusion, image processing, visualization", hours=7.0),
    e("Ran initial YOLO11n-OBB training for single-VV and Pauli inputs and collected precision, recall, mAP50 and mAP50-95. Saved best checkpoints and validation plots.", "Detection evaluation requires both loose-IoU mAP50 and stricter mAP50-95; one metric alone can hide localization weakness.", "Training time and runtime disconnections required checkpoint persistence on Drive.", "Ultralytics, YOLO-OBB, GPU training, experiment tracking", hours=7.0),
    e("Designed a learnable polarimetric fusion stem using HH, HV, VH and VV magnitude/phase or real/imaginary channels to produce a three-channel detector input.", "A learnable fusion module can adapt channel weighting to data, but image-level fusion may still discard task-specific information.", "Integrating a custom trainable stem with a pretrained detector required interface decisions.", "Neural fusion, PyTorch, polarimetric representation learning", refs="POLISAR_OBB_LearnableFusion_YOLO_Comparison.ipynb", hours=7.0),
    e("Prepared the learnable-fusion dataset and matched its split, image size and detector settings to the fixed baselines. Added caching to avoid regenerating fused images after disconnections.", "Controlled comparisons require changing only the representation while holding training protocol constant.", "Generating fused representations repeatedly was slow without persistent caching.", "Experimental controls, caching, dataset engineering", hours=6.5),
    e("Compared completed YOLO11n-OBB results: single VV achieved mAP50 0.821 and mAP50-95 0.582; Pauli achieved 0.783/0.573; learned fusion achieved 0.743/0.518.", "The simplest VV input was strongest overall. Pauli improved precision in one run but reduced recall, and the first learned fusion did not improve detection.", "The result contradicted the expectation that more channels would automatically improve performance.", "Metric interpretation, ablation study, scientific reasoning", hours=7.0),
    e("Reviewed oriented-detection methods beyond YOLO and shortlisted CenterNet-OBB, DETR-style OBB, Rotated RetinaNet, Rotated Faster R-CNN, RoI Transformer and Oriented R-CNN.", "One-stage, two-stage and transformer detectors offer different localization and computational trade-offs.", "Many mature rotated detectors depend on MMRotate/MMCV versions incompatible with current Colab Python.", "Literature survey, object detection, model selection", hours=7.0),
    e("Implemented a lightweight pure-PyTorch CenterNet-OBB baseline with heatmap, size, offset and angle outputs. Added oriented-box decoding and metric computation.", "Anchor-free centre-based detection requires balanced heatmap and regression losses and careful angle representation.", "The initial model converged slowly and lagged far behind YOLO.", "CenterNet, PyTorch, multi-head losses, OBB decoding", refs="POLISAR_NonYOLO_OBB_PureTorch_Comparison.ipynb", hours=7.0),
    e("Implemented a compact DETR-OBB diagnostic baseline with learned object queries, transformer decoding and oriented box parameters. Verified matching and no-object handling.", "Set-based detection avoids NMS but requires effective bipartite matching and substantially more training to converge.", "The lightweight DETR configuration produced weak recall under limited epochs.", "Transformers, DETR, Hungarian matching, oriented boxes", hours=7.0),
    e("Reviewed CenterNet-OBB and DETR-OBB training curves and corrected comparison code. Marked them as diagnostic baselines rather than competitive SOTA claims.", "Negative results remain useful when their protocol and limitations are stated accurately.", "Loss reduction did not translate into competitive mAP, requiring careful interpretation.", "Model evaluation, debugging, research integrity", refs="POLISAR_NonYOLO_OBB_PureTorch_Comparison_REVIEWED.ipynb", hours=6.5),
    e("Attempted an MMRotate environment for stronger non-YOLO methods and documented package conflicts involving Python, MMCV, MIM and setuptools. Evaluated isolated-environment options.", "Research code ecosystems can be tightly version-coupled; environment compatibility should be treated as part of experiment design.", "MMCV-full failed to build under the available Colab Python version.", "Dependency management, MMRotate, environment debugging", hours=7.0),
    e("Consolidated August classification and detection findings into a guide update. Defined the next priority as long-epoch YOLO comparison, paper translation and a detection-aware fusion method.", "The current evidence supports strengthening the VV baseline before adding architectural complexity.", "Results came from notebooks with different run lengths, so direct comparisons needed protocol labels.", "Technical communication, research planning, results synthesis", hours=6.0),
]

SEPTEMBER = [
    e("Extended pure-PyTorch non-YOLO training toward 80 epochs with checkpoint recovery and monitored overfitting. Exported loss and metric histories for later comparison.", "Longer training alone does not guarantee better detection when architecture, assignment and data regime are limiting factors.", "Long runs were vulnerable to Colab disconnects and required resumable checkpoints.", "Long-running experiments, checkpointing, performance analysis", refs="POLISAR_NonYOLO_OBB_PureTorch_Comparison_E80_FIXED.ipynb", hours=7.0),
    e("Prepared a controlled 100-epoch YOLO11n-OBB versus YOLO11s-OBB comparison using single VV. Added best/last checkpoints, patience and test evaluation.", "Model-capacity comparisons must share the same split, input and augmentation policy.", "GPU time had to be managed carefully across two long detector runs.", "YOLO11, hyperparameter control, experiment automation", refs="POLISAR_YOLO_OBB_LongEpoch_Comparison.ipynb", hours=7.0),
    e("Audited training and validation plots required by the guide, including box loss, classification loss, distribution focal loss and mAP curves. Defined a consistent reporting table.", "Training/validation curves reveal capacity and overfitting more clearly than a final precision value alone.", "Older runs did not all preserve identical histories, so only traceable plots were retained.", "Learning-curve analysis, visualization, technical reporting", hours=6.5),
    e("Compared larger OBB models on T4 GPU and prepared YOLO26s-OBB and YOLO26m-OBB runs. Recorded resource use and controlled single-VV metrics.", "The medium model improved the available result, reaching approximately 0.880 mAP50 and 0.667 mAP50-95 in the recorded run.", "Model naming and Ultralytics-version support had to be verified to avoid comparing incompatible checkpoints.", "Large-model training, Ultralytics, GPU resource management", refs="POLISAR_YOLO26_OBB_T4_Comparison.ipynb", hours=7.0),
    e("Diagnosed Colab package failures involving Pillow, Ultralytics and Python 3.13. Repaired the environment without downgrading unrelated core packages.", "Aggressive pip installs can break preinstalled Colab dependencies; minimal pinned installation is safer.", "A mixed Pillow installation caused ImportError for PIL._typing._Ink.", "Python packaging, dependency debugging, Colab", hours=7.0),
    e("Reviewed and translated the Chinese Pol-ISAR dataset paper into English. Extracted dataset construction, simulation workflow, annotation protocol and baseline objectives.", "The paper primarily contributes a simulated full-polarimetric dataset and benchmark; it leaves room for stronger fusion and robust split evaluation.", "Technical terminology required context-aware translation rather than literal word substitution.", "Technical translation, paper reading, radar terminology", refs="Pol-ISARSpaceTarget-1.0_English_Translation_and_Summary.pdf", hours=7.0),
    e("Mapped the paper's methods against the implemented notebooks and identified reproducibility gaps: exact partition logic, model configuration, robustness evaluation and real-data transfer.", "A useful literature gap must be tied to measurable experiments rather than only proposing a new module.", "Some implementation details were not fully specified in the paper.", "Gap analysis, reproducibility, literature comparison", hours=6.5),
    e("Prepared an isolated MMRotate notebook for Rotated RetinaNet, Rotated Faster R-CNN, RoI Transformer and Oriented R-CNN. Documented the need for a compatible Python 3.10 runtime.", "Environment isolation is preferable to forcing legacy CV packages into modern Colab runtimes.", "The standard Colab image remained incompatible with required compiled MMCV components.", "MMRotate, environment design, rotated detection", refs="POLISAR_PaperMethods_MMRotate_Isolated.ipynb", hours=7.0),
    e("Prepared a DINOv3/ResNet18 backbone comparison concept and augmentation/backbone study notebook. Defined fair frozen-backbone and fine-tuning protocols.", "Backbone replacement is meaningful only when preprocessing, data split and detector head are held constant.", "Compute limits made full multi-backbone tuning impractical within a single session.", "Representation learning, backbone evaluation, experimental design", refs="POLISAR_DINOv3_ResNet18_Backbone_Study.ipynb", hours=7.0),
    e("Designed OBB-safe augmentation experiments with geometric transformations that update both image and oriented polygons. Separated photometric changes from coordinate-changing operations.", "For detection, rotating an image without rotating its labels creates incorrect supervision.", "General image augmentation libraries do not always preserve oriented-box geometry directly.", "Data augmentation, computational geometry, imgaug", refs="POLISAR_YOLO_OBB_Augmentation_Backbone_Study.ipynb", hours=7.0),
    e("Reorganized the ComplexCNN classification material for presentation: motivation, complex arithmetic, architecture, attention maps, loss curves and confusion matrix.", "A technical presentation should tell why each model was introduced, not merely list architectures and scores.", "Some earlier diagrams were too generic and did not exactly match the notebook implementation.", "Architecture documentation, presentation design, model interpretation", hours=6.5),
    e("Prepared a concise project review covering literature survey, dataset, conventional CNN, ComplexCNN, K-fold results, normalization, object detection and future work.", "Plots and confusion matrices communicate model behaviour more reliably than dense text.", "Fitting the complete project into a limited slide count required prioritizing verifiable results.", "Scientific presentation, results curation, visual communication", refs="Polarimetric ISAR Space Target Recognition and Detection.pptx", hours=7.0),
    e("Corrected claims in the presentation by separating completed results, provisional runs and planned work. Added proper IEEE-style citation for the dataset paper.", "Clear status labels prevent planned experiments from being mistaken for completed evidence.", "Several plots originated from different protocols and could not be combined without qualification.", "Citation management, research integrity, technical editing", hours=6.5),
    e("Implemented the paper-aligned elevation/azimuth partitioning workflow and generated reproducible manifests. Added assertions for class counts, overlap and missing files.", "Partition design determines what generalization claim is valid; sample-random splits can overestimate performance on neighboring views.", "The paper's split description had to be translated into explicit, auditable code.", "Dataset partitioning, validation, reproducible research", refs="POLISAR_PaperPartition_YOLO_OBB.ipynb", hours=7.0),
    e("Applied YOLO-OBB to the paper-aligned partition and exported test metrics, predictions and learning curves. Compared the protocol with the authors' reported task setup.", "A protocol-matched baseline is more informative than comparing headline numbers from different splits.", "Direct superiority claims were limited by incomplete equivalence of implementations and hardware.", "YOLO-OBB, benchmark reproduction, comparative analysis", refs="POLISAR_PaperPartition_YOLO_OBB.ipynb", hours=7.0),
    e("Built a rigorous 100-epoch conventional ResNet18 notebook using four-channel log-magnitude input, train-only normalization, strict split checks and checkpoint selection.", "A conventional baseline must be as carefully controlled as the proposed complex model to make the comparison fair.", "An earlier cache file was empty after interruption, producing an EOFError and requiring integrity validation.", "ResNet18, normalization, cache validation, checkpointing", refs="POLISAR_ResNet18_100Epoch_Rigorous.ipynb", hours=7.0),
    e("Ran and reviewed the 100-epoch ResNet18 experiment. Investigated the very high official-split result using class balance, basename overlap and nearest-angle analysis.", "No exact filename overlap was found, but validation/test views were often only one azimuth degree from training views, making the task an easy within-domain interpolation problem.", "A 100% score could be misinterpreted as universal generalization without the angle-proximity audit.", "Leakage analysis, nearest-neighbour geometry, critical model evaluation", hours=7.0),
    e("Designed a stronger validation experiment combining data shuffling, five-fold cross-validation, azimuth grouping and an untouched official test set. Defined matched augmented and non-augmented regimes.", "Grouping by azimuth reduces direct view correlation across folds and gives a more defensible estimate than ordinary random K-fold.", "Five folds times two regimes significantly increased total training cost.", "Cross-validation, grouped splitting, experiment design", hours=7.0),
    e("Implemented a four-channel imgaug pipeline with mild affine transforms, blur, intensity scaling and Gaussian noise. Ensured the same geometric transformation was applied jointly to HH, HV, VH and VV.", "Independent geometric transforms per polarization would destroy channel alignment; transforms must operate on the stacked four-channel image.", "imgaug 0.4.0 required NumPy 2 compatibility aliases without downgrading Colab packages.", "imgaug, multi-channel augmentation, NumPy compatibility", refs="POLISAR_ResNet18_ImgAug_Shuffled_5Fold_FIXED.ipynb", hours=7.0),
    e("Added per-fold train-only normalization, shuffled DataLoaders, 100-epoch resume-safe training and five-model soft-voting ensembles. Diagnosed checkpoint resumption when torch.load mapped the DataLoader RNG state to CUDA, restored it as a CPU ByteTensor and reran structural tests.", "Each fold must fit its own normalization statistics, and checkpoint files can contain device-sensitive state beyond model weights.", "The ten-run study required robust persistence; torch.Generator.set_state initially rejected the CUDA-mapped RNG tensor.", "PyTorch, StratifiedGroupKFold, ensembling, systematic debugging", refs="POLISAR_ResNet18_ImgAug_Shuffled_5Fold_FIXED.ipynb", hours=7.0),
]

OCTOBER = [
    e("Planned work: resume the five-fold augmented and non-augmented ResNet18 study, verify fold completion and consolidate histories from Drive. Check that no fold silently restarted after a runtime interruption.", "Expected outcome: a complete auditable run table containing ten matched 100-epoch experiments.", "Risk: total GPU duration may exceed a single Colab allocation; checkpoint continuity must be monitored.", "Experiment monitoring, checkpoint recovery, Colab", hours=6.5),
    e("Planned work: calculate cross-validation mean and standard deviation for accuracy, macro-precision, macro-recall, macro-F1 and loss under both regimes.", "Expected outcome: quantify whether augmentation improves average generalization or only changes individual folds.", "Risk: fold variance may be more informative than a very small difference in mean score.", "Statistical analysis, cross-validation, pandas", hours=7.0),
    e("Planned work: compare mean training and validation loss curves with fold-wise uncertainty bands. Identify convergence speed, capacity limits and possible overfitting.", "Expected outcome: select the defensible checkpoint epoch range rather than relying only on the final epoch.", "Risk: label smoothing creates a non-zero loss floor and must be considered during interpretation.", "Learning-curve analysis, Matplotlib, model diagnostics", hours=7.0),
    e("Planned work: evaluate five-fold soft-voting ensembles on the untouched 162-image official test set and export class-wise reports and confusion matrices.", "Expected outcome: obtain one final test result per regime without using test data for model selection.", "Risk: the official test still contains views close to training angles, limiting external-generalization claims.", "Model ensembling, classification metrics, leakage-aware evaluation", hours=7.0),
    e("Planned work: inspect augmented samples from all polarizations and confirm that transformations remain physically plausible. Adjust rotation, scaling and noise ranges only if artefacts are observed.", "Expected outcome: establish an augmentation policy that improves robustness without changing target identity or destroying scattering structure.", "Risk: excessive transformations may create unrealistic ISAR signatures.", "Image augmentation, radar-domain validation, visualization", hours=6.5),
    e("Planned work: perform an augmentation ablation separating affine transforms, blur, amplitude scaling and additive noise. Record which operations influence fold stability.", "Expected outcome: identify useful augmentation components rather than attributing improvement to the combined pipeline alone.", "Risk: additional ablations multiply training time, so a reduced epoch screening stage may be necessary.", "Ablation study, imgaug, experimental design", hours=7.0),
    e("Planned work: compare official-split performance with azimuth-grouped K-fold performance and explain the expected reduction caused by harder view separation.", "Expected outcome: produce a clear evidence-based discussion of interpolation versus view-generalization performance.", "Risk: protocols cannot be ranked solely by score because they measure different difficulty levels.", "Benchmark analysis, dataset bias, scientific writing", hours=7.0),
    e("Planned work: consolidate classification results from conventional ResNet18, ComplexCNN, normalization and feature-selection studies into a protocol-labelled table.", "Expected outcome: prevent misleading direct comparisons by listing input representation, split, epochs and selection criterion for every result.", "Risk: some older experiments may lack complete logs and must be labelled exploratory.", "Results curation, reproducibility, technical documentation", hours=6.5),
    e("Planned work: design the custom detection-aware polarimetric fusion architecture with separate co-polar and cross-polar stems followed by channel and spatial attention.", "Expected outcome: fuse HH/HV/VH/VV features inside the detector instead of producing a fixed RGB-like image before detection.", "Risk: custom Ultralytics integration may require modifying model parsing and pretrained-weight loading.", "Neural architecture design, polarimetric fusion, attention mechanisms", hours=7.0),
    e("Planned work: specify exact tensor dimensions, convolution blocks and fusion points for the custom detector. Estimate parameters and memory use against YOLO11n/YOLO26m baselines.", "Expected outcome: a buildable architecture diagram with no unspecified modules.", "Risk: early fusion may be memory-heavy, while late fusion may lose fine scattering interactions.", "Tensor-shape design, computational budgeting, model architecture", hours=7.0),
    e("Planned work: implement a small polarimetric stem accepting either eight real/imaginary channels or eight magnitude/phase channels. Unit-test output shapes and gradients.", "Expected outcome: verify the fusion module independently before coupling it to an OBB detector.", "Risk: phase discontinuity near +/-pi may require sine/cosine encoding.", "PyTorch modules, complex features, unit testing", hours=7.0),
    e("Planned work: add channel-attention weights and log their values for HH, HV, VH and VV. Create visualizations showing polarization importance by component class.", "Expected outcome: improve interpretability and determine whether the network consistently favours VV or uses cross-pol information selectively.", "Risk: attention weights are not automatically causal explanations.", "Attention mechanisms, interpretability, visualization", hours=7.0),
    e("Planned work: integrate the fusion stem with the OBB backbone and run a small smoke test on a reduced split. Verify loss calculation, label scaling and checkpoint serialization.", "Expected outcome: an end-to-end trainable custom detector before committing to a full run.", "Risk: interface mismatches between custom tensors and the detector neck/head may require adapters.", "Model integration, OBB losses, debugging", hours=7.0),
    e("Planned work: define physics-informed consistency rules, including reciprocal HV/VH behaviour, polarization-energy relationships and augmentation constraints.", "Expected outcome: convert radar-domain knowledge into measurable regularization terms or data-quality checks.", "Risk: simulation-specific assumptions must not be presented as universal physical laws.", "Physics-informed ML, polarimetry, regularization design", hours=6.5),
    e("Planned work: implement a soft HV/VH reciprocity consistency penalty and test it first on cached complex samples. Measure its scale relative to classification/detection loss.", "Expected outcome: encourage physically coherent feature representations without overwhelming the primary task loss.", "Risk: the appropriate regularization coefficient will require controlled tuning.", "Custom loss functions, complex data, hyperparameter tuning", hours=7.0),
    e("Planned work: establish the final detector comparison matrix: single VV, fixed Pauli, image-level learned fusion and internal detection-aware fusion under the same partition.", "Expected outcome: isolate whether performance gains come from representation, capacity or training protocol.", "Risk: all runs must use identical epochs, image size and evaluation code to remain comparable.", "Controlled experiments, OBB evaluation, ablation planning", hours=7.0),
    e("Planned work: run initial custom-detector training with conservative batch size and automatic mixed precision. Monitor GPU memory, loss components and gradient stability.", "Expected outcome: confirm convergence and identify the most expensive module before the full experiment.", "Risk: multi-polarization inputs may reduce feasible batch size and destabilize normalization.", "GPU optimization, mixed precision, training diagnostics", hours=7.0),
    e("Planned work: evaluate custom detections qualitatively on several satellites and view angles. Compare localization of solar arrays, bodies, antennas and small fittings.", "Expected outcome: identify class-specific benefits and failure modes hidden by aggregate mAP.", "Risk: small component classes may remain difficult because of scale and annotation density.", "Qualitative detection analysis, error analysis, visualization", hours=6.5),
    e("Planned work: prepare the research-paper experiment section with dataset protocol, preprocessing, baselines, metrics, ablations and reproducibility settings.", "Expected outcome: a methods section that can be traced directly to saved notebooks, configurations and result files.", "Risk: provisional experiments must not be written as completed findings.", "Scientific writing, experiment documentation, IEEE structure", hours=7.0),
    e("Planned work: summarize completed internship progress and prepare the November review plan. Prioritize remaining full custom-detector runs, robustness tests and paper figures.", "Expected outcome: a defensible milestone report linking each claim to an artifact or experiment log.", "Risk: limited remaining GPU time may require prioritizing the highest-value ablations.", "Project management, technical reporting, research planning", hours=6.0),
]


def working_days(start: date, end: date):
    current = start
    while current <= end:
        if current.weekday() < 5 and current not in HOLIDAYS:
            yield current
        current += timedelta(days=1)


def render_entry(day: date, item: dict) -> str:
    status = "Planned work entry; confirm after completion." if day > date(2026, 9, 30) else "Reconstructed completed-work entry."
    return fr"""Internship Diary Entry Details

Internship

**Intern at DRDO - DRDO**

Date

**{day.strftime('%d %b %Y')}**

## What I worked on?

**Work Summary \***

{item['summary']}

**Hours worked \***

{item['hours']:.2f}

## Show Your Work (Links)

**Reference Links**

{item['refs']}

## Blockers & Learnings

**Learnings / Outcomes \***

{item['learning']}

**Blockers / Risks**

{item['blocker']}

## Skills

**Skills Used\***

{item['skills']}

Entry status: {status}
"""


def main():
    days = list(working_days(date(2026, 7, 1), date(2026, 11, 1)))
    items = JULY + AUGUST + SEPTEMBER + OCTOBER
    assert len(days) == 82, len(days)
    assert [len(JULY), len(AUGUST), len(SEPTEMBER), len(OCTOBER)] == [23, 19, 20, 20]
    assert len(items) == len(days)
    assert all(0 <= item["hours"] <= 24 and item["hours"] * 4 == round(item["hours"] * 4) for item in items)
    assert all(len(item["summary"]) <= 2000 for item in items)
    assert all(len(item["learning"]) <= 2000 for item in items)
    assert all(len(item["blocker"]) <= 1000 for item in items)
    assert all(len(item["refs"]) <= 5000 for item in items)

    header = """DRDO INTERNSHIP DIARY - RECONSTRUCTED WORKING DRAFT

Coverage: 01 Jul 2026 to 01 Nov 2026
Generated entries: 82 working days (last eligible day: 30 Oct 2026)

IMPORTANT VERIFICATION NOTE
This diary was reconstructed from project notebooks, file history, experiment
records and guide discussions. Hours are reasonable 6-7 hour estimates. Verify
each date and hour against actual attendance before university submission.
Entries after 30 Sep 2026 are explicitly marked as planned because they were
future dates when this document was prepared.

Excluded: all Saturdays, all Sundays, and the following weekday holidays:
26 Aug, 28 Aug, 04 Sep, 14 Sep, 02 Oct and 20 Oct 2026.

======================================================================

"""
    body = "\n\n======================================================================\n\n".join(
        render_entry(day, item) for day, item in zip(days, items)
    )
    OUTPUT.write_text(header + body + "\n", encoding="utf-8")
    print(OUTPUT)
    print("Entries:", len(days))


if __name__ == "__main__":
    main()
