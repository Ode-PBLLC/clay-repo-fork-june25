## Segmentation with Clay using Segformer
Refer to the segmentation tutorial for a conceptual overview. https://clay-foundation.github.io/model/finetune/segment.html

### Basic use:
I've attempted to flag things that need to be changed with "Edit -" in the scripts. Be sure to use Python 3.11.

**Workflow -**
Place your tif with labels in data/mydata. Add the model root to path.

Data folder setup:
data/
└── mydata/
    ├── train/
    │   ├── chips/
    │   └── labels/
    └── val/
        ├── chips/
        └── labels/
1. Generate chunked tif files for training and validation data using generate_tifs.py
2. Convert tif files into numpy arrays for processing using preprocess_data.py
3. Run model training with python /home/ubuntu/model/finetune/segment/segment.py fit --config /home/ubuntu/model/configs/segment_chesapeake.yaml
4. Run inference and inspect results in chesapeake_inference.ipynb
