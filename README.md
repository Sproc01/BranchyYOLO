# BranchyYOLO
A Deep Learning model built using the same idea as YOLO, but with a 2-branch topology.
This is its PyTorch implementation for Risiko! table game's pieces object detection.

In this repository there is also an ablated version of YOLOv9-C which has been studied during the development of BranchyYOLO.

For more information read the [report](Report.pdf).

# Requirements
- PyTorch
- [YOLOv9-C official implementation](https://github.com/WongKinYiu/yolov9.git): `git clone 'https://github.com/WongKinYiu/yolov9.git'`
  After downloading it run the following commands:
    - `sed -i 's/opt.min_items/min_items/' yolov9/val.py`
    - `sed -i 's/opt.min_items/min_items/' yolov9/val_dual.py`
- Install the yolov9 requirements: `pip install -r yolov9/requirements.txt`

# Files explanation
We don't include our training dataset because it is too big

- The file `run.py` contains the code to train the model and also to test it
  > **Important**: modify `run.py` imports according to the model being trained: use `*_dual` files if and only if the ablated model is going to be used (since it has the DualDetect instead of Detect at the end).
- The file `BranchyYOLO.yaml` contains the definition of *BranchyYOLO* model; it will be imported by `models.yolo.parse_model`
- The file `AblatedYOLOv9-C.yaml` contains the definition of the ablated version of YOLOv9-C
- The file `hyp.yaml` contains the definition of some hyperparameters used during the training phase
- The file `coco.yaml` contains the definition of the dataset used for training, validation and testing
- The file `Detection.ipynb` can be used to perform object detection in images

