# BranchyYOLO
Deep learning model constructed using the same idea of yolo but for Risiko! object detection. In this repository there is also an ablated version of YOLOv9-C that it has been studied during the devolpment of BranchyYOLO. 

# Requirements
- Pytorch
- Yolov9-c official implementation: [GitHub Repository of YOLOv9-c](https://github.com/WongKinYiu/yolov9.git)
- After downloading it do this following command in the terminal in the folder that contains the repository:
  - sed -i 's/opt.min_items/min_items/' yolov9/val.py
  - sed -i 's/opt.min_items/min_items/' yolov9/val_dual.py
- Install the yolov9 requirments: pip install -r yolov9/requirements.txt

# How to use
- The file run.py contains the code to train the model and also to test it
- The file BranchyYOLO.yaml contains the definition of the model
- The file AblatedYOLOv9-C.yaml contains the definition of the ablated version of YOLOv9-C
- The file hyp.yaml contains the definition of some parameters used in the training phase
- The file coco.yaml contains the definition of the dataset used for training and testing
- The file detection can be used to detect the objects in some images
- We don't include our training dataset because it was too big

