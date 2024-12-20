# WCUA-YOLO

Enhanced Fabric Defect Detection using WCUA-YOLO: Integrating Comprehensive Feature Extraction and Multi-Scale Fusion

# Abstract

Fabric surface defect detection is crucial for ensuring product quality in the textile industry. However, the complexity of fabric defects and industrial environment interference pose significant challenges to accurate detection. This paper presents the WCUA-YOLO model, an enhancement of the YOLOv8 architecture specifically designed for fabric defect detection. The model incorporates the WIoU loss function to optimize detection performance, the CUConv module to enhance feature extraction, and the CSSF module to effectively fuse multi-scale features. Additionally, the ASFF-Head detection head further improves detection accuracy while maintaining feature scale consistency. Extensive experiments conducted on customized and public datasets demonstrate that WCUA-YOLO achieves superior performance with mAP values of 93.8% and 51.4%, respectively, outperforming other state-of-the-art models. These results validate the effectiveness and reliability of WCUA-YOLO in fabric defect detection, underscoring its potential for real-world applications.

## Usage:

We have created a public repository on GitHub containing all relevant source code, including but not limited to the CSFF module, CUConv module, and ASFF module. The code files of these modules are all located in the nn/models directory, while the model definition files are located in the cfg/models directory; the paths of the test data used as well as the paths of the models are called in the TRAIN file, and parameters such as the imgsz value, the PATIENCE value, and so on, are set.
