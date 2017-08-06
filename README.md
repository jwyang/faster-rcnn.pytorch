# graphdetection

This is the code for image graph detection. It conducts the object detection, attribute recognition and relation detection in images jointly, and obtain the graph representations. This module is expected to help a bunch of high level tasks, such as image captioning, visual question answering, expression reference, etc.

### Modules

#### DataLoader

1. Data Preparation Class (load data from given datasets)

2. Data Loader Class (collect training data in parallel)

#### Faster-RCNN

1. Pretrained Bottom Network (e.g., AlexNet, VGG, ResNet, etc.) 

2. RPN (Region Proposal Network)

- RPN_Conv, Conv, o512-f3-p1-s1, one more conv layer on feature map
- RPN_ReLU, ReLU, one more relu layer on feature map
- RPN_Cls_Score, Conv, o18-f1-p0-s1, 18 = 2(bg/fg) * 9(anchors), get region proposal classifcation scores
- RPN_Bbox_Pred, Conv, o36-f1-po-s1, 36 = 4(coordinates) * 9(anchors), get region proposal coordinates
- RPN_Data, /*Python*/, get rpn_labels_gt, rpn_bboxes_gt based on ground truth annotations
- RPN_Loss_Cls, SoftmaxWithLoass, classifcation loss between rpn_cls_score and rpn_labels_gt
- RPN_Loss_Bbox, SmoothL1Loss, regression loss between rpn_bbox_pred and prn_bboxes_gt

3. ROI (Region of Interest)

- RPN_Cls_Prob, Softmax, get rpn_cls_prob given rpn_cls_score
- Proposal, /*Python*/, return top proposals based on the rpn_cls_prob and anchors
- ROI-Data, /*Python*/, compare proposals and ground truth bboxes

4. RCNN (Region Convolutional Neural Network)

- ROI_Pooling, ROIPooling, take conv feature map and rois and return pooled features
- FC6+FC7, InnerProduct, o4096, work on roi pooled features
- CLS_Score, InnerProduct, o21, return classifcation scores on all rois
- Bbox_Pred, Innerproduct, o84, predict bboxes for all categories on all rois
- Loss_Cls, SoftmaxWithLoss, classification loss between CLS_score and ground truth labels
- Loss_bbox, SmoothL1Loss, regression loss between bbox_pred and ground truth bboxes

#### Graph Detection Network

1. Attribute Classifcation Network

2. Relation Classification Network

### TODO list

- [x] Try Pytorch faster RCNN [code](https://github.com/longcw/faster_rcnn_pytorch), and ensure it works.
- [ ] Go through Pytorch faster RCNN code and get familiar with it.
- [ ] Data preparation for Visual Genome Dataset.
- [ ] Train extended faster RCNN model on Visual Genome Dataset.


### Implementation TODO list

- [ ] Re-implement the RoIDataLayer using python multi-thread API.
- [ ] faster_rcnn.py: classes loading from external file. 
- [ ] network.py is not necessary.


