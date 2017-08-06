# graphdetection

This is the code for image graph detection. It conducts the object detection, attribute recognition and relation detection in images jointly, and obtain the graph representations. This module is expected to help a bunch of high level tasks, such as image captioning, visual question answering, expression reference, etc.

### TODO list

- [x] Try Pytorch faster RCNN [code](https://github.com/longcw/faster_rcnn_pytorch), and ensure it works.
- [ ] Go through Pytorch faster RCNN code and get familiar with it.
- [ ] Data preparation for Visual Genome Dataset.
- [ ] Train extended faster RCNN model on Visual Genome Dataset.


### Implementation TODO list

- [ ] Re-implement the RoIDataLayer using python multi-thread API.
- [ ] faster_rcnn.py: classes loading from external file. 
- [ ] network.py is not necessary.

### Modules

#### DataLoader

1. Data Preparation Class (load data from given datasets)

2. Data Loader Class (collect training data in parallel)

#### Faster-RCNN

1. Pretrained Bottom Network (e.g., AlexNet, VGG, ResNet, etc.) 

2. RPN (Region Proposal Network)

3. RCNN (Region Convolutional Neural Network)

#### Graph Detection Network

1. Attribute Classifcation Network

2. Relation Classification Network



