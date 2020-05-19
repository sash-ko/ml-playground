[Social distancing detection tool](https://www.analyticsvidhya.com/blog/2020/05/social-distancing-detection-tool-deep-learning/)

<img src="./images/distancing.png" width="40%" align="right">

The idea is to take a video with people walking on a street, detect people, measure distance between them and if the distance between two persons is less than a certain threshold draw red rectangles around each of them.

[Detectron2](https://github.com/facebookresearch/detectron2) is used to detect people on each frame. People are detected each frame separately, not tracking.

_Faster R-CNN_ is the model used for object detection.

<br/>

**Runs only on [Google colab](https://colab.research.google.com/)**
