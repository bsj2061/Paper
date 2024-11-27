# ImageNet Classification with Deep Convolutional Nerual networks
이 논문은 2012년 University of Toronto의 Alex Krizhevsky, Ilya Sutskever, Geoffrey E. Hinton이 쓴 논문이며, CNN으로 ILSVRC대회에서 우수한 성적을 냈던 AlexNet에 관한 논문입니다.
## Abstract
- 신경망은 5개의 convolutional layers와 3개의 FC layers로 구성되어 있으며 convolutional layers중 몇몇은 maxpooling layers뒤에 옴
- 빠른 학습을 위해 **non-saturating neurons**과 **convolution operation에 매우 효율적인 GPU**를 사용함
- FC에서 Overfitting문제를 해결하기 위해 **dropout**을 사용함
- ILSVRC-2012 competition에서 top-5 test error rate를 15.3%까지 낮춰서 우승함

## Introduction
- Machine Learning 모델은 성능을 향상시키기 위해서 많은 양의 labeled images가 필요하고, 최근 LabelMe나 ImageNet을 포함한 대량의 image data가 나옴

+  object recognition task의 엄청난 복잡성은 ImageNet과 같은 대량의 데이터셋으로도 명시될 수 없기 때문에, 많은 prior knowledge로 우리가 갖고 있지 않은 데이터들을 보완해야 함
+ CNN구조는 depth와 breadth를 다르게 하여 capacity를 조정할 수 있고, image의 특성(stationary of statistics, locality of pixel dependencies)에 관한 강력하고 올바른 가정을 할 수 있음
+ CNN은 비슷한 사이즈의 layer를 가진 standard feedforward NN보다 더 적은 connection과 parameter를 가지기 때문에 학습이 더 쉬움  

- CNN의 매력적인 특성과 local architecture의 효율성에도 불구하고, 여전히 고해상도의 이미지에 적용하기에는 너무 비쌈
- 2D convolution의 매우 최적화된 구현과 함께하여 현재의 GPU로도 심각한 overfitting없이 ImageNet과 같은 데이터셋으로 large CNN의 학습이 가능함 

+ 이 논문은 ILSVRC-2010과 ILSVRC-2012 competitions에서 가장 좋은 결과를 냈음
+ network의 사이즈는 현재 GPU의 memory에 제약을 받아서 이 논문에서는 두 개의 GTX 580 3GB GPU를 사용하여 5~6일의 학습 시간이 걸렸지만, 더 빠른 GPU와 더 큰 데이터셋이 이용 가능해진다면, 더 향상된 결과가 나올 것임

## The Dataset
-  ILSVRC에서는 ImageNet에서 1000개의 카테고리에서 각각 1000개의 이미지로 구성된 데이터셋을 활용하여, 120만 개의 training data, 5만 개의 validation data, 15만 개의 test data가 있음
	-  ImageNet은 약 22,000개의 카테고리에 속해있는 1500만 개 이상의 labeled된 고해상도 이미지로 구성되어 있음


- ImageNet의 데이터의 resolution은 바뀌는 반면, 이 모델에서의 input size는 고정되어 있으므로, image데이터를 256 by 256의 resolution으로 바꿈
	1. Rectangular image에서 더 짧은 면의 길이를 256이 되도록 크기를 조정함
	2. 그 사진의 중앙에서 256 by 256 크기의 patch를 crop함
	3. 모든 픽셀의 RGB값에서 그 사진의 평균 RGB값을 빼줌
	 (그 외 아무 전처리도 하지 않았음)

## The Architecture
### 1. ReLU Nonlinearity
