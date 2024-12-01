
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
+ILSVRC에서는 전통적으로 top-1 error rates와 top-5 error rates를 사용함
>**top-n error rates**란 모델의 전체 예측 중에서 가능성 높은 상위 n개의 categories에 실제 category가 없는 경우의 비율
- ImageNet의 데이터의 resolution은 바뀌는 반면, 이 모델에서의 input size는 고정되어 있으므로, image데이터를 256 by 256의 resolution으로 바꿈
	1. Rectangular image에서 더 짧은 면의 길이를 256이 되도록 크기를 조정함
	2. 그 사진의 중앙에서 256 by 256 크기의 patch를 crop함
	3. 모든 픽셀의 RGB값에서 그 사진의 평균 RGB값을 빼줌
	 (그 외 아무 전처리도 하지 않았음)

## The Architecture
### 1. ReLU Nonlinearity
- 기존의 모델에서는 뉴런의 결과값 $f$를 $f(x)=tanh(x)$나 $f(x)=(1+e^{-x})^{-1}$로 모델링했지만, gradient descent를 이용한 학습에서 앞선 함수와 같은 saturating nonlinearity들은 ReLU와 같은 non-saturating nonlinearity에 비해 학습 속도가 느림을 아래의 사진을 통해 알 수 있음

<center>
	<figure>
		<img  src="https://github.com/user-attachments/assets/1be1cf6b-8348-4552-bb25-76166bf6e643 "  width="400"  height="400"/>
			<figcaption>
				<font size=2>
					위 그래프는 각각의 함수들을 이용하여 학습했을 때 	CIFAR-10데이터에서 학습 에러율이 25%까지 가는 데에 걸린 epoch 수이고, 실선은 ReLU, 점선은 tanh함수임. 약 6배 정도의 속도 차이를 보임
				</font>
		</figcaption>
	</figure>
</center>
<br></br>

>**saturation**이란  활성화함수로 특정 nonlinear function (ex. sigmoid, tanh)을 사용해서 반복하여 학습시키다보면 가중치가 업데이트되지 않는 지점이 나타나는 현상을 말한다. 
>
>- saturating nonlinear function의 예로는 tanh, sigmoid가 있다.
>   - sigmoid 함수는 입력 $x$의 값이 커지거나 작아지면 gradient가 0에 가까워진다. 추가로 sigmoid 함수는 $x=0$일 때, 미분 값이 $\frac{1}{4}$로 최대가 된다. 따라서 연쇄 법칙을 통해 back propagate한다면, 1보다 작은 값인 sigmoid의 미분 값을 계속 곱하게 되고, 결국 학습 함수 $L$의 가중치 $W$에 대한 미분 값은 0으로 수렴하게 된다. 
>- non-saturating nonlinear function의 예로는 ReLU가 있다.

<br></br>

- 이 논문 이전에도 sigmoid나 tanh함수의 대안으로 $|tanh(x)|$함수가 나왔었음(Jarrett et al.) 하지만 이 함수 $|tanh(x)|$는 Caltech-101 dataset에 대하여 overfitting을 방지하기 위해 제안되었으며, 이 논문에서는 overfitting 방지보다는 학습 속도를 높이는 것이 중요하기 때문에 ReLU를 사용함
>**contrast normalization**이란 다양한 밝기와 환경에서 촬영된 다양한 이미지들을 잘 훈련될 수 있도록 통일해 주는 과정이다. 아래의 수식을 이용한다.
>$$ \hat X = \frac{X - E(X)}{\sigma(X)} \ $$
>추가로 contrast normalization은 Global Contrast Normalization(GCN)과 local Contrast Normalization(LCN)으로 나뉜다. 전체 이미지에 대해서 Contrast Normalization을 해주면 GCN(feature-wise)이고, 개별 샘플에 대해서 Contrast Normalization을 해주면 LCN(sample-wise)이다.

>**이 부분을 읽고 생각해본 점**
>
>ReLU 함수는 x=0에서 continuous하지만 differentiable하지 않다. 그렇다면 backpropagation할 때 문제가 생기진 않는가? 하는 궁금증이 들었다. 그러나 실제에서 x=0이 나오는 경우는 극히 드물기 때문에 이 부분은 무시하고 사용한다고 한다. (Numerical influence of ReLU'(0) on backpropagation 참고) 그럼에도 불구하고 ReLU의 장점을 살리면서도 모든 domain에서 미분가능한 함수가 있지 않을까 생각해봤다.
>$$f(x)=\begin{cases}e^x-1 & x<0 \\ x &x\geq 0 \end{cases}$$
>이 함수는 미분하면
>$$f'(x)=\begin{cases}e^x & x<0 \\ 1 &x\geq 0 \end{cases}$$
>이므로 모든 domain에서 differentiable하다. 이에 대해서 실험해봐야겠다.

### 2. Training on Multiple GPUs
- 한 개의 GTX 580 GPU를 사용하여 이 모델을 훈련하는 것은 메모리상의 제약이 있음
- 특정 layer에서 2개의 GPU가 상호작용하도록 parallelize하여 훈련함
	- GPU는 다른 GPU의 메모리를 직접 읽고 쓸 수 있어서 cross-parallelization에 적합함
	- layer3에서는 layer2에 있는 모든 kernel maps로부터 입력을 받는 반면, layer4에서는 layer3에서 같은 GPU에 있는 kernel maps로부터만 입력을 받음
- 그 결과 top-1 error rates를 1.7%까지, top-5 error rates를 1.2%까지 낮췄으며, 학습 속도의 면에서도 단일 GPU를 사용한 network보다 약간 더 빨랐음

### 3. Local Response Normalization
> **Lateral Inhibition**
> Lateral Inhibition이란 신경생리학 용어로, 한 영역에 있는 신경세포가 상호 간 연결되어 있을 때, 흥분된 뉴런이 그 주변에 있는 뉴런들에 억제성 신경전달물질을 전달하여 그 뉴런들의 활동을 억제하려는 경향이다. Lateral Inhibition은 주변의 noise를 최소화하고, 중요한 신호들을 포착하여 감각 정보의 정확도를 높인다. Lateral Inhibition은 색채의 구별에 도움이 된다고 알려져 있다.

- Local Response Normalization(이하 LRN)은 Lateral Inhibition의 매커니즘에 착안해서 만들어지며, 다음의 식을 통해 이루어짐
$$ b^i_{x,y}=a^i_{x,y}/{\Big( k+\alpha \sum_{j=\max(0,i-\frac{n}{2})}^{\min(N-1,i+\frac{n}{2})}}(a^i_{x,y})^2\Big)^\beta $$
- 위 식에서 $a^i_{x,y}$은 $(x,y)$의 위치에 있는 픽셀에  $i$번째 커널을 적용하여 계산된 값에 ReLU를 적용한 값임
- $N$은 그 layer에 있는 모든 kernel의 수임
- $k,\alpha,\beta,n$은 모두 hyperparameter임
- $k$는 분모가 0이 되는 것을 막기 위한 값인 것 같음
- $n$은 몇 개의 인접한 픽셀에 대하여 LRN을 시행할 것인지에 관한 hyperparameter임
- AlexNet에서는 activation fuction으로 ReLU를 쓰고 있기 때문에, 픽셀에서의 값이 양수이면 그대로 값이 나옴. 그렇게 되면 ReLU적용 이후 Convolution layer나 pooling layer를 지날 때, 값이 매우 큰 한 픽셀에 큰 영향을 받아 값이 작은 픽셀은 무시되어 학습이 제대로 이루어지지 않을 수 있음.  그렇기 때문에 LRN으로 주변 픽셀에 대해서 normalization을 하고, pooling이나 convolution을 함.

-LRN을 사용하지 않았을 때는 test error rate가 13%이고, LRN을 사용했을 때에는 test error rate가 11%로 그 효율성이 입증됨 

> **이 부분을 읽고 생각해본 점**
> 
> 1. 왜 $\alpha,\beta$를 hyperparameter로 두었을까? $\alpha=\frac{1}{n},\beta=\frac{1}{2}$이어도 lateral inhibition의 효과는 달라지지 않지 않았을까?  LRN에서 $\alpha,\beta$의 역할이 무엇일까?
>  $\alpha,\beta$는 모두 주변 픽셀에 대한 영향을 얼마나 고려할 것 인지에 관한 파라미터라고 생각한다. $\alpha$나 $\beta$가 클수록 주변 픽셀에 대한 영향력이 증가하여 개별 뉴런의 출력이 더 약해진다. 반대로 $\alpha$나 $\beta$가 작다면,  개별 뉴런의 출력의 영향력이 더 강해져 각 픽셀의 출력이 독립적으로 작용한다. 그렇기 때문에 적절한 $\alpha,\beta$의 값을 설정하여야 한다. LRN에서는 hyperparameter tuning이 아주 중요하게 작용할 것이라고 생각된다. 그럼에도 불구하고 비슷한 역할을 하는 hyperparameter를 왜 2개씩이나 설정하였는지에 관한 의문은 남는다.
>  
> 2. LRN이 과도하게 큰 activation value를 억제하기 위한 것이라면 오히려 주변 뉴런의 영향으로 이미지의 feature가 제대로 학습되지 않을 수 있다는 우려가 든다. 뉴런이 지나치게 normalize된다면 중요한 feature가 압축되거나 소실될 수도 있을 것 같다. 그렇기 때문에 normalization 정도를 바꿔가며 adaptive한 방식의 LRN을 구현하는 것도 좋을 것 같다는 생각이 든다.

### 4. Overlapping Pooling
- Pooling layer는 같은 kernel map에 있는 뉴런과 그 주변의 뉴런들의 결과를 요약함
- 전통적인 방식의 pooling은 겹치는 부분 없이 pooling size와 stride가 일치했음
- Alexnet에서는 pooling size > stride로 설정하여 pooling을 할 때 겹치는 부분이 있도록함
- 이 논문에서는 pooling size = 3, stride = 2로 함
- overlapping pooling은 모델이 overffing되는 것을 방지함
- 그 결과 top-1 error rates 는 0.4%만큼, top-5 error rates는 0.3%만큼 줄었음

> **이 부분을 읽고 생각해본 점**
> 
> 1. overlap되는 부분이 많아지면 summarize하는 것의 의미가 사라지고, overlap되는 부분이 적어지면 정보 손실이 커진다. 그렇다면 pooling size와 stride는 어떻게 결정하는 것이 좋을까? pooling size와 stride도 adaptive하게 할 순 없을까?
> 2. 앞서 data가 discrete한 경우 downsampling을 위해 pooling을 진행했다. 그렇다면 continuous한 data의 경우 어떻게 downsampling을 할까?
> 3. AlexNet에서는 maxpooling을 사용했다. maxpooling은 다른 작은 픽셀 값을 무시하므로 정보의 손실이 크지 않은가? average pooling이나 다른 pooling이 더 효과적이지는 않은가?

### 5. Overall Architecture
<center>
	<figure>
		<img  src="https://github.com/user-attachments/assets/d9dc3d32-dd4b-4e12-8033-cd490be339b0"  width="800"  height="400"/>
			<figcaption>
				<font size=2>
					image caption
				</font>
		</figcaption>
	</figure>
</center>

- 첫 번째 layer(Conv):
	- input size: $224\times 224\times3$ 
	- kernel size: $11\times11\times3$
	- kernel 개수: 96
	- stride: 4
	- activation: ReLU + LRN + Max pooling(overlapping pooling)
> $$O = \frac{I-F+2P}{S}+1$$ I: 입력 feature map의 크기, F: Filter의 크기, P: padding, S: stride 이므로 첫 번째 layer를 계산해보면 input size가  $227\times 227\times3$이 되어야 함
- 두 번째 layer(Conv):
	- kernel size: $5\times5\times48$
	- kernel 개수: 256
	- activation: ReLU + LRN + Max pooling(overlapping pooling)
- 세 번째 layer(Conv):
	- kernel size: $3\times3\times256$
	- kernel 개수: 384
	- activation: ReLU
	- 유일하게 이전 layer에서 모든 kernel map들과 연결됨 
- 네 번째 layer(Conv):
	- kernel size: $3\times3\times192$
	- kernel 개수: 384
	- activation: ReLU
- 다섯 번째 layer(Conv):
	- kernel size: $3\times3\times192$
	- kernel 개수: 256
	- activation: ReLU
- 여섯 번째 layer(FC):
	- Neuron 수:  4096
	- activation: ReLU
- 일곱 번째 layer(FC):
	- Neuron 수:  4096
	- activation: ReLU
- 여덟 번째 layer(FC):
	- Neuron 수:  1000
	- activation: Softmax


## Reducing Overfitting

### 1. Data Augmentation

### 2. DropOut

## Details of Learning

## Results

### 1. Qualitative Evaluations

## Discussion 
