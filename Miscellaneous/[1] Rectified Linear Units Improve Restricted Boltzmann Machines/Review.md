
# Rectified Linear Units Improve Restricted Boltzmann Machines
이 논문은 2010년 토론토 대학교의 Vinod Nair와 Geoffrey E. Hinton이 쓴 논문이다. Restricted Boltzmann Machines(RBM)에서 Rectified Linear Units(ReLU)이 효과적이라는 것을 보여주었다. 

## Abstract
Binary stochastic hidden unit을 사용하는 RBM은 각각의 binary unit은 **같은 가중치와 편향을 공유하는 많은 copies로 대체**될 수 있다. 그리고 이 경우에, 기존에 사용했던 RBM의 **학습 방법을 그대로 사용**할 수 있다는 장점이 있다.  많은 copies로 대체한 RBM은 **Noisy Rectified Linear Unit(NReLU)** 로 효율적으로 근사할 수 있다. ReLU는 binary unit과 비교하여 object recognition(NORB dataset)과 face verification(Labeled Faces in the Wild dataset)에서 더 나은 feature를 학습했다. ReLU는 binary unit과 달리 **intensity equivariance**라는 특성을 갖기 때문이다.


## Introduction
- RBM은 생성모델로 많은 분야에서 사용됨

### 1. Learning a Restricted Boltzmann Machine
- [그림 1]과 같이 Visible unit들 간, hidden unit들 간 직접적인 interaction이 없는 Boltzmann Machine을 RBM(Restricted Boltzmann Machine)이라고 함

  <center>
	<figure>
		<img  src="https://github.com/user-attachments/assets/1be74df7-4c3e-44f5-b4ce-2942c98bf89c"  width="400"  height="400"/>
			<figcaption>
				<font size=2>
					[그림 1]
				</font>
		</figcaption>
	</figure>
</center>
<br></br>

- RBM은 **Contrastive Divergence**를 사용하여 학습이 가능함
> **Contrastive Divergence**
	>  - Contrastive Divergence(이하 CD)는 RBM과 같은 에너지 기반 모델을 학습하는 데에 사용되는 알고리즘임
	> - $\Delta w_{ij}=\epsilon(<v_ih_j>_{data}-<v_ih_j>_{recon})$를 사용하여 학습함
	> - $\epsilon$은 learning rate,  $<v_ih_j>_{data}$는 input으로 실제 data를 넣었을 때 visible unit $i$와 hidden unit $j$가 함께 켜지는 빈도($v_ih_j$의 기댓값 정도로 생각하면 될 듯),   $<v_ih_j>_{recon}$는 input으로 reconstructed data를 넣었을 때 visible unit $i$와 hidden unit $j$가 함께 켜지는 빈도를 나타냄
	
> - CD의 과정을 나타내면
		1. visible unit에 training data(실제 data)를 넣고, hidden unit이 켜질 확률을 구함(확률을 구하는 식은 뒤에 나옴)
		2. 앞서 구한 확률로  hidden unit을 sampling하고, 그 sampling된 값으로 reconstruction을 함
		3. training data와 reconstructed data로 구한  $<v_ih_j>_{data}$와  $<v_ih_j>_{recon}$의 차이를 통해 모델의 parameter(weight, bias)를 업데이트함
	
- training data에 대해서 hidden unit이 1이 될 확률(켜질 확률)은 
$$p(h_j = 1)=\frac{1}{1+\exp(-b_j-\sum_{i\in vis}v_iw_{ij})}$$임

- 여기에서 $b_j$는 $j$의 bias이고, $v_i$는 pixel $i$의 binary state임

- 앞서 구한 hidden state의 configuration을 통해 reconstruction을 구할 수 있는데, 각각의 pixel이 1이 될 확률(켜질 확률)은 $$p(h_j = 1)=\frac{1}{1+\exp(-b_i-\sum_{i\in hid}h_jw_{ij})}$$임
- RBM과 같은 에너지 기반 네트워크 모델은 학습이 잘 되었는지를 평가할 때, 에너지를 사용하며 그 식은 다음과 같음
$$E(\mathbf{v},\mathbf{h})=-\sum\limits_{i,j}{v_ih_jw_{ij}-\sum\limits_{i}{v_ib_i}-\sum\limits_j{h_jb_j}}$$
- 에너지가 낮을수록 학습이 잘 된 것임
- 이렇게 학습된 RBM의 확률분포는 다음과 같음
$$p(\mathbf{v}) = \frac{\sum\limits_{\mathbf{h}}e^{-E(\mathbf{v},\mathbf{h})}}{\sum\limits_{\mathbf{u},\mathbf{g}}e^{-E(\mathbf{u},\mathbf{g})}}$$
- 여기에서 $\mathbf{u},\mathbf{g}$는 각각 visible, hidden layer의 가능한 모든 이진 상태를 나타내는 듯함 

### 2. Gaussian Units
- real-valued data를 다루기 위해 binary visible unit을 independent Gaussian noise를 가진 linear unit으로 대체함
- 이렇게 대체하여도 기존의 학습 방식인 Contrastive Divergence(이하 CD)를 사용할 수 있는지에 관해서는 exponential family harmoniums에서 CD가 잘 작동한다는 것을 보여주는 [Exponential Family Harmoniums with an Application to Information Retrieval](https://papers.nips.cc/paper_files/paper/2004/hash/0e900ad84f63618452210ab8baae0218-Abstract.html)에 잘 나와있음
    
- linear unit으로 대체한 RBM의 에너지는 $$E(\mathbf{v},\mathbf{h})=\sum\limits_{i\in vis}{\frac{(v_i-b_i)^2}{2\sigma_i^2}}-\sum\limits_{j\in hid}{b_jh_j}-\sum\limits_{i,j}{\frac{v_i}{\sigma_i}h_jw_{ij}}$$ ($\sigma_i$는 visble unit $i$에 대한 Gaussian noise의 표준편차임)
- 에너지의 첫째항 $\sum\limits_{i\in vis}{\frac{(v_i-b_i)^2}{2\sigma_i^2}}$는  $-\sum\limits_{i\in vis}{\ln{e^{-\frac{(v_i-b_i)^2}{2\sigma_i^2}}}}$로 Gaussian distribution에 로그를 취한 형태임. 이는 visible unit이 Gaussian distribution을 따르도록 함

- $\sigma_i$를 학습시키는 것이 가능하긴 하지만 binary hidden unit으로는 어려움([Geoffry hinton 강의 참고](https://www.youtube.com/watch?v=SnbfQwJLNk8))
- 따라서 데이터를 평균이 0, 분산이 단위 분산이 되도록 nomalize하고, reconstruction시에  $\sigma_i^2$이 1이 되도록 하여 noise-free reconstruction을 사용함

- 그러면 $v_i=\sum\limits_{j\in hid}\mathbf{h}_j\mathbf{w}_{ij}+\mathbf{b}_i$

## Rectified Linear Units
- hidden unit에서 더 많은 정보를 표현하기 위해 binomial unit을 도입함 (N개의 같은 가중치와 편향을 공유하는 binary unit을 합친 것으로 볼 수 있음)
- $N$개의 binary unit에 대해서 각각 연산을 진행할 수도 있겠지만, 이는 computational and implementational 관점에서 비효율적이므로, $N$개의 binary unit을 합친 binomial unit으로 대체함
- binary unit의 $N$개가 있는 것과 같으므로 기존의 learning 및 inference 방법에서 바뀌지 않음
- $N$개의 binary unit이 켜질 확률이 $p$로 같으므로, 켜지는 binary units의 평균은 $Np$, 분산은 $Np(1-p)$임
- 하지만 여기에서 binomial unit의 문제가 드러남
> **문제점**
N개의 복사본 중에서 켜지는 unit의 평균은 $Np$이고, 분산은 $Np(1-p)$이다. p가 1에 가까워지면 대부분의 unit들이 항상 켜지게 되어 분산이 작아진다. 이는 모델이 충분한 정보를 표현하지 못하도록 하기 때문에 문제가 있다. $p$가 작으면 Poisson unit처럼 행동한다. 이는 약간의 증가만으로도 exponentially하게 값이 증가하게 되므로 learning을 불안정하게 만든다.

- 이 문제를 해결하기 위해 Stepped Sigmoid Units(SSU)를 도입함
> **Stepped Sigmoid Unit(SSU)**
SSU는 같은 가중치와 편향을 공유하는 binary unit을 무한개 복제하고, 이 복제본들에 각각 다르고 고정된 offset(ex. -0.5, -1.5, -2.5 ...)을 준 것을 의미한다. 그렇게 되면 hidden unit이 켜질 확률은 $\sigma(\mathbf{v}\mathbf{w}^T+b-i+0.5)$ ($\sigma()$는 sigmoid함수)이 된다. 따라서 모든 복제본들의 전체 활성도(각각의 복제본이 켜질 확률의 합)은 $\sum\limits_{i=1}^N\sigma(\mathbf{v}\mathbf{w}^T+b-i+0.5)$이 되고, 이는 $\log(1+e^{\mathbf{v}\mathbf{w}^T+b})$로 근사할 수 있다. SSU는 일반적인 binary unit보다 더 많은 parameter를 필요로 하지 않으면서, 더 많이 표현할 수 있다.

- 하지만 SSU는 활성도를 정확하게 샘플링하기 위해서 logistic sigmoid 함수를 여러 번 불러야 함
- 이를 해결하기 위해서 logistic sigmoid 함수의 빠른 근사인 Noisy Rectified Linear Unit(NReLU, $\max(0, x+N(0,\sigma(x)))$)를 사용함

## Intensity Equivariance
- Intensity Equivariance는 ReLU의 장점 중 하나임
- Equivariance와 Invariace에 대한 설명은 [Equivariance vs Invariance](https://jrc-park.tistory.com/312)에 잘 나와있음

- $x<0$에서 ReLU의 값은 $0$이므로 $\alpha$를 곱해도 그대로 $0$이고, ReLU가 zero biases를 갖고 noise-free이므로 $x>0$에서 이미지에 $\alpha>0$만큼 곱한 값은 원래의 결과에 $\alpha$만큼 곱한 값으로 나옴 ($f(\alpha x) = \alpha f(x)$)
- 즉, ReLU는 binary unit과 다르게 Intensity Equivariant함
- 여기에 intensity invariant한 cosine similarity를 적용함
  Cosine Similarity$(\alpha x,\alpha y)$ = Cosine Similarity$(x,y)$
- 따라서 이미지의 intensity에 $\alpha$만큼 곱해져도 결과가 변하지 않기 때문에 밝기나 조명에 의해 결과가 달라지지 않고, intensity의 변화에 더 강건한 feature vector 비교가 가능함
## Empirical Evaluation
- Jittered-Cluttered NORM 데이터셋에 대한 객체 인식과 Labeled Faces in the Wild 데이터셋에 대한 face verification을 binary hidden unit과 NReLU에 대해서 각각 평가를 진행함
- 두 데이터셋 모두 binary hidden unit보다 NReLU가 더 뛰어난 성능을 보임
## Jittered-Cluttered NORB
- NORB는 3D 객체 인식용 합성 데이터셋으로, 5개의 객체 클래스(인간, 동물, 자동차, 비행기, 트럭)로 구성됨
- Jittered-Cluttered 버전의 NORB는 배경에 잡음이 포함되고, 객체의 위치, 크기, 밝기 등이 무작위로 변형됨
- 각 클래스에 대해서 10개의 인스턴스가 있으며, 5개는 학습용, 5개는 시험용임
- 5개의 클래스에 더하여, 중앙에 객체가 없이 배경만 있는 6번째 클래스가 있음

<center>
	<figure>
		<img  src="https://velog.velcdn.com/images/bsj2061/post/15895fb4-0d3f-479f-9e71-d24936e30d5a/image.png"  width="400"  height="400"/>
			<figcaption>
				<font size=2>
					[그림 2] Jittered-Cluttered NORB의 예시
				</font>
		</figcaption>
	</figure>
</center>
<br></br>

### 1. Training
- 이미지들을 $108\times108\times2$에서 $32\times32\times2$의 해상도로 다운샘플링하고, 이를 정규화함(zero-mean이 되도록하고, 모든 training 이미지에 있는 pixel의 표준편차의 평균으로 나눠줌) 
- CD를 사용하여 두 개의 feature 레이어를 사전학습시킴
- 가장 상위의 hidden layer에서 다항 회귀를 사용하여 label을 예측하고, 분류기의 모든 parameter들을 fine-tuning함

<center>
	<figure>
		<img  src= "https://velog.velcdn.com/images/bsj2061/post/f647808d-ad4c-4da1-bcd3-0ae2f788b990/image.png"  width="300"  height="300"/>
			<figcaption>
				<font size=2>
					[그림 3] Jittered-Cluttered NORB 데이터셋에 대한 네트워크 구조
				</font>
		</figcaption>
	</figure>
</center>
<br></br>

- 첫번째 hidden layer에 1000, 2000, 4000개의 units, 두번째 hidden layer에 1000, 2000개의 units으로 실험해본 결과, unit의 수가 많을수록 더 정확한 분류 결과를 얻었음
- visible unit은 모두 Gaussian unit이며, hidden unit은 NReLU, stochastic binary unit에 대해서 모두 실험을 진행함

### 2. Classification Results
- 그 결과는 아래의 표와 같음

<center>
	<figure>
		<img  src= "https://velog.velcdn.com/images/bsj2061/post/e8f52f03-a048-4a15-8ade-4f75c885d166/image.png"  width="200"  height="100"/>
			<figcaption>
				<font size=2>
					[표 1] Test error rates for classifiers with 4000 hidden units trained on 32x32x2 Jittered-Cluttered NORB images.
				</font>
		</figcaption>
	</figure>
</center>
<br></br>

<center>
	<figure>
		<img  src= "https://velog.velcdn.com/images/bsj2061/post/1e5885be-116c-4fa5-9068-cad1b7421336/image.png"  width="200"  height="170"/>
			<figcaption>
				<font size=2>
					[표 2] Test error rates for classifier with two hidden layers (4000 units in the first, 2000 in the second), trained on 32x32x2 Jittered-Cluttered NORB images.
				</font>
		</figcaption>
	</figure>
</center>
<br></br>

- 모든 결과에서 Binary보다 NReLU가 더 좋은 분류 결과를 보임

## Labeled Faces in the Wild(LFW)
- 두 개의 얼굴 이미지가 주어지고, 이 두 얼굴이 같은지 다른지 예측하는 작업
<center>
	<figure>
		<img  src= "https://velog.velcdn.com/images/bsj2061/post/b08fec77-0304-4a29-8628-4b0aa6cd6c9f/image.png"  width="340"  height="170"/>
			<figcaption>
				<font size=2>
					[그림 4] Labeled Faces in the Wild dataset의 예시 
				</font>
		</figcaption>
	</figure>
</center>
<br></br>
                                                                                                                                         
### 1. Network Architecture
- 이 task는 두 개의 input을 받는데, 두 input을 합치면 이진 분류의 결과가 합치는 순서에 의존하게 됨
- 따라서 symmetric한 분류기를 만들기 위해 siamese architecture를 사용함
- 하나의 얼굴 사진에 대한 feature를 계산하는 함수를 학습시킴(feature extractor)
- 두 얼굴 사진이 주어지면 각각의 사진에 대해서 feature extractor로 feature를 계산하고, 두 feature들을 cosine similarity와 같이 input의 순서에 invariant한 연산을 적용하여 하나의 representation으로 합침
- feature extractor가 equivariant하도록 하기 위해 hidden layer에 편향을 사용하지 않음
<center>
	<figure>
		<img  src= "https://velog.velcdn.com/images/bsj2061/post/6cfd4635-864c-4288-99ba-6390a8004f82/image.png"  width="200"  height="300"/>
			<figcaption>
				<font size=2>
					[그림 5] Siamese network used for the Labeled Faces in the Wild task.
				</font>
		</figcaption>
	</figure>
</center>
<br></br>



### 2. Training
- LFW는 $250\times250$ $( \times 3$ channels$)$의 크기를 가지고 있지만, 배경정보를 막기위해 중앙의 $144\times144$ $( \times 3$ channels$)$ 크기의 이미지만을 사용함
- 이 이미지를 회전시키고 크기를 조정해서 모든 이미지에서 눈의 좌표를 일치시킴
- 이후 $32\times32$ $( \times 3$ channels$)$로 다운샘플링하고, 이를 정규화함(Jittered-Cluttered NORB에서와 동일)
- RBM으로 feature distractor를 사전학습시키고, [그림 5]에 나와있듯이 이를 Siamese architecture에 넣고, 두 얼굴 사진에 대해서 parameter들을 fine-tuning함
- Jittered-Cluttered NORB에서처럼 RBM의 visible unit은 Gaussian unit을, hidden unit은 NReLU와 Stochastic Binary Unit을 모두 사용함

### 3. Classification Results
- 그 결과는 [표 3]과 같음
<center>
	<figure>
		<img  src= "https://velog.velcdn.com/images/bsj2061/post/5e519459-377d-4f51-ab5a-c32235a6a36d/image.png"  width="350"/>
			<figcaption>
				<font size=2>
					[표 3] Accuracy on the LFW task for various models trained on 32x32 color images.
				</font>
		</figcaption>
	</figure>
</center>
<br></br>

## Mixtures of Exponentially Many Linear Models
- 기존에 real-valued, high-dimensional data의 density를 모델링하기 위해서 a mixture of diagonal Gaussians나 a mixture of factor analysis 방식을 사용했지만, 이 방식들은 데이터가 componential structure를 포함하면 exponentially inefficient함 ( 독립적인 숫자들로 이루어진 이미지 쌍을 생각해보자. 하나의 이미지에 대한 혼합 모델(mixture model)이 $N$개의 component를 필요로 한다면, 이미지 쌍은 $N^2$의 component를 필요로 한다. 즉, component가 지수적으로 증가한다.)
- 반면 NReLU는 잠재 변수(latent variable)에 대해서는 선형적으로, parameter에 대해서는 이차적(quadratic)적으로 증가함


## 느낀점
수학적인 기초나 이전 연구에 대한 이해가 부족한 상태에서 이 논문을 읽으니 어렵게 느껴졌다. 연속적인 값을 다루기 위해 linear unit with independent Gaussian noise를 사용한다거나 N개의 binary unit들을 binomial unit으로 생각한다거나 SSU가 NReLU로 approximate된다는 것 등 그냥 읽으면 읽히긴 하지만 왜 이게 성립하고, 왜 이렇게 되는지에 대해서 꼼꼼하게 살펴보다보면 점점 머리가 아파왔다. 그래도 기초가 부족한만큼 이 논문을 읽으면서 모든 문장을 당연하게 넘어가지 않고 주로 '왜?'라는 질문을 던지며 읽으려고 노력했다. 그 결과로 아래의 논문들을 읽어보게 되었다.
> - Welling, Max, Michal Rosen-Zvi and Geoffrey E. Hinton. “[Exponential Family Harmoniums with an Application to Information Retrieval.](https://papers.nips.cc/paper_files/paper/2004/file/0e900ad84f63618452210ab8baae0218-Paper.pdf)” Neural Information Processing Systems (2004).
- G. E. Hinton, R. R. Salakhutdinov ,[Reducing the Dimensionality of Data with Neural Networks.](https://www.cs.toronto.edu/~hinton/absps/science.pdf)Science313,504-507(2006).DOI:10.1126/science.1127647
- Yoav Freund and David Haussler. 1991. [Unsupervised learning of distributions on binary vectors using two layer networks.](https://proceedings.nips.cc/paper/1991/file/33e8075e9970de0cfea955afd4644bb2-Paper.pdf) In Proceedings of the 5th International Conference on Neural Information Processing Systems (NIPS'91). Morgan Kaufmann Publishers Inc., San Francisco, CA, USA, 912–919.
- Hinton, Geoffrey. (2002). ARTICLE [Training Products of Experts by Minimizing Contrastive Divergence.](https://www.cs.toronto.edu/~fritz/absps/tr00-004.pdf) Neural computation. 14. 1771-800. 10.1162/089976602760128018. 
- Marks, Tim & Movellan, Javier. (2001). [Diffusion Networks, Products of Experts, and Factor Analysis.](https://inc.ucsd.edu/mplab/68/media/mplab2001.02.pdf)

이 외에도 많은 논문과 자료들을 찾아보며 깊은 이해를 하려고 노력했다. 이렇게까지 논문을 오랫동안 꼼꼼하게 읽어본 경험이 처음이라서 새롭기도 하면서, 추상적으로만 느껴지던 논문이 점점 머릿속에서 구체화되면서 흥미를 느꼈던 것 같다. 모든 논문을 이렇게 읽는 것은 힘들겠지만, 앞으로도 기초가 부족한 분야의 논문에 대해서는 이렇게 읽어볼 예정이다. 그리고 간단하게 생긴 ReLU에 이렇게 많은 수학적인 지식들이 내포되어있다는 사실이 놀라웠다.
