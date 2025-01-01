
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
$$E(\bold{v},\bold{h})=-\sum\limits_{i,j}{v_ih_jw_{ij}-\sum\limits_{i}{v_ib_i}-\sum\limits_j{h_jb_j}}$$
- 에너지가 낮을수록 학습이 잘 된 것임
- 이렇게 학습된 RBM의 확률분포는 다음과 같음
$$p(\bold{v}) = \frac{\sum\limits_{\bold{h}}e^{-E(\bold{v},\bold{h})}}{\sum\limits_{\bold{u},\bold{g}}e^{-E(\bold{u},\bold{g})}}$$
- 여기에서 $\bold{u},\bold{g}$는 각각 visible, hidden layer의 가능한 모든 이진 상태를 나타내는 듯함 

### 2. Gaussian Units
- real-valued data를 다루기 위해 binary visible unit을 independent Gaussian noise를 가진 linear unit으로 대체함
- 이렇게 대체하여도 기존의 학습 방식인 Contrastive Divergence(이하 CD)를 사용할 수 있는지에 관해서는 exponential family harmoniums에서 CD가 잘 작동한다는 것을 보여주는 [Exponential Family Harmoniums with an Application to Information Retrieval](https://papers.nips.cc/paper_files/paper/2004/hash/0e900ad84f63618452210ab8baae0218-Abstract.html)에 잘 나와있음
    
- linear unit으로 대체한 RBM의 에너지는 다음과 같음
$$E(\bold{v},\bold{h})=\sum\limits_{i\in vis}{\frac{(v_i-b_i)^2}{2\sigma_i^2}}-\sum\limits_{j\in hid}{b_jh_j}-\sum\limits_{i,j}{\frac{v_i}{\sigma_i}h_jw_{ij}}$$ ($\sigma_i$는 visble unit $i$에 대한 Gaussian noise의 표준편차임)
- $\sum\limits_{i\in vis}{\frac{(v_i-b_i)^2}{2\sigma_i^2}}$는  $-\sum\limits_{i\in vis}{\ln{e^{-\frac{(v_i-b_i)^2}{2\sigma_i^2}}}}$로 Gaussian distribution에 로그를 취한 형태임. 이는 visible unit이 Gaussian distribution을 따르도록 함

- $\sigma_i$를 학습시키는 것이 가능하긴 하지만 binary hidden unit으로는 어려움([Geoffry hinton 강의 참고](https://www.youtube.com/watch?v=SnbfQwJLNk8))
- 따라서 데이터를 평균이 0, 분산이 단위 분산이 되도록 nomalize하고, reconstruction시에  $\sigma_i^2$이 1이 되도록 하여 noise-free reconstruction을 사용함
- $v_i$는
## Rectified Linear Units
- hidden unit에서도 똑같이 binary unit의 $N$개의 복제본을 만들어 표현할 수 있지만 다음과 같은 문제가 있음
> **문제점**
N개의 복사본 중에서 켜지는 unit의 평균은 $Np$이고, 분산은 $Np(1-p)$이다. p가 1에 가까워지면 대부분의 unit들이 항상 켜지게 되어 분산이 작아진다. 이는 모델이 충분한 정보를 표현하지 못하도록 하기 때문에 문제가 있다. $p$가 작으면 Poisson unit처럼 행동한다. 이는 약간의 증가만으로도 exponentially하게 값이 증가하게 되므로 learning을 불안정하게 만든다.

픽셀의 값이 $n\in[0,10]$ ($n$은 정수)라고 하자. binary 값이 아닌 실수 값을 표현하기 위해 visible unit $i$와 10개의 같은 가중치와 편향을 공유하는 복제본을 생성하면, 각 복제본이 켜질 확률은 $p(v_i=1)$ (이하 $p_i$)이다. computation과 implementation의 측면에서 각각의 복제본들에 대해서 켜질 것인지 말 것인지를 계산하는 것은 비효율적이다. 따라서 10개의 unit 중 n개의 unit이 켜질 확률을 구하면 이 확률은 ${10 \choose n}p_i^n(1-p_i)^{10-n}$의 binomial distribution을 따를 것이다. 이 이항분포의 평균은 $10p_i$이므로, $10p_i$에 작은 Gaussian noise를 추가하면 켜지는 unit의 개수인 n을 근사할 수 있다.
## Intensity Equivariance

## Empirical Evaluation

## Jittered-Cluttered NORB

## Labeled Faces in the Wild

## Mixtures of Exponentially Many Linear Models

## Summary
