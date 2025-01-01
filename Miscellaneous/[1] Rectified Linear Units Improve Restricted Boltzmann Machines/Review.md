
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

- 그러면 $v_i = \sum\limits_{j\in hid}\mathbf{h}_j\mathbf{w}_{ij}+\mathbf{b}_i$

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

## Labeled Faces in the Wild

## Mixtures of Exponentially Many Linear Models

## Summary
