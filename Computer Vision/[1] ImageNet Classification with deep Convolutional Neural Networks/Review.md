
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
	$\Delta w_{ij}=\epsilon(<v_ih_j>_{data}-<v_ih_j>_{recon})$를 사용하여 학습하는데, 여기에서 $\epsilon$은 learning rate, $<v_ih_j>_{data}$는 
-
### 2. Gaussian Units

## Rectified Linear Units

## Intensity Equivariance

## Empirical Evaluation

## Jittered-Cluttered NORB

## Labeled Faces in the Wild

## Mixtures of Exponentially Many Linear Models

## Summary
