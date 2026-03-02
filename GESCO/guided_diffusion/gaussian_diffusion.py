"""
This code started out as a PyTorch port of Ho et al's diffusion models:
https://github.com/hojonathanho/diffusion/blob/1e0dceb3b3495bbe19116a5e1b3596cd0706c543/diffusion_tf/diffusion_utils_2.py

Docstrings have been added, as well as DDIM sampling and a new collection of beta schedules.
"""

import enum
import math

import numpy as np
import torch as th

from .nn import mean_flat
from .losses import normal_kl, discretized_gaussian_log_likelihood

import sys
import matplotlib.pyplot as plt


def get_named_beta_schedule(schedule_name, num_diffusion_timesteps): #beta 값은 매 스텝마다 노이즈 양을 의미
    #scedule_name으로 linear나 cosine 방식 중 하나를 택할 수 있음. num_diffusion_timesteps는 diffusion process의 step수 일반적으로는 1000
    """
    Get a pre-defined beta schedule for the given name.

    The beta schedule library consists of beta schedules which remain similar
    in the limit of num_diffusion_timesteps.
    Beta schedules may be added, but should not be removed or changed once
    they are committed to maintain backwards compatibility.
    """
    if schedule_name == "linear": #linear일때
        # Linear schedule from Ho et al, extended to work for any number of
        # diffusion steps.
        scale = 1000 / num_diffusion_timesteps
        beta_start = scale * 0.0001
        beta_end = scale * 0.02 #0.0001 ~ 0.02 사이를 균등하게 나눔
        return np.linspace(
            beta_start, beta_end, num_diffusion_timesteps, dtype=np.float64
        )
    elif schedule_name == "cosine": #조금 더 부드럽고 자연스러움
        return betas_for_alpha_bar(
            num_diffusion_timesteps,
            lambda t: math.cos((t + 0.008) / 1.008 * math.pi / 2) ** 2, #t = 0 → cos(0) = 1 → alpha_bar(0) = 1 → 노이즈 없음 #t = 1 → cos(pi/2) = 0 →apha_bar(1) = 0 → 노이즈 없음
        )
    else:
        raise NotImplementedError(f"unknown beta schedule: {schedule_name}")


def betas_for_alpha_bar(num_diffusion_timesteps, alpha_bar, max_beta=0.999): #alpha_bar: 누적된 alpha_t, # ᾱₜ = ∏ₛ₌₁ᵗ (1 − βₛ) max_beta: 너무 큰 beta 방지용
    """
    Create a beta schedule that discretizes the given alpha_t_bar function,
    which defines the cumulative product of (1-beta) over time from t = [0,1].

    :param num_diffusion_timesteps: the number of betas to produce.
    :param alpha_bar: a lambda that takes an argument t from 0 to 1 and
                      produces the cumulative product of (1-beta) up to that
                      part of the diffusion process.
    :param max_beta: the maximum beta to use; use values lower than 1 to
                     prevent singularities.
    """
    betas = [] #beta 시퀀스를 담을 리스트 생성 반복하면서 각 timestep마다 베타값 계산
    for i in range(num_diffusion_timesteps):
        t1 = i / num_diffusion_timesteps #t1은 현재 step
        t2 = (i + 1) / num_diffusion_timesteps #t2는 다음 step
        betas.append(min(1 - alpha_bar(t2) / alpha_bar(t1), max_beta)) #list에 # ᾱₜ = ∏ₛ₌₁ᵗ (1 − βₛ) , βₜ = 1 − ᾱₜ₊₁ / ᾱₜ 추가
        #min은 1에 너무 가까운 값이 나오지 않게 막아줌

    return np.array(betas)


class ModelMeanType(enum.Enum): 
    """
    Which type of output the model predicts. Enum.auto로 자동 배정
    """

    PREVIOUS_X = enum.auto()  # the model predicts x_{t-1}
    START_X = enum.auto()  # the model predicts x_0
    EPSILON = enum.auto()  # the model predicts epsilon noise를 예측함


class ModelVarType(enum.Enum):
    """
    What is used as the model's output variance.

    The LEARNED_RANGE option has been added to allow the model to predict
    values between FIXED_SMALL and FIXED_LARGE, making its job easier.
    """

    LEARNED = enum.auto() #모델이 직접 분산값을 예측함
    FIXED_SMALL = enum.auto() #분산값을 아주 작게 고정시켜 noise 제거에 집중함. 고정된 분산이라서 안정적 그러나 유연성이 부족
    FIXED_LARGE = enum.auto() #분산을 좀 더 크게 고정, 샘플 다양성이 높아질 수 있음
    LEARNED_RANGE = enum.auto() #FIXED_SMALL ~ FIXED_LARGE 사이에서만 나오게 유도함


class LossType(enum.Enum):
    MSE = enum.auto()  # use raw MSE loss (and KL when learning variances) 모델이 예측한 노이즈 값과 실제 노이즈를 비교 L_MSE = E[(ε − ε_θ(x_t, t))²]
    RESCALED_MSE = (
        enum.auto()
    )  # use raw MSE loss (with RESCALED_KL when learning variances) 모델이 분산을 예측할 때 KL loss를 곁들여 씀 근데 KL 값이 너무 크면 MSE 손실이 묻히니까 rescale
    KL = enum.auto()  # use the variational lower-bound #KL Divergence할 때 사용 L_KL = D_KL(q(x_{t-1} | x_t, x_0) || p_θ(x_{t-1} | x_t))
    RESCALED_KL = enum.auto()  # like KL, but rescale to estimate the full VLB 전체 Variational lower bound(VLB)를 추정하려고 너무 크면 안돼서 rescale

    def is_vb(self):
        return self == LossType.KL or self == LossType.RESCALED_KL


class GaussianDiffusion:
    """
    Utilities for training and sampling diffusion models.

    Ported directly from here, and then adapted over time to further experimentation.
    https://github.com/hojonathanho/diffusion/blob/1e0dceb3b3495bbe19116a5e1b3596cd0706c543/diffusion_tf/diffusion_utils_2.py#L42

    :param betas: a 1-D numpy array of betas for each diffusion timestep,
                  starting at T and going to 1.
    :param model_mean_type: a ModelMeanType determining what the model outputs.
    :param model_var_type: a ModelVarType determining how variance is output.
    :param loss_type: a LossType determining the loss function to use.
    :param rescale_timesteps: if True, pass floating point timesteps into the
                              model so that they are always scaled like in the
                              original paper (0 to 1000).
    """

    def __init__(
        self,
        *,
        betas,
        model_mean_type,
        model_var_type,
        loss_type,
        rescale_timesteps=False,
    ):
        self.model_mean_type = model_mean_type
        self.model_var_type = model_var_type
        self.loss_type = loss_type
        self.rescale_timesteps = rescale_timesteps
        #enum들을 그대로 저장

        # Use float64 for accuracy.
        betas = np.array(betas, dtype=np.float64) #Diffusion Schedul에서의 beta들을 numpy array로 저장
        self.betas = betas
        assert len(betas.shape) == 1, "betas must be 1-D" #beta는 1차원 배열이여야 함
        assert (betas > 0).all() and (betas <= 1).all() #모든 베타 값들이 0보다 크고 1 이하인지 검사 1보다 크면 모델 터짐 ㅋ 0이하면 노이즈 안들어감

        self.num_timesteps = int(betas.shape[0]) 

        alphas = 1.0 - betas #alpha는 noise가 아닌 부분의 비율
        self.alphas_cumprod = np.cumprod(alphas, axis=0) ## alpha_bar_t = product_{s=1}^t (1 - beta_s) forward process 
        self.alphas_cumprod_prev = np.append(1.0, self.alphas_cumprod[:-1]) #posterior 계산할 때 필요, 1.0으로 시작하고 alpha_t-1값을 만들기 위함
        self.alphas_cumprod_next = np.append(self.alphas_cumprod[1:], 0.0) #다음 step의 alpha
        assert self.alphas_cumprod_prev.shape == (self.num_timesteps,) #길이가 전체 timestep 수랑 같은지 검사 틀리면 수식 전개 다 망함

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.sqrt_alphas_cumprod = np.sqrt(self.alphas_cumprod) #alphas_cumprod에 root를 씌움 x_0의 비율이 얼마인지 구하기 위함, 노이즈가 없는 정보의 강도
        self.sqrt_one_minus_alphas_cumprod = np.sqrt(1.0 - self.alphas_cumprod) #1에서 alphas_cumprod를 빼고 root를 씌움, 노이즈가 얼마나 들어가는 지 계산
        self.log_one_minus_alphas_cumprod = np.log(1.0 - self.alphas_cumprod) #posterior 분산의 로그값 구할 때 사용
        self.sqrt_recip_alphas_cumprod = np.sqrt(1.0 / self.alphas_cumprod) #reverse process에서 x_0를 예측할 때 사용 # sqrt(1 / alpha_bar_t)
        self.sqrt_recipm1_alphas_cumprod = np.sqrt(1.0 / self.alphas_cumprod - 1) #노이즈 역 계산용

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        self.posterior_variance = (
            betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        ) # 현재 상태에서 한 스텝 돌아갈 대 되돌아가는 Gaussian 분포
        # log calculation clipped because the posterior variance is 0 at the
        # beginning of the diffusion chain.
        self.posterior_log_variance_clipped = np.log(
            np.append(self.posterior_variance[1], self.posterior_variance[1:]) #로그 분산이 필요할 때  log(σ²) 대신 사용, 첫번째 값이 0이 되면 터지므로 두 번째 값을 복붙해서 안정화한 로그 분산
        )
        self.posterior_mean_coef1 = (
            betas * np.sqrt(self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod) #x0 쪽 계수
        )
        self.posterior_mean_coef2 = (
            (1.0 - self.alphas_cumprod_prev)
            * np.sqrt(alphas)
            / (1.0 - self.alphas_cumprod) #xt쪽 계수
        ) # posterior_mean = coef1 * x_0 + coef2 * x_t을 만들기 위한 계수


    def q_mean_variance(self, x_start, t): #위의 계수들을 실제로 조합하여 q(x_t | x_0)의 평균과 분산을 return , forward process 수식 그대로 구현
        """
        Get the distribution q(x_t | x_0).

        :param x_start: the [N x C x ...] tensor of noiseless inputs.
        :param t: the number of diffusion steps (minus 1). Here, 0 means one step.
        :return: A tuple (mean, variance, log_variance), all of x_start's shape.
        """
        mean = (
            _extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start 
        ) #self.sqrt_alphas_cumprod[t]값을 배치별 shape로 브로드 캐스팅해서 가져옴, 즉 sqrt_alpha_bar_t * x_0
        variance = _extract_into_tensor(1.0 - self.alphas_cumprod, t, x_start.shape) # var_t = 1 - alpha_bar_t
        log_variance = _extract_into_tensor(
            self.log_one_minus_alphas_cumprod, t, x_start.shape
        ) # 앞에서 계산 해둔 log(1 - alpha_bar_t) 값 꺼내서 shape 맞춰줌
        return mean, variance, log_variance #tuple로 return

    def q_sample(self, x_start, t, noise=None): #sample from q(x_t | x_0), 즉 forward process로 노이즈를 t스텝만큼 추가한 xt 생성
        """
        Diffuse the data for a given number of diffusion steps.

        In other words, sample from q(x_t | x_0).

        :param x_start: the initial data batch.
        :param t: the number of diffusion steps (minus 1). Here, 0 means one step.
        :param noise: if specified, the split-out normal noise.
        :return: A noisy version of x_start.
        """
        if noise is None: #Noise parameter를 안주면 Normal Distribution ε ~ N(0, I) 샘플을 만듦
            noise = th.randn_like(x_start)
        assert noise.shape == x_start.shape # x0와 noise 텐서 shape 동일성 체크
        return (
            _extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start #sqrt(alpha_bar_t) * x_0
            + _extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) #sqrt(1 - alpha_bar_t) * ε
            * noise
        ) #noise가 섞인 xt 텐서 반환, 학습 때 training_losses()가 xt를 만들어 모델 입력으로 사용한다.

    def q_posterior_mean_variance(self, x_start, x_t, t):
        """
        Compute the mean and variance of the diffusion posterior:

            q(x_{t-1} | x_t, x_0)의 평균과 분산 로그분산을 계산

        """
        assert x_start.shape == x_t.shape #x0(x_start)와 xt 텐서 크기가 같아야 한다.
        posterior_mean = (
            _extract_into_tensor(self.posterior_mean_coef1, t, x_t.shape) * x_start
            + _extract_into_tensor(self.posterior_mean_coef2, t, x_t.shape) * x_t
        ) # mean = c1_t * x_0 + c2_t * x_t
        posterior_variance = _extract_into_tensor(self.posterior_variance, t, x_t.shape) # var = beta_t * (1 - alpha_bar_{t-1}) / (1 - alpha_bar_t)
        posterior_log_variance_clipped = _extract_into_tensor(
            self.posterior_log_variance_clipped, t, x_t.shape
        )
        assert (
            posterior_mean.shape[0]
            == posterior_variance.shape[0]
            == posterior_log_variance_clipped.shape[0]
            == x_start.shape[0]
        ) #배치 크기가 셋 다 동일한지 확인
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def p_mean_variance(
        self, model, x, t, clip_denoised=True, denoised_fn=None, model_kwargs=None
    ):
        """
        Apply the model to get p(x_{t-1} | x_t), as well as a prediction of
        the initial x, x_0.

        :param model: the model, which takes a signal and a batch of timesteps
                      as input. 	ε̂ 또는 x̂₀ 등을 예측하는 U-Net
        :param x: the [N x C x ...] tensor at time t. 현재 단 계의 noisy 이미지 𝑥t
        :param t: a 1-D Tensor of timesteps.
        :param clip_denoised: if True, clip the denoised signal into [-1, 1]. clip이란 값이 튀지 않도록 잘라내는 것 -1는 완전 검정, +1은 완전 흰색
        :param denoised_fn: if not None, a function which applies to the
            x_start prediction before it is used to sample. Applies before
            clip_denoised.
        :param model_kwargs: if not None, a dict of extra keyword arguments to
            pass to the model. This can be used for conditioning. y(semantic map), s(guidance) 같은 부가인자
        :return: a dict with the following keys:
                 - 'mean': the model mean output.
                 - 'variance': the model variance output.
                 - 'log_variance': the log of 'variance'.
                 - 'pred_xstart': the prediction for x_0.
        """
        #모델 호출
        if model_kwargs is None:
            model_kwargs = {} #model_kwargs 초기화, 비어있으면 dictionary 생성

        B, C = x.shape[:2]

        assert t.shape == (B,) #배치 길이 일치 확인
        if 'y' in model_kwargs: # y 조건이 있을 때
            model_output = model(x, self._scale_timesteps(t), y=model_kwargs['y']) # 0~T를 0~1000 scale로
        else:
            model_output = model(x, self._scale_timesteps(t), **model_kwargs)

        if 's' in model_kwargs and model_kwargs['s'] > 1.0: #Classifer-free guidance
            model_output_zero = model(x, self._scale_timesteps(t), y=th.zeros_like(model_kwargs['y']))     # ε̂ = ε_θ(x_t|y) + s*(ε_θ(x_t|y) - ε_θ(x_t|∅))
            model_output[:, :3] = model_output_zero[:, :3] + model_kwargs['s'] * (model_output[:, :3] - model_output_zero[:, :3])

        if self.model_var_type in [ModelVarType.LEARNED, ModelVarType.LEARNED_RANGE]: #LEARNED μ, log σ² 로 더 유연하게, LEARNED_RANGE: 예측하되 최소~최대를 지정

            assert model_output.shape == (B, C * 2, *x.shape[2:])
            model_output, model_var_values = th.split(model_output, C, dim=1) #model_output : ε̂(또는 x̂₀), model_var_values : log σ² 후보.
            if self.model_var_type == ModelVarType.LEARNED:
                model_log_variance = model_var_values #네트워크가 출력한 log σ²를 그대로 사용
                model_variance = th.exp(model_log_variance) #KL loss로 적당히 맞춰지도록
            else: #범위제한 LEARNED_RANGE일 때
                min_log = _extract_into_tensor(
                    self.posterior_log_variance_clipped, t, x.shape
                ) #클립된 posterior log σ² (아주 작음)
                max_log = _extract_into_tensor(np.log(self.betas), t, x.shape) # max_log = log βₜ (상당히 큼)
                # The model_var_values is [-1, 1] for [min_var, max_var].
                frac = (model_var_values + 1) / 2 #[-1,1]을 [0,1]로 Remapping
                model_log_variance = frac * max_log + (1 - frac) * min_log #log σ²_θ = frac·max + (1-frac)·min → min ≤ log σ²_θ ≤ max 보장
                model_variance = th.exp(model_log_variance) #log variance를 다시 로그를 풀어 model_variance로
                #그러면 왜 log σ²로 예측함? σ²는 반드시 양수여야하는데 로그를 취하면 음수도 허용됨. + 로그를 취하면 덧셈 뺄셈으로 처리하여 overflow/underflow에 강함, 그레디언트 폭주 방지, 학습에 안정적임
        else: #모델이 분산을 직접 예측하지 않을 때 (고정 분산)
            model_variance, model_log_variance = {
                # for fixedlarge, we set the initial (log-)variance like so
                # to get a better decoder log likelihood.
                ModelVarType.FIXED_LARGE: ( # β_t(= forward 노이즈) 로 근사, posterior 값을 끼워 넣어 log-likelihood를 개선 → 다양성을 높이고 발산 방지
                    np.append(self.posterior_variance[1], self.betas[1:]), #t=0 을 posterior_variance[1]로 대체하여 σ²값을 너무 크게 두지 않고 조금 줄여서 품질을 높임 
                    np.log(np.append(self.posterior_variance[1], self.betas[1:])), # t >= 1은 그냥 betas[1: ]사용하여 σ²을 상대적으로 크게함
                ),
                ModelVarType.FIXED_SMALL: ( #노이즈를 적게 넣어 샤프하게 복원
                    self.posterior_variance,
                    self.posterior_log_variance_clipped,
                ),
            }[self.model_var_type]
            model_variance = _extract_into_tensor(model_variance, t, x.shape)
            model_log_variance = _extract_into_tensor(model_log_variance, t, x.shape) #_extract_into_tensor로 배치 shape 맞추기

        def process_xstart(x): #사용자 커스텀 후가공 함수가 있으면 실행
            if denoised_fn is not None: 
                x = denoised_fn(x)
            if clip_denoised: #값 폭주 방지 Clipping
                x = x.clamp(-1, 1)
                if 'mean' in model_kwargs and 'std' in model_kwargs: #(μ, σ) 원래대로 복원
                    x = (x - x.mean(dim=(2, 3), keepdim=True)) / x.std(dim=(2, 3), keepdim=True)
                    x = x * model_kwargs["std"][None, :, None, None] + model_kwargs["mean"][None, :, None, None]
            return x

        if self.model_mean_type == ModelMeanType.PREVIOUS_X:
            pred_xstart = process_xstart( # 후가공 + Clipping
                self._predict_xstart_from_xprev(x_t=x, t=t, xprev=model_output) #x̂₀ = (x_{t-1} - c2·x_t) / c1 , x_{t-1}에서 x̂₀ 예측 (역공식으로)
            )
            model_mean = model_output # model_output을 그대로 model_mean으로 사용
        elif self.model_mean_type in [ModelMeanType.START_X, ModelMeanType.EPSILON]:
            if self.model_mean_type == ModelMeanType.START_X: #START_X 모드는 네트워크가 Noise나 전단계를 거치지 않고 x0를 그대로
                pred_xstart = process_xstart(model_output)
            else: #Epsilon 모드 노이즈 ε를 바로 예측, x0는 수식으로 환산할 뿐 네트워크 출력 자체는 노이즈 값으로 출력
                pred_xstart = process_xstart(
                    self._predict_xstart_from_eps(x_t=x, t=t, eps=model_output)
                )
            model_mean, _, _ = self.q_posterior_mean_variance(
                x_start=pred_xstart, x_t=x, t=t
            )
        else:
            raise NotImplementedError(self.model_mean_type) #다른 type Error 처리

        assert (
            model_mean.shape == model_log_variance.shape == pred_xstart.shape == x.shape
        )# 네 개의 tensor 모두 동일한 shape여야 함
        return {
            "mean": model_mean, # μθ(x_{t-1}|x_t) x_{t-1}뽑을 때 쓰는 μ
            "variance": model_variance, # σ²θ 샘플 σ²
            "log_variance": model_log_variance, # log σ²θ
            "pred_xstart": pred_xstart, # x̂₀ 모델이 보기에 깨끗한 이미지 noise없는 이미지
        } 

    def _predict_xstart_from_eps(self, x_t, t, eps): #ε̂ 과 x_t가 주어졌을 때 노이즈가 없는 x̂₀ 복원 (역변환)
        assert x_t.shape == eps.shape
        return (
            _extract_into_tensor(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t #sqrt_recip_alphas_comprod =  1 / ᾱₜ
            - _extract_into_tensor(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * eps #sqrt_recipm1_alphas_cumprod =  √((1 - ᾱₜ) / ᾱₜ)
        ) # x̂₀ = (1/√ᾱ_t) * x_t  -  √((1-ᾱ_t)/ᾱ_t) * ε̂

    def _predict_xstart_from_xprev(self, x_t, t, xprev): # x_{t-1} 로부터 x̂₀ 역산 ,PREVIOUS_X 모드에서 사용
        assert x_t.shape == xprev.shape # x_t와 x_{t-1} 텐서 shape 동일한지 확인
        return (  # (xprev - coef2*x_t) / coef1
            _extract_into_tensor(1.0 / self.posterior_mean_coef1, t, x_t.shape) * xprev
            - _extract_into_tensor(
                self.posterior_mean_coef2 / self.posterior_mean_coef1, t, x_t.shape
            )
            * x_t
        ) # x̂₀ = (x_{t-1} - c2_t * x_t) / c1_t
          # where c1_t = posterior_mean_coef1[t]
          #       c2_t = posterior_mean_coef2[t]

    def _predict_eps_from_xstart(self, x_t, t, pred_xstart): # x̂₀ → ε̂ 역산 함수 , x̂₀와 현재 noise x_t를 이용해 노이즈 ε̂ 를 다시 계산
        # 왜 사용함? PREVIOUS_X 모드처럼 x̂₀만 output으로 내는 상황에서 KL 계산할 때 ε̂ 이 필요할 때 가 있음.  x̂₀ ↔ ε̂  자유롭게 변환 가능
        return (
            _extract_into_tensor(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t
            - pred_xstart
        ) / _extract_into_tensor(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) # ε̂ = ( 1 / √ᾱₜ · x_t  –  x̂₀ ) / √((1 − ᾱₜ) / ᾱₜ)

    def _scale_timesteps(self, t):
        if self.rescale_timesteps:
            return t.float() * (1000.0 / self.num_timesteps) #timestep tensor를 float로 변환 후 * (1000.0 / self.num_timesteps)로 모델이 다른 T로 학습할 때에도 0~1000로 상대 위치를 맞춰줌
        return t #rescale을 안하면 원본 t 그대로 반환

    def condition_mean(self, cond_fn, p_mean_var, x, t, model_kwargs=None):
        """
        Compute the mean for the previous step, given a function cond_fn that
        computes the gradient of a conditional log probability with respect to
        x. In particular, cond_fn computes grad(log(p(y|x))), and we want to
        condition on y.

        This uses the conditioning strategy from Sohl-Dickstein et al. (2015).
        
        이전 step의 평균 µ( p_θ(x_{t-1} | x_t) 의 평균)을 구하고 외부에서 제공된 cond_fn(조건부 로그확률의 그래디언트)을 이용해
        조건 y를 반영(조건부 샘플링)한다. cond_fn 은 ∇ₓ log p(y | x)를 반환해야 함 그래디언트를 이용해 샘플이 y를 만족하도록 유도
        """
        gradient = cond_fn(x, self._scale_timesteps(t), **model_kwargs)
        new_mean = (
            p_mean_var["mean"].float() + p_mean_var["variance"] * gradient.float() # μ̃ = μ_θ + σ_θ² · ∇_x log p(y | x_t) 분포의 평균을 그래디언트 방향으로 살짝 이동
        )
        return new_mean

    def condition_score(self, cond_fn, p_mean_var, x, t, model_kwargs=None):
        """
        Compute what the p_mean_variance output would have been, should the
        model's score function be conditioned by cond_fn.

        See condition_mean() for details on cond_fn.

        Unlike condition_mean(), this instead uses the conditioning strategy
        from Song et al (2020).
        """
        alpha_bar = _extract_into_tensor(self.alphas_cumprod, t, x.shape) 

        eps = self._predict_eps_from_xstart(x, t, p_mean_var["pred_xstart"]) #모델이 예측한 x̂₀ 로부터 ε̂ 를 역산.
        eps = eps - (1 - alpha_bar).sqrt() * cond_fn(
            x, self._scale_timesteps(t), **model_kwargs
        ) #Score-based 보정 ε̂ − √(1−ᾱ) · ∇log p : 안정성, Classfier-free Guidance-SDE sampler와 호환성 ↑, 매 step마다 x̂₀를 재계산하여 자연스러움

        out = p_mean_var.copy() # 	기존 dict(μ,σ²,x̂₀)을 복사해 수정 예정.
        out["pred_xstart"] = self._predict_xstart_from_eps(x, t, eps) # 보정된 ε̂′로부터 새로운 x̂₀′ 계산.
        out["mean"], _, _ = self.q_posterior_mean_variance( # x̂₀′를 토대로 새로운 μ′ 재계산.
            x_start=out["pred_xstart"], x_t=x, t=t
        )
        return out # μ′, σ² (변화 없음), x̂₀′ 가 담긴 dict 반환 (sampler가 사용)
        #condition_mean은 μ를 직접 이동시키는 데 반해, condition_score는 ε를 이동하여 x̂₀·μ를 재계산(자연스러움)
    def p_sample( #주어진 noisy 이미지 x_t에서 한 스텝 뒤인 x_{t-1}을 뽑아냄 (역환산)
        self,
        model, # U-Net (ε̂, x̂₀, or x_{t-1} 예측)
        x, # 현재 noisy 이미지 x_t
        t, #timesteps
        clip_denoised=True, #[-1,1]
        denoised_fn=None,
        cond_fn=None,
        model_kwargs=None, # y, s, mean, std ...
    ):
        """
        Sample x_{t-1} from the model at the given timestep.

        :param model: the model to sample from.
        :param x: the current tensor at x_{t-1}.
        :param t: the value of t, starting at 0 for the first diffusion step.
        :param clip_denoised: if True, clip the x_start prediction to [-1, 1].
        :param denoised_fn: if not None, a function which applies to the
            x_start prediction before it is used to sample.
        :param cond_fn: if not None, this is a gradient function that acts
                        similarly to the model.
        :param model_kwargs: if not None, a dict of extra keyword arguments to
            pass to the model. This can be used for conditioning.
        :return: a dict containing the following keys:
                 - 'sample': a random sample from the model.
                 - 'pred_xstart': a prediction of x_0.
        """
        out = self.p_mean_variance(
            model,
            x,
            t,
            clip_denoised=clip_denoised,
            denoised_fn=denoised_fn,
            model_kwargs=model_kwargs,
        )
        noise = th.randn_like(x) #Gaussian 분포의 노이즈 잡음 생성
        nonzero_mask = (
            (t != 0).float().view(-1, *([1] * (len(x.shape) - 1)))
        )  # no noise when t == 0
        if cond_fn is not None: #cond_fn이 None이 아니면 condition_mean 보정
            out["mean"] = self.condition_mean(
                cond_fn, out, x, t, model_kwargs=model_kwargs
            )
        sample = out["mean"] + nonzero_mask * th.exp(0.5 * out["log_variance"]) * noise # x_{t-1} Sample 생성 μ에 nonzero_mask * σ 추가 t=0 이면 더 이상 노이즈 추가 X
        #   x_{t-1} = μθ + σθ * ε,      ε ~ N(0, I)
        return {"sample": sample, "pred_xstart": out["pred_xstart"]} #pred_xstart 같은 step에서 예측된 x̂₀

    def p_sample_loop(
        self,
        model,
        shape,
        noise=None,
        clip_denoised=True,
        denoised_fn=None,
        cond_fn=None,
        model_kwargs=None,
        device=None,
        progress=False,
    ):
        """
        Generate samples from the model.

        :param model: the model module.
        :param shape: the shape of the samples, (N, C, H, W).
        :param noise: if specified, the noise from the encoder to sample.
                      Should be of the same shape as `shape`.
        :param clip_denoised: if True, clip x_start predictions to [-1, 1].
        :param denoised_fn: if not None, a function which applies to the
            x_start prediction before it is used to sample.
        :param cond_fn: if not None, this is a gradient function that acts
                        similarly to the model.
        :param model_kwargs: if not None, a dict of extra keyword arguments to
            pass to the model. This can be used for conditioning.
        :param device: if specified, the device to create the samples on.
                       If not specified, use a model parameter's device.
        :param progress: if True, show a tqdm progress bar.
        :return: a non-differentiable batch of samples.
        """
        final = None
        for sample in self.p_sample_loop_progressive( # 각 timestep 결과를 yield (밑에 있음)
            model,
            shape,
            noise=noise,
            clip_denoised=clip_denoised,
            denoised_fn=denoised_fn,
            cond_fn=cond_fn,
            model_kwargs=model_kwargs,
            device=device,
            progress=progress,
        ):
            final = sample # final 변수에 마지막 step dict 저장
        return final["sample"] #x_0을 반환

    def p_sample_loop_progressive(
        self,
        model,
        shape,
        noise=None,
        clip_denoised=True,
        denoised_fn=None,
        cond_fn=None,
        model_kwargs=None,
        device=None,
        progress=False,
    ):
        """
        Generate samples from the model and yield intermediate samples from
        each timestep of diffusion. sample을 생성하고 각 timestep 중간의 sample들을 yield

        Arguments are the same as p_sample_loop().
        Returns a generator over dicts, where each dict is the return value of
        p_sample().
        """
        if device is None: #Device 초기 상태 준비
            device = next(model.parameters()).device # 사용자가 device를 안 주면 모델 파라미터가 있는 GPU를 자동으로 따라감
        assert isinstance(shape, (tuple, list)) # shape 는 (B, C, H, W) 형태 tuple 혹은 list 여야 한다.
        if noise is not None:
            img = noise # noise가 있으면 사용자가 미리 준 x_t(가장 노이즈가 큰 단계)로
        else:
            img = th.randn(*shape, device=device) # noise가 없으면 torch.randn으로 초기화 𝓝(0,I) 샘플 생성
        if 'y' in model_kwargs: # y(semantic map)이 model_kwargs에 존재하면 현재 device로 이동시켜 mismatch를 방지함
            model_kwargs['y'] = model_kwargs['y'].to(device)
        indices = list(range(self.num_timesteps))[::-1] #timestep을 역순으로 만든 list

        if progress: # progress = True이면 실행
            # Lazy import so that we don't depend on tqdm.
            # tqdm이란 빠르고 확장 가능한 progress 막대 라이브러리 : python에서 Iterable 처리할 때 남은 작업량과 시간을 실시간으로 보여줌 그 모델 돌릴 때 남은 작업량 막대로 보여주는거 그건인듯 ㅋㅋ
            from tqdm.auto import tqdm #lazy import 평소에는 tqdm 라이브러리를 의존성 없이 두다가 막대가 실제로 필요할 때만 가져옴

            indices = tqdm(indices) #	기존 리스트 [T-1, …, 0] 를 tqdm 래퍼로 감싼다. → for i in indices: 루프를 돌 때 자동 진행 막대가 출력됨.
        for i in indices: # T-1 에서 0 으로 역순 (indices가 역순이니까)
            t = th.tensor([i] * shape[0], device=device) # 배치의 모든 이미지가 동일해야함 timestep(i) 에서
            with th.no_grad(): # 연산 그래프를 만들지 마셈 → 그래프 저장 안해서 GPU 메모리 줄어듦 (메모리 절약), 속도향상(그래디언트 추적 x 계산량 줄어듦), 샘플링 단계인데 학습 안해서 gradient 필요없음
                out = self.p_sample( #p_sample 호출
                    model,
                    img,
                    t,
                    clip_denoised=clip_denoised,
                    denoised_fn=denoised_fn,
                    cond_fn=cond_fn,
                    model_kwargs=model_kwargs,
                )
                yield out # 중간 결과 (샘플·x̂₀)를 바로 수신
                img = out["sample"] # x_{t-1}이 다음 반복의 x_t가 되어 갱신함. 루프가 이어지며 마지막엔 x_0 도출

    def ddim_sample( #DDIM(Denodising Diffusion Implicit Model) T = 1000 step을 다 안해도 적은 step만으로 똑같이 빠르게 Denoise해줌
        self,
        model,
        x,
        t,
        clip_denoised=True,
        denoised_fn=None,
        cond_fn=None,
        model_kwargs=None,
        eta=0.0,
    ):
        """
        Sample x_{t-1} from the model using DDIM.

        Same usage as p_sample().
        """
        out = self.p_mean_variance( #p_mean_variance를 호출하여 현재 x_t를 보고 x_{t-1}의 평균과 분산 x0 예측을 뽑아옴
            model,
            x,
            t,
            clip_denoised=clip_denoised,
            denoised_fn=denoised_fn,
            model_kwargs=model_kwargs,
        )
        if cond_fn is not None: #score guidance로 보정
            out = self.condition_score(cond_fn, out, x, t, model_kwargs=model_kwargs)

        # Usually our model outputs epsilon, but we re-derive it
        # in case we used x_start or x_prev prediction.
        eps = self._predict_eps_from_xstart(x, t, out["pred_xstart"]) #p_sample에서 예측한 x0을 이용해 ε 재구성  ε = (1 / √ᾱ_t) · x_t − x_0

        alpha_bar = _extract_into_tensor(self.alphas_cumprod, t, x.shape)
        alpha_bar_prev = _extract_into_tensor(self.alphas_cumprod_prev, t, x.shape)
        sigma = (
            eta
            * th.sqrt((1 - alpha_bar_prev) / (1 - alpha_bar))
            * th.sqrt(1 - alpha_bar / alpha_bar_prev)
        ) # η = 0 이면 Deterministic 한 ODE 경로. η = 1 이면 DDPM과 동일한 노이즈 분산 (DDPM은 DDIM 이전의 모델)
        # Equation 12.  mean_pred = √ᾱ_{t-1} · x0  +  √(1−ᾱ_{t-1}−σ_t²) · ε
        noise = th.randn_like(x) # z ~ 𝒩(0,I) η = 0이면 사용 X
        mean_pred = ( # mean_pred = √ᾱ_{t-1} · x0  +  √(1−ᾱ_{t-1}−σ_t²) · ε
            out["pred_xstart"] * th.sqrt(alpha_bar_prev)
            + th.sqrt(1 - alpha_bar_prev - sigma ** 2) * eps
        )
        nonzero_mask = (
            (t != 0).float().view(-1, *([1] * (len(x.shape) - 1)))
        )  # no noise when t == 0
        sample = mean_pred + nonzero_mask * sigma * noise
        return {"sample": sample, "pred_xstart": out["pred_xstart"]}

    def ddim_reverse_sample(
        self,
        model,
        x, # x_t
        t,
        clip_denoised=True,
        denoised_fn=None,
        model_kwargs=None,
        eta=0.0, # 무조건 η = 0 Deterministic ODE 사용
    ):
        """
        Sample x_{t+1} from the model using DDIM reverse ODE.
        """
        assert eta == 0.0, "Reverse ODE only for deterministic path" # 무조건 η = 0 Deterministic ODE 사용
        out = self.p_mean_variance( # p_mean_variance 호출하여 x0과  µₜ, σₜ²를 얻음
            model,
            x,
            t,
            clip_denoised=clip_denoised,
            denoised_fn=denoised_fn,
            model_kwargs=model_kwargs,
        )
        # Usually our model outputs epsilon, but we re-derive it
        # in case we used x_start or x_prev prediction.
        eps = ( # ε 재계산 (xₜ, x̂₀ → ε)
            _extract_into_tensor(self.sqrt_recip_alphas_cumprod, t, x.shape) * x
            - out["pred_xstart"]
        ) / _extract_into_tensor(self.sqrt_recipm1_alphas_cumprod, t, x.shape)  # eps = (x_t / √ᾱ_t  -  x0_hat) / √(1/ᾱ_t − 1)
        alpha_bar_next = _extract_into_tensor(self.alphas_cumprod_next, t, x.shape) # ᾱ_{t+1} 추출

        # Equation 12. reversed
        mean_pred = (
            out["pred_xstart"] * th.sqrt(alpha_bar_next)
            + th.sqrt(1 - alpha_bar_next) * eps
        ) # x_{t+1} = √ᾱ_{t+1}·x0_hat + √(1−ᾱ_{t+1})·eps 결과적으로 x_t 에서 x_{t+1}로

        return {"sample": mean_pred, "pred_xstart": out["pred_xstart"]}

    def ddim_sample_loop( #DDIM으로 한 번에 최종 이미지를 생성
        self,
        model,
        shape,
        noise=None,
        clip_denoised=True,
        denoised_fn=None,
        cond_fn=None,
        model_kwargs=None,
        device=None,
        progress=False,
        eta=0.0,
    ):
        """
        Generate samples from the model using DDIM.

        Same usage as p_sample_loop().
        """
        final = None
        for sample in self.ddim_sample_loop_progressive( #p_sample loop 랑 비슷함 모든 t를 역순으로 돌리며 중간 sample을 yield
            model,
            shape,
            noise=noise,
            clip_denoised=clip_denoised,
            denoised_fn=denoised_fn,
            cond_fn=cond_fn,
            model_kwargs=model_kwargs,
            device=device,
            progress=progress,
            eta=eta,
        ):
            final = sample # 루프가 끝나면 final은 t=0 단계의 x_0
        return final["sample"]

    def ddim_sample_loop_progressive(
        self,
        model,
        shape,
        noise=None,
        clip_denoised=True,
        denoised_fn=None,
        cond_fn=None,
        model_kwargs=None,
        device=None,
        progress=False, #tqdm 막대 표시 여부
        eta=0.0,
    ):
        """
        Use DDIM to sample from the model and yield intermediate samples from
        each timestep of DDIM.

        Same usage as p_sample_loop_progressive().
        """
        if device is None: # 모델이 이미 올라가 있는 GPU/CPU를 그대로 사용
            device = next(model.parameters()).device
        assert isinstance(shape, (tuple, list))
        if noise is not None:
            img = noise #사용자 지정 x_T
        else:
            img = th.randn(*shape, device=device) # 무작위 가우시안 x_T
            indices = list(range(self.num_timesteps))[::-1] #[T-1, ... , 0]

        if progress:
            # Lazy import so that we don't depend on tqdm.
            from tqdm.auto import tqdm

            indices = tqdm(indices)

        for i in indices: # i = T-1 , ... , 0
            t = th.tensor([i] * shape[0], device=device)
            with th.no_grad():
                out = self.ddim_sample(
                    model,
                    img,
                    t,
                    clip_denoised=clip_denoised,
                    denoised_fn=denoised_fn,
                    cond_fn=cond_fn,
                    model_kwargs=model_kwargs,
                    eta=eta,
                )
                yield out # {"sample": x_{t-1}, "pred_xstart": x̂0} 중간 결과 도출
                img = out["sample"] # 다음 루프의 x_t ← x_{t-1} 다음 스텝으로 넘어가서 다시 반복

    def _vb_terms_bpd( # 각 timestep t에서 KL을 bits-per-dimension(bpd) 단위로 계산해서 학습/평가에 쓰도록 리턴
        self, model, x_start, x_t, t, clip_denoised=True, model_kwargs=None
    ):
        """
        Get a term for the variational lower-bound.

        The resulting units are bits (rather than nats, as one might expect).
        This allows for comparison to other papers.

        :return: a dict with the following keys:
                 - 'output': a shape [N] tensor of NLLs or KLs.
                 - 'pred_xstart': the x_0 predictions.
        """
        true_mean, _, true_log_variance_clipped = self.q_posterior_mean_variance( # q(x_{t-1}|x_t,x0) = N(μ_q, σ_q² I)
            x_start=x_start, x_t=x_t, t=t
        ) # true posterior q: 𝒩(μ_q, σ_q²),   μ_q = coef1·x0 + coef2·x_t
        # μ_q = (β_t √ᾱ_{t-1} / (1-ᾱ_t)) · x0 + (√α_t (1-ᾱ_{t-1}) / (1-ᾱ_t)) · x_t
        # σ_q² = ((1-ᾱ_{t-1}) / (1-ᾱ_t)) · β_t
        out = self.p_mean_variance( # 모델 예측 p_mean_variance 사용 pθ(x_{t-1}|x_t) = N(μθ, σθ² I)
            model, x_t, t, clip_denoised=clip_denoised, model_kwargs=model_kwargs
        ) # μθ = (1/√α_t) * (x_t - (β_t / √(1-ᾱ_t)) * eps_theta)
        # σθ² = {β_t, 𝛽_big_t, or learned}
        kl = normal_kl( #KL Divergence
            true_mean, true_log_variance_clipped, out["mean"], out["log_variance"]
        )
        kl = mean_flat(kl) / np.log(2.0) # KL은 자연로그로 계산 되기 때문에 bit로 바꿔줌 로그 밑을 e 에서 2로 바꿔줌

        decoder_nll = -discretized_gaussian_log_likelihood(
            x_start, means=out["mean"], log_scales=0.5 * out["log_variance"]
        ) # t = 0 에서 KL 대신 실제 데이터 x0에 대한 negative log-likelihood 사용
        assert decoder_nll.shape == x_start.shape
        decoder_nll = mean_flat(decoder_nll) / np.log(2.0)

        # At the first timestep return the decoder NLL,
        # otherwise return KL(q(x_{t-1}|x_t,x_0) || p(x_{t-1}|x_t))
        output = th.where((t == 0), decoder_nll, kl) # t == 0 이면 decoder_nll 선택, 다른 값이면 KL divergence 선택
        return {"output": output, "pred_xstart": out["pred_xstart"]} # output = 배치별 bpd tensor

    def training_losses(self, model, x_start, t, model_kwargs=None, noise=None):
        """
        Compute training losses for a single timestep. 매 timestep 마다 training loss 계산

        :param model: the model to evaluate loss on.
        :param x_start: the [N x C x ...] tensor of inputs.
        :param t: a batch of timestep indices.
        :param model_kwargs: if not None, a dict of extra keyword arguments to
            pass to the model. This can be used for conditioning.
        :param noise: if specified, the specific Gaussian noise to try to remove.
        :return: a dict with the key "loss" containing a tensor of shape [N].
                 Some mean or variance settings may also have other keys.
        """
        if model_kwargs is None:
            model_kwargs = {} #model_kwargs 없으면 dict로 초기화
        if noise is None:
            noise = th.randn_like(x_start) # noise 없으면 표준 gaussian으로 초기화
        x_t = self.q_sample(x_start, t, noise=noise) # q_sample 호출하여 x_t = √ᾱ_t·x0 + √(1-ᾱ_t)·noise 생성

        terms = {}

        if self.loss_type == LossType.KL or self.loss_type == LossType.RESCALED_KL:
            terms["loss"] = self._vb_terms_bpd( # 
                model=model,
                x_start=x_start,
                x_t=x_t,
                t=t,
                clip_denoised=False,
                model_kwargs=model_kwargs,
            )["output"]
            if self.loss_type == LossType.RESCALED_KL: # 그냥 KL이나 NLL은 학습 과정마다 하나의 t를 샘플해오므로 VLB를 1/T만큼 줄인 것과 같으므로 다시 timestep만큼 곱하여 Rescaling 해준다.
                terms["loss"] *= self.num_timesteps
        elif self.loss_type == LossType.MSE or self.loss_type == LossType.RESCALED_MSE:
            model_output = model(x_t, self._scale_timesteps(t), y=model_kwargs['y'])

            if self.model_var_type in [ # 분산을 예측하는 Mode 일때
                ModelVarType.LEARNED,
                ModelVarType.LEARNED_RANGE,
            ]:
                B, C = x_t.shape[:2] # B = 배치, C = 채널 수

                assert model_output.shape == (B, C * 2, *x_t.shape[2:]) # assert로 2C 채널인지 확인, 분산학습 모델은 채널이 2C여야함
                model_output, model_var_values = th.split(model_output, C, dim=1) 
                # split으로 2개로 쪼갬 model_output: 분리한 첫 C채널((모델이 예측한 μ / ε / x₀ )
                # model_var_values : 분리한 뒤 C채널 (모델이 예측한 log σ²)
                # th.cat( , dim=1) 두 개의 tensor를  2 × C 채널로 합쳐 dim = 1로 이어줌 shape: (B, 2C, H, W) 재구성
                # Learn the variance using the variational bound, but don't let
                # it affect our mean prediction.
                frozen_out = th.cat([model_output.detach(), model_var_values], dim=1) # .detach()로 그래프에서 분리하여 gradient 전달 x 즉 μ는 고정, σ²는 학습 가능한 새 텐서
                if self.loss_type == LossType.RESCALED_MSE:
                    # Divide by 1000 for equivalence with initial implementation.
                    # Without a factor of 1/1000, the VB term hurts the MSE term.
                    # terms["vb"] *= self.num_timesteps / 1000.0
                    pass

            target = {
                ModelMeanType.PREVIOUS_X: self.q_posterior_mean_variance(
                    x_start=x_start, x_t=x_t, t=t
                )[0],
                ModelMeanType.START_X: x_start, # 원본 x0
                ModelMeanType.EPSILON: noise, # 샘플된 노이즈 ε
            }[self.model_mean_type]
            assert model_output.shape == target.shape == x_start.shape
            terms["loss"] = mean_flat((target - model_output) ** 2) # loss = ⟨‖ target − output ‖²⟩  
        else:
            raise NotImplementedError(self.loss_type)

        return terms # terms에 배치별 스칼라 lose tensor return

    def _prior_bpd(self, x_start): # variational lower-bound(VLB)의 첫항 KL(q(xT|x0)||p(xT))를 bpd 단위로 계산
        """
        Get the prior KL term for the variational lower-bound, measured in
        bits-per-dim.

        This term can't be optimized, as it only depends on the encoder.

        :param x_start: the [N x C x ...] tensor of inputs.
        :return: a batch of [N] KL values (in bits), one per batch element.
        """
        batch_size = x_start.shape[0] # 배치사이즈를 (N, C, H, W) 텐서로
        t = th.tensor([self.num_timesteps - 1] * batch_size, device=x_start.device) # self.num_timesteps - 1 = T-1 : 즉, x_T에 해당
        qt_mean, _, qt_log_variance = self.q_mean_variance(x_start, t) # q_mean_variance로 x_T = √ᾱ_T · x0 + √(1−ᾱ_T) · ε  , μ_q = √ᾱ_T · x0  ,  σ_q² = 1 − ᾱ_T  계산
        kl_prior = normal_kl( # KL 계산
            mean1=qt_mean, logvar1=qt_log_variance, mean2=0.0, logvar2=0.0
        )
        return mean_flat(kl_prior) / np.log(2.0) # 평균 취한 다음 bit로 변환

    def calc_bpd_loop(self, model, x_start, clip_denoised=True, model_kwargs=None): # x0에 대해 T-1 ~ 0 의 모든 timestep에 대하여 x0 MSE와 ε 오차 MSE를 [N x T]로 반환해 디버그, 로그에 활용
        """
        Compute the entire variational lower-bound, measured in bits-per-dim,
        as well as other related quantities.

        :param model: the model to evaluate loss on.
        :param x_start: the [N x C x ...] tensor of inputs.
        :param clip_denoised: if True, clip denoised samples.
        :param model_kwargs: if not None, a dict of extra keyword arguments to
            pass to the model. This can be used for conditioning.

        :return: a dict containing the following keys:
                 - total_bpd: the total variational lower-bound, per batch element.
                 - prior_bpd: the prior term in the lower-bound.
                 - vb: an [N x T] tensor of terms in the lower-bound.
                 - xstart_mse: an [N x T] tensor of x_0 MSEs for each timestep.
                 - mse: an [N x T] tensor of epsilon MSEs for each timestep.
        """
        device = x_start.device
        batch_size = x_start.shape[0]

        # vb = []
        xstart_mse = [] # 각 timestep t 에서 x̂0 MSE 저장
        mse = [] # 각 timestep t 에서 ε MSE 저장
        for t in list(range(self.num_timesteps))[::-1]: # t = T-1, … ,0
            t_batch = th.tensor([t] * batch_size, device=device)
            noise = th.randn_like(x_start)  # ε ∼ N(0,I)
            x_t = self.q_sample(x_start=x_start, t=t_batch, noise=noise)
            # Calculate VLB term at the current timestep
            with th.no_grad():
                out = self._vb_terms_bpd( # 모델이 추정한 x0를 얻기 위해서 호출 KL 값은 버리고 x0값만 얻음 (품질 측정용)
                    model,
                    x_start=x_start,
                    x_t=x_t,
                    t=t_batch,
                    clip_denoised=clip_denoised,
                    model_kwargs=model_kwargs,
                )
            xstart_mse.append(mean_flat((out["pred_xstart"] - x_start) ** 2)) # x̂₀ MSE 저장
            eps = self._predict_eps_from_xstart(x_t, t_batch, out["pred_xstart"]) # ε MSE 계산
            mse.append(mean_flat((eps - noise) ** 2)) # ε MSE 저장

        xstart_mse = th.stack(xstart_mse, dim=1)  # shape: [N, T]
        mse = th.stack(mse, dim=1) # shape: [N, T]
        # t축(두 번째 차원) 으로 쌓아 [배치 × 타임스텝] 형태 완성.
        return {
            "xstart_mse": xstart_mse,
            "mse": mse,
        }


def _extract_into_tensor(arr, timesteps, broadcast_shape):
    """
    Extract values from a 1-D numpy array for a batch of indices. 

    :param arr: the 1-D numpy array.
    :param timesteps: a tensor of indices into the array to extract.
    :param broadcast_shape: a larger shape of K dimensions with the batch
                            dimension equal to the length of timesteps. 맞춰야 할 배치 포함 N차원 모양 (ex. (batch, 1, 1, 1) -> [B,1,1,1])
    :return: a tensor of shape [batch_size, 1, ...] where the shape has K dims.
    """
    res = th.from_numpy(arr).to(device=timesteps.device)[timesteps].float() # Numpy view를 만들고 timesteps.devide와 동일한 device로 옮김, dtype 통일시킴
    while len(res.shape) < len(broadcast_shape): # rank 맞을 때 까지
        res = res[..., None] # B (batch) 하나뿐인 벡터를 차원 확장시킴
    return res.expand(broadcast_shape) # 메모리 복사 없이 원하는 모양으로 view 확장
