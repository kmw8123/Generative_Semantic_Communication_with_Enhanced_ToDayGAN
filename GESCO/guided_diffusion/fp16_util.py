"""
Helpers to train with 16-bit precision.
"""
'''
GESCO 훈련 파이프라인에서 16-bit 혼합 정밀도 학(Mixed Precision Training)을 지원하는 함수와 클래스(MixedPrecisionTrainer)에 대한 코드이다.
모델의 메모리 사용량을 줄이고 학습 속도를 높이면서 정확도를 유지하기 위한 방안을 제시.
'''

# Top 모듈의 느낌으로 작성된 코드이다. 전체 class와 parameter의 타입, 형식 선언 및 변환을 관장하는 여러가지 Top 함수와 선언부가 제시되어 있음!

import numpy as np
import torch as th
import torch.nn as nn
from torch._utils import _flatten_dense_tensors, _unflatten_dense_tensors

from . import logger

INITIAL_LOG_LOSS_SCALE = 20.0


def convert_module_to_f16(l):
    """
    Convert primitive modules to float16.
    """ # primitive modules를 바꾼다는 건 사용자가 임의로 설정한 각 layer의 특정 영역이 아니라 1d,2d,3d layer 전반적으로 다루겠다는 의미이다.
        # PyTorch에서 layer를 만들면 기본값이 float32으로 생성된다. 이를 float16으로 변환해 연산량을 절반으로 줄이겠다는 코드이다.
        # 당연히 이렇게 bit를 반으로 줄이면 작은 변화량에 대해서 정보가 손실되어 정확도가 줄어들 것이다. 이를 보완하고자 하단의 def MixedPrecisionTrainer에 backward()함수를 사용한다.
    if isinstance(l, (nn.Conv1d, nn.Conv2d, nn.Conv3d)): #isinstance함수 : (확인하고자 하는 인스턴스 값, 확인하고자 하는 데이터, 클래스 타입)
                                                        # 인스턴스와 타입이 같으면 True를 반환.
                                                        # Conv1d~3d는 각각 convolution layer의 차원을 의미(1d,2d,3d)
        l.weight.data = l.weight.data.half() # float32 -> float16으로 바꾸는 것은 곧 절반으로 줄이는 것과 같은 의미이므로 half()함수를 이용한다.
        if l.bias is not None:
            l.bias.data = l.bias.data.half() # bias 또한 마찬가지의 작업을 취한다.


def convert_module_to_f32(l):
    """
    Convert primitive modules to float32, undoing convert_module_to_f16().
    """ # 위에서 float16으로 바꾼 layer를 다시 float32로 복원하는 코드이다. 디버깅 시에는 기존 값과 출력값을 비교해야 하므로 동일 형식(float32)으로 변환하는 것이 작업에 용이하기에 복원한다.
    if isinstance(l, (nn.Conv1d, nn.Conv2d, nn.Conv3d)):
        l.weight.data = l.weight.data.float() # float()함수로 연산하면 결과값이 float32로 반환되므로 이를 이용한다.
        if l.bias is not None:
            l.bias.data = l.bias.data.float() # bias도 마찬가지로 복원한다.


def make_master_params(param_groups_and_shapes): # 하단의 def get_param_groups_and_shapes의 반환값을 input으로 받는다.
    '''
    [
      ( [(name1, param1), (name2, param2), ...], shape1 ),
      ( [(name3, param3), ...], shape2 )
    ] param_groups_and_shapes는 요래 생겼다.
    '''
    """
    Copy model parameters into a (differently-shaped) list of full-precision
    parameters.
    """ # 모델 parameter들을 full precision(float32형식)으로 복사해서 master parameter list를 만드는 코드
        # FP16으로 훈련할 때를 제외하고는 parameter 업데이트를 실제 값 기준으로 해야되므로 복사해둔다.
    master_params = []
    for param_group, shape in param_groups_and_shapes:
        master_param = nn.Parameter(
            _flatten_dense_tensors( # ()안의 값(텐서)들로 1행n열 텐서(가로로 쭉 이어진, 즉 flat한 텐서)를 만든다.
                [param.detach().float() for (_, param) in param_group] # param_group튜플 리스트의 각 튜플 ("이름", tensor)에서 뒤의 tensor 값에 대한 for 반복문 실행(이름부분은 _로 무시한다는 뜻)
                                                                       # param.detach()로 autograd()연산으로 gradient 추적 중인 parameter들에 대해 gradient 연산 성분을 제거하고 단순 데이터 텐서만 복사
            ).view(shape) # 맨 처음 input값인 param_groups_and_shapes의 shape 형태로 flatten tensor를 재구성한다. 즉 param_group의 원래 형식으로 복원하는 함수다.
        )
        master_param.requires_grad = True # requires_grad=True값을 갖는 개체만 gradient 연산을 수행하고 있다. 즉 새로이 만든 master_param 값을 gradient 연산의 input으로 추가하는 코드이다.
        master_params.append(master_param) # 결과 리스트에 하나씩 추가, for 반복문으로 위 과정을 반복.
    return master_params


def model_grads_to_master_grads(param_groups_and_shapes, master_params):
    """
    Copy the gradients from the model parameters into the master parameters
    from make_master_params().
    """ # 위에서 만든 master_params 리스트에 gradient 값을 추가하는 코드이다.
    for master_param, (param_group, shape) in zip(master_params, param_groups_and_shapes): # for 반복문 수행
        master_param.grad = _flatten_dense_tensors(
            [param_grad_or_zeros(param) for (_, param) in param_group] # param_grad_or_zeros()함수는 하단부에 정의되어 있음. 각 parameter들의 gradient값을 반환하는 함수로 gradient가 없으면 0을 반환.
        ).view(shape) # 위와 비슷한 역할의 형태 복원 함수이다.


def master_params_to_model_params(param_groups_and_shapes, master_params):
    """
    Copy the master parameter data back into the model parameters.
    """
    # Without copying to a list, if a generator is passed, this will silently not copy any parameters.
    for master_param, (param_group, _) in zip(master_params, param_groups_and_shapes):
        for (_, param), unflat_master_param in zip(param_group, unflatten_master_params(param_group, master_param.view(-1))): # 위에서 한 flatten을 되돌리는 unflatten함수 실행. 하단에 정의되어 있음.
                                                                                                                              # 이걸 master_params의 모든 값에 대해 반복
            param.detach().copy_(unflat_master_param) # 각 값들을 unflat_master_params로 복사한다.
            '''
            주의! 위에서 master_params를 .view(shape)로 flat하지 않게 원상복구 시킨 거 아니냐? 라고 되물을 수 있다. 그 말이 맞지만, 절반만 맞다.
            위에서 한 원상복구는 가장 바깥쪽의 형식, 즉 각 master_param값을 하나로 모은 master_params라는 리스트를 flat하지 않게 바꾼 거다.
            그러나 정작 master_param 값 자체는 그걸 만드는 for 반복문 안의 코드로 인해 flat하게 처리되어 있고 이것이 유지되고 있다.
            즉, master_params라는 묶음(리스트) 자체는 unflat하게 바꿨지만 그 안의 데이터값(master_param)들은 여전히 flat하다.
            ''' # 하단의 backward()나 optimizer.step() 등의 함수들은 서로 다른 종류의 parameter(ex: 가중치, 바이어스, 오차 등등) 성분들이 하나의 list로 연속되게 묶여있는 flat한 master_param을
                # 각 parameter shape(shape은 성분 개수를 의미한다고 생각하면 된다. ex: 3차원 벡터는 shape=3)에 맞게 구분하여 parameter각각을 input으로 사용할 수 있어야 한다. 그래서 위 과정을 수행하는 것임.


def unflatten_master_params(param_group, master_param):
    return _unflatten_dense_tensors(master_param, [param for (_, param) in param_group]) # unflatten_dense_tensors는 PyTorch 내장 함수이다.
                                                                                        # param_group 튜플 리스트에서 name 빼고 tensor값(parameter 데이터)만 뽑는 과정이다.(위에서 다룬 for문과 비슷한 구조)


def get_param_groups_and_shapes(named_model_params):
    named_model_params = list(named_model_params) # iterator 형식을 list로 변환(반복 연산에 용이)
    scalar_vector_named_params = (
        [(n, p) for (n, p) in named_model_params if p.ndim <= 1], # scalar(0차원, ndim = 0)와 1차원 vector(ndim = 1)에 대해서 params 생성
        (-1), # 여기의 -1은 추후 있을 view(-1) 함수로 1차원 행렬을 만들겠다는 의미로 이해하면 된다.(연산 오류 방지를 위한 padding작업이라 생각하자.)
    )
    matrix_named_params = (
        [(n, p) for (n, p) in named_model_params if p.ndim > 1], # ndim 2이상, 즉 2,3차원 vector들에 대해서 동일 작업 수행
        (1, -1), # 역시 나중에 쓰일 view()로 2차원 행렬을 만들겠다는 의미로 이해하면 된다.
    )
    return [scalar_vector_named_params, matrix_named_params]
    '''
    위의 함수들로 최종 named_model_params를 만들면 아래와 같
    [
        ( [ (name, param), ... ], (-1) ),       # 스칼라/벡터 파라미터들
        ( [ (name, param), ... ], (1, -1) )     # 행렬/텐서 파라미터들
    ]
    '''


def master_params_to_state_dict(
    model, param_groups_and_shapes, master_params, use_fp16
):
    if use_fp16:
        state_dict = model.state_dict() # FP16 훈련 때에는 기존 model의 state_dict를 그대로 가져온다.
                                        # 이 때 model은 float32 타입이었으므로 우리의 연산 결과인 unflatten_maste_param들(float 16)과 타입, 형식이 대응되지 않는다. 따라서 이를 수정하는 사전 작업이 수반된다.
        for master_param, (param_group, _) in zip(master_params, param_groups_and_shapes):
                                            # master_params : flatten된 float32 master parameter 리스트
                                            # param_groups_and_shapes : [ (param_group1, shape1), (param_group2, shape2) ] 형식의 리스트
                                            #                               각 param_group은 [("weight", W), ("bias", b)] 형식의 튜플 리스트
            # for A, B in zip(A_list, B_list): 형식으로 A와 B를 각각 대응되는 리스트에서 뽑아오는 for 반복문 구조. 이 때 (param_group, _)으로 뽑아오니 param_groups_and_shape에서 shape은 무시한다.
            for (name, _), unflat_master_param in zip(param_group, unflatten_master_params(param_group, master_param.view(-1))):
                                                # master_param : master_params의 각 데이터로 1차원 벡터(서로 다른 종류의 parameter 성분들의 연속체), float32 타입
                                                # .view(-1) : PyTorch 내장함수 .view(shape)는 데이터 복사 없이 입력 shape로 모양을 빠르게 재구성하는 함수이다.
                                                #               shape = -1로 설정하면 tensor의 원소 전체 개수는 유지하면서 모양을 1차원으로 바꾸라는 뜻이다.
                                                #               근데 master_param은 이미 1차원 벡터인데 뭘 또 .view(-1) 써서 1차원 벡터로 바꾼다는 거임? 이라고 물으신다면, 기존 master_param은
                                                #               같은 내용이래도 형식이 (1, 6) : 1행 6열 이런 느낌의 2차원 행렬 요소로 저장이 되어 있었다면, .view(-1)을 거치면
                                                #               (6) : 6개 요소 행벡터 이런 느낌으로 저장 형식이 사뭇 다른 느낌으로 다루어진다. 이는 우리가 볼 땐 그게 그거지만 컴퓨터가 연산할 때는 큰 차이다.
                                                
                assert name in state_dict # assert는 뒤의 조건문이 True인 경우에만 에러 발생없이 지나가는 boolean함수이다. 조건문을 만족하지 않는다면 에러가 발생하고 컴파일이 멈춘다.
                                          # 우선 name들이 state_dict에, 즉 기존 model값에 원래 있던 애들인지부터 검사하는 코드
                                          
                state_dict[name] = unflat_master_param # 업데이트한 unflat_master_param 값으로 state_dict를 수정하는 코드(결과 반영식)
    else:
        state_dict = model.state_dict() # FP32 훈련 때에는 FP16일 때보다 좀 더 간단하게 동일 작업을 수행할 수 있다. 우선 똑같이 model에서 state_dict를 그대로 불러오고
        for i, (name, _value) in enumerate(model.named_parameters()): # master_params가 model의 state_dict parmeter들과 알맞게 전부 대응되므로 name을 기준으로 state_dict업데이트 작업만 수행하면 끝이다.
            assert name in state_dict
            state_dict[name] = master_params[i]
    return state_dict


def state_dict_to_master_params(model, state_dict, use_fp16): # state_dict를 master_params와 같은 타입, 형식으로 재구성하는 함수다. 이는 직관적으로,
    if use_fp16:                            # FP16 훈련 때는 state_dict를 flatten된 master 구조로 재구성하는 작업이 자세하게 필요하겠지만,
        named_model_params = [
            (name, state_dict[name]) for name, _ in model.named_parameters()
        ]
        param_groups_and_shapes = get_param_groups_and_shapes(named_model_params)
        master_params = make_master_params(param_groups_and_shapes)
    else:                                   # FP32 훈련 때는 이미 타입과 형식이 알맞게 대응되므로 그냥 state_dict에서 바로 parameter들을 가져오면 끝이니 간단한 작업일 것이라는 예측을 어렵지 않게 해볼 수 있다.
        master_params = [state_dict[name] for name, _ in model.named_parameters()]
    return master_params


def zero_master_grads(master_params): # parameter들의 gradient 성분을 없애는 함수. 위에서 gradient 성분과 무관하게 연산하는 make_master_params 과정에서 사용된다.
    for param in master_params:
        param.grad = None


def zero_grad(model_params):
    for param in model_params:
        # Taken from https://pytorch.org/docs/stable/_modules/torch/optim/optimizer.html#Optimizer.add_param_group
        if param.grad is not None: # parameter에 gradient 성분이 존재할 경우
            param.grad.detach_() # gradient tensor를 autograd연산에서 분리시켜두어 불필요한 연산 작업이 수행을 방지한다.(메모리 부하 절감)
            param.grad.zero_() # backward에서 gradient tensor값이 누적연산되지 않게 하려면 매 autograd 연산 과정 수행 후에 tensor값을 0으로 초기화하는 작업이 필수적이다.


def param_grad_or_zeros(param): # 위에서 언급한 함수.
    if param.grad is not None:
        return param.grad.data.detach()
    else:
        return th.zeros_like(param)


class MixedPrecisionTrainer: # 훈련 main 코드
    def __init__( # parameter 선언 및 그룹화, float타입 변환 작업
        self,
        *,
        model,
        use_fp16=False,
        fp16_scale_growth=1e-3,
        initial_lg_loss_scale=INITIAL_LOG_LOSS_SCALE,
    ):
        self.model = model
        self.use_fp16 = use_fp16
        self.fp16_scale_growth = fp16_scale_growth

        self.model_params = list(self.model.parameters())
        self.master_params = self.model_params
        self.param_groups_and_shapes = None
        self.lg_loss_scale = initial_lg_loss_scale

        if self.use_fp16:
            self.param_groups_and_shapes = get_param_groups_and_shapes(
                self.model.named_parameters()
            )
            self.master_params = make_master_params(self.param_groups_and_shapes)
            self.model.convert_to_fp16()

    def zero_grad(self):
        zero_grad(self.model_params)

    def backward(self, loss: th.Tensor): # 위에서 언급한 float32 layer를 float16 layer로 변환함에 따라 생긴 부정확도를 보완하는 함수
        if self.use_fp16: # 이 값이 True라는 건 MixedPrecisionTraining을 하고 있다는 뜻
            loss_scale = 2 ** self.lg_loss_scale #lg_loss_scale은 초기값이 INITIAL_LOG_LOSS_SCALE = 20.0이므로, 처음에는 loss_scale = 2^20 = 1,048,576
            (loss * loss_scale).backward() # 기존 loss에 scale(가중치)를 곱해 loss값을 크게 한 후, 역전파 방식(backward함수)으로 선형 함수의 기울기를 구한다.
                                           # 역전파로 기울기를 구한다는 것은 데이터 선형 회귀에서 매우 흔히 쓰이는 알고리즘으로, 아래 링크에 자세히 설명되어 있으니 참고할 것!
                                           # https://velog.io/@cherry0319/%EA%B8%B0%EC%9A%B8%EA%B8%B0-%EA%B5%AC%ED%95%98%EA%B8%B0%EC%88%98%EC%B9%98%EB%AF%B8%EB%B6%84%EC%98%A4%EC%B0%A8%EC%97%AD%EC%A0%84%ED%8C%8C
                                           # 근데 이렇게 멋대로 loss값을 늘려버린 걸 방치하면 데이터가 변질되는 것이므로 원상복구를 해줘야겠죠? 그걸 하단의 def _optimize_fp16에서 하고 있다.
        else:
            loss.backward()

    def optimize(self, opt: th.optim.Optimizer):
        if self.use_fp16:
            return self._optimize_fp16(opt)
        else:
            return self._optimize_normal(opt)

    def _optimize_fp16(self, opt: th.optim.Optimizer):  # gradient를 모델에서 → master로 복사
                                                        # norm 계산 후 overflow 확인
                                                        # 문제가 없으면 scaling 복원: grad *= 1 / scale
                                                        # optimizer가 master_param 기준으로 step()
                                                        # master → 모델 파라미터로 복사 (float32 → float16)  
                                                        # lg_loss_scale 증가 (성공했으므로 더 큰 scale 시도)
        logger.logkv_mean("lg_loss_scale", self.lg_loss_scale)
        model_grads_to_master_grads(self.param_groups_and_shapes, self.master_params)
        grad_norm, param_norm = self._compute_norms(grad_scale=2 ** self.lg_loss_scale)
        if check_overflow(grad_norm):
            self.lg_loss_scale -= 1
            logger.log(f"Found NaN, decreased lg_loss_scale to {self.lg_loss_scale}")
            zero_master_grads(self.master_params)
            return False

        logger.logkv_mean("grad_norm", grad_norm)
        logger.logkv_mean("param_norm", param_norm)

        self.master_params[0].grad.mul_(1.0 / (2 ** self.lg_loss_scale)) # def MixedPrecisionTrainer의 backward()에서 늘린 loss값을 원상복구 하는 코드
                                                                        # loss을 loss_scale배만큼 증가 = gradient 증가이므로 이를 원상복구하려면 1/loss_scale만큼 다시 곱하면 된다.
        opt.step()
        zero_master_grads(self.master_params)
        master_params_to_model_params(self.param_groups_and_shapes, self.master_params)
        self.lg_loss_scale += self.fp16_scale_growth
        return True

    def _optimize_normal(self, opt: th.optim.Optimizer): # FP32 훈련 시에는 별도의 scale값이나 복사가 필요없으므로 위의 optimize연산을 이처럼 훨씬 간단히 수행할 수 있다.
        grad_norm, param_norm = self._compute_norms()
        logger.logkv_mean("grad_norm", grad_norm)
        logger.logkv_mean("param_norm", param_norm)
        opt.step()
        return True

    def _compute_norms(self, grad_scale=1.0): # parameter의 norm, gradient norm을 계산(이 값들은 추후 안전성 모니터링에 사용함)
        grad_norm = 0.0
        param_norm = 0.0
        for p in self.master_params:
            with th.no_grad():
                param_norm += th.norm(p, p=2, dtype=th.float32).item() ** 2
                if p.grad is not None:
                    grad_norm += th.norm(p.grad, p=2, dtype=th.float32).item() ** 2
        return np.sqrt(grad_norm) / grad_scale, np.sqrt(param_norm)

    def master_params_to_state_dict(self, master_params): # 저장 및 복원 시에 필요한 변환, 재구성 함수
        return master_params_to_state_dict(
            self.model, self.param_groups_and_shapes, master_params, self.use_fp16
        )

    def state_dict_to_master_params(self, state_dict): # 저장 및 복원 시에 필요한 변환, 재구성 함수
        return state_dict_to_master_params(self.model, state_dict, self.use_fp16)


def check_overflow(value):
    return (value == float("inf")) or (value == -float("inf")) or (value != value)
