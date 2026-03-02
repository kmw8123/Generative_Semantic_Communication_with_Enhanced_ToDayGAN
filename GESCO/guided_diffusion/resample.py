from abc import ABC, abstractmethod

import numpy as np
import torch as th
import torch.distributed as dist


def create_named_schedule_sampler(name, diffusion):
    """
    Create a ScheduleSampler from a library of pre-defined samplers.

    :param name: the name of the sampler.
    :param diffusion: the diffusion object to sample for.
    """
    if name == "uniform": # name == "uniform" 이면 모든 timestep을 동일 확률로 뽑음
        return UniformSampler(diffusion)
    elif name == "loss-second-moment": # name == "loss-second-momoent" 이면 최근 손실의 2차 모멘트(E[X^2])을 기반으로 가중치를 조정
        return LossSecondMomentResampler(diffusion)
    else:
        raise NotImplementedError(f"unknown schedule sampler: {name}") # 다른 name이 들어오면 바로 error 처리


class ScheduleSampler(ABC):
    """
    A distribution over timesteps in the diffusion process, intended to reduce
    variance of the objective.

    By default, samplers perform unbiased importance sampling, in which the
    objective's mean is unchanged.
    However, subclasses may override sample() to change how the resampled
    terms are reweighted, allowing for actual changes in the objective.
    """
    # 목표 함수 분산 감소를 위해 Diffusion 과정의 timestep을 뽑는 확률 분포를 정의.
    @abstractmethod # weights()가 반드시 하위 클래스에서 구현되어야 함을 강제시킴
    def weights(self): # 길이 : Diffusion step 수 인 1-D Numpy 배열을 반환해야하고, 모든 원소는 양수여야함
        """
        Get a numpy array of weights, one per diffusion step.

        The weights needn't be normalized, but must be positive.
        """

    def sample(self, batch_size, device):
        """
        Importance-sample timesteps for a batch.

        :param batch_size: the number of timesteps.
        :param device: the torch device to save to.
        :return: a tuple (timesteps, weights):
                 - timesteps: a tensor of timestep indices.
                 - weights: a tensor of weights to scale the resulting losses.
        """
        w = self.weights() # 원본 가중치 벡터를 가져옴
        p = w / np.sum(w) # 확률 분포로 정규화 시킴.
        indices_np = np.random.choice(len(p), size=(batch_size,), p=p) # batch_size 개의 timestep을 중복을 허용해서 뽑고 확률은 분포 p에 비례
        indices = th.from_numpy(indices_np).long().to(device) # Numpy 에서 Pytorch Longtensor 변환 후 Device로 이동
        weights_np = 1 / (len(p) * p[indices_np]) # 중요도 샘플링 보정 계수 : # w_i = 1 / (N * p(t_i)) N : 전체 timestep 수, sample() 함수가 편향되지 않도록 보정
        weights = th.from_numpy(weights_np).float().to(device) # 가중치를 float32 tensor로 변환하여 동일 device로 이동
        return indices, weights


class UniformSampler(ScheduleSampler): # SchedueSampler를 상속하여 모든 timesteps를 동일 확률로 뽑자!
    def __init__(self, diffusion):
        self.diffusion = diffusion
        self._weights = np.ones([diffusion.num_timesteps]) # 전체 timestep 수인 1-D Numpy 배열을 1로 채워 만들어 전부 동일한 가중치를 돌려주어 분포가 완전 Uniform하게 됨 
        #따라서 UniformSampler를 사용하면 가중치가 모두 1이 되어 평균 손실 그대로를 학습

    def weights(self):
        return self._weights # 외부에서 요청 시 그대로 가중치를 돌려줌


class LossAwareSampler(ScheduleSampler): # 최근 손실을 기반으로 가중치 분포를 바꿔줌
    def update_with_local_losses(self, local_ts, local_losses):
        """
        Update the reweighting using losses from a model.

        Call this method from each rank with a batch of timesteps and the
        corresponding losses for each of those timesteps.
        This method will perform synchronization to make sure all of the ranks
        maintain the exact same reweighting.

        각 GPU(rank)가 자신의 미니배치에서 얻은 (t, loss) 쌍을 전달하면 전 랭크가 동일한 재가중치 상태를 유지하도록 동기화(synchronization) 를 수행한다.

        :param local_ts: an integer Tensor of timesteps.
        :param local_losses: a 1D Tensor of losses.
        """
        batch_sizes = [
            th.tensor([0], dtype=th.int32, device=local_ts.device)
            for _ in range(dist.get_world_size()) #모든 rank의 배치 크기를 저장할 1-element 텐서를 리스트로 준비
        ]
        dist.all_gather(
            batch_sizes,
            th.tensor([len(local_ts)], dtype=th.int32, device=local_ts.device),
        ) # 현재 랭크의 배치 길이를 보내고 각 랭크의 길이를 모두 가져옴. Padding 크기를 맞추기 위해 필요

        # Pad all_gather batches to be the maximum batch size.
        batch_sizes = [x.item() for x in batch_sizes]
        max_bs = max(batch_sizes) # 최대 배치 길이를 계산

        timestep_batches = [th.zeros(max_bs).to(local_ts) for bs in batch_sizes] # max_bs 길이의 0-tensor를 만들어 배치를 Padding
        loss_batches = [th.zeros(max_bs).to(local_losses) for bs in batch_sizes]
        dist.all_gather(timestep_batches, local_ts) # all_gather로 timestep tensor와 loss tensor를 각각 모음
        dist.all_gather(loss_batches, local_losses)
        timesteps = [
            x.item() for y, bs in zip(timestep_batches, batch_sizes) for x in y[:bs]
        ] # Padding을 제거 후 batch_size만큼 잘라 Python list로 변환
        losses = [x.item() for y, bs in zip(loss_batches, batch_sizes) for x in y[:bs]] # 위와 동일
        self.update_with_all_losses(timesteps, losses) # 동기화가 끝난 후(모든 랭크가 동일한 데이터가 된 후) update_with_all_losses() 호출, 분포 업데이트가 각 process에서 동일하게 일어나도록 보장

    @abstractmethod # 하위 클래스가 반드시 update_with_all_losses()를 구현하게 강제.
    def update_with_all_losses(self, ts, losses):
        """
        Update the reweighting using losses from a model.

        Sub-classes should override this method to update the reweighting
        using losses from the model.

        This method directly updates the reweighting without synchronizing
        between workers. It is called by update_with_local_losses from all
        ranks with identical arguments. Thus, it should have deterministic
        behavior to maintain state across workers.

        :param ts: a list of int timesteps.
        :param losses: a list of float losses, one per timestep.
        """


class LossSecondMomentResampler(LossAwareSampler):
    def __init__(self, diffusion, history_per_term=10, uniform_prob=0.001):
        self.diffusion = diffusion
        self.history_per_term = history_per_term # 각 timestep마다 최근 10개의 loss를 보존할 버퍼의 길이
        self.uniform_prob = uniform_prob
        self._loss_history = np.zeros(
            [diffusion.num_timesteps, history_per_term], dtype=np.float64
        ) # shape = [T, N] 인 2-D 배열 생성 (T : timestep, N : history_per_term) 각 칸에는 최근 loss 값을 순서대로 저장
        self._loss_counts = np.zeros([diffusion.num_timesteps], dtype=np.int) # 각 timestep이 현재까지 수집한 loss 개수를 수집하는 1-D Counter

    def weights(self):
        if not self._warmed_up():
            return np.ones([self.diffusion.num_timesteps], dtype=np.float64)
        weights = np.sqrt(np.mean(self._loss_history ** 2, axis=-1)) # 각 timestep마다 loss의 제곱의 평균 후 루트를 취함, 표준편차를 추정해 학습 난이도로 사용
        weights /= np.sum(weights) # 확률 분포로 정규화
        weights *= 1 - self.uniform_prob # 분포에 uniform_prob을 섞어 exploration과 zero division 방지
        # exploration이란? : 아직 충분히 평가하지 못한 영역을 시도해 새로운 정보를 얻는 것 즉, 여기서는 explitation : 이미 좋은 것만을 쓰겠다!
        weights += self.uniform_prob / len(weights)
        return weights

    def update_with_all_losses(self, ts, losses):
        for t, loss in zip(ts, losses):
            if self._loss_counts[t] == self.history_per_term: #이미 버퍼가 가득 찬 timestep이면
                # Shift out the oldest loss term.
                self._loss_history[t, :-1] = self._loss_history[t, 1:] # 가장 오래된 값을 왼쪽으로 밀어내고
                self._loss_history[t, -1] = loss # 새 loss를 맨 뒤에 삽입
            else:
                self._loss_history[t, self._loss_counts[t]] = loss # 덜 찼으면 append하고
                self._loss_counts[t] += 1 # counter + 1

    def _warmed_up(self): # 모든 timestep이 history_per_term 개수만큼 loss를 모았는지 검사
        return (self._loss_counts == self.history_per_term).all() # .all : 전부 loss를 모아야 True 그 때부터 중요도 샘플링이 활성화
