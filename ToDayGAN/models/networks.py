import torch
import torch.nn as nn
from torch.nn import init
import functools, itertools
import numpy as np
from util.util import gkern_2d


# Helper function: weight 초기화
# Conv → N(0, 0.02), BN → N(1, 0.02)
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
        if hasattr(m.bias, 'data'):
            m.bias.data.fill_(0)
    elif classname.find('BatchNorm2d') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


# normalization 선택 함수 ('batch' or 'instance')
def get_norm_layer(norm_type='instance'):
    if norm_type == 'batch':
        return functools.partial(nn.BatchNorm2d, affine=True)
    elif norm_type == 'instance':
        return functools.partial(nn.InstanceNorm2d, affine=False)
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm_type)







############################################################################################################################
###                                                                                                                      ###
###                         보완 3-1. define_G 함수가 segmentation map을 조건부 입력으로 받도록 설정 정의                ###
###                                                                                                                      ###
############################################################################################################################
# Generator 정의: 각 도메인별 Encoder/Decoder + 공유 block (optional)
def define_G(input_nc, output_nc, ngf, n_blocks, n_blocks_shared, n_domains, norm='batch', use_dropout=False, gpu_ids=[], seg_nc=0):
    norm_layer = get_norm_layer(norm_type=norm)
    use_bias = norm_layer.func == nn.InstanceNorm2d if isinstance(norm_layer, functools.partial) else norm_layer == nn.InstanceNorm2d

    # seg_nc만큼 입력 채널 증가
    input_nc += seg_nc

    # 전체 ResNet 블록 수를 encoding/decoding/shared로 분리
    n_blocks -= n_blocks_shared
    n_blocks_enc = n_blocks // 2
    n_blocks_dec = n_blocks - n_blocks_enc

    # 인코더/디코더/공유 블록에 들어갈 인자 정의
    dup_args = (ngf, norm_layer, use_dropout, gpu_ids, use_bias)
    enc_args = (input_nc, n_blocks_enc) + dup_args
    dec_args = (output_nc, n_blocks_dec) + dup_args

    if n_blocks_shared > 0:
        n_blocks_shdec = n_blocks_shared // 2
        n_blocks_shenc = n_blocks_shared - n_blocks_shdec
        shenc_args = (n_domains, n_blocks_shenc) + dup_args
        shdec_args = (n_domains, n_blocks_shdec) + dup_args
        plex_netG = G_Plexer(n_domains, ResnetGenEncoder, enc_args, ResnetGenDecoder, dec_args, ResnetGenShared, shenc_args, shdec_args) # 순서대로 Encoder, Decoder, (optional)공유 Block의 3개로 Generator G 구성
    else:
        plex_netG = G_Plexer(n_domains, ResnetGenEncoder, enc_args, ResnetGenDecoder, dec_args)

    if len(gpu_ids) > 0:
        assert(torch.cuda.is_available())
        plex_netG.cuda(gpu_ids[0])

    plex_netG.apply(weights_init)
    return plex_netG


# Discriminator 정의: 도메인별로 PatchGAN 기반 모델 구성
# RGB, Grayscale, Gradient 3개 feature branch 사용

def define_D(input_nc, ndf, netD_n_layers, n_domains, tensor, norm='batch', gpu_ids=[]):
    norm_layer = get_norm_layer(norm_type=norm)
    model_args = (input_nc, ndf, netD_n_layers, tensor, norm_layer, gpu_ids)
    plex_netD = D_Plexer(n_domains, NLayerDiscriminator, model_args)

    if len(gpu_ids) > 0:
        assert(torch.cuda.is_available())
        plex_netD.cuda(gpu_ids[0])

    plex_netD.apply(weights_init)
    return plex_netD








##############################################################################
# Classes
##############################################################################

# Relativistic Least Squares GAN loss (세 feature 평균)
def GANLoss(inputs_real, inputs_fake, is_discr):
    if is_discr:
        y = -1
    else:
        y = 1
        inputs_real = [i.detach() for i in inputs_real]  # G를 위한 backward 시 detach
    loss = lambda r,f : torch.mean((r-f+y)**2)
    losses = [loss(r,f) for r,f in zip(inputs_real, inputs_fake)]
    multipliers = list(range(1, len(inputs_real)+1));  multipliers[-1] += 1
    losses = [m*l for m,l in zip(multipliers, losses)]
    return sum(losses) / (sum(multipliers) * len(losses))

############################################################################################################################
###                                                                                                                      ###
###                                           보완 2-1. Feature Discriminator 정의                                       ###
###                                                                                                                      ###
############################################################################################################################
class FeatureDiscriminator(nn, module):
    def __init__(self, input_nc=512):
        super(FeatureDiscriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(input_nc, 256, kernel_size=3, stride=1, padding=1),      # input_nc는 netG에서 추출한 중간 feature 채널 수를 의미한다. 즉, 본 코드는 512 채널에서 256 채널로 중간 feature 추출하는 코드.
            nn.LeakyReLU(0.2, inplace=True),                                   # GAN 안정화
            nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1),           # 256 채널에서 128 채널로 다시 한 번 압축
            nn.LeakyReLU(0.2, inplace=True),                                 
            nn.Conv2d(128, 1, kernel_size=3, stride=1, padding=1)              # Patch-level에서 최종 real / fake 판별
            )
    def forward(self, x):
        return self.model(x)



# ResNet 기반 인코더 (downsampling → residual blocks)
class ResnetGenEncoder(nn.Module):
    def __init__(self, input_nc, n_blocks=4, ngf=64, norm_layer=nn.BatchNorm2d,
                 use_dropout=False, gpu_ids=[], use_bias=False, padding_type='reflect'):
        super(ResnetGenEncoder, self).__init__()
        self.gpu_ids = gpu_ids

        model = [
            nn.ReflectionPad2d(3),
            nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0, bias=use_bias),
            norm_layer(ngf),
            nn.PReLU()
        ]

        # 다운샘플링 (stride=2 conv)
        n_downsampling = 2
        for i in range(n_downsampling):
            mult = 2**i
            model += [
                nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1, bias=use_bias),
                norm_layer(ngf * mult * 2),
                nn.PReLU()
            ]

        # ResNet 블록
        mult = 2**n_downsampling
        for _ in range(n_blocks):
            model += [ResnetBlock(ngf * mult, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias, padding_type=padding_type)]

        self.model = nn.Sequential(*model)

    def forward(self, input):
        if self.gpu_ids and isinstance(input.data, torch.cuda.FloatTensor):
            return nn.parallel.data_parallel(self.model, input, self.gpu_ids)
        return self.model(input)




class ResnetGenShared(nn.Module):
    def __init__(self, n_domains, n_blocks=2, ngf=64, norm_layer=nn.BatchNorm2d,
                 use_dropout=False, gpu_ids=[], use_bias=False, padding_type='reflect'):
        assert(n_blocks >= 0)
        super(ResnetGenShared, self).__init__()
        self.gpu_ids = gpu_ids

        model = []
        n_downsampling = 2
        mult = 2**n_downsampling  # 다운샘플링으로 채널이 4배로 증가했다고 가정

        # 공유 블록에 들어갈 ResNet 블록들을 생성 (도메인 정보를 함께 받음)
        for _ in range(n_blocks):
            model += [ResnetBlock(ngf * mult, norm_layer=norm_layer, n_domains=n_domains,
                                  use_dropout=use_dropout, use_bias=use_bias, padding_type=padding_type)]

        # SequentialContext는 도메인 정보를 각 레이어에 주입하는 nn.Sequential 래퍼
        self.model = SequentialContext(n_domains, *model)

    def forward(self, input, domain):
        # 멀티 GPU 지원 (입력 텐서가 GPU에 있을 경우)
        if self.gpu_ids and isinstance(input.data, torch.cuda.FloatTensor):
            return nn.parallel.data_parallel(self.model, (input, domain), self.gpu_ids)
        return self.model(input, domain)


class ResnetGenDecoder(nn.Module):
    def __init__(self, output_nc, n_blocks=5, ngf=64, norm_layer=nn.BatchNorm2d,
                 use_dropout=False, gpu_ids=[], use_bias=False, padding_type='reflect'):
        assert(n_blocks >= 0)
        super(ResnetGenDecoder, self).__init__()
        self.gpu_ids = gpu_ids

        model = []
        n_downsampling = 2
        mult = 2**n_downsampling  # 업샘플링 전 채널 수

        # ResNet 블록들 (변환된 latent feature 처리)
        for _ in range(n_blocks):
            model += [ResnetBlock(ngf * mult, norm_layer=norm_layer,
                                  use_dropout=use_dropout, use_bias=use_bias, padding_type=padding_type)]

        # 업샘플링 단계 (ConvTranspose2d 사용)
        for i in range(n_downsampling):
            mult = 2**(n_downsampling - i)
            model += [
                nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2),
                                   kernel_size=4, stride=2,
                                   padding=1, output_padding=0,
                                   bias=use_bias),
                norm_layer(int(ngf * mult / 2)),
                nn.PReLU()
            ]

        # 마지막에 출력 채널 수를 맞추고, tanh로 정규화 (-1~1 사이)
        model += [nn.ReflectionPad2d(3),
                  nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0),
                  nn.Tanh()]

        self.model = nn.Sequential(*model)

    def forward(self, input):
        if self.gpu_ids and isinstance(input.data, torch.cuda.FloatTensor):
            return nn.parallel.data_parallel(self.model, input, self.gpu_ids)
        return self.model(input)



# ResNet 블록 정의 (skip connection 포함)
class ResnetBlock(nn.Module):
    def __init__(self, dim, norm_layer, use_dropout, use_bias, padding_type='reflect', n_domains=0):
        super(ResnetBlock, self).__init__()

        conv_block = []
        p = 0

        # 패딩 방식 지정
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1  # zero padding은 padding 파라미터로 지정
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)

        # 첫 번째 conv → norm → PReLU (+ 도메인 context 추가 가능)
        conv_block += [
            nn.Conv2d(dim + n_domains, dim, kernel_size=3, padding=p, bias=use_bias),
            norm_layer(dim),
            nn.PReLU()
        ]

        if use_dropout:
            conv_block += [nn.Dropout(0.5)]

        # 두 번째 conv 블록 구성
        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)

        conv_block += [
            nn.Conv2d(dim + n_domains, dim, kernel_size=3, padding=p, bias=use_bias),
            norm_layer(dim)
        ]

        # context-aware sequential 레이어
        self.conv_block = SequentialContext(n_domains, *conv_block)

    def forward(self, input):
        # context-aware forward: 도메인 context를 함께 처리
        if isinstance(input, tuple):
            return input[0] + self.conv_block(*input)
        return input + self.conv_block(input)



 # PatchGAN 기반 Discriminator 정의 (ToDayGAN에서 3가지 입력 방식 지원)
class NLayerDiscriminator(nn.Module):
    def __init__(self, input_nc, ndf=64, n_layers=3, tensor=torch.FloatTensor, norm_layer=nn.BatchNorm2d, gpu_ids=[]):
        super(NLayerDiscriminator, self).__init__()
        self.gpu_ids = gpu_ids

        # 필터 정의: gradient, downsample, blur 필터
        self.grad_filter = tensor([0,0,0,-1,0,1,0,0,0]).view(1,1,3,3)  # 수평 gradient 검출 필터
        self.dsamp_filter = tensor([1]).view(1,1,1,1)  # downsampling 필터 (stride=2)
        self.blur_filter = tensor(gkern_2d())  # Gaussian blur 필터

        # 세 가지 형태의 이미지에 대한 Discriminator 각각 정의
        self.model_rgb = self.model(input_nc, ndf, n_layers, norm_layer)  # RGB 이미지
        self.model_gray = self.model(1, ndf, n_layers, norm_layer)        # Grayscale 이미지
        self.model_grad = self.model(2, ndf, n_layers-1, norm_layer)      # Gradient 이미지 (x, y 방향)

    def model(self, input_nc, ndf, n_layers, norm_layer):
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        kw = 4  # 커널 크기
        padw = int(np.ceil((kw-1)/2))  # padding 설정

        sequences = [[
            nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw),
            nn.PReLU()
        ]]

        nf_mult = 1
        nf_mult_prev = 1
        # 중간 convolution layer들 (stride=2)
        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2**n, 8)
            sequences += [[
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult + 1,
                          kernel_size=kw, stride=2, padding=padw, bias=use_bias),
                norm_layer(ndf * nf_mult + 1),
                nn.PReLU()
            ]]

        # 마지막 output layer
        nf_mult_prev = nf_mult
        nf_mult = min(2**n_layers, 8)
        sequences += [[
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult,
                      kernel_size=kw, stride=1, padding=padw, bias=use_bias),
            norm_layer(ndf * nf_mult),
            nn.PReLU(),
            nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)
        ]]

        # SequentialOutput은 각 블록별 출력을 따로 반환할 수 있는 모듈
        return SequentialOutput(*sequences)

    def forward(self, input):
        # 입력 이미지를 3가지 형태로 변환
        blurred = torch.nn.functional.conv2d(input, self.blur_filter, groups=3, padding=2)
        gray = (.299*input[:,0,:,:] + .587*input[:,1,:,:] + .114*input[:,2,:,:]).unsqueeze_(1)  # grayscale 변환

        gray_dsamp = nn.functional.conv2d(gray, self.dsamp_filter, stride=2)
        dx = nn.functional.conv2d(gray_dsamp, self.grad_filter)  # x방향 gradient
        dy = nn.functional.conv2d(gray_dsamp, self.grad_filter.transpose(-2,-1))  # y방향 gradient
        gradient = torch.cat([dx,dy], 1)  # 2채널 (x,y) gradient 병합

        # 병렬 연산 지원 (멀티 GPU)
        if len(self.gpu_ids) and isinstance(input.data, torch.cuda.FloatTensor):
            outs1 = nn.parallel.data_parallel(self.model_rgb, blurred, self.gpu_ids)
            outs2 = nn.parallel.data_parallel(self.model_gray, gray, self.gpu_ids)
            outs3 = nn.parallel.data_parallel(self.model_grad, gradient, self.gpu_ids)
        else:
            outs1 = self.model_rgb(blurred)
            outs2 = self.model_gray(gray)
            outs3 = self.model_grad(gradient)
        return outs1, outs2, outs3






# Generator/Discriminator 공통 기반 클래스: 여러 네트워크를 동시에 다룸
class Plexer(nn.Module):
    def __init__(self):
        super(Plexer, self).__init__()

    def apply(self, func):
        # 모든 네트워크에 동일한 함수 적용 (ex: weight 초기화)
        for net in self.networks:
            net.apply(func)

    def cuda(self, device_id):
        # 모든 네트워크를 지정 GPU로 이동
        for net in self.networks:
            net.cuda(device_id)

    def init_optimizers(self, opt, lr, betas):
        # 모든 네트워크에 대한 개별 optimizer 생성
        self.optimizers = [opt(net.parameters(), lr=lr, betas=betas)
                           for net in self.networks]

    def zero_grads(self, dom_a, dom_b):
        # 지정한 도메인의 optimizer에 대해 gradient 초기화
        self.optimizers[dom_a].zero_grad()
        self.optimizers[dom_b].zero_grad()

    def step_grads(self, dom_a, dom_b):
        # 지정한 도메인의 optimizer에 대해 weight 업데이트 수행
        self.optimizers[dom_a].step()
        self.optimizers[dom_b].step()

    def update_lr(self, new_lr):
        # 모든 optimizer의 learning rate 갱신
        for opt in self.optimizers:
            for param_group in opt.param_groups:
                param_group['lr'] = new_lr

    def save(self, save_path):
        # 각 네트워크의 state_dict를 개별 파일로 저장
        for i, net in enumerate(self.networks):
            filename = save_path + ('%d.pth' % i)
            torch.save(net.cpu().state_dict(), filename)

    def load(self, save_path):
        # 저장된 state_dict 로드하여 각 네트워크에 복원
        for i, net in enumerate(self.networks):
            filename = save_path + ('%d.pth' % i)
            net.load_state_dict(torch.load(filename))


# Generator Plexer: 도메인별 인코더/디코더 쌍 + 공유 블록 지원
class G_Plexer(Plexer):
    def __init__(self, n_domains, encoder, enc_args, decoder, dec_args,
                 block=None, shenc_args=None, shdec_args=None):
        super(G_Plexer, self).__init__()
        self.encoders = [encoder(*enc_args) for _ in range(n_domains)]  # 도메인별 인코더
        self.decoders = [decoder(*dec_args) for _ in range(n_domains)]  # 도메인별 디코더

        # 공유 블록이 존재하면 shared encoder/decoder 정의
        self.sharing = block is not None
        if self.sharing:
            self.shared_encoder = block(*shenc_args)
            self.shared_decoder = block(*shdec_args)
            self.encoders.append(self.shared_encoder)
            self.decoders.append(self.shared_decoder)

        # 전체 네트워크 리스트 등록 (Plexer 내부용)
        self.networks = self.encoders + self.decoders

    def init_optimizers(self, opt, lr, betas):
        # 인코더 + 디코더를 묶어서 optimizer 초기화
        self.optimizers = []
        for enc, dec in zip(self.encoders, self.decoders):
            params = itertools.chain(enc.parameters(), dec.parameters())
            self.optimizers.append(opt(params, lr=lr, betas=betas))

    def forward(self, input, in_domain, out_domain):
        # 인코더 → (공유 블록) → 디코더를 거쳐 변환 수행
        encoded = self.encode(input, in_domain)
        return self.decode(encoded, out_domain)

    def encode(self, input, domain):
        output = self.encoders[domain].forward(input)
        if self.sharing:
            return self.shared_encoder.forward(output, domain)
        return output



    ############################################################################################################################
    ###                                                                                                                      ###
    ###                              보완 2-2. decode() 함수에서 feture를 함께 return하도록 수정                             ###
    ###                                                                                                                      ###
    ############################################################################################################################
    def decode(self, input, domain, return_feature=False):
        if self.sharing:
            input = self.shared_decoder.forward(input, domain)
        feature = input # 중간 feature 추출
        output = self.decoders[domain].forward(input)
        if return_feature:
            return output, feature
        # return self.decoders[domain].forward(input)
        return output # 위 코드처럼 return하면 결과를 direct하게 전달하는 것이라 따로 feature를 추출, 추가 정보 활용이 불가함. 본 줄처럼 작성해야 후속 작업에 용이하다.

    def zero_grads(self, dom_a, dom_b):
        # 도메인 A, B + 공유 블록(있을 경우) optimizer 초기화
        self.optimizers[dom_a].zero_grad()
        if self.sharing:
            self.optimizers[-1].zero_grad()
        self.optimizers[dom_b].zero_grad()

    def step_grads(self, dom_a, dom_b):
        # 도메인 A, B + 공유 블록(있을 경우) optimizer 업데이트
        self.optimizers[dom_a].step()
        if self.sharing:
            self.optimizers[-1].step()
        self.optimizers[dom_b].step()

    def __repr__(self):
        # 인코더/디코더 구조 출력 및 파라미터 수 요약
        e, d = self.encoders[0], self.decoders[0]
        e_params = sum([p.numel() for p in e.parameters()])
        d_params = sum([p.numel() for p in d.parameters()])
        return repr(e) + '\n' + repr(d) + '\n' + \
            'Created %d Encoder-Decoder pairs' % len(self.encoders) + '\n' + \
            'Number of parameters per Encoder: %d' % e_params + '\n' + \
            'Number of parameters per Deocder: %d' % d_params


# Discriminator Plexer: 도메인마다 별도의 판별기 (Discriminator) 구성
class D_Plexer(Plexer):
    def __init__(self, n_domains, model, model_args):
        super(D_Plexer, self).__init__()
        self.networks = [model(*model_args) for _ in range(n_domains)]  # 도메인별 D 구성

    def forward(self, input, domain):
        discriminator = self.networks[domain]
        return discriminator.forward(input)

    def __repr__(self):
        # 모델 구조 및 파라미터 수 요약
        t = self.networks[0]
        t_params = sum([p.numel() for p in t.parameters()])
        return repr(t) + '\n' + \
            'Created %d Discriminators' % len(self.networks) + '\n' + \
            'Number of parameters per Discriminator: %d' % t_params





# SequentialContext: 도메인 context 벡터를 conv layer 입력에 붙여주는 커스텀 Sequential
class SequentialContext(nn.Sequential):
    def __init__(self, n_classes, *args):
        super(SequentialContext, self).__init__(*args)
        self.n_classes = n_classes  # 도메인 개수 (예: 낮/밤 → 2)
        self.context_var = None     # 도메인 context 벡터 캐시

    def prepare_context(self, input, domain):
        # 입력과 동일한 spatial 크기의 one-hot context 벡터 생성
        if self.context_var is None or self.context_var.size()[-2:] != input.size()[-2:]:
            tensor = torch.cuda.FloatTensor if isinstance(input.data, torch.cuda.FloatTensor) \
                     else torch.FloatTensor
            self.context_var = tensor(*((1, self.n_classes) + input.size()[-2:]))

        # 도메인 인덱스 위치만 1.0, 나머지는 -1.0
        self.context_var.data.fill_(-1.0)
        self.context_var.data[:,domain,:,:] = 1.0
        return self.context_var

    def forward(self, *input):
        # 입력이 1개면 context가 없다고 보고 일반 sequential 수행
        if self.n_classes < 2 or len(input) < 2:
            return super(SequentialContext, self).forward(input[0])

        x, domain = input  # 입력 feature, 도메인 인덱스

        for module in self._modules.values():
            # Conv 계층에서는 context 채널을 concat
            if 'Conv' in module.__class__.__name__:
                context_var = self.prepare_context(x, domain) 
                x = torch.cat([x, context_var], dim=1)
            # ResnetBlock처럼 튜플을 인자로 받는 계층도 처리
            elif 'Block' in module.__class__.__name__:
                x = (x,) + input[1:]  # domain 정보까지 함께 넘김
            x = module(x)  # forward 연산 수행
        return x





# Sequential 모델 리스트를 블록 단위로 묶고 중간 출력값 추출 가능하도록 정의
class SequentialOutput(nn.Sequential):
    def __init__(self, *args):
        # args: 여러 블록의 리스트를 순서대로 nn.Sequential로 감쌈
        args = [nn.Sequential(*arg) for arg in args]
        super(SequentialOutput, self).__init__(*args)

    def forward(self, input):
        predictions = []
        layers = self._modules.values()
        for i, module in enumerate(layers):
            output = module(input)
            if i == 0:
                input = output
                continue
            predictions.append( output[:,-1,:,:] )  # 가장 마지막 채널을 추출 (판별 결과)
            if i != len(layers) - 1:
                input = output[:,:-1,:,:]  # 다음 layer의 입력은 마지막 채널 제외한 feature
        return predictions