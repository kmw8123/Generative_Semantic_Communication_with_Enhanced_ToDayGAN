import numpy as np
import torch
from collections import OrderedDict
import util.util as util
from util.image_pool import ImagePool
from .base_model import BaseModel
from . import networks


class ComboGANModel(BaseModel):
    def name(self):
        return 'ComboGANModel'




    # 1. 초기화
    def __init__(self, opt):
        '''
        self.netG: 여러 도메인을 처리할 수 있는 공통 Generator
        self.netD: 각 도메인 쌍마다 처리할 수 있는 다중 Discriminator
        self.criterionGAN: Discriminator 3개 (Color, Texture, Gradient)에 대한 loss 평균
        self.criterionCycle: Cycle consistency loss (L1)
        self.criterionIdt: Identity loss (Downsampling 후 L1)
        self.lambda_*: 각 loss에 대한 가중치

        # ToDayGAN은 이 구조 위에 feature-wise Discriminator (3개)를 얹어 사용한다.
          따라서 netD.forward()는 각 Discriminator의 결과 (3개)를 튜플로 리턴하고 criterionGAN은 그 평균을 계산한다.
        '''
        super(ComboGANModel, self).__init__(opt)

        self.n_domains = opt.n_domains  # 이미지 도메인의 개수 (예: 낮/밤 → 2)
        self.DA, self.DB = None, None  # 현재 배치의 도메인 인덱스 (도메인 A, 도메인 B)

        # 입력 이미지 tensor 초기화 (크기: 배치 x 채널 x 높이 x 너비)
        self.real_A = self.Tensor(opt.batchSize, opt.input_nc, opt.fineSize, opt.fineSize)
        self.real_B = self.Tensor(opt.batchSize, opt.output_nc, opt.fineSize, opt.fineSize)

        # Generator 정의: encode-decode 기반 공유 네트워크 구조
        self.netG = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf,
                                      opt.netG_n_blocks, opt.netG_n_shared,
                                      self.n_domains, opt.norm, opt.use_dropout, self.gpu_ids)

        # 학습일 경우 Discriminator도 정의 (도메인마다 3-way feature 분기 포함 가능)
        if self.isTrain:
            self.netD = networks.define_D(opt.output_nc, opt.ndf, opt.netD_n_layers,
                                          self.n_domains, self.Tensor, opt.norm, self.gpu_ids)
            ############################################################################################################################
            ###                                                                                                                      ###
            ###                                           보완 2-1. Feature Discriminator 추가                                       ###
            ###                                                                                                                      ###
            ############################################################################################################################
            # Feature Discriminator 추가
            self.netD_feat = networks.FeatureDiscriminator(input_nc=512).to(self.device)
            self.optimizer_D_feat = torch.optim.Adam(self.netD_feat.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))

            # Feature loss 가중치(hyper-parameter) 추가
            self.lambda_feat = opt.lambda_feature if hasattr(opt, 'lambda_feature') else 1.0 # opt 객체에 lambda_feature 속성 값을 넣었으면 그대로 사용하고, 만약 아무것도 안 넣어져 있으면 1.0을 사용한다.




        # 학습 재시작 or 테스트인 경우 weight 로딩
        if not self.isTrain or opt.continue_train:
            which_epoch = opt.which_epoch
            self.load_network(self.netG, 'G', which_epoch)
            if self.isTrain:
                self.load_network(self.netD, 'D', which_epoch)

        if self.isTrain:
            # 최근 생성 이미지를 저장해두는 이미지 풀 (Discriminator의 학습 안정화)
            self.fake_pools = [ImagePool(opt.pool_size) for _ in range(self.n_domains)]

            # Loss 함수 정의
            self.L1 = torch.nn.SmoothL1Loss()  # 기본 L1 손실
            self.downsample = torch.nn.AvgPool2d(3, stride=2)  # ID loss를 위한 다운샘플러
            self.criterionCycle = self.L1  # Cycle Consistency Loss
            self.criterionIdt = lambda y,t : self.L1(self.downsample(y), self.downsample(t))  # ID Loss
            self.criterionLatent = lambda y,t : self.L1(y, t.detach())  # 잠재 공간 loss (선택적)

            # 3개의 Feature Discriminator 출력을 평균하는 GAN loss 함수
            self.criterionGAN = lambda r,f,v : (networks.GANLoss(r[0],f[0],v) + \
                                                networks.GANLoss(r[1],f[1],v) + \
                                                networks.GANLoss(r[2],f[2],v)) / 3

            # Optimizer 정의 (G/D 모두 Adam)
            self.netG.init_optimizers(torch.optim.Adam, opt.lr, (opt.beta1, 0.999))
            self.netD.init_optimizers(torch.optim.Adam, opt.lr, (opt.beta1, 0.999))

            # 손실 저장용 변수 초기화
            self.loss_D, self.loss_G = [0]*self.n_domains, [0]*self.n_domains
            self.loss_cycle = [0]*self.n_domains

            # 각 loss에 대한 가중치 설정
            self.lambda_cyc, self.lambda_enc = opt.lambda_cycle, (0 * opt.lambda_latent)
            self.lambda_idt, self.lambda_fwd = opt.lambda_identity, opt.lambda_forward

        # 네트워크 구조 요약 출력
        print('---------- Networks initialized -------------')
        print(self.netG)
        if self.isTrain:
            print(self.netD)
        print('-----------------------------------------------')






    # 2. 입력 설정
    def set_input(self, input):
        # 입력 데이터(batch)를 받아 내부 변수로 저장
        input_A = input['A']  # 도메인 A 이미지
        self.real_A.resize_(input_A.size()).copy_(input_A)
        self.DA = input['DA'][0]  # 도메인 A 인덱스

        if self.isTrain:
            input_B = input['B']  # 도메인 B 이미지
            self.real_B.resize_(input_B.size()).copy_(input_B)
            self.DB = input['DB'][0]  # 도메인 B 인덱스

        ############################################################################################################################
        ###                                                                                                                      ###
        ###                         보완 3-1. define_G 함수가 segmentation map을 조건부 입력으로 받도록 설정 정의                ###
        ###                                                                                                                      ###
        ############################################################################################################################
        self.seg_A = input['seg_A'] # shape: [B, C, H, W], one-hot encoded map
        self.seg_B = input['seg_B']



        self.image_paths = input['path']  # 이미지 경로 저장






    # 3. 테스트 실행: generator의 인코딩, 디코딩 과정을 수행
    def test(self):
        with torch.no_grad(): # 테스트용이므로 gradient 계산을 비활성화해 연산 효율을 높인다.
            self.visuals = [self.real_A]
            self.labels = ['real_%d' % self.DA]


            ############################################################################################################################
            ###                                                                                                                      ###
            ###                                            보완 3-2. Generator에 넣기 전 concat                                      ###
            ###                                                                                                                      ###
            ############################################################################################################################
            input_A_with_seg = torch.cat([self.real_A, self.seg_A], dim=1)
            # 입력 이미지(real_A)를 다양한 도메인으로 변환(encoding) 후 결과를 저장
            encoded = self.netG.encode(input_A_with_seg, self.DA)

            # 도메인 전체에 대해 변환 결과 생성
            for d in range(self.n_domains):
                # 옵션에 따라 autoencode: G(x) -> x    /  reconstruction: G -> F -> G     중 선택
                if d == self.DA and not self.opt.autoencode:
                    continue
                fake = self.netG.decode(encoded, d)
                self.visuals.append(fake)
                self.labels.append('fake_%d' % d)
                if self.opt.reconstruct:
                    rec = self.netG.forward(fake, d, self.DA)  # 재변환
                    self.visuals.append(rec)
                    self.labels.append('rec_%d' % d)




    ############################################################################################################################
    ###                                                                                                                      ###
    ###                                        보완 1-1. 중심 가중치 함수 추가                                               ###
    ###                                                                                                                      ###
    ############################################################################################################################
    def get_center_weight(self, H, W, sigma=0.25): # 중심에서 멀어질수록 weight가 작아지게 설정
        x = torch.linspace(-1, 1, W, device=self.real_A.device).repeat(H, 1)
        # x만 사용하여 좌우 중심 강조, 위아래(y)는 무시하여 연산 속도 up
        weight = torch.exp(-x**2 / (2 * sigma**2)) 
        return weight.unsqueeze(0).unsqueeze(0)  # shape: [1, 1, H, W]


    
    def get_image_paths(self):
        return self.image_paths






    # 4. Discriminator 학습
    def backward_D_basic(self, pred_real, fake, domain):
        # 진짜 이미지와 가짜 이미지의 dicriminator 출력 비교하여 loss 계산(Least Square GAN loss 사용 - 설명은 GAN 개념정리글 참고)
        pred_fake = self.netD.forward(fake.detach(), domain)
        loss_D = self.criterionGAN(pred_real, pred_fake, True) * 0.5
        loss_D.backward()
        return loss_D

    def backward_D(self):
        # 두 도메인에 대해 discriminator 손실 계산 및 역전파 -> discriminator를 업데이트한다는 뜻
        fake_B = self.fake_pools[self.DB].query(self.fake_B)
        self.loss_D[self.DA] = self.backward_D_basic(self.pred_real_B, fake_B, self.DB)

        fake_A = self.fake_pools[self.DA].query(self.fake_A)
        self.loss_D[self.DB] = self.backward_D_basic(self.pred_real_A, fake_A, self.DA)





    ############################################################################################################################
    ###                                                                                                                      ###
    ###             보완 3-1. define_G 함수가 segmentation map을 조건부 입력으로 받도록 변경, 보완 1, 2와 함께 적용          ###
    ###                                                                                                                      ###
    ############################################################################################################################
    # 5. Generator 학습 - 각 loss에 대한 설명은 GAN 개념정리글 참고
    def backward_G(self):
        # 1. 인코딩
        input_A_with_seg = torch.cat([self.real_A, self.seg_A], dim=1)
        input_B_with_seg = torch.cat([self.real_B, self.seg_B], dim=1)

        encoded_A = self.netG.encode(input_A_with_seg, self.DA)
        encoded_B = self.netG.encode(input_B_with_seg, self.DB)

        # 2. Identity Loss (중앙 가중치 적용)
         if self.lambda_idt > 0:
             center_weight = self.get_center_weight(H=self.real_A.shape[2], W=self.real_A.shape[3])
             idt_A = self.netG.decode(encoded_A, self.DA)
             loss_idt_A = torch.mean(center_weight * torch.abs(self.downsample(idt_A) - self.downsample(self.real_A)))
             idt_B = self.netG.decode(encoded_B, self.DB)
             loss_idt_B = torch.mean(center_weight * torch.abs(self.downsample(idt_B) - self.downsample(self.real_B)))
         else:
             loss_idt_A, loss_idt_B = 0, 0

        # 3. Generator output (야간 → 주간 변환)
        self.fake_B, feat_B = self.netG.decode(encoded_A, self.DB, return_feature=True)
        self.fake_A = self.netG.decode(encoded_B, self.DA)

        # 4. GAN Loss
        pred_fake_B = self.netD(self.fake_B, self.DB)
        self.loss_G[self.DA] = self.criterionGAN(self.pred_real_B, pred_fake_B, False)
        pred_fake_A = self.netD(self.fake_A, self.DA)
        self.loss_G[self.DB] = self.criterionGAN(self.pred_real_A, pred_fake_A, False)

        # 5. Feature Discriminator Loss (보완 2-1)
        if self.lambda_feat > 0:
            pred_feat_fake = self.netD_feat(feat_B)
            target_real = torch.ones_like(pred_feat_fake)
            loss_feat_G = torch.nn.functional.mse_loss(pred_feat_fake, target_real) * self.lambda_feat
        else:
            loss_feat_G = 0

        # 6. Cycle Consistency Loss
        rec_encoded_A = self.netG.encode(self.fake_B, self.DB)
        self.rec_A = self.netG.decode(rec_encoded_A, self.DA)
        self.loss_cycle[self.DA] = self.criterionCycle(self.rec_A, self.real_A)

        rec_encoded_B = self.netG.encode(self.fake_A, self.DA)
        self.rec_B = self.netG.decode(rec_encoded_B, self.DB)
        self.loss_cycle[self.DB] = self.criterionCycle(self.rec_B, self.real_B)

        # 7. Latent Consistency Loss
        if self.lambda_enc > 0:
            loss_enc_A = self.criterionLatent(rec_encoded_A, encoded_A)
            loss_enc_B = self.criterionLatent(rec_encoded_B, encoded_B)
        else:
            loss_enc_A, loss_enc_B = 0, 0

        # 8. Forward Consistency Loss
        if self.lambda_fwd > 0:
            loss_fwd_A = self.criterionIdt(self.fake_B, self.real_A)
            loss_fwd_B = self.criterionIdt(self.fake_A, self.real_B)
        else:
            loss_fwd_A, loss_fwd_B = 0, 0

        # 9. 최종 Loss 합산 및 Backward
        loss_G = self.loss_G[self.DA] + self.loss_G[self.DB] + \
                    (self.loss_cycle[self.DA] + self.loss_cycle[self.DB]) * self.lambda_cyc + \
                    (loss_idt_A + loss_idt_B) * self.lambda_idt + \
                    (loss_enc_A + loss_enc_B) * self.lambda_enc + \
                    (loss_fwd_A + loss_fwd_B) * self.lambda_fwd + \
                    loss_feat_G

        loss_G.backward()






    # 6. 전체 학습 최적화
    def optimize_parameters(self): # train.py에서 매 step마다 호출되는 핵심 함수
        # Discriminator에 real 이미지 전달하여 예측값 저장
        self.pred_real_A = self.netD.forward(self.real_A, self.DA)
        self.pred_real_B = self.netD.forward(self.real_B, self.DB)

        # Generator 학습, 업데이트 -> backward_G()
        self.netG.zero_grads(self.DA, self.DB)
        self.backward_G()
        self.netG.step_grads(self.DA, self.DB)

        # Discriminator 학습, 업데이트 -> backward_D()
        self.netD.zero_grads(self.DA, self.DB)
        self.backward_D()
        self.netD.step_grads(self.DA, self.DB)

        ############################################################################################################################
        ###                                                                                                                      ###
        ###                                        보완 2-1. Feature Discriminator 학습 루프 추가                                ###
        ###                                                                                                                      ###
        ############################################################################################################################
        # Feature Discriminator 학습
        self.netD_feat.zero_grad()

        with torch.no_grad():
            _, feat_real = self.netG.decode(encoded_B, self.DB, return_feature=True)  # Real feature
            _, feat_fake = self.netG.decode(encoded_A, self.DB, return_feature=True)  # Fake feature (detach X)

        pred_real = self.netD_feat(feat_real.detach())
        pred_fake = self.netD_feat(feat_fake.detach())

        loss_D_feat = 0.5 * (
            torch.nn.functional.mse_loss(pred_real, torch.ones_like(pred_real)) +
            torch.nn.functional.mse_loss(pred_fake, torch.zeros_like(pred_fake))
        )
        loss_D_feat.backward()
        self.optimizer_D_feat.step()






    # 7. 학습 상태 시각화
    def get_current_errors(self):
        # 현재 손실값들 추출
        extract = lambda l: [(i if type(i) is int or type(i) is float else i.item()) for i in l]
        D_losses, G_losses, cyc_losses = extract(self.loss_D), extract(self.loss_G), extract(self.loss_cycle)
        return OrderedDict([('D', D_losses), ('G', G_losses), ('Cyc', cyc_losses)])

    def get_current_visuals(self, testing=False):
        # 시각화용 이미지 (입력, 변환, 복원 결과 등)
        if not testing:
            self.visuals = [self.real_A, self.fake_B, self.rec_A, self.real_B, self.fake_A, self.rec_B]
            self.labels = ['real_A', 'fake_B', 'rec_A', 'real_B', 'fake_A', 'rec_B']
        images = [util.tensor2im(v.data) for v in self.visuals]
        return OrderedDict(zip(self.labels, images))






    # 8. 모델 저장 및 학습률 업데이트(일정 에폭 이후 학습률을 선형적으로 줄이는 decay 전략 적용)
    def save(self, label):
        # 모델 저장
        self.save_network(self.netG, 'G', label, self.gpu_ids)
        self.save_network(self.netD, 'D', label, self.gpu_ids)

    def update_hyperparams(self, curr_iter):
        # 학습률 선형 감소 적용
        if curr_iter > self.opt.niter:
            decay_frac = (curr_iter - self.opt.niter) / self.opt.niter_decay
            new_lr = self.opt.lr * (1 - decay_frac)
            self.netG.update_lr(new_lr)
            self.netD.update_lr(new_lr)
            print('updated learning rate: %f' % new_lr)

        # latent loss 가중치 점진 적용
        if self.opt.lambda_latent > 0:
            decay_frac = curr_iter / (self.opt.niter + self.opt.niter_decay)
            self.lambda_enc = self.opt.lambda_latent * decay_frac
