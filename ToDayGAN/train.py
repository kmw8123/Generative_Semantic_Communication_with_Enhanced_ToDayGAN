import time
from options.train_options import TrainOptions           # 커맨드라인 인자 파서 (ex. --niter, --batchSize 등)
from data.data_loader import DataLoader                 # 학습 데이터셋을 로드하는 클래스
from models.combogan_model import ComboGANModel         # Generator + 다중 Discriminator가 정의된 모델 클래스
from util.visualizer import Visualizer                  # 학습 중 결과를 시각화 (HTML/Visdom 등)

# 1. 학습 설정값 불러오기 (ex. 에폭 수, 배치 사이즈, 데이터 경로 등)
opt = TrainOptions().parse()

############################################################################################################################
###                                                                                                                      ###
###                                            보완 3-2. Generator에 넣기 전 concat                                      ###
###                                                                                                                      ###
############################################################################################################################
opt.seg_nc = 35
opt.input_nc += opt.seg_nc # Generator G에 RGB + segmentation을 같이 입력하도록 설정







# 2. 데이터셋 로드 (예: train0 = night 도메인, train1 = day 도메인)
dataset = DataLoader(opt)
print('# training images = %d' % len(dataset))

# 3. ComboGAN 모델 초기화 (ToDayGAN 구조 포함)
model = ComboGANModel(opt)
'''
ComboGAN 기반 모델을 불러와 초기화
내부적으로 Generator 𝐺, Discriminator 𝐷_C, 𝐷_T, 𝐷_G 등을 설정하고 weight 초기화
'''

# 4. 시각화 도구 초기화 (결과 이미지 및 loss 그래프 저장)
visualizer = Visualizer(opt)

# 전체 학습 iteration 수 카운터
total_steps = 0

# 5. 이어서 학습하는 경우, 이전 에폭에 맞게 하이퍼파라미터 재설정
if opt.which_epoch > 0:
    model.update_hyperparams(opt.which_epoch)

# 6. 본격적인 학습 루프 시작 (Epoch 단위)
for epoch in range(opt.which_epoch + 1, opt.niter + opt.niter_decay + 1):
    # opt.niter: 고정 학습률 구간
	# opt.niter_decay: 학습률 감소 구간
    epoch_start_time = time.time()      # 한 에폭당 시간 측정용
    epoch_iter = 0                      # 해당 에폭 내에서 진행된 iteration 수

    # 7. 배치 단위로 이미지를 받아 모델에 공급, 학습 루프
    for i, data in enumerate(dataset):
        iter_start_time = time.time()
        total_steps += opt.batchSize   # 전체 학습 데이터 수 누적
        epoch_iter += opt.batchSize    # 현재 에폭 내 진행된 데이터 수 누적

        # 7-1. 현재 배치를 모델에 전달 (야간/주간 이미지 한 쌍 포함)
        model.set_input(data)

        # 7-2. Generator, Discriminators 모두 학습 (Adversarial, Cycle, Identity loss 사용 -> 각각 L_GAN, L_cycle, L_identity)
        model.optimize_parameters()

        # 7-3. 일정 step마다 현재 생성 이미지들을 화면에 출력(이미지 저장 or Visdom으로 전송)
        if total_steps % opt.display_freq == 0:
            visualizer.display_current_results(model.get_current_visuals(), epoch)

        # 7-4. 일정 step마다 현재 손실값(error) 출력 또는 그래프로 저장
        if total_steps % opt.print_freq == 0:
            errors = model.get_current_errors()  # G, D_C, D_T, D_G 등의 loss 값 딕셔너리
            t = (time.time() - iter_start_time) / opt.batchSize  # 1 이미지당 연산 시간
            visualizer.print_current_errors(epoch, epoch_iter, errors, t)

            # Visdom 서버로 실시간 그래프 전송 (선택 사항)
            if opt.display_id > 0:
                visualizer.plot_current_errors(epoch, float(epoch_iter)/len(dataset), opt, errors)

    # 8. 지정된 주기마다 모델 체크포인트 저장
    if epoch % opt.save_epoch_freq == 0:
        print('saving the model at the end of epoch %d, iters %d' % (epoch, total_steps))
        model.save(epoch)  # 지정된 주기마다 Generator와 Discriminators weight 저장

    # 9. 한 에폭 종료 후 로그 출력
    print('End of epoch %d / %d \t Time Taken: %d sec' %
          (epoch, opt.niter + opt.niter_decay, time.time() - epoch_start_time))

    # 10. 학습률 감소 적용 (Epoch 수가 증가할수록 learning rate를 줄임)
    model.update_hyperparams(epoch)
