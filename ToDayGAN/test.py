import time
import os
from options.test_options import TestOptions             # 테스트용 커맨드라인 옵션
from data.data_loader import DataLoader                 # 입력 이미지 로딩
from models.combogan_model import ComboGANModel         # G + D 구조 포함된 모델
from util.visualizer import Visualizer                  # 이미지 시각화 및 저장 도구
from util import html                                   # HTML 결과 페이지 자동 생성기

# 1. 테스트용 설정값 읽기
opt = TestOptions().parse()
'''
opt는 커맨드라인에서 입력한 옵션들을 저장하는 객체이다.
    어떤 폴더에서 이미지를 읽을지 (--dataroot)
    어떤 에폭의 모델을 쓸지 (--which_epoch)
    몇 장의 이미지만 테스트할지 (--how_many)
    결과를 어디에 저장할지 (--results_dir) 등등
'''

# 2. 테스트는 단일 이미지(batch=1), 단일 스레드로만 동작하도록 설정
opt.nThreads = 1
opt.batchSize = 1

# 3. 데이터셋 로드(ex: test0 = day, test1 = night 이미지 폴더)
dataset = DataLoader(opt)

# 4. 학습된 Generator/Discriminator 모델 로드 -> 주로 generator만 test에 쓰일 예정이다.
model = ComboGANModel(opt)

# 5. 시각화 도구 준비
visualizer = Visualizer(opt)

# 6. 생성 이미지를 HTML 형태로 저장하기 위한 경로 지정(ex: results/robotcar_2day/test_150/)
web_dir = os.path.join(opt.results_dir, opt.name, '%s_%d' % (opt.phase, opt.which_epoch))
webpage = html.HTML(web_dir, 'Experiment = %s, Phase = %s, Epoch = %d' % 
                    (opt.name, opt.phase, opt.which_epoch))

# 7. show_matrix 옵션이 켜졌을 때, 결과를 행렬 형식으로 저장하기 위한 버퍼
vis_buffer = []

# 8. 테스트 이미지 순차 처리
for i, data in enumerate(dataset):

    # --how_many개수의 데이터만 테스트 하도록 최대 이미지 수 제한(serial_test=False일 때만)
    if not opt.serial_test and i >= opt.how_many:
        break

    # 8-1. 입력 배치(night/day 모두)를 모델에 설정
    model.set_input(data)

    # 8-2. Generator G를 통해 이미지 변환 수행(forward pass only = 밤->낮 변환 only)
    model.test()

    # 8-3. 밤->낮 변환된 이미지 결과 딕셔너리 가져오기
    visuals = model.get_current_visuals(testing=True) # visauls: input, output 이미지를 포함하는 딕셔너리

    # 8-4. 원본 이미지 경로 리스트 가져오기
    img_path = model.get_image_paths() # img_path: 현재 처리 중인 이미지의 경로 -> 로그 찍을 때 사용한다.
    print('process image... %s' % img_path)

    # 8-5. 결과 이미지(visuals를, 즉 input/output 모두를, 즉 night/day 모두를)를 HTML로 저장
    visualizer.save_images(webpage, visuals, img_path)

    # 8-6. (Option) 여러 도메인을 사용하는 모델일 경우 각 변환 결과를 하나의 matrix imgae로 저장(ex: test0→test1 변환 결과 matrix)
    if opt.show_matrix:
        vis_buffer.append(visuals)

        # 도메인 수만큼 이미지가 쌓이면 matrix image로 저장 후 초기화
        if (i+1) % opt.n_domains == 0:
            save_path = os.path.join(web_dir, 'mat_%d.png' % (i // opt.n_domains))
            visualizer.save_image_matrix(vis_buffer, save_path)
            vis_buffer.clear()

# 9. HTML 페이지 최종 저장
webpage.save()
