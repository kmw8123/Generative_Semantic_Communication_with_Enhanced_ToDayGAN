from .base_options import BaseOptions


class TrainOptions(BaseOptions):
    def initialize(self):
        BaseOptions.initialize(self)
        self.isTrain = True

        self.parser.add_argument('--continue_train', action='store_true', help='continue training: load the latest model')
        self.parser.add_argument('--which_epoch', type=int, default=0, help='which epoch to load if continuing training')
        self.parser.add_argument('--phase', type=str, default='train', help='train, val, test, etc (determines name of folder to load from)')

        self.parser.add_argument('--niter', required=True, type=int, help='# of epochs at starting learning rate (try 50*n_domains)')
        self.parser.add_argument('--niter_decay', required=True, type=int, help='# of epochs to linearly decay learning rate to zero (try 50*n_domains)')

        self.parser.add_argument('--lr', type=float, default=0.0002, help='initial learning rate for ADAM')
        self.parser.add_argument('--beta1', type=float, default=0.5, help='momentum term of ADAM')

        self.parser.add_argument('--lambda_cycle', type=float, default=10.0, help='weight for cycle loss (A -> B -> A)')
        self.parser.add_argument('--lambda_identity', type=float, default=0.0, help='weight for identity "autoencode" mapping (A -> A)')
        self.parser.add_argument('--lambda_latent', type=float, default=0.0, help='weight for latent-space loss (A -> z -> B -> z)')
        self.parser.add_argument('--lambda_forward', type=float, default=0.0, help='weight for forward loss (A -> B; try 0.2)')

        self.parser.add_argument('--save_epoch_freq', type=int, default=5, help='frequency of saving checkpoints at the end of epochs')
        self.parser.add_argument('--display_freq', type=int, default=100, help='frequency of showing training results on screen')
        self.parser.add_argument('--print_freq', type=int, default=100, help='frequency of showing training results on console')

        self.parser.add_argument('--pool_size', type=int, default=50, help='the size of image buffer that stores previously generated images')
        self.parser.add_argument('--no_html', action='store_true', help='do not save intermediate training results to [opt.checkpoints_dir]/[opt.name]/web/')


        ############################################################################################################################
        ###                                                                                                                      ###
        ###                                보완 2-1. Feature Discriminator의 optional hyper-parameter 추가                       ###
        ###                                                                                                                      ###
        ############################################################################################################################
        self.parser.add_argument('--lambda_feature', type=float, default=1.0, help='weight for feature-level GAN loss (default: 1.0)') # lambda_feature 추가




        ############################################################################################################################
        ###                                                                                                                      ###
        ###                    보완 3-1. define_G 함수가 segmentation map을 입력으로 받을 떄의 hyper-parameter 추가              ###
        ###                                                                                                                      ###
        ############################################################################################################################
        self.parser.add_argument('--seg_nc', type=int, default=19, help='number of segmentation map channels (e.g., one-hot 19 for Cityscapes)') # seg_nc 추가

