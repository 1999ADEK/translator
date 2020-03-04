from .base_options import BaseOptions


class TrainOptions(BaseOptions):
    """This class includes training options.
    It also includes shared options defined in BaseOptions.
    """

    def initialize(self, parser):
        parser = BaseOptions.initialize(self, parser)
        # network saving and loading parameters
        parser.add_argument('--save_latest_freq', type=int, default=5, help='frequency of saving the latest results')
        parser.add_argument('--save_epoch_freq', type=int, default=10, help='frequency of saving checkpoints at the end of epochs')
        parser.add_argument('--continue_train', action='store_true', help='continue training: load the latest model')
        parser.add_argument('--num_epoch', type=int, default=100, help='train for this number of epochs')
        parser.add_argument('--epoch_count', type=int, default=0, help='the starting epoch count, save the model by <epoch_count>, <epoch_count>+<save_latest_freq>, ...')
        return parser