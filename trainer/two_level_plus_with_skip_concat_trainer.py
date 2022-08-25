from models.multi_scale_mutil_level_loss import MultiScaleMultiLevelLoss
from models.dccan_two_level_plus_with_skip_con_concate import DCCANShareTwoLevelPlusWithSkipConcat
from .base import Base


class TwoLevelPlusWithSkipConcatTrainer(Base):

    def __init__(self, args):
        super(TwoLevelPlusWithSkipConcatTrainer, self).__init__(args)
        # 定义模型
        self.model = DCCANShareTwoLevelPlusWithSkipConcat()
        self.loss_fn = MultiScaleMultiLevelLoss().cuda()
        super(TwoLevelPlusWithSkipConcatTrainer, self).trainer_initial(self.model, self.loss_fn)
        print('=========================>three level three scale trainer')

    def get_model_out(self, blur):
        output = self.model(blur)
        return output['I1']
