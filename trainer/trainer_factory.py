from trainer.two_level_plus_with_skip_concat_trainer import TwoLevelPlusWithSkipConcatTrainer


class TrainerFactory(object):

    def __init__(self, args):
        self._args = args

    def get_trainer(self, key):
        if key == 'TwoLevelPlusWithSkipConcat':
            return TwoLevelPlusWithSkipConcatTrainer(self._args)

        return None
