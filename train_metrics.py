
from keras.callbacks import BaseLogger
from json import JSONEncoder

class FloydhubKerasCallback(BaseLogger):
    def __init__(self, mode='epoch', stateful_metrics=None):
        super().__init__(stateful_metrics)

        if mode not in ('epoch', 'batch'):
            raise ValueError('Mode parameter should be "epoch" or "batch"')
        self.mode = mode
        self.encoder = JSONEncoder()

    def report(self, metric, value, **kwargs):
        info = {'metric': metric, 'value': value}
        info.update(kwargs)
        print(self.encoder.encode(info))

    def on_batch_end(self, batch, logs):
        if not self.mode == 'batch':
            return
        for metric in frozenset(logs.keys()) - frozenset(['batch', 'size']):
            self.report(metric, logs[metric].item(), step=batch)

    def on_epoch_end(self, epoch, logs):
        if not self.mode == 'epoch':
            return
        for metric in frozenset(logs.keys()):
            self.report(metric, logs[metric].item(), step=epoch)
