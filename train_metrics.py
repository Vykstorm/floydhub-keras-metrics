
from keras.callbacks import BaseLogger
from json import JSONEncoder

class FloydhubKerasCallback(BaseLogger):
    '''
    This class can be used as a callback object that can be passed to the method fit()
    when training your model (inside 'callbacks' argument)
    If it is used while your model is running on a floydhub server, training metrics
    will be plotted at real time under the 'Training metrics' panel.
    '''
    def __init__(self, mode='epoch', metrics=None, stateful_metrics=None):
        super().__init__(stateful_metrics)

        if mode not in ('epoch', 'batch'):
            raise ValueError('Mode parameter should be "epoch" or "batch"')

        if metrics is not None and not isinstance(metrics, (list, tuple)):
            raise ValueError('Metrics parameter should be a list of training metric names to track')

        if  stateful_metrics is not None and not isinstance(metrics, (list, tuple)):
            raise ValueError('Stateful metrics parameter should be a list of training metric names to track')

        self.mode = mode
        self.metrics = frozenset(metrics) if metrics is not None else None
        self.encoder = JSONEncoder()

    def report(self, metric, value, **kwargs):
        info = {'metric': metric, 'value': value}
        info.update(kwargs)
        print(self.encoder.encode(info))

    def on_batch_end(self, batch, logs):
        if not self.mode == 'batch':
            return

        metrics = frozenset(logs.keys()) - frozenset(['batch', 'size'])
        if self.metrics:
            metrics &= self.metrics
        for metric in metrics:
            self.report(metric, round(logs[metric].item(), 5), step=batch)

    def on_epoch_end(self, epoch, logs):
        if not self.mode == 'epoch':
            return

        metrics = frozenset(logs.keys())
        if self.metrics:
            metrics &= self.metrics
        for metric in metrics:
            self.report(metric, round(logs[metric].item(), 5), step=epoch)
