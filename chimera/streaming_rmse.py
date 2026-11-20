"""Accumulation de l'erreur quadratique sans carrés débordants."""
import math


class StreamingRMSE:
    def __init__(self):
        self.scale = 0.0
        self.squared = 0.0
        self.nonfinite = None

    def update(self, errors):
        maximum = errors.abs().amax().item()
        if not math.isfinite(maximum):
            if math.isnan(maximum) or self.nonfinite is None:
                self.nonfinite = maximum
            return
        if self.nonfinite is not None or maximum == 0:
            return
        if maximum > self.scale:
            self.squared *= (self.scale / maximum) ** 2
            self.scale = maximum
        self.squared += (errors / self.scale).square().sum().item()

    def result(self, count):
        if not count:
            return None
        if self.nonfinite is not None:
            return self.nonfinite
        return self.scale * math.sqrt(self.squared / count)
