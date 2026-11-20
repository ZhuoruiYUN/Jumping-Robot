"""Bounded per-episode diagnostic records. No effect on physics, scoring or actor input."""


class GeometryAudit:
    def __init__(self):
        self.previous = None
        self.peak = None
        self.crossings = {}

    def sample(self, snapshot):
        error = snapshot["error_m"]
        for name, threshold in (("first_over_1mm", .001), ("first_over_1cm", .01)):
            if error > threshold and name not in self.crossings:
                self.crossings[name] = dict(previous=self.previous, current=snapshot)
        if self.peak is None or error > self.peak["error_m"]:
            self.peak = snapshot
        self.previous = snapshot

    def result(self):
        return dict(**self.crossings, peak=self.peak, last=self.previous)
