import numpy as np
from StringModel import StringModel

class Recorder():
    def __init__(self, model: StringModel, fs: int, d0, dist):
        """
        Initializes a Recorder object.

        Args:
            model (StringModel): The string model object.
            fs (int): The sampling frequency.
            d0: The position of the recording (ranging from 0 to 1).
            dist: The distance from the recording position.
        """
        self.model = model
        self.fs = fs
        self.d0 = int(d0 * model.y.shape[0])
        self.dist = dist # m
        self.x_unit = model.L / model.x.shape[0]
        self.output = []

        # recording positions
        self.dd_pattern = np.zeros(model.y.shape[0])
        for i in range(len(self.model.y[1:-1])):
            self.dd_pattern[i] = np.sqrt(((i - self.d0) * self.x_unit) ** 2 + self.dist ** 2)

    def step_record(self):
        """
        Records a step of the string model.

        Calculates the sample at each recording position and appends it to the output list.
        """
        sample = self.model.y * (1 / (self.dd_pattern + 1e-8) ** 2)
        self.output.append(np.mean(sample[1:-1]))
    
    def export(self):
        """
        Exports the recorded samples.

        Returns:
            numpy.ndarray: An array containing the recorded samples.
        """
        return np.array(self.output)