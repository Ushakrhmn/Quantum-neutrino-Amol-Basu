import json
import os
import numpy as np

class RunSaver(object):
    """
    A class designed to work with the SNB scripts. It saves various details about a run for future reference.
    """
    def __init__(self, name=None):
        self.name = name
        self.data = {}

    def save(self, name, value):
        """
        Saves a key-value pair in the data dictionary.
        """
        self.data[name] = value

    def save_multiple(self, pairs):
        """
        Save the multiple values by (name, value) pairs
        """
        for name, value in pairs:
            self.save(name, value)

    def log(self, path):
        """
        Log all collected data to json file under name.json
        :param path: The path dir to save the log file
        """

        save_path = os.path.join(path, f"{self.name}.json")
        if not os.path.exists(path):
            os.makedirs(path)

        try:
            with open(save_path, 'w') as f:
                json.dump(self.data, f, indent=4)
        except TypeError:
            for key, value in self.data.items():
                if isinstance(value, np.ndarray):
                    self.data[key] = value.tolist()
            with open(save_path, 'w') as f:
                json.dump(self.data, f, indent=4)