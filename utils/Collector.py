import numpy as np
import os

class Collector(object):
    """
    A class to collect and store data
    """
    def __init__(self, run_prefix = ""):
        """
        :param run_prefix: The result lists will be stored in the format
        <run_prefix>_<metric_name>.npy
        """
        self.run_prefix = run_prefix
        self.data = {}
    
    def add(self, metric_name, value):
        """
        Adds a value to the collector under the specified metric name
        :param metric_name: The name of the metric
        :param value: The value to add
        """
        if metric_name not in self.data:
            self.data[metric_name] = []
        self.data[metric_name].append(value)
    
    def save(self, path='./'):
        """
        Saves the collected data to files
        Each metric will be saved in a separate file named <run_prefix>_<metric_name>.npy
        """
        if not os.path.exists(path):
            os.makedirs(path)

        for metric_name, values in self.data.items():
            filename = f"{self.run_prefix}_{metric_name}.npy"
            np.save(os.path.join(path, filename), np.array(values))
            print(f"Saved {len(values)} values for {metric_name} to {filename}")

    def get(self, metric_name):
        """
        Returns the collected values for the specified metric name
        :param metric_name: The name of the metric
        :return: List of collected values for the metric
        """
        return self.data.get(metric_name, [])
    