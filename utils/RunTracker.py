"""
This is a tracker interface for tracking a series of jobs submitted to ibm,
for which we store the job type and job id for later retrieval of the results.
"""

import json
import os

from torch import Value

class RunTracker(object):
    """
    A class that internally stores type of jobs and their id for later retrieval
    from the IBM cloud service.
    """

    def verbose_print(self, message):
        """
        Print the message if verbose is set to True
        :param message: str
            The message to print
        """
        if self.verbose:
            print(message)

    def __init__(self, verbose=True):
        """
        :param verbose: bool
            If True, print messages to the console.
        """
        self.verbose = verbose
        self._data = {}
        self.temporal_counters = {}
        self.verbose_print("RunTracker initialized.")

    def add_job(self, job_type, job_id, target_state):
        """
        Add a job type and its corresponding job id to the tracker.
        
        :param job_type: str
            The type identifier of the job
        :param job_id: str
            The id of the job from ibm cloud (or other backend)
        :param
        """
        if job_type not in self.temporal_counters:
            self.temporal_counters[job_type] = 0
        save_name = f"{job_type}_{self.temporal_counters[job_type]}"
        self.temporal_counters[job_type] += 1
        
        self._data[save_name] = {'job_id':job_id, 'target_state':target_state}

        self.verbose_print(f"Added job: {save_name} with id: {job_id} and target state: {target_state}")
    
    def get_job_id(self, job_type, temporal_index):
        """
        Retrieve the job id for a given job type and a temporal index.
        
        :param job_type: str
            The type identifier of the job
        :param temporal_index: int
            The index of the job of that type, 0 for the first job, 1 for second, etc.
        :return: str

        :rasie: KeyError
            If the job type and temporal index combination does not exist.
        """
        save_name = f"{job_type}_{temporal_index}"
        if save_name not in self._data:
            raise KeyError(f"Job type '{job_type}' with index '{temporal_index}' not found.")
        data = self._data.get(save_name, None)
        if data is None:
            raise ValueError(f"Got None for the requested job.")
        else:
            return data.get('job_id', None)
        
    def get_target_state(self, job_type, temporal_index):
        """
        Retrieve the target state for a given job type and a temporal index.
        
        :param job_type: str
            The type identifier of the job
        :param temporal_index: int
            The index of the job of that type, 0 for the first job, 1 for second, etc.
        :return: str
            The target state of the job.
        
        :raises KeyError: If the job type and temporal index combination does not exist.
        """
        save_name = f"{job_type}_{temporal_index}"
        if save_name not in self._data:
            raise KeyError(f"Job type '{job_type}' with index '{temporal_index}' not found.")
        data = self._data.get(save_name, None)
        if data is None:
            raise ValueError(f"Got None for the requested job.")
        else:
            return data.get('target_state', None)

    def dump(self, savepath, file_name):
        """
        dump the tracked jobs to a json file.
        :param savepath: str
            The path to the directory where the file should be saved.
        :param file_name: str
            The name of the file to save the tracked jobs.
        """
        if not os.path.exists(savepath):
            self.verbose_print(f"Creating directory: {savepath}")
            os.makedirs(savepath)
        file_path = os.path.join(savepath, file_name+".json")
        with open(file_path, 'w') as f:
            json.dump(self._data, f, indent=4)
        self.verbose_print(f"Tracked jobs saved to {file_path}")
    
    def load(self, loadpath, file_name):
        """
        Load the tracked jobs from a json file.
        :param loadpath: str
            The path to the directory where the file is located.
        :param file_name: str
            The name of the file to load the tracked jobs from.
        
        :raises FileNotFoundError: If the file does not exist.
        """
        file_path = os.path.join(loadpath, file_name+".json")
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Tracked jobs file not found: {file_path}")
        with open(file_path, 'r') as f:
            self._data = json.load(f)
        self.verbose_print(f"Tracked jobs loaded from {file_path}")

    def __len__(self):
        """
        Return the number of tracked jobs.
        :return: int
            The number of tracked jobs.
        """
        return len(self._data)
    
    def __iter__(self):
        """
        Iterate over the tracked jobs.
        :return: iterator
            An iterator over the tracked jobs.
        """
        id_list = []

        for key in self._data.keys():
            value = self._data.get(key, None)
            if value is None:
                raise ValueError("Found None in value.")
            else:
                id_list.append(value['job_id'])
        
        return iter(id_list)