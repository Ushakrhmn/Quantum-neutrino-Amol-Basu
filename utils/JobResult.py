from qiskit_ibm_runtime import QiskitRuntimeService
from qiskit_ibm_runtime import RuntimeEncoder, RuntimeDecoder
import qiskit as qk
import json

"""
A class that encompass a quantum computer job and the acompanying results.
"""
class JobResult(object):
    """
    A class that encompass a quantum computer job and the acompanying results.
    Provides a straightforward interface to access the counts from a job.
    """

    def verbose_print(self, message):
        """
        Print the message if verbose is set to True
        :param message: str
            The message to print
        """
        if self.verbose:
            print(message)

    def __init__(self, service = None, job_id = None, job = None, result = None, verbose = False):
        """
        :param service: the quantum computing service, can be a sampler, estimator, or a simulator
        :param job_id: alternative constructor option, directly provide a job id
        :param job: alternative constructor, directly provide a job
        :param result: alternative constructor, directly provide a job result
        """
        self.service = service
        self.job_id = job_id
        self.job = job
        self.result = result
        self.verbose = verbose

    def get_service(self):
        return self.service

    def set_service(self, service):
        self.service = service

    def get_job_id(self):
        return self.job_id
    
    def set_job_id(self, job_id):
        self.job_id = job_id
    
    def get_job(self):
        return self.job
    
    def set_job(self, job):
        self.job = job
    
    def get_result(self):
        return self.result
    
    def set_result(self, result):
        self.result = result
    
    def retrieve_result_with_id(self):
        """
        Try to retrieve job result from the service, if the job id is provided.

        This is a blocking call, it will wait until the job is finished.
        """
        if self.job_id is None:
            raise ValueError("Job ID does not exist")

        self.job = self.service.job(self.job_id)
        self.result = self.job.result()
        return self.result

    def get_result_from_job(self):
        """
        Retrive the job result assuming we have a job object
        this is a blocking call, it will wait until the job is finished
        """
        if (self.job is None) and (self.result is None):
            raise ValueError("Job does not exist")
        if self.job is not None:
            self.result = self.job.result()
        else:
            return self.result
        self.verbose_print("Job result retrieved")
        return self.result

    def save_job_to_file(self, path):
        """
        Save the job result to a file
        Will try to save the job result to one file and the id to another
        :param path: path save location
        """
        if self.job_id is not None:
            with open(path+self.job_id+".json", 'w') as f:
                json.dump(self.job_id, f)
                self.verbose_print(f"Job ID: {self.job_id} saved to {self.job_id}.json")
        

        if self.result is not None:
            with open(path+self.job_id+"_result.json", 'w') as f:
                json.dump(self.result, f, cls=RuntimeEncoder)
                self.verbose_print(f"Job result saved to {self.job_id}"+"_result.json".json)

    def load_result_from_id(self, path):
        """
        Will try to load job result from file
        """
        # check is file exists
        if self.job_id is None:
            raise ValueError("Provide a job id firstW")
        try:
            with open(path+self.job_id+"_result.json", 'r') as f:
                self.result = json.load(f, cls=RuntimeDecoder)
                self.verbose_print(f"Job ID: {self.job_id} loaded from {self.job_id}._result.json")
        except FileNotFoundError:
            self.verbose_print(f"Job ID: {self.job_id} not found in {self.job_id}.json")
            self.result = None

    def run(self, circuit, run_options):
        """
        Run the job, does not wait for it to finish
        """
        self.job = self.service.run([circuit], **run_options)
        self.job_id = self.job.job_id()
        self.verbose_print(f"Job ID: {self.job_id} submitted")
    
    def check_job_status(self):
        """
        Check the job status
        """
        if self.job is None and self.job_id is None:
            raise ValueError("No Job or ID found.")
        if self.job is None:
            return self.service.job(self.job_id).status()
        else:
            return self.job.status()
        
    def load_job_from_id(self):
        """
        Load the job from clodu service with the job id
        """
        if self.job_id is None:
            raise ValueError("Job ID does not exist")
        
        self.job = self.service.job(self.job_id)
        self.verbose_print(f"Job ID: {self.job_id} loaded")
        return self.job
    
    def get_estimator_result(self):
        """
        Get the estimator result fromm the job result
        """
        if self.result is None:
            if self.job is None:
                raise ValueError("Result does not exist")
            else:
                self.get_result_from_job()
        
        return self.result[0].data.evs
    
    def get_counts(self):
        """
        Get the counts from the job result
        """
        if self.result is None:
            if self.job is None:
                raise ValueError("Result does not exist")
            else:
                self.get_result_from_job()

        count = None
        
        try:
            count = self.result.get_counts()
        except AttributeError:
            count = self.result[0].data.c.get_counts()
        finally:
            return count

    

            

    

        
        


    