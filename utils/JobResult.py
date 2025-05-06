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

    def __init__(self, service = None, job_id = None, job = None, result = None):
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

    def default_ibm_service(self, token = "a1a173ef5427a0e110ac33b0fb03add9d211ffae97a6eca6da26474feb765c2b722d2eb2460417c36411e35943f7fe7f2bd1ee2d180cef9d0ee71e180029f770"):
        """
        Creates a default ibm service using quantum experience token
        """
        self.service = QiskitRuntimeService(
            channel='ibm_quantum',
            instance='ibm-q/open/main',
            token='***'
        )
    
    def retrieve_result_with_id(self):
        """
        Try to retrieve job result from the service, if the job id is provided.

        This is a blocking call, it will wait until the job is finished.
        """
        if self.job_id is None:
            raise ValueError("Job ID does not exist")

        self.job = self.service.job(self.job_id)
        self.result = self.job.result()

    def retrieve_result(self):
        """
        Retrive the job result assuming we have a job object
        this is a blocking call, it will wait until the job is finished
        """
        if self.job is None:
            raise ValueError("Job does not exist")
        self.result = self.job.result()
        print("Job result retrieved")

    def save_job_to_file(self, path):
        """
        Save the job result to a file
        Will try to save the job result to one file and the id to another
        :param path: path save location
        """
        if self.job_id is not None:
            with open(self.job_id+".json", 'w') as f:
                json.dump(self.job_id, f)
                print(f"Job ID: {self.job_id} saved to {self.job_id}.json")
        

        if self.result is not None:
            with open(self.job_id+"_result.json", 'w') as f:
                json.dump(self.result, f, cls=RuntimeEncoder)
                print(f"Job result saved to {self.job_id+"_result.json"}.json")

    def load_result_from_id(self):
        """
        Will try to load job result from file
        """
        # check is file exists
        if self.job_id is None:
            raise ValueError("Provide a job id first")
        try:
            with open(self.job_id+".json", 'r') as f:
                result = json.load(file, cls=RuntimeDecoder)
                print(f"Job ID: {self.job_id} loaded from {self.job_id}.json")
        except FileNotFoundError:
            print(f"Job ID: {self.job_id} not found in {self.job_id}.json")
            result = None

    def run_job(self, circuit, run_options):
        """
        Run the job, does not wait for it to finish
        """
        self.job = self.service.run([circuit], **run_options)
        self.job_id = self.job.job_id()
        print(f"Job ID: {self.job_id} submitted")
    
    def check_job_status(self):
        """
        Check the job status
        """
        if self.job is None:
            raise ValueError("Job does not exist")
        return self.job.status()

            

    

        
        


    