from collections import UserList
from typing import Callable
from typing import Iterable
from typing import Optional
from uuid import uuid4

from configs.make_config import Config


class Job:

	"""This is a basic abstraction of a process (a function or a method of a class) that can be run as a step in a flow"""

	def __init__(self, func: Callable, conf: Config):
		"""This function can be constructed using a callable object or function and a Config object"""
		self._func = func
		self._conf = conf

	@property
	def allowed_attributes(self):
		return ('_func', '_conf')

	def __setattr__(self, key, value):
		"""This function ensures immutability of every instance of this class"""
		if key in self.allowed_attributes:
			if hasattr(self, key):
				raise Exception(f"{key} is already set")
			else:
				self.__dict__[key] = value
		else:
			raise Exception(f"{key} is not a valid attribute")

	@property
	def name(self) -> str:
		return self._func.__name__

	@property
	def config(self) -> Config:
		return self._conf

	def __call__(self):
		return self._func(conf=self.config)


class Pipeline(UserList):
	"""This class is aims to implement the behaviour of a DAG-like flow/pipeline.
	A typical example of a pipeline could be as shown below
		(X)->(Y)->(Z)
	where X, Y & Z are a "Job" each and "->" is to be read as 'is executed before'
	The main objective of the pipeline is to 'connect' a bunch of Jobs together.
	"""

	def __init__(self, start_step: Optional[Job] = None, unique_run_id: str = "",
				 list_of_steps: Optional[Iterable[Job]] = None):
		"""This class must be constructed either using one Job OR a collection of Jobs but NOT both
		"""
		super().__init__()

		if start_step and list_of_steps:
			raise Exception("Pipeline must be constructed with either 'start_step' or 'list_of_steps' but not both")

		if (not start_step) and (not list_of_steps):
			raise Exception("Pipeline must be constructed with atleast 'start_step' or 'list_of_steps'")

		if unique_run_id == "":
			self._run_id: str = str(uuid4())
		else:
			self._run_id: str = unique_run_id

		if not list_of_steps:
			assert isinstance(start_step, Job), "step must be of type Job"
			assert start_step.config.input_data_path, "First step must have a valid input configuration"

			start_step.config.set_run_id(self.run_id)

			self.data.append(start_step)
		else:
			flg = True
			for step in list_of_steps:
				assert isinstance(step, Job), "step must be of type Job"
				# do one more check if the first job in the collection
				if flg:
					assert step.config.input_data_path, "First step must have a valid input configuration"
					flg = False

				step.config.set_run_id(self.run_id)

				self.data.append(step)

	@property
	def allowed_attributes(self):
		return ('_run_id', 'data')

	def __setattr__(self, key, value):
		"""This function overrides the default method of the UserList class
		so that immutability of the '_run_id' attribute can be ensured.
		"""
		if key in self.allowed_attributes:
			if key != 'data':
				if hasattr(self, key):
					raise Exception(f"value of {key} is already set")
				else:
					self.__dict__[key] = value
			else:
				self.__dict__[key] = value
		else:
			raise Exception(f"{key} is not a valid attribute")

	@property
	def run_id(self):
		return self._run_id

	def add_job(self, step: Job) -> None:
		self.append(step)

	def append(self, step: Job) -> None:
		assert isinstance(step, Job), "step must be of type Job"
		assert step.name not in [job.name for job in self], "same step cannot be a part of a pipeline"

		step.config.set_input_data_path(self[-1].config.output_data_path)
		step.config.set_run_id(self[-1].config.run_id)
		self.data.append(step)

	def __call__(self):
		"""This function ensures 'lazy' execution of the pipeline"""
		# make the collection of jobs immutable before executing each job
		self.data = tuple(self.data)
		print("Starting the following pipeline: ")
		print(self)
		for job in self:
			res = job()
			if not res:
				raise Exception(f"Step - {job.name} failed")
		print("Pipeline completed successfully ! ")

	def __repr__(self):
		return " -> ".join([job.name for job in self])
