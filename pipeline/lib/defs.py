
from typing import Callable
from typing import List

from uuid import uuid4

from configs.make_config import Config


class Job:
	"""This is a basic abstraction of a process (a function or a method of a class) that can be run as a step in a flow"""
	__slots__ = (
		'_conf',
		'_func',
		'_job_id'
	)

	def __init__(self,
				 func: Callable,
				 conf: Config):
		self._func = func
		self._conf = conf

	@property
	def name(self) -> str:
		return self._func.__name__

	@property
	def config(self) -> Config:
		return self._conf

	def set_config(self, new_conf: Config) -> None:
		self._conf = new_conf

	def __call__(self):
		return self._func(conf=self.config)


class Pipeline:
	"""This class is aims to implement the behaviour of a DAG-like flow/pipeline.
	A typical example of a pipeline could be as shown below
		(X)->(Y)->(Z)
	where X, Y & Z are a "Job" each.
	The main objective of the pipeline is to connect a bunch of Jobs together.
	"""

	__slots__ = (
		'_jobs',
		'_run_id'
	)

	def __init__(self, start_step: Job, unique_run_id: str = ""):
		assert isinstance(start_step, Job), "step must be of type Job"
		assert start_step.config.input_data_path, "First step must have a valid input configuration"
		assert start_step.config.output_data_path, "First step must have a valid output configuration"

		if unique_run_id == "":
			self._run_id: str = str(uuid4())
		else:
			self._run_id: str = unique_run_id

		start_step.config.set_run_id(self.run_id)

		self._jobs: List[Job] = [start_step]

	@property
	def run_id(self):
		return self._run_id

	@property
	def previous_job(self):
		return self._jobs[-1]

	def add_job(self, step: Job) -> None:
		assert isinstance(step, Job), "step must be of type Job"
		assert step.name not in [job.name for job in self._jobs], "same step cannot be a part of a pipeline"

		step.config.set_input_data_path(self.previous_job.config.output_data_path)
		step.config.set_run_id(self.previous_job.config.run_id)
		self._jobs.append(step)

	def __call__(self):
		print("Starting the following pipeline: ")
		print(self)
		for job in self._jobs:
			res = job()
			if not res:
				raise Exception(f"Step - {job.name} failed")
		print("Pipeline completed successfully ! ")

	def __repr__(self):
		return " -> ".join([job.name for job in self._jobs])