
from typing import Callable
from typing import List
from configs.make_config import Config


class Job:

	__slots__ = ('_conf', '_func')

	def __init__(self, func: Callable, conf: Config):
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
		self._func(conf=self.config)


class Pipeline:

	__slots__ = (
		'_jobs'
	)

	def __init__(self, start_step: Job):
		assert isinstance(start_step, Job), "step must be of type Job"
		assert start_step.config.input_data_path, "First step must have a valid input configuration"
		assert start_step.config.output_data_path, "First step must have a valid output configuration"

		self._jobs: List[Job] = [start_step]

	def add_job(self, step: Job) -> None:
		assert isinstance(step, Job), "step must be of type Job"
		assert step.name not in [job.name for job in self._jobs], "same step cannot be a part of a pipeline"

		step.config.set_input_data_path(self._jobs[-1].config.output_data_path)
		self._jobs.append(step)

	def __call__(self):
		for job in self._jobs:
			job()
		print("Pipeline completed successfully ! ")