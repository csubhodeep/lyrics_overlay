import time
from collections import UserList
from datetime import timedelta
from pathlib import Path
from typing import Any
from typing import Callable
from typing import Iterable
from typing import Optional
from typing import Tuple
from uuid import uuid4

from configs.make_config import Config
from pipeline.lib.decorators import make_immutable


class Job:
    """This is a basic abstraction of a process (a function or a method of a class)
        that can be run as a step in a flow"""

    def __init__(self, func: Callable, conf: Config):
        """This function can be constructed using a callable object or function and a Config object"""
        assert isinstance(
            func, Callable  # type: ignore
        ), "'func' should be of type Callable"
        assert isinstance(conf, Config), "'conf' should be of type Config"
        self._func = func
        self._conf = conf

    @make_immutable(allowed_settable_attributes=("_func", "_conf", "_time_exec"))
    def __setattr__(self, key, value):
        self.__dict__[key] = value

    @property
    def name(self) -> str:
        return self._func.__name__

    @property
    def config(self) -> Config:
        return self._conf

    @property
    def time_exec(self) -> str:
        return str(timedelta(seconds=self._time_exec))

    @staticmethod
    def __clear_files(path: Path, exclude_files: Tuple[str, ...], run_id: str) -> None:
        for item in path.iterdir():
            if item.is_dir() and item.name == run_id:
                for ele in item.iterdir():
                    if not ele.is_dir():
                        if ele.name not in exclude_files:
                            ele.unlink(missing_ok=True)
                    else:
                        for i in ele.iterdir():
                            i.unlink(missing_ok=True)
                        ele.rmdir()
                item.rmdir()
            else:
                if item.name not in exclude_files and item.name.startswith(run_id):
                    item.unlink(missing_ok=True)
                else:
                    pass

    def clear_files(
        self, is_last_job: bool = False, exclude_files: Tuple[str, ...] = (".gitkeep",)
    ) -> None:
        if not is_last_job:
            self.__clear_files(
                path=self.config.output_data_path,
                exclude_files=exclude_files,
                run_id=self.config.run_id,
            )
        else:
            pass

    def __call__(self):
        msg = f"*** Running job: {self.name} ***"
        print("=" * len(msg))
        print(msg)
        start_time = time.time()
        is_successful = self._func(conf=self.config)
        self._time_exec = time.time() - start_time
        return is_successful


class Pipeline(UserList):
    """This class is aims to implement the behaviour of a DAG-like flow/pipeline.
        A typical example of a pipeline could be as shown below
            (X)->(Y)->(Z)
        where X, Y & Z are a "Job" each and "->" is to be read as 'is executed before'
        The main objective of the pipeline is to 'connect' a bunch of Jobs together."""

    __RUN_IDS: Tuple[str, ...] = tuple([])

    @classmethod
    def __register_run_id(cls, run_id: str):
        if run_id not in cls.__RUN_IDS:
            run_ids = list(cls.__RUN_IDS)
            run_ids.append(run_id)
            cls.__RUN_IDS = tuple(run_ids)
        else:
            raise Exception(f"Pipeline with run-id: {run_id} already exists !")

    @classmethod
    def __remove_run_id(cls, run_id: str):
        if run_id in cls.__RUN_IDS:
            run_ids = list(cls.__RUN_IDS)
            run_ids.remove(run_id)
            cls.__RUN_IDS = tuple(run_ids)

    def __init__(
        self,
        start_step: Optional[Job] = None,
        unique_run_id: str = "",
        list_of_steps: Optional[Tuple[Job, ...]] = None,
    ):
        """This class must be constructed either using one Job OR a collection of Jobs but NOT both"""
        super().__init__()

        if start_step and list_of_steps:
            raise Exception(
                "Pipeline must be constructed with either 'start_step' or 'list_of_steps' but not both"
            )

        if (not start_step) and (not list_of_steps):
            raise Exception(
                "Pipeline must be constructed with atleast 'start_step' or 'list_of_steps'"
            )

        if unique_run_id == "":
            self._run_id = str(uuid4())
        else:
            self._run_id = unique_run_id

        self.__register_run_id(self.run_id)

        if not list_of_steps:
            assert isinstance(start_step, Job), "step must be of type Job"

            if hasattr(start_step.config, "input_data_path"):
                assert (
                    start_step.config.input_data_path
                ), "First step must have a valid input configuration"
            else:
                raise Exception("First step must have a valid input configuration")

            start_step.config.run_id = self.run_id

            self.data.append(start_step)
        else:
            flg = True
            for step in list_of_steps:
                assert isinstance(step, Job), "step must be of type Job"
                # do one more check if it is the first job in the collection
                if flg:
                    if hasattr(step.config, "input_data_path"):
                        assert (
                            step.config.input_data_path
                        ), "First step must have a valid input configuration"
                    else:
                        raise Exception(
                            "First step must have a valid input configuration"
                        )
                    flg = False
                    step.config.run_id = self.run_id
                    self.data.append(step)
                else:
                    self.add_job(step)

    def __setattr__(self, key, value):
        """This function overrides the default method of the UserList class
        so that immutability of the '_run_id' attribute can be ensured.

        Args:
            key:
            value:

        Returns:
        """
        if key == "_run_id":
            if hasattr(self, key):
                raise Exception(f"value of {key} is already set")
            else:
                self.__dict__[key] = value
        elif key == "data":
            self.__dict__[key] = value
        else:
            raise Exception(f"{key} is not a valid attribute")

    @property
    def run_id(self) -> str:
        return self._run_id

    def add_job(self, step: Job) -> None:
        self.append(step)

    def append(self, step: Job) -> None:
        assert isinstance(step, Job), "step must be of type Job"
        assert step.name not in [
            job.name for job in self
        ], "same step cannot be a part of a pipeline"

        step.config.input_data_path = self[-1].config.output_data_path
        step.config.run_id = self[-1].config.run_id
        self.data.append(step)

    def clear(self, exclude_files: Tuple[str, ...] = (".gitkeep",)) -> None:
        for i, job in enumerate(self):
            job.clear_files(is_last_job=i == len(self) - 1, exclude_files=exclude_files)

        del self.data
        self.__remove_run_id(self.run_id)

    def insert(self, i: int, item) -> None:
        raise NotImplementedError

    def pop(self, i: int = ...) -> None:
        raise NotImplementedError

    def copy(self):
        raise NotImplementedError

    def sort(self, *args: Any, **kwds: Any) -> None:
        raise NotImplementedError

    def reverse(self) -> None:
        raise NotImplementedError

    def extend(self, other: Iterable) -> None:
        raise NotImplementedError

    def __call__(self):
        """This function ensures 'lazy' execution of the pipeline"""
        # make the collection of jobs immutable before executing each job
        self.data = tuple(self.data)
        print(f"Starting pipeline - {self.run_id}: ")
        print(self)
        start_time = time.time()
        for job in self:
            res = job()
            if not res:
                raise Exception(f"Step - {job.name} failed")
            print(f"This job took: {job.time_exec}")
        print("=================================")
        print(
            f"Pipeline completed successfully in {str(timedelta(seconds=time.time()-start_time))} !"
        )

    def __repr__(self):
        return " -> ".join([job.name for job in self])
