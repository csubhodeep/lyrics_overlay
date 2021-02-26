import unittest

from configs.make_config import Config
from pipeline.lib.defs import Job
from pipeline.lib.defs import Pipeline


class TestDefs(unittest.TestCase):
    def setUp(self) -> None:
        self.conf = Config(output_data_path=".", input_data_path=".")
        self.dummy_function = lambda x: x
        self.job = Job(func=self.dummy_function, conf=self.conf)
        self.pipeline = Pipeline(start_step=self.job)

    def test_modify_attribute(self) -> None:

        # 1. value of attributes must NOT be modifiable
        try:
            self.job._conf = "nikhil"
        except Exception as ex:
            assert str(ex) == "_conf is already set"

    def test_add_attribute(self) -> None:

        # 2. no attributes should be added on the fly
        try:
            self.job.y = 2
        except Exception as ex:
            assert str(ex) == "y is not a valid attribute"

    def test_make_pipeline_with_same_run_ids(self):
        run_id = getattr(self.pipeline, "run_id")
        # run_id = self.pipeline.run_id
        try:
            new_pipeline = Pipeline(start_step=self.job, unique_run_id=run_id)
        except Exception as ex:
            assert str(ex) == f"Pipeline with run-id: {run_id} already exists !"


if __name__ == "__main__":
    unittest.main()
