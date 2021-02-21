
import unittest

from configs.make_config import Config
from pipeline.lib.defs import Job
from pipeline.lib.defs import Pipeline


class TestDefs(unittest.TestCase):

	def setUp(self) -> None:
		self.conf = Config(".")
		self.dummy_function = lambda x: x
		self.job = Job(func=self.dummy_function, conf=self.conf)

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


if __name__ == "__main__":
	unittest.main()
