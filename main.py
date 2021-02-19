
from configs.make_config import get_config
from optimizer.optimize import optimize
from overlayer.overlay import overlay
from overlayer.upload import upload_video
from person_box_detector.inference import detect_persons
from pipeline.lib.defs import Job
from pipeline.lib.defs import Pipeline
from pre_processor.data_fetcher import fetch_data
from pre_processor.sampler import sample
from splitter.splitter import split



if __name__ == "__main__":

	# get all details from config file
	collection_of_configs = get_config(path_to_config="./configs/config.json")

	# declare jobs
	fetch_data_step = Job(func=fetch_data, conf=collection_of_configs['fetch_data'])
	sample_step = Job(func=sample, conf=collection_of_configs['sample'])
	detect_persons_step = Job(func=detect_persons, conf=collection_of_configs['detect_persons'])
	split_step = Job(func=split, conf=collection_of_configs['split'])
	optimization_step = Job(func=optimize, conf=collection_of_configs['optimization'])
	overlay_step = Job(func=overlay, conf=collection_of_configs['overlay'])
	upload_step = Job(func=upload_video, conf=collection_of_configs['upload'])

	# instantiate a pipeline
	pipeline = Pipeline(start_step=fetch_data_step)
	# add steps to the pipeline - the sequence is important
	pipeline.add_job(sample_step)
	pipeline.add_job(detect_persons_step)
	pipeline.add_job(split_step)
	pipeline.add_job(optimization_step)
	pipeline.add_job(overlay_step)
	pipeline.add_job(upload_step)

	# execute pipeline
	pipeline()

	# # declare another pipeline
	# pipeline2 = Pipeline()

	# # bundle the pipelines together
	# collection_of_pipelines = [pipeline, pipeline2]

	# # run the pipelines in parallel
	# from multiprocessing import Pool
	# with Pool() as p:
	# 	res = p.map(collection_of_pipelines)