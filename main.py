
from pre_processor.data_fetcher import fetch_data
from pre_processor.sampler import sample
from person_box_detector.inference import detect_persons
from splitter.splitter import split
from optimizer.optimize import optimize
from overlayer.overlay import overlay
from overlayer.upload import upload_video
from pipeline.lib.defs import Job, Pipeline
from configs.make_config import Config
from configs.make_config import get_config

if __name__ == "__main__":

	collection_of_configs = get_config(path_to_config="./configs/config.json")


	fetch_data_step = Job(func=fetch_data, conf=collection_of_configs['data_fetcher'])
	sample_step = Job(func=sample, conf=collection_of_configs['sample'])
	detect_persons_step = Job(func=detect_persons, conf=collection_of_configs['detected_persons'])
	split_step = Job(func=split, conf=collection_of_configs['split'])
	optimization_step = Job(func=optimize, conf=collection_of_configs['optimization'])
	overlay_step = Job(func=overlay, conf=collection_of_configs['overlay'])
	upload_step = Job(func=upload_video, conf=collection_of_configs['upload'])

	pipeline = Pipeline(start_step=fetch_data_step)
	pipeline.add_job(sample_step)
	pipeline.add_job(detect_persons_step)
	pipeline.add_job(split_step)
	pipeline.add_job(optimization_step)
	pipeline.add_job(overlay_step)
	pipeline.add_job(upload_step)

	pipeline()

	# pipeline2 = Pipeline()
	#
	# collection_of_pipelines = [pipeline, pipeline2]
	#
	# from multiprocessing import Pool
	# with Pool() as p:
	# 	res = p.map(collection_of_pipelines)