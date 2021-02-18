
from pre_processor.data_fetcher import fetch_data
from pre_processor.sampler import sample
from person_box_detector.inference import detect_persons
from splitter.splitter import split
from optimizer.optimize import optimize
from overlayer.overlay import overlay
from overlayer.upload import upload_video
from pipeline.lib.defs import Job, Pipeline
from configs.make_config import Config




if __name__ == "__main__":
	fetch_data_step = Job(func=fetch_data, conf=Config(input_data_path='./data/input',
													   output_data_path='./data/input'))
	sample_step = Job(func=sample, conf=Config(output_data_path='./data/pre_processed_output'))
	detect_persons_step = Job(func=detect_persons, conf=Config(output_data_path='./data/detected_persons_output'))
	split_step = Job(func=split, conf=Config(output_data_path='./data/splitter_output'))
	optimization_step = Job(func=optimize, conf=Config(output_data_path='./data/optimizer_output'))
	overlay_step = Job(func=overlay, conf=Config(output_data_path='./data/final_output'))
	upload_step = Job(func=upload_video, conf=Config(output_data_path='./data/final_output'))

	pipeline = Pipeline(start_step=fetch_data_step)
	pipeline.add_job(detect_persons_step)
	pipeline.add_job(split_step)
	pipeline.add_job(optimization_step)
	pipeline.add_job(overlay_step)
	pipeline.add_job(upload_step)

	pipeline()
