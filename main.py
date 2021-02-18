
from pre_process.data_fetcher import fetch_data
from pre_process.sampler import sample
from person_box_detector.inference import detect_image
from split.splitter import split
from optimizer.optimize import optimize
from overlayer.overlay import overlay
from overlayer.upload import upload_video


pipeline_jobs = [
	fetch_data,
	sample,
	detect_image, # parallelised
	split,
 	optimize, # parallelised
	overlay,
	upload_video
]

