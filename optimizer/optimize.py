from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import differential_evolution

from statistics import variance
from typing import Iterable, Tuple

from configs.make_config import Config

from optimizer.lib.defs import Box, Point, Lyrics
from optimizer.utils.utils import text_fits_box
from optimizer.utils.utils import get_distance_from_image_edges


def get_loss(x,
			 canvas_shape: Tuple[int, int],
			 forbidden_zones: Iterable[Box],
			 text: Lyrics,
			 ) -> float:

	try:
		lyrics_box = Box(first_diagonal_coords=Point(coords=(x[0], x[1])),
						 second_diagonal_coords=Point(coords=(x[2], x[3])))
	except AssertionError as ex:
		return 10000

	if any([lyrics_box.is_overlapping(zone) for zone in forbidden_zones]):
		return 5000

	if not text_fits_box(text, font_size=int(round(x[4])), box=lyrics_box, form=2):
		return 2500

	w1 = 0.50
	w2 = 0.50
	# include the following:
	# distance from all person-boxes - w1

	# iterate over all the edges of all person-boxes and find the distances of them from the lyrics-box
	if len(forbidden_zones):
		distance_persons = tuple([lyrics_box.get_distance_from(zone) for zone in forbidden_zones])
	else:
		distance_persons = tuple([])

	# balance_1 = np.nan_to_num(np.var(distance_persons))

	## distance from all 4 edges - w2
	distance_edges = get_distance_from_image_edges(canvas_shape, lyrics_box)

	# balance_2 = variance(distance_edges)

	# loss = w1*balance_1 + w2*balance_2
	all_distances = distance_edges+distance_persons

	if min(all_distances) < 5:
		return 1000
	else:
		return variance(all_distances)

	# loss = np.var(all_distances)


def get_optimal_boxes(row, conf: Config):

	persons = (
		Box(first_diagonal_coords=Point(coords=(row['x1'], row['y1'])),
			second_diagonal_coords=Point(coords=(row['x3'], row['y3']))),
	)

	lyrics = Lyrics(row['text'])

	limits = (
		(0, conf.img_size),
		(0, conf.img_size),
		(0, conf.img_size),
		(0, conf.img_size),
		(conf.font_size_min_limit, conf.font_size_max_limit)
	)

	res = differential_evolution(get_loss,
								 bounds=limits,
								 args=((conf.img_size, conf.img_size), persons, lyrics),
								 popsize=100
								 )

	return int(round(res.x[0])), int(round(res.x[1])), int(round(res.x[2])), int(round(res.x[3])), res.x[4]


def optimize(conf: Config) -> bool:

	input_file_path = Path.cwd().joinpath(conf.input_data_path).joinpath(f"{conf.run_id}.feather")
	output_file_path = Path.cwd().joinpath(conf.output_data_path).joinpath(f"{conf.run_id}.feather")

	df_input = pd.read_feather(input_file_path)

	df_output = df_input[['start_time', 'end_time', 'text', 'font_type']]

	df_output[['x1', 'y1', 'x3', 'y3', 'font_size']] = df_input.apply(get_optimal_boxes, axis=1, args=(conf,), result_type='expand')

	df_output[['x1', 'y1', 'x3', 'y3']] = df_output[['x1', 'y1', 'x3', 'y3']].astype(int)

	df_output.to_feather(output_file_path)

	return True


if __name__ == "__main__":

	config = Config(output_data_path="../data/optimizer_output",
					input_data_path="../data/splitter_output")
	config.set_run_id(run_id="341e3c19-74db-48b0-b986-73689231a268")
	config.img_size = 416
	config.font_size_max_limit = 5
	config.font_size_min_limit = 1

	optimize(conf=config)

	# lyrics = Lyrics("I love you I love you I love you I love you")
	#
	# limits = (
	# 	(0, 100),
	# 	(0, 100),
	# 	(0, 100),
	# 	(0, 100),
	# 	(1, 5)
	# )

	# #binary_mask[y1:y3, x1:x3]

	### case 0
	# binary_mask = np.zeros([100, 100])
	#
	# persons = ()


	### case 1
	# binary_mask = np.zeros([100, 100])
	# binary_mask[40:60, 60:80] = 1
	#
	# persons = (
	# 		Box(first_diagonal_coords=Point(coords=(60,40)), second_diagonal_coords=Point(coords=(80,60))),
	# 	)

	### case 2
	# binary_mask = np.zeros([100, 100])
	#
	# binary_mask[10:30, 10:30] = 1
	# binary_mask[40:60, 40:60] = 1
	# binary_mask[70:90, 70:90] = 1
	#
	# persons = (
	# 	Box(first_diagonal_coords=Point(coords=(10, 10)), second_diagonal_coords=Point(coords=(30, 30))),
	# 	Box(first_diagonal_coords=Point(coords=(40, 40)), second_diagonal_coords=Point(coords=(60, 60))),
	# 	Box(first_diagonal_coords=Point(coords=(70, 70)), second_diagonal_coords=Point(coords=(90, 90)))
	# )

	### case 3
	# binary_mask = np.zeros([100, 100])
	# binary_mask[20:60, 10:40] = 1
	# binary_mask[10:30, 65:85] = 1
	# binary_mask[70:90, 50:70] = 1
	#
	# persons = (
	# 	Box(first_diagonal_coords=Point(coords=(10, 20)), second_diagonal_coords=Point(coords=(40, 60))),
	# 	Box(first_diagonal_coords=Point(coords=(65, 10)), second_diagonal_coords=Point(coords=(85, 30))),
	# 	Box(first_diagonal_coords=Point(coords=(50, 70)), second_diagonal_coords=Point(coords=(70, 90)))
	# )

	### case 4
	# binary_mask = np.zeros([100, 100])
	# binary_mask[20:80, 10:25] = 1
	# binary_mask[20:80, 47:52] = 1
	# binary_mask[20:80, 70:85] = 1
	#
	# persons = (
	# 	Box(first_diagonal_coords=Point(coords=(10, 20)), second_diagonal_coords=Point(coords=(25, 80))),
	# 	Box(first_diagonal_coords=Point(coords=(47, 20)), second_diagonal_coords=Point(coords=(52, 80))),
	# 	Box(first_diagonal_coords=Point(coords=(70, 20)), second_diagonal_coords=Point(coords=(85, 80)))
	# )

	# TODO: why not exclude certain solutions before triggering the opti algo for faster convergence

	# res = differential_evolution(get_loss,
	# 							 bounds=limits,
	# 							 args=(binary_mask.shape, persons, lyrics),
	# 							 popsize=100
	# 							 )
	#
	# if res.success:
	# 	optimal_box = Box(first_diagonal_coords=Point((res.x[0], res.x[1])),
	# 				  second_diagonal_coords=Point((res.x[2], res.x[3])))
	#
	# 	plt.imshow(optimal_box.overlay_on_image(binary_mask))
	# 	plt.show()
	# 	print(res)



