import numpy as np
from scipy.optimize import differential_evolution

from typing import Iterable

from lib.defs import Box, Point, Lyrics
from utils.utils import text_fits_box
from utils.utils import	is_lyrics_box_overlaps_person_box
from utils.utils import get_distance_between_boxes
from utils.utils import get_distance_from_image_edges
from utils.utils import overlay_box_on_image

LOSS = []

# define loss/cost function
def get_loss(x,  # diagonal coords only,
			 binary_mask: np.ndarray,
			 person_boxes: Iterable[Box],
			 text: Lyrics,
			 ) -> float:

	try:
		lyrics_box = Box(first_diagonal_coords=Point(coords=(x[0], x[1])),
						 second_diagonal_coords=Point(coords=(x[2], x[3])))
		# TODO: why not exclude certain solutions before triggering the opti algo for faster convergence
		if not text_fits_box(text, font_size=3, box=lyrics_box, form=int(round(x[4]))):
			return 2500
		# TODO: elif not is_text_tight_in_box():
		if any([is_lyrics_box_overlaps_person_box(lyrics_box, person_box) for person_box in person_boxes]):
			return 5000
		w1 = 0.50
		w2 = 0.50
		# include the following:
		# distance from all person-boxes - w1

		# iterate over all the edges of all person-boxes and find the distances of them from the lyrics-box
		if len(person_boxes):
			distance_persons = tuple([get_distance_between_boxes(person_box, lyrics_box) for person_box in person_boxes])
		else:
			distance_persons = tuple([])

		# balance_1 = np.nan_to_num(np.var(distance_persons))


		# distance from all 4 edges - w2
		distance_edges = get_distance_from_image_edges(binary_mask, lyrics_box)

		# balance_2 = np.var(distance_edges)

		# loss = w1*balance_1 + w2*balance_2

		all_distances = distance_edges+distance_persons

		if min(all_distances) < 5:
			loss = 1000
		else:
			loss = np.var(all_distances)

		# loss = np.var(all_distances)
	except AssertionError as ex:
		loss = 10000

	LOSS.append(loss)

	return loss

if __name__ == "__main__":
	lyrics = Lyrics("I love you I love you I love you I love you")

	limits = (
		(0, 100),
		(0, 100),
		(0, 100),
		(0, 100),
		(1,5)
	)

	# #binary_mask[y1:y3, x1:x3]

	### case 0
	binary_mask = np.zeros([100, 100])

	persons = ()


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



	res = differential_evolution(get_loss,
								 bounds=limits,
								 args=(binary_mask, persons, lyrics),
								 popsize=100
								 )
	import matplotlib.pyplot as plt
	plt.plot(LOSS)
	plt.show()
	if res.success:
		optimal_box = Box(first_diagonal_coords=Point((res.x[0], res.x[1])),
					  second_diagonal_coords=Point((res.x[2], res.x[3])))

		overlay_box_on_image(optimal_box, binary_mask)
		print(res)



