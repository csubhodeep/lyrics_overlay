import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import differential_evolution

from typing import Tuple, Iterable

from lib.defs import Box, Point, Lyrics, LineSegment
from utils.utils import text_fits_box
from utils.utils import	is_lyrics_box_overlaps_person_box
from utils.utils import get_distance_between_boxes
from utils.utils import get_distance_from_image_edges


# define loss/cost function
def get_loss(x,  # diagonal coords only,
			 binary_mask: np.ndarray,
			 person_boxes: Iterable[Box],
			 text: Lyrics,
			 font_size: int = 1,  # this should be varied from outside
			 ) -> float:

	try:
		lyrics_box = Box(first_diagonal_coords=Point(coords=(x[0], x[1])),
						 second_diagonal_coords=Point(coords=(x[2], x[3])))
		# TODO: why not exclude certain solutions before triggering the opti algo for faster convergence
		if not text_fits_box(text, font_size, box=lyrics_box, form=int(round(x[4]))):
			loss = 800
		elif any([is_lyrics_box_overlaps_person_box(lyrics_box, person_box) for person_box in person_boxes]):
			loss = 1000
		else:
			w1 = 0.50
			w2 = 0.50
			# include the following:
			# distance from all person-boxes - w1

			# iterate over all the edges of all person-boxes and find the distances of them from the lyrics-box
			distances = [get_distance_between_boxes(person_box, lyrics_box) for person_box in person_boxes]
			balance_1 = np.var(distances)

			# distance from all 4 edges - w2
			balance_2 = np.var(get_distance_from_image_edges(binary_mask, lyrics_box))

			loss = w1*balance_1 + w2*balance_2
	except AssertionError as ex:
		loss = 1000

	return loss

if __name__ == "__main__":
	binary_mask_4 = np.zeros([100, 100])
	# binary_mask[y1:y3, x1:x3]
	binary_mask_4[20:80, 10:25] = 1
	binary_mask_4[20:80, 47:52] = 1
	binary_mask_4[20:80, 70:85] = 1
	plt.imshow(binary_mask_4)
	lyrics = Lyrics("I love you")

	limits = (
		(0,100),
		(0,100),
		(0,100),
		(0,100)
	)

	persons = (
		Box(first_diagonal_coords=Point(coords=(10,20)), second_diagonal_coords=Point(coords=(25,80))),
		Box(first_diagonal_coords=Point(coords=(47,20)), second_diagonal_coords=Point(coords=(52,80))),
		Box(first_diagonal_coords=Point(coords=(70,20)), second_diagonal_coords=Point(coords=(85,80)))
	)


	res = differential_evolution(get_loss,
								 bounds=limits,
								 args=(binary_mask_4, persons, lyrics)
								 )


