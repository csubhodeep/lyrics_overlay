import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import differential_evolution

from typing import Tuple, Iterable

from lib.defs import Box, Point, Lyrics, LineSegment
from utils.utils import text_fits_box, get_nearness_to_preferred_centre, get_overlap_with_mask, get_preferred_centre

def get_distance_between_boxes(box_1: Box,
						       box_2: Box) -> int:
	"""This function calculates the distance between the two closest points of two boxes
	"""
	return 10


def get_distance_from_image_edges(image: np.ndarray,
								  box: Box) -> Tuple[int, int, int, int]:

	distance_edge_1 = box.x1
	distance_edge_2 = image.shape[1] - box.x3
	distance_edge_3 = box.y1
	distance_edge_4 = image.shape[0] - box.y3

	return (distance_edge_1, distance_edge_2, distance_edge_3, distance_edge_4)


# define loss/cost function
def get_loss(x,  # diagonal coords only,
			 binary_mask: np.ndarray,
			 perons_boxes: Iterable[Box],
			 text: Lyrics,
			 padding: int = 10,
			 font_size: int = 1,  # this should be varied from outside
			 n_bbox: int = 1) -> float:

	try:
		lyrics_box = Box(first_diagonal_coords=Point(coords=(x[0], x[1])),
						 second_diagonal_coords=Point(coords=(x[2], x[3])))
		# TODO: why not exclude certain solutions before triggering the opti algo for faster convergence
		if not text_fits_box(text, font_size, box=lyrics_box, form=int(round(x[4]))):
			loss = 800
		else:
			w1 = 0.50
			w2 = 0.50
			# include the following:
			# distance from all person-boxes - w1

			# iterate over all the edges of all person-boxes and find the distances of them from the lyrics-box
			distances = [get_distance_between_boxes(person_box, lyrics_box) for person_box in perons_boxes]
			balance_1 = np.var(distances)

			# distance from all 4 edges - w2
			balance_2 = np.var(get_distance_from_image_edges(binary_mask, lyrics_box))

			loss = w1*balance_1 + w2*balance_2
	except AssertionError as ex:
		loss = 1000

	return loss

if __name__ == "__main__":
	binary_mask = np.zeros([100, 100])

	binary_mask[20:80, 20:30] = 1
	binary_mask[40:60, 60:80] = 1

	plt.imshow(binary_mask)

	# hyper parameters
	preferred_centre = get_preferred_centre()
	lyrics = "I love you"
	padding = 10
	min_ = 0 + padding // 2
	max_ = 100 - padding // 2
	limits = ((min_, max_), (min_, max_), (min_, max_), (min_, max_),)

	res = differential_evolution(get_loss,
								 bounds=limits,
								 args=(preferred_centre, binary_mask, lyrics, padding)
								 )
	optimal_coords = ((int(round(res.x[0])), int(round(res.x[1]))), (int(round(res.x[2])), int(round(res.x[3]))))
	new_binary_mask = binary_mask
	new_binary_mask[optimal_coords[0][1]:optimal_coords[1][1], optimal_coords[0][0]:optimal_coords[1][0]] = -1
	plt.imshow(new_binary_mask)



