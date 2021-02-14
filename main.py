import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import differential_evolution

from lib.defs import LyricsBox, Point, Lyrics
from utils.utils import text_fits_box, get_nearness_to_preferred_centre, get_overlap_with_mask


# define loss/cost function
def get_loss(x,  # diagonal coords only,
			 preffered_centre_coords: Point,
			 binary_mask: np.ndarray,
			 text: Lyrics,
			 padding: int = 10,
			 font_size: int = 1,  # this should be varied from outside
			 n_bbox: int = 1) -> float:
	w1 = 0.8
	w2 = 0.2

	try:
		lyrics_box = LyricsBox(first_diagonal_coords=Point(coords=(int(x[0]), int(x[1]))),
							   second_diagonal_coords=Point(coords=(int(x[2]), int(x[3]))))
		# TODO: why not exclude certain solutions before triggering the opti algo for faster convergence
		if not text_fits_box(text, font_size, lyrics_box):
			loss = 800
		else:
			nearness_to_preferred_centre = get_nearness_to_preferred_centre(centre_1=preffered_centre_coords,
																			centre_2=lyrics_box.centre)
			overlap_with_mask = get_overlap_with_mask(image=binary_mask,
													  lyrics_box=lyrics_box,
													  padding=padding)
			loss = w1 * overlap_with_mask + w2 * nearness_to_preferred_centre
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



