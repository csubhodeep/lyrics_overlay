
from typing import Iterable
from statistics import mean

import numpy as np

from lib.defs import LyricsBox, Point, LineSegment, Lyrics

def text_fits_box(text: Lyrics,
				  font_size: int,
				  box: LyricsBox
				  ) -> bool:
	# assuming a particular formation e.g. for now one word comes under another


	return box.area > 20 and box.width > 10 and box.height > 10 and box.height == box.width

def get_nearness_to_preferred_centre(centre_1: Point,
                                     centre_2: Point)->int:
	line_seg = LineSegment(centre_1, centre_2)

	return line_seg.length

def get_overlap_with_mask(image: np.ndarray,
						  lyrics_box: LyricsBox,
						  padding: int):
	box_array = np.ones(shape=[lyrics_box.height + padding, lyrics_box.width + padding])

	cropped_image_array = image[lyrics_box.y1 - padding // 2:lyrics_box.y3 + padding // 2,
						        lyrics_box.x1 - padding // 2:lyrics_box.x3 + padding // 2]

	score = (box_array * cropped_image_array).sum()

	return score

def get_preferred_centre(boxes: Iterable[LyricsBox]) -> Point:

	list_of_centres = [box.centre for box in boxes]

	naive_centre_x = mean([centre.x for centre in list_of_centres])
	naive_centre_y = mean([centre.y for centre in list_of_centres])

	naive_centre = Point(coords=(naive_centre_x, naive_centre_y))

	return naive_centre