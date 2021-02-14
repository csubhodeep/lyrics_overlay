
from typing import Iterable
from statistics import mean

import numpy as np

from lib.defs import Box, Point, LineSegment, Lyrics

# def text_fits_box(text: Lyrics,
# 				  font_size: int,
# 				  box: LyricsBox
# 				  ) -> bool:
# 	# assuming a particular formation e.g. for now one word comes under another
#
#
# 	return box.area > 20 and box.width > 10 and box.height > 10 and box.height == box.width

def len_of_text_list(text: Iterable[str]) -> int:
	text_pad = 1
	length = text_pad  # text pad for left side
	for word in text:
		length += len(word)
		length += 1  # added for a space between word
	length += text_pad  # adding text pad for right side
	return length

def text_fits_box(lyrics: Lyrics,
				  font_size: int,
				  form: int,  # 1,2,3
				  box: Box,
				  ) -> bool:

	n_words = len(lyrics.text)
	lengths_of_lines = []
	for i in range(0, n_words - form, form):
		if i + form < n_words:
			last_index = i + form
		else:
			last_index = n_words - 1
		lengths_of_lines.append(len_of_text_list(lyrics.text[i:last_index]))

	# max length will never be zero
	expected_width = max(lengths_of_lines) * font_size
	expected_height = (n_words / form) * font_size

	return expected_width <= box.width and expected_height <= box.height


def get_nearness_to_preferred_centre(centre_1: Point,
                                     centre_2: Point)->int:
	line_seg = LineSegment(centre_1, centre_2)

	return line_seg.length

def get_overlap_with_mask(image: np.ndarray,
						  lyrics_box: Box,
						  padding: int):
	box_array = np.ones(shape=[lyrics_box.height + padding, lyrics_box.width + padding])

	cropped_image_array = image[lyrics_box.y1 - padding // 2:lyrics_box.y3 + padding // 2,
						        lyrics_box.x1 - padding // 2:lyrics_box.x3 + padding // 2]

	score = (box_array * cropped_image_array).sum()

	return score

def is_inside_box(point: Point,
				  box: Box) -> bool:

	return box.y1 <= point.y <= box.y3 and box.x1 <= point.x <= box.x3

def get_combined_box(boxes: Iterable[Box]) -> Box:

	min_x = min([box.x1 for box in boxes])
	min_y = min([box.y1 for box in boxes])

	max_x = max([box.x3 for box in boxes])
	max_y = max([box.y3 for box in boxes])

	new_box = Box(first_diagonal_coords=Point((min_x, min_y)),
				  second_diagonal_coords=Point((max_x, max_y)))

	return new_box

def get_preferred_centre(boxes: Iterable[Box],
						 image: np.ndarray) -> Point:

	# check the spread of boxes
	combi_box = get_combined_box(boxes)

	total_area_of_all_boxes = sum([box.area for box in boxes])

	available_area_in_between_boxes = combi_box.area - total_area_of_all_boxes

	# if all boxes together occupy > 50% of the image area and more than 50% of the area in between boxes is available
	if combi_box.area/(image.shape[0]*image.shape[1]) > 0.5 and available_area_in_between_boxes/combi_box.area > 0.5:
		# search for an optimal location within the boxes
		list_of_centres = [box.centre for box in boxes]

		naive_centre_x = mean([centre.x for centre in list_of_centres])
		naive_centre_y = mean([centre.y for centre in list_of_centres])

		naive_centre = Point(coords=(naive_centre_x, naive_centre_y))

		# if the naive centre is inside any box
		if any([is_inside_box(naive_centre, box) for box in boxes]):
			# TODO: develop a better logic here !
			# then return the centre of the imaage as the preferred centre
			preferred_centre = Point(coords=(image.shape[0] // 2, image.shape[1] // 2))
		else:
			preferred_centre = naive_centre
	else:
		# search for an optimal location in the area of the image where there are no boxes

		print("do something else")



	return preferred_centre