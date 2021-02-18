
from typing import Iterable, Tuple
from statistics import mean

import numpy as np

from optimizer.lib.defs import Box, Point, LineSegment, Lyrics


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

	# 	return box.area > 20 and box.width > 10 and box.height > 10 and box.height == box.width
	return 0.8*box.width < expected_width <= box.width and 0.8*box.height < expected_height <= box.height


def get_overlap_with_mask(image: np.ndarray,
						  lyrics_box: Box,
						  padding: int):
	box_array = np.ones(shape=[lyrics_box.height + padding, lyrics_box.width + padding])

	cropped_image_array = image[lyrics_box.vertex_1.y - padding // 2:lyrics_box.vertex_3.y + padding // 2,
						        lyrics_box.vertex_1.x - padding // 2:lyrics_box.vertex_3.x + padding // 2]

	score = (box_array * cropped_image_array).sum()

	return score


def get_distance_from_image_edges(canvas_shape: Tuple[int, int],
								  box: Box) -> Tuple[int, int, int, int]:
	distance_edge_1 = box.vertex_1.x
	distance_edge_2 = canvas_shape[1] - box.vertex_3.x
	distance_edge_3 = box.vertex_1.y
	distance_edge_4 = canvas_shape[0] - box.vertex_3.y

	return (distance_edge_1, distance_edge_2, distance_edge_3, distance_edge_4)


def get_combined_box(boxes: Iterable[Box]) -> Box:

	min_x = min([box.vertex_1.x for box in boxes])
	min_y = min([box.vertex_1.y for box in boxes])

	max_x = max([box.vertex_3.x for box in boxes])
	max_y = max([box.vertex_3.y for box in boxes])

	new_box = Box(first_diagonal_coords=Point((min_x, min_y)),
				  second_diagonal_coords=Point((max_x, max_y)))

	return new_box


def get_nearness_to_preferred_centre(centre_1: Point,
                                     centre_2: Point)->int:
	line_seg = LineSegment(centre_1, centre_2)

	return line_seg.length


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
		if any([box.is_enclosing(naive_centre) for box in boxes]):
			# TODO: develop a better logic here !
			# then return the centre of the imaage as the preferred centre
			preferred_centre = Point(coords=(image.shape[0] // 2, image.shape[1] // 2))
		else:
			preferred_centre = naive_centre
	else:
		# search for an optimal location in the area of the image where there are no boxes

		print("do something else")

	return preferred_centre