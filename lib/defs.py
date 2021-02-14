from typing import Tuple
from typing import Union
from typing import Iterable


class Point:
	def __init__(self, coords: Tuple[Union[int, float], Union[int, float]]):
		assert coords[0] >= 0 and coords[1] >= 0, "Negative coordinates are not allowed !"
		self._x = int(round(coords[0]))
		self._y = int(round(coords[1]))

	@property
	def x(self):
		return self._x

	@property
	def y(self):
		return self._y

class LineSegment:
	def __init__(self,
				 first_coord: Point,
				 second_coord: Point):
		self.first_coord = first_coord
		self.second_coord = second_coord

	@property
	def length(self):
		return ((self.first_coord.x-self.second_coord.x)**2 + (self.first_coord.y-self.second_coord.y)**2)**0.5

class Box:
	"""
	A box is represented as below
		(x1, y1)              (x2, y2)
				*-------------*
				|			  |
				|			  |
				*-------------*
		(x4, y4)              (x3, y3)

	"""

	def __init__(self,
				 first_diagonal_coords: Point,
				 second_diagonal_coords: Point):
		"""
		:param first_diagonal_coords: Point(x1, y1)
		:param second_diagonal_coords: Point(x3, y3)
		"""
		assert first_diagonal_coords.x < second_diagonal_coords.x and first_diagonal_coords.y < second_diagonal_coords.y, "Wrong coordinates !"
		self.first_diagonal_coords = first_diagonal_coords
		self.second_diagonal_coords = second_diagonal_coords

	@property
	def vertices(self):
		return (
			Point(coords=(self.x1, self.y1)),
			Point(coords=(self.x2, self.y2)),
			Point(coords=(self.x3, self.y3)),
			Point(coords=(self.x4, self.y4)),
		)

	@property
	def vertex_1(self):
		return Point(coords=(self.x1, self.y1))

	@property
	def vertex_2(self):
		return Point(coords=(self.x2, self.y2))

	@property
	def vertex_3(self):
		return Point(coords=(self.x3, self.y3))

	@property
	def vertex_4(self):
		return Point(coords=(self.x4, self.y4))

	@property
	def x1(self) -> int:
		return self.first_diagonal_coords.x

	@property
	def y1(self) -> int:
		return self.first_diagonal_coords.y

	@property
	def x2(self) -> int:
		return self.second_diagonal_coords.x

	@property
	def y2(self) -> int:
		return self.first_diagonal_coords.y

	@property
	def x3(self) -> int:
		return self.second_diagonal_coords.x

	@property
	def y3(self) -> int:
		return self.second_diagonal_coords.y

	@property
	def x4(self) -> int:
		return self.first_diagonal_coords.x

	@property
	def y4(self) -> int:
		return self.second_diagonal_coords.y

	@property
	def height(self) -> int:
		return self.y4 - self.y1

	@property
	def width(self) -> int:
		return self.x2 - self.x1

	@property
	def area(self) -> int:
		return self.height * self.width

	@property
	def centre(self) -> Point:
		x_centre = (self.x1+self.x3)//2
		y_centre = (self.y1+self.y3)//2
		return Point(coords=(x_centre, y_centre))

	def is_x_overlapping(self,
					 box_2) -> bool:
		return not ((self.x1 < box_2.x1 and self.x3 < box_2.x1) or (self.x1 > box_2.x3 and self.x3 > box_2.x3))

	def is_y_overlapping(self,
					 box_2) -> bool:
		return not ((self.y1 < box_2.y1 and self.y3 < box_2.y1) or (self.y1 > box_2.y3 and self.y3 > box_2.y3))

	def is_overlapping(self, box):
		return self.is_x_overlapping(box) and self.is_y_overlapping(box)

	def is_enclosing(self, point: Point):
		return self.y1 <= point.y <= self.y3 and self.x1 <= point.x <= self.x3


class Lyrics:
	def __init__(self, text: Union[str, Iterable[str]]):
		assert len(text)

		if isinstance(text, str):
			self._text = tuple(text.split(" "))
		else:
			self._text = tuple(text)

	@property
	def text(self) -> Iterable[str]:
		return self._text

	@property
	def longest_word(self):
		word = self.text[0]
		for w in self.text:
			if len(w) > len(word):
				word = w
			else:
				pass

		return word

	@property
	def length_of_longest_word(self):
		return len(self.longest_word)