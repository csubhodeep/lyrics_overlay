from typing import Tuple
from typing import Union
from typing import Iterable

class Point:
	def __init__(self, coords: Union[Tuple[int, int], Tuple[float, float]]):
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

class LyricsBox:
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
		assert first_diagonal_coords.x < second_diagonal_coords.x and first_diagonal_coords.y and second_diagonal_coords.y, "Wrong coordinates !"
		self.first_diagonal_coords = first_diagonal_coords
		self.second_diagonal_coords = second_diagonal_coords

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

class Lyrics:
	def __init__(self, text: Union[str, Iterable[str]]):
		assert len(text)

		if isinstance(text, str):
			self.text = tuple(text.split(" "))
		else:
			self.text = tuple(text)

	@property
	def lyrics(self) -> Iterable[str]:
		return self.text

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



