from itertools import product
from typing import Tuple
from typing import Union


class Point:

    __slots__ = ("_x", "_y")

    def __init__(self, coords: Tuple[Union[int, float], Union[int, float]]):
        assert (
            coords[0] >= 0 and coords[1] >= 0
        ), "Negative coordinates are not allowed !"
        self._x = int(round(coords[0]))
        self._y = int(round(coords[1]))

    @property
    def x(self) -> int:
        return self._x

    @property
    def y(self) -> int:
        return self._y


class LineSegment:

    __slots__ = ("_first_coord", "_second_coord")

    def __init__(self, first_coord: Point, second_coord: Point):
        self._first_coord = first_coord
        self._second_coord = second_coord

    @property
    def point_1(self) -> Point:
        return self._first_coord

    @property
    def point_2(self) -> Point:
        return self._second_coord

    @property
    def length(self) -> float:
        return (
            (self.point_1.x - self.point_2.x) ** 2
            + (self.point_1.y - self.point_2.y) ** 2
        ) ** 0.5

    @property
    def slope(self) -> float:
        return (self.point_2.y - self.point_1.y) / (self.point_2.x - self.point_1.x)


class Box:
    """A box is represented as below
        (x1, y1)              (x2, y2)
                *-------------*
                |			  |
                |			  |
                *-------------*
        (x4, y4)              (x3, y3)"""

    __slots__ = ("_first_diagonal_coords", "_second_diagonal_coords")

    def __init__(self, first_diagonal_coords: Point, second_diagonal_coords: Point):
        """
        :param first_diagonal_coords: Point(x1, y1)
        :param second_diagonal_coords: Point(x3, y3)
		"""
        assert (
            first_diagonal_coords.x < second_diagonal_coords.x
            and first_diagonal_coords.y < second_diagonal_coords.y
        ), "Wrong coordinates !"
        self._first_diagonal_coords = first_diagonal_coords
        self._second_diagonal_coords = second_diagonal_coords

    @property
    def vertices(self) -> Tuple[Point, Point, Point, Point]:
        return (self.vertex_1, self.vertex_2, self.vertex_3, self.vertex_4)

    @property
    def vertex_1(self) -> Point:
        return self._first_diagonal_coords

    @property
    def vertex_2(self) -> Point:
        return Point(
            coords=(self._second_diagonal_coords.x, self._first_diagonal_coords.y)
        )

    @property
    def vertex_3(self) -> Point:
        return self._second_diagonal_coords

    @property
    def vertex_4(self) -> Point:
        return Point(
            coords=(self._first_diagonal_coords.x, self._second_diagonal_coords.y)
        )

    @property
    def height(self) -> int:
        return self.vertex_4.y - self.vertex_1.y

    @property
    def width(self) -> int:
        return self.vertex_2.x - self.vertex_1.x

    @property
    def area(self) -> int:
        return self.height * self.width

    @property
    def centre(self) -> Point:
        x_centre = (self.vertex_1.x + self.vertex_3.x) // 2
        y_centre = (self.vertex_1.y + self.vertex_3.y) // 2
        return Point(coords=(x_centre, y_centre))

    @property
    def diagonal_length(self) -> float:
        return LineSegment(first_coord=self.vertex_1, second_coord=self.vertex_3).length

    @property
    def aspect_ratio(self) -> Union[float, int]:
        return self.width / self.height

    def get_distance_from(self, box_2) -> Union[float, int]:
        """This function calculates the distance between the two closest points of two boxes"""
        if self.is_overlapping(box_2):
            return 0
        elif self.is_x_overlapping(box_2):
            if self.vertex_1.y > box_2.vertex_1.y:
                return self.vertex_1.y - box_2.vertex_3.y
            else:
                return box_2.vertex_1.y - self.vertex_3.y
        elif self.is_y_overlapping(box_2):
            if self.vertex_1.x > box_2.vertex_1.x:
                return self.vertex_1.x - box_2.vertex_3.x
            else:
                return box_2.vertex_1.x - self.vertex_3.x
        else:
            return min(
                [
                    LineSegment(x[0], x[1]).length
                    for x in product(self.vertices, box_2.vertices)
                ]
            )

    def is_x_overlapping(self, box_2) -> bool:
        return not (
            (self.vertex_1.x < box_2.vertex_1.x and self.vertex_3.x < box_2.vertex_1.x)
            or (
                self.vertex_1.x > box_2.vertex_3.x
                and self.vertex_3.x > box_2.vertex_3.x
            )
        )

    def is_y_overlapping(self, box_2) -> bool:
        return not (
            (self.vertex_1.y < box_2.vertex_1.y and self.vertex_3.y < box_2.vertex_1.y)
            or (
                self.vertex_1.y > box_2.vertex_3.y
                and self.vertex_3.y > box_2.vertex_3.y
            )
        )

    def is_overlapping(self, box_2) -> bool:
        return self.is_x_overlapping(box_2) and self.is_y_overlapping(box_2)

    def is_enclosing(self, point: Point) -> bool:
        return (
            self.vertex_1.y <= point.y <= self.vertex_3.y
            and self.vertex_1.x <= point.x <= self.vertex_3.x
        )

    def overlay_on_image(self, image):
        overlaid_image = image.copy()
        overlaid_image[
            self.vertex_1.y : self.vertex_3.y, self.vertex_1.x : self.vertex_3.x
        ] = -1
        return overlaid_image


class Lyrics:

    __slots__ = ("_text", "_start_time", "_end_time", "_font_size", "_font_type")

    def __init__(
        self,
        text: Union[str, Tuple[str, ...]],
        start_time: Union[int, float] = 0,
        end_time: Union[int, float] = 1,
        font_size: Union[int, float] = 3,
        font_type: str = "",
    ):
        assert len(text)

        assert start_time != end_time

        assert font_size > 0

        if isinstance(text, str):
            self._text = tuple(text.split(" "))
        else:
            self._text = tuple(text)

        self._start_time = start_time
        self._end_time = end_time
        self._font_size = font_size
        self._font_type = font_type

    @property
    def start_time(self) -> Union[int, float]:
        return self._start_time

    @property
    def end_time(self) -> Union[int, float]:
        return self._end_time

    @property
    def font_size(self) -> Union[int, float]:
        return self._font_size

    @property
    def font_type(self) -> str:
        return self._font_type

    @property
    def text(self) -> Tuple[str, ...]:
        return self._text

    @property
    def longest_word(self) -> str:
        word = self.text[0]
        for w in self.text:
            if len(w) > len(word):
                word = w
            else:
                pass

        return word

    @property
    def length_of_longest_word(self) -> int:
        return len(self.longest_word)
