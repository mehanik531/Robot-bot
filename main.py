import numpy
import heapq
import matplotlib.pyplot as plt
from typing import List, Tuple, Union


def generate_map(height: int = 15, width: int = 15, wall_chance: float = 0.2) -> numpy.ndarray:
    """
    Функция создаёт поле, по которому перемещается робот.
    Важно: начальная позиция робота всегда равна (0, 0), а координаты его цели (height - 1, width - 1).

    :param height: int - "Высота" поля перемещения, если None - выбирается случайным образом.
    :param width: int - "Ширина" поля перемещения, если None - выбирается случайным образом.
    :param wall_chance: float - Вероятность появления стены для каждой ячейки поля перемещения, если None - 0.3.
    :return: numpy.ndarray - Матрица, представляющая собой поле перемещения. 0 - свободная ячейка, 1 - стена.
    """
    if height is None:
        height = numpy.random.randint(2, 100)
    if width is None:
        width = numpy.random.randint(2, 100)

    assert height > 1
    assert width > 1
    assert 1 >= wall_chance >= 0

    return numpy.random.choice(
        [0, 1],
        size=(height, width),
        p=[1 - wall_chance, wall_chance],
    )


def heuristic(p1: Tuple[int, int], p2: Tuple[int, int]) -> int:
    """
    Функция для оценки расстояния между точками p1 и p2.

    :param p1: tuple[int, int] - Координаты первой точки.
    :param p2: tuple[int, int] - Координаты второй точки.
    :return: int - Манхэтэнское расстояние aka эвристика для точек p1 и p2.
    """
    return abs(p1[0] - p2[0]) + abs(p1[1] - p2[1])


def find_path(
        robo_map: numpy.ndarray,
        start: Tuple[int, int] = None,
        target: Tuple[int, int] = None
) -> Union[List[Tuple[int, int]], None]:
    """
    Функция вычисляет путь, применяя алгоритм A*.
    Если путь не найден, возвращается None.

    :param robo_map: numpy.ndarray - Матрица, представляющая собой поле перемещения. 0 - свободная ячейка, 1 - стена.
    :param start: tuple[int, int] - Координаты первого участка пути, если None - (0, 0).
    :param target: tuple[int, int] - Координаты конечного участка пути, если None - последний элемент матрицы robo_map.
    :return: list[tuple[int, int]] - Список координат, принадлежащих найденному пути.
    """
    if start is None:
        start = (0, 0)
    if target is None:
        target = (robo_map.shape[0] - 1, robo_map.shape[1] - 1)

    assert robo_map.shape[0] > 1
    assert robo_map.shape[1] > 1
    assert 0 <= start[0] <= robo_map.shape[0] - 1
    assert 0 <= start[1] <= robo_map.shape[1] - 1
    assert 0 <= target[0] <= robo_map.shape[0] - 1
    assert 0 <= target[1] <= robo_map.shape[1] - 1
    if robo_map[target] == 1:
        return None

    to_be_processed = []
    came_from = {}
    g_len = {start: 0}
    f_len = {start: heuristic(start, target)}

    heapq.heappush(to_be_processed, (0, start))
    while to_be_processed:
        current = heapq.heappop(to_be_processed)[1]

        if current == target:
            found_path = []
            while current in came_from:
                found_path.append(current)
                current = came_from[current]
            found_path.append(start)
            return found_path[::-1]

        for cell in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
            neighbor = (current[0] + cell[0], current[1] + cell[1])

            if 0 <= neighbor[0] < robo_map.shape[0] and 0 <= neighbor[1] < robo_map.shape[1]:
                if robo_map[neighbor[0]][neighbor[1]] == 0:
                    tmp_g_len = g_len[current] + 1

                    if neighbor not in g_len or tmp_g_len < g_len[neighbor]:
                        came_from[neighbor] = current
                        g_len[neighbor] = tmp_g_len
                        f_len[neighbor] = heuristic(neighbor, target) + tmp_g_len
                        heapq.heappush(to_be_processed, (f_len[neighbor], neighbor))


def optimize_path(default_path: List[Tuple[int, int]]) -> Union[List[Tuple[int, int]], None]:
    """
    Функция оптимизирует путь робота.
    Основная идея заключается в добавлении "срезок" под 45 граудов, позволяя уменьшать количество поротов робота.

    :param default_path: list[tuple[int, int]] - Исходный список точек маршрута.
    :return: list[tuple[int, int]] - Оптимизированный список точек маршрута.
    """
    if default_path is None or len(default_path) < 3:
        return default_path

    optimal_path = []
    is_next_bad = False

    for i in range(len(default_path) - 2):
        if not is_next_bad:
            if default_path[i + 2][0] - default_path[i][0] == 1 and default_path[i + 2][1] - default_path[i][1] == 1:
                optimal_path.append(default_path[i])
                is_next_bad = True
            else:
                optimal_path.append(default_path[i])
        else:
            is_next_bad = False

    optimal_path.append(default_path[-1])

    return optimal_path


if __name__ == '__main__':
    mappy = generate_map()
    default_path = find_path(mappy)
    opt = optimize_path(default_path)

    plt.imshow(mappy, cmap='binary')
    if default_path:
        path = numpy.array(default_path)
        opt = numpy.array(opt)
        plt.plot(path[:, 1], path[:, 0], color='red', label="Исходный путь")
        plt.plot(opt[:, 1], opt[:, 0], color='blue', label="Оптимизированный путь")
        plt.legend()
        plt.title('Маршрут найден!')
    else:
        plt.title("Маршрут не найден!")
    plt.show()
