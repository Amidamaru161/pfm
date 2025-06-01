#from __future__ import annotations

from collections import deque
from random import randint
from random import random

import numpy as np
from torch.utils.data.dataset import IterableDataset


class PVM:
    def __init__(self, capacity, portfolio_size):
        """Инициализирует память векторного портфеля.

        Аргументы:
          capacity: Максимальная вместимость памяти.
          portfolio_size: Размер портфеля.
        """
        # изначально память будет содержать одинаковые действия
        self.capacity = capacity
        self.portfolio_size = portfolio_size
        self.reset()

    def reset(self):
        self.memory = [np.array([1] + [0] * self.portfolio_size, dtype=np.float32)] * (
            self.capacity + 1
        )
        self.index = 0  # начальный индекс для получения данных

    def retrieve(self):
        last_action = self.memory[self.index]
        self.index = 0 if self.index == self.capacity else self.index + 1
        return last_action

    def add(self, action):
        self.memory[self.index] = action


class ReplayBuffer:
    def __init__(self, capacity):
        """Инициализирует буфер воспроизведения.

        Аргументы:
          capacity: Максимальная вместимость буфера.
        """
        self.buffer = deque(maxlen=capacity)

    def __len__(self):
        """Представляет размер буфера

        Возвращает:
          Размер буфера.
        """
        return len(self.buffer)

    def append(self, experience):
        """Добавляет опыт в буфер. Когда буфер заполнен, он удаляет
           старый опыт.

        Аргументы:
          experience: опыт для сохранения.
        """
        self.buffer.append(experience)

    def sample(self):
        """Выборка из буфера воспроизведения. Все данные из буфера воспроизведения
        возвращаются, и буфер очищается.

        Возвращает:
          Выборка размером batch_size.
        """
        buffer = list(self.buffer)
        self.buffer.clear()
        return buffer


class RLDataset(IterableDataset):
    def __init__(self, buffer):
        """Инициализирует набор данных для обучения с подкреплением.

        Аргументы:
            buffer: буфер воспроизведения, который станет итерируемым набором данных.

        Примечание:
            Это подкласс IterableDataset из pytorch,
            см. https://pytorch.org/docs/stable/data.html#torch.utils.data.IterableDataset
        """
        self.buffer = buffer

    def __iter__(self):
        """Итерирует по RLDataset.

        Возвращает:
          Каждый опыт из выборки из буфера воспроизведения.
        """
        yield from self.buffer.sample()


def apply_portfolio_noise(portfolio, epsilon=0.1):
    """Применяет шум к распределению портфеля с учетом его ограничений.

    Аргументы:
        portfolio: начальное распределение портфеля.
        epsilon: максимальная ребалансировка.

    Возвращает:
        Новое распределение портфеля с примененным шумом.
    """
    portfolio_size = portfolio.shape[0]
    new_portfolio = portfolio.copy()
    for i in range(portfolio_size):
        target_index = randint(0, portfolio_size - 1)
        difference = epsilon * random()
        # проверка ограничений
        max_diff = min(new_portfolio[i], 1 - new_portfolio[target_index])
        difference = min(difference, max_diff)
        # применение разницы
        new_portfolio[i] -= difference
        new_portfolio[target_index] += difference
    return new_portfolio