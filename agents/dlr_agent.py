"""
DRL-модели для решения задачи оптимизации портфеля с помощью обучения с подкреплением.
Этот агент был разработан для работы с такими окружениями, как PortfolioOptimizationEnv.
"""

#from __future__ import annotations
import sys 
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from agents.policy_gradient import PolicyGradient
from agents.hyper_policy import HyperNetPolicyGradient


MODELS = {
    "pg": PolicyGradient,
    "hyper_pg": HyperNetPolicyGradient
}


class DRLAgent:
    """Реализация DRL-алгоритмов для оптимизации портфеля.

    Примечание:
        Во время тестирования агент оптимизируется с помощью онлайн-обучения.
        Параметры политики обновляются многократно через постоянный
        промежуток времени. Чтобы отключить это, установите скорость обучения в 0.

    Атрибуты:
        env: Класс gym-окружения.
    """

    def __init__(self, env):
        """Инициализация агента.

        Аргументы:
            env: Gym-окружение, используемое для обучения.
        """
        self.env = env

    def get_model(
        self, model_name, device="cpu", model_kwargs=None, policy_kwargs=None
    ):
        """Настраивает DRL-модель.

        Аргументы:
            model_name: Имя модели согласно списку MODELS.
            device: Устройство для инициализации нейросетей.
            model_kwargs: Аргументы, передаваемые в класс модели.
            policy_kwargs: Аргументы, передаваемые в класс политики.

        Примечание:
            model_kwargs и policy_kwargs — это словари. Ключи должны быть строками
            с такими же именами, как аргументы класса. Пример для model_kwargs::

            { "lr": 0.01, "policy": EIIE }

        Возвращает:
            Экземпляр модели.
        """
        if model_name not in MODELS:
            raise NotImplementedError("The model requested was not implemented.")

        model = MODELS[model_name]
        model_kwargs = {} if model_kwargs is None else model_kwargs
        policy_kwargs = {} if policy_kwargs is None else policy_kwargs

        # add device settings
        model_kwargs["device"] = device
        policy_kwargs["device"] = device

        # add policy_kwargs inside model_kwargs
        model_kwargs["policy_kwargs"] = policy_kwargs

        return model(self.env, **model_kwargs)

    @staticmethod
    def train_model(model, episodes=100):
        """Обучает модель оптимизации портфеля.

        Аргументы:
            model: Экземпляр модели.
            episodes: Количество эпизодов.

        Возвращает:
            Экземпляр обученной модели.
        """
        model.train(episodes)
        return model

    @staticmethod
    def DRL_validation(
        model,
        test_env,
        policy=None,
        online_training_period=10,
        learning_rate=None,
        optimizer=None,
    ):
        """Тестирует модель в тестовом окружении.

        Аргументы:
            model: Экземпляр модели.
            test_env: Gym-окружение для тестирования.
            policy: Архитектура политики, которая будет использоваться. Если None, будет использоваться архитектура обучения.
            online_training_period: Период, через который происходит онлайн-обучение. Чтобы
                отключить онлайн-обучение, используйте очень большое значение.
            batch_size: Размер батча для обучения нейросети. Если None, будет использоваться
                размер батча обучения.
            lr: Скорость обучения нейросети политики. Если None, будет использоваться скорость
                обучения при обучении.
            optimizer: Оптимизатор нейросети. Если None, будет использоваться оптимизатор обучения.
        """
        model.test(test_env, policy, online_training_period, learning_rate, optimizer)