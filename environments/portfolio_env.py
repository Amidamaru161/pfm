

from __future__ import annotations

import math

import gym
import matplotlib
import numpy as np
import pandas as pd
from gym import spaces
from gym.utils import seeding

#matplotlib.use("Agg")
import matplotlib.pyplot as plt
from stable_baselines3.common.vec_env import DummyVecEnv
from pathlib import Path

try:
    import quantstats as qs
except ModuleNotFoundError:
    raise ModuleNotFoundError(
        """QuantStats module not found, environment can't plot results and calculate indicadors.
        This module is not installed with FinRL. Install by running one of the options:
        pip install quantstats --upgrade --no-cache-dir
        conda install -c ranaroussi quantstats
        """
    )


class PortfolioOptimizationEnv(gym.Env):
    """Среда аллокации портфеля для OpenAI gym.

    Эта среда имитирует взаимодействие между агентом и финансовым рынком
    на основе данных, предоставленных в виде датафрейма. Датафрейм содержит временные ряды
    признаков, определённых пользователем (например, цены закрытия, максимума и минимума), и должен иметь
    столбцы времени и тикера с соответствующими датами и символами тикеров.
    Пример датафрейма приведён ниже::

            date        high            low             close           tic
        0   2020-12-23  0.157414        0.127420        0.136394        ADA-USD
        1   2020-12-23  34.381519       30.074295       31.097898       BNB-USD
        2   2020-12-23  24024.490234    22802.646484    23241.345703    BTC-USD
        3   2020-12-23  0.004735        0.003640        0.003768        DOGE-USD
        4   2020-12-23  637.122803      560.364258      583.714600      ETH-USD
        ... ...         ...             ...             ...             ...

    На основе этого датафрейма среда создаёт пространство наблюдений, которое может
    быть либо Dict, либо Box. Пространство наблюдений Box — это трёхмерный массив формы
    (f, n, t), где f — количество признаков, n — количество акций в портфеле,
    t — размер временного окна, определённый пользователем. Если среда создана с параметром
    return_last_action=True, пространство наблюдений будет Dict со следующими ключами::

        {
        "state": трёхмерный Box (f, n, t), представляющий временные ряды,
        "last_action": одномерный Box (n+1,), представляющий веса портфеля
        }

    Обратите внимание, что пространство действий этой среды — одномерный Box размерности
    n + 1, так как веса портфеля должны содержать веса для всех акций в портфеле и остатка денежных средств.

    Атрибуты:
        action_space: Пространство действий.
        observation_space: Пространство наблюдений.
        episode_length: Количество временных шагов в эпизоде.
        portfolio_size: Количество акций в портфеле.
    """

    metadata = {"render.modes": ["human"]}

    def __init__(
        self,
        df,
        initial_amount,
        order_df=True,
        return_last_action=False,
        normalize_df="by_previous_time",
        reward_scaling=1,
        comission_fee_model="trf",
        comission_fee_pct=0,
        features=["close", "high", "low"],
        valuation_feature="close",
        time_column="date",
        time_format="%Y-%m-%d",
        tic_column="tic",
        tics_in_portfolio="all",
        time_window=1,
        cwd="./",
        new_gym_api=False,
    ):
        """Инициализация экземпляра среды.

        Аргументы:
            df: Датафрейм с рыночной информацией за период времени.
            initial_amount: Начальная сумма денежных средств для инвестирования.
            order_df: Если True, входной датафрейм сортируется по времени.
            return_last_action: Если True, наблюдения также возвращают последнее выполненное
                действие. В этом случае пространство наблюдений — Dict.
            normalize_df: Определяет метод нормализации, применяемый к датафрейму.
                Возможные значения: "by_previous_time", "by_fist_time_window_value",
                "by_COLUMN_NAME" (где COLUMN_NAME — имя столбца) и пользовательская функция.
                Если None, нормализация не выполняется.
            reward_scaling: Множитель для масштабирования функции вознаграждения.
                Может помочь в обучении.
            comission_fee_model: Модель для симуляции комиссии. Возможные значения:
                "trf" (transaction remainder factor) и "wvm" (weights vector modifier).
                Если None, комиссия не учитывается.
            comission_fee_pct: Процент комиссии (от 0 до 1).
            features: Список признаков, используемых в пространстве наблюдений.
                Элементы списка — имена столбцов датафрейма.
            valuation_feature: Признак, используемый для расчёта стоимости портфеля.
            time_column: Имя столбца с датами.
            time_format: Формат строк дат.
            tic_name: Имя столбца с тикерами.
            tics_in_portfolio: Список тикеров, входящих в портфель. Если "all", используются все тикеры.
            time_window: Размер временного окна.
            cwd: Локальный путь для сохранения графиков.
            new_gym_api: Если True, используются новые стандарты step/reset gym.
        """
        self._time_window = time_window
        self._time_index = time_window - 1
        self._time_column = time_column
        self._time_format = time_format
        self._tic_column = tic_column
        self._df = df
        self._initial_amount = initial_amount
        self._return_last_action = return_last_action
        self._reward_scaling = reward_scaling
        self._comission_fee_pct = comission_fee_pct
        self._comission_fee_model = comission_fee_model
        self._features = features
        self._valuation_feature = valuation_feature
        self._cwd = Path(cwd)
        self._new_gym_api = new_gym_api

        # results file
        self._results_file = self._cwd / "results" / "rl"
        self._results_file.mkdir(parents=True, exist_ok=True)

        # initialize price variation
        self._df_price_variation = None

        # preprocess data
        self._preprocess_data(order_df, normalize_df, tics_in_portfolio)

        # dims and spaces
        self._tic_list = self._df[self._tic_column].unique()
        self.portfolio_size = (
            len(self._tic_list)
            if tics_in_portfolio == "all"
            else len(tics_in_portfolio)
        )
        action_space = 1 + self.portfolio_size

        # sort datetimes and define episode length
        self._sorted_times = sorted(set(self._df[time_column]))
        self.episode_length = len(self._sorted_times) - time_window + 1

        # define action space
        self.action_space = spaces.Box(low=0, high=1, shape=(action_space,))

        # define observation state
        if self._return_last_action:
            # if  last action must be returned, a dict observation
            # is defined
            self.observation_space = spaces.Dict(
                {
                    "state": spaces.Box(
                        low=-np.inf,
                        high=np.inf,
                        shape=(
                            len(self._features),
                            len(self._tic_list),
                            self._time_window,
                        ),
                    ),
                    "last_action": spaces.Box(low=0, high=1, shape=(action_space,)),
                }
            )
        else:
            # if information about last action is not relevant,
            # a 3D observation space is defined
            self.observation_space = spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(len(self._features), len(self._tic_list), self._time_window),
            )

        self._reset_memory()

        self._portfolio_value = self._initial_amount
        self._terminal = False

    def step(self, actions):
        """Выполняет шаг симуляции.

        Аргументы:
            actions: Одномерный массив с новыми весами портфеля.

        Примечание:
            Если среда создана с return_last_action=True, следующее состояние будет Dict.
            Если False — Box. Проверить тип можно через observation_space.

        Возвращает:
            Если new_gym_api=True, возвращается кортеж:
            (state, reward, terminal, truncated, info). Если False:
            (state, reward, terminal, info).

            state: Следующее состояние симуляции.
            reward: Вознаграждение за последнее действие.
            terminal: True, если достигнуто терминальное состояние.
            truncated: True, если превышен лимит времени (всегда False).
            info: Словарь с информацией о последнем состоянии.
        """
        self._terminal = self._time_index >= len(self._sorted_times) - 1

        if self._terminal:
            metrics_df = pd.DataFrame(
                {
                    "date": self._date_memory,
                    "returns": self._portfolio_return_memory,
                    "rewards": self._portfolio_reward_memory,
                    "portfolio_values": self._asset_memory["final"],
                }
            )
            metrics_df.set_index("date", inplace=True)

            plt.plot(metrics_df["portfolio_values"], "r")
            plt.title("Portfolio Value Over Time")
            plt.xlabel("Time")
            plt.ylabel("Portfolio value")
            plt.savefig(self._results_file / "portfolio_value.png")
            plt.close()

            plt.plot(self._portfolio_reward_memory, "r")
            plt.title("Reward Over Time")
            plt.xlabel("Time")
            plt.ylabel("Reward")
            plt.savefig(self._results_file / "reward.png")
            plt.close()

            plt.plot(self._actions_memory)
            plt.title("Actions performed")
            plt.xlabel("Time")
            plt.ylabel("Weight")
            plt.savefig(self._results_file / "actions.png")
            plt.close()

            print("=================================")
            print("Initial portfolio value:{}".format(self._asset_memory["final"][0]))
            print(f"Final portfolio value: {self._portfolio_value}")
            print(
                "Final accumulative portfolio value: {}".format(
                    self._portfolio_value / self._asset_memory["final"][0]
                )
            )
            print(
                "Maximum DrawDown: {}".format(
                    qs.stats.max_drawdown(metrics_df["portfolio_values"])
                )
            )
            print("Sortino ratio: {}".format(qs.stats.sortino(metrics_df["returns"])))
            print("Sharpe ratio: {}".format(qs.stats.sharpe(metrics_df["returns"])))
            print("=================================")

            qs.plots.snapshot(
                metrics_df["returns"],
                show=False,
                savefig=self._results_file / "portfolio_summary.png",
            )

            if self._new_gym_api:
                return self._state, self._reward, self._terminal, False, self._info
            return self._state, self._reward, self._terminal, self._info

        else:
            # transform action to numpy array (if it's a list)
            actions = np.array(actions, dtype=np.float32)

            # if necessary, normalize weights
            if math.isclose(np.sum(actions), 1, abs_tol=1e-6) and np.min(actions) >= 0:
                weights = actions
            else:
                weights = self._softmax_normalization(actions)

            # save initial portfolio weights for this time step
            self._actions_memory.append(weights)

            # get last step final weights and portfolio_value
            last_weights = self._final_weights[-1]

            # load next state
            self._time_index += 1
            self._state, self._info = self._get_state_and_info_from_time_index(
                self._time_index
            )

            # если используется модификатор вектора весов, необходимо изменить вектор весов
            if self._comission_fee_model == "wvm":
                delta_weights = weights - last_weights
                delta_assets = delta_weights[1:]  # disconsider
                # calculate fees considering weights modification
                fees = np.sum(np.abs(delta_assets * self._portfolio_value))
                if fees > weights[0] * self._portfolio_value:
                    weights = last_weights
                    # maybe add negative reward
                else:
                    portfolio = weights * self._portfolio_value
                    portfolio[0] -= fees
                    self._portfolio_value = np.sum(portfolio)  # new portfolio value
                    weights = portfolio / self._portfolio_value  # new weights
            elif self._comission_fee_model == "trf":
                last_mu = 1
                mu = 1 - 2 * self._comission_fee_pct + self._comission_fee_pct**2
                while abs(mu - last_mu) > 1e-10:
                    last_mu = mu
                    mu = (
                        1
                        - self._comission_fee_pct * weights[0]
                        - (2 * self._comission_fee_pct - self._comission_fee_pct**2)
                        * np.sum(np.maximum(last_weights[1:] - mu * weights[1:], 0))
                    ) / (1 - self._comission_fee_pct * weights[0])
                self._info["trf_mu"] = mu
                self._portfolio_value = mu * self._portfolio_value

            # save initial portfolio value of this time step
            self._asset_memory["initial"].append(self._portfolio_value)

            # time passes and time variation changes the portfolio distribution
            portfolio = self._portfolio_value * (weights * self._price_variation)

            # calculate new portfolio value and weights
            self._portfolio_value = np.sum(portfolio)
            weights = portfolio / self._portfolio_value

            # save final portfolio value and weights of this time step
            self._asset_memory["final"].append(self._portfolio_value)
            self._final_weights.append(weights)

            # save date memory
            self._date_memory.append(self._info["end_time"])

            # define portfolio return
            rate_of_return = (
                self._asset_memory["final"][-1] / self._asset_memory["final"][-2]
            )
            portfolio_return = rate_of_return - 1
            portfolio_reward = np.log(rate_of_return)

            # save portfolio return memory
            self._portfolio_return_memory.append(portfolio_return)
            self._portfolio_reward_memory.append(portfolio_reward)

            # Define portfolio return
            self._reward = portfolio_reward
            self._reward = self._reward * self._reward_scaling

        if self._new_gym_api:
            return self._state, self._reward, self._terminal, False, self._info
        return self._state, self._reward, self._terminal, self._info

    def reset(self):
        """Сбрасывает среду и возвращает её в начальное состояние (первая дата датафрейма).

        Примечание:
            Если среда создана с return_last_action=True, начальное состояние — Dict.
            Если False — Box. Проверить тип можно через observation_space.

        Возвращает:
            Если new_gym_api=True, возвращается кортеж (state, info). Если False — только state.

            state: Начальное состояние.
            info: Информация о начальном состоянии.
        """
        # time_index must start a little bit in the future to implement lookback
        self._time_index = self._time_window - 1
        self._reset_memory()

        self._state, self._info = self._get_state_and_info_from_time_index(
            self._time_index
        )
        self._portfolio_value = self._initial_amount
        self._terminal = False

        if self._new_gym_api:
            return self._state, self._info
        return self._state

    def _get_state_and_info_from_time_index(self, time_index):
        """Получает состояние и информацию по индексу времени. Также обновляет атрибут "data"
        информацией о текущем шаге симуляции.

        Аргументы:
            time_index: Целое число — индекс определённой даты.
                Начальная дата датафрейма соответствует 0.

        Примечание:
            Если среда создана с return_last_action=True, возвращаемое состояние — Dict.
            Если False — Box. Проверить тип можно через observation_space.

        Возвращает:
            Кортеж (state, info):

            state: Состояние для текущего индекса времени (Box или Dict).
            info: Словарь с информацией о текущем шаге симуляции:

                {
                "tics": Список тикеров,
                "start_time": Начало текущего окна,
                "start_time_index": Индекс начала окна,
                "end_time": Конец окна,
                "end_time_index": Индекс конца окна,
                "data": Данные текущего окна,
                "price_variation": Изменение цены на текущем шаге
                }
        """
        # returns state in form (channels, tics, timesteps)
        end_time = self._sorted_times[time_index]
        start_time = self._sorted_times[time_index - (self._time_window - 1)]

        # define data to be used in this time step
        self._data = self._df[
            (self._df[self._time_column] >= start_time)
            & (self._df[self._time_column] <= end_time)
        ][[self._time_column, self._tic_column] + self._features]

        # define price variation of this time_step
        self._price_variation = self._df_price_variation[
            self._df_price_variation[self._time_column] == end_time
        ][self._valuation_feature].to_numpy()
        self._price_variation = np.insert(self._price_variation, 0, 1)

        # define state to be returned
        state = None
        for tic in self._tic_list:
            tic_data = self._data[self._data[self._tic_column] == tic]
            tic_data = tic_data[self._features].to_numpy().T
            tic_data = tic_data[..., np.newaxis]
            state = tic_data if state is None else np.append(state, tic_data, axis=2)
        state = state.transpose((0, 2, 1))
        info = {
            "tics": self._tic_list,
            "start_time": start_time,
            "start_time_index": time_index - (self._time_window - 1),
            "end_time": end_time,
            "end_time_index": time_index,
            "data": self._data,
            "price_variation": self._price_variation,
        }
        return self._standardize_state(state), info

    def render(self, mode="human"):
        """Визуализирует среду.

        Возвращает:
            Наблюдение текущего шага симуляции.
        """
        return self._state

    def _softmax_normalization(self, actions):
        """Нормализует вектор действий с помощью softmax-функции.

        Возвращает:
            Нормализованный вектор действий (портфель).
        """
        numerator = np.exp(actions)
        denominator = np.sum(np.exp(actions))
        softmax_output = numerator / denominator
        return softmax_output

    def enumerate_portfolio(self):
        """Выводит текущий портфель, показывая тикеры всех активов в портфеле."""
        print("Index: 0. Tic: Cash")
        for index, tic in enumerate(self._tic_list):
            print(f"Index: {index + 1}. Tic: {tic}")

    def _preprocess_data(self, order, normalize, tics_in_portfolio):
        """Сортирует и нормализует датафрейм среды.

        Аргументы:
            order: Если True, датафрейм сортируется по тикеру и времени.
            normalize: Метод нормализации датафрейма.
                Возможные значения: "by_previous_time", "by_fist_time_window_value",
                "by_COLUMN_NAME" (где COLUMN_NAME — имя столбца) и пользовательская функция.
                Если None, нормализация не выполняется.
            tics_in_portfolio: Список тикеров, входящих в портфель. Если "all", используются все тикеры.
        """
        # order time dataframe by tic and time
        if order:
            self._df = self._df.sort_values(by=[self._tic_column, self._time_column])
        # defining price variation after ordering dataframe
        self._df_price_variation = self._temporal_variation_df()
        # select only stocks in portfolio
        if tics_in_portfolio != "all":
            self._df_price_variation = self._df_price_variation[
                self._df_price_variation[self._tic_column].isin(tics_in_portfolio)
            ]
        # apply normalization
        if normalize:
            self._normalize_dataframe(normalize)
        # transform str to datetime
        self._df[self._time_column] = pd.to_datetime(self._df[self._time_column])
        self._df_price_variation[self._time_column] = pd.to_datetime(
            self._df_price_variation[self._time_column]
        )
        # transform numeric variables to float32 (compatibility with pytorch)
        self._df[self._features] = self._df[self._features].astype("float32")
        self._df_price_variation[self._features] = self._df_price_variation[
            self._features
        ].astype("float32")

    def _reset_memory(self):
        """Сбрасывает память среды."""
        date_time = self._sorted_times[self._time_index]
        # memorize portfolio value each step
        self._asset_memory = {
            "initial": [self._initial_amount],
            "final": [self._initial_amount],
        }
        # memorize portfolio return and reward each step
        self._portfolio_return_memory = [0]
        self._portfolio_reward_memory = [0]
        # initial action: all money is allocated in cash
        self._actions_memory = [
            np.array([1] + [0] * self.portfolio_size, dtype=np.float32)
        ]
        # memorize portfolio weights at the ending of time step
        self._final_weights = [
            np.array([1] + [0] * self.portfolio_size, dtype=np.float32)
        ]
        # memorize datetimes
        self._date_memory = [date_time]

    def _standardize_state(self, state):
        """Стандартизирует состояние в зависимости от пространства наблюдений. Если return_last_action=False,
        возвращается трёхмерный Box. Если True — словарь следующего вида::

            {
            "state": Трёхмерный Box текущего состояния,
            "last_action": Одномерный Box последнего действия
            }
        """
        last_action = self._actions_memory[-1]
        if self._return_last_action:
            return {"state": state, "last_action": last_action}
        else:
            return state

    def _normalize_dataframe(self, normalize):
        """Нормализует датафрейм среды.

        Аргументы:
            normalize: Метод нормализации датафрейма.
                Возможные значения: "by_previous_time", "by_fist_time_window_value",
                "by_COLUMN_NAME" (где COLUMN_NAME — имя столбца) и пользовательская функция.
                Если None, нормализация не выполняется.

        Примечание:
            Если используется пользовательская функция, она должна принимать датафрейм среды.
        """
        if type(normalize) == str:
            if normalize == "by_fist_time_window_value":
                print(
                    "Normalizing {} by first time window value...".format(
                        self._features
                    )
                )
                self._df = self._temporal_variation_df(self._time_window - 1)
            elif normalize == "by_previous_time":
                print(f"Normalizing {self._features} by previous time...")
                self._df = self._temporal_variation_df()
            elif normalize.startswith("by_"):
                normalizer_column = normalize[3:]
                print(f"Normalizing {self._features} by {normalizer_column}")
                for column in self._features:
                    self._df[column] = self._df[column] / self._df[normalizer_column]
        elif callable(normalize):
            print("Applying custom normalization function...")
            self._df = normalize(self._df)
        else:
            print("No normalization was performed.")

    def _temporal_variation_df(self, periods=1):
        """
        Вычисляет датафрейм, отражающий временные изменения. Для каждого признака (feature) 
        в этот датафрейм записывается отношение текущего значения признака к предыдущему 
        значению за заданный период. Это используется для нормализации датафрейма.

        Аргументы:
            periods: Количество временных шагов (индексов), за которые рассчитывается изменение.

        Возвращает:
            Датафрейм с временными изменениями (temporal variation).
        """

        df_temporal_variation = self._df.copy()
        prev_columns = []
        for column in self._features:
            prev_column = f"prev_{column}"
            prev_columns.append(prev_column)
            df_temporal_variation[prev_column] = df_temporal_variation.groupby(
                self._tic_column
            )[column].shift(periods=periods)
            df_temporal_variation[column] = (
                df_temporal_variation[column] / df_temporal_variation[prev_column]
            )
        df_temporal_variation = (
            df_temporal_variation.drop(columns=prev_columns)
            .fillna(1)
            .reset_index(drop=True)
        )
        return df_temporal_variation

    def _seed(self, seed=None):
        """Устанавливает сид генератора случайных чисел для воспроизводимости.

        Аргументы:
            seed: Значение сида.

        Возвращает:
            Применённое значение сида.
        """
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def get_sb_env(self, env_number=1):
        """Генерирует среду, совместимую с Stable Baselines 3. Возвращает векторизованную версию текущей среды.

        Возвращает:
            Кортеж с созданной средой и начальным наблюдением.
        """
        e = DummyVecEnv([lambda: self] * env_number)
        obs = e.reset()
        return e, obs