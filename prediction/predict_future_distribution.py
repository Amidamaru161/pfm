"""
Модуль для прогнозирования распределения портфеля на будущее, используя обученные DRL-модели.
"""

import numpy as np
import pandas as pd
import torch
import sys 
import os 
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.models import EIIE,EI3,GPM
from agents.dlr_agent import DRLAgent
from environments.portfolio_env import PortfolioOptimizationEnv
import copy



from environments.portfolio_memory import PVM, ReplayBuffer, RLDataset,apply_portfolio_noise

class PortfolioPredictor:
    """Класс для прогнозирования распределения портфеля на будущее.
    
    Этот класс использует обученную DRL-модель для прогнозирования
    распределения портфеля на основе исторических данных и 
    генерации прогнозов на будущие периоды.
    """
    
    def __init__(self, model_path, model_type='EIIE', model_params=None, device='cpu'):
        """Инициализирует предиктор портфеля.
        
        Аргументы:
            model_path: Путь к файлу с сохраненными весами модели.
            model_type: Тип модели ('EIIE', 'EI3', 'GPM').
            model_params: Словарь с параметрами модели.
           
        """
        self.model_path = model_path
        self.model_type = model_type
        self.model_params = {} if model_params is None else model_params
        self.device=device
        
        # Загрузить модель
        self.model = self._load_model()
        
    def _load_model(self):
        """Загружает модель из файла.
        
        Возвращает:
            Загруженную модель.
        """
        # Загружаем чекпоинт
        checkpoint = torch.load(self.model_path)
        
        # Получаем параметры модели из чекпоинта
        model_params = checkpoint['hyperparameters']
        
        # Создаем экземпляр модели
        if self.model_type == 'EIIE':
            model = EIIE(**model_params)
        elif self.model_type == 'EI3':
            model = EI3(**model_params)
        elif self.model_type == 'GPM':
            model = GPM(**model_params)
        else:
            raise ValueError(f"Неизвестный тип модели: {self.model_type}")
        
        # Загружаем веса модели из чекпоинта
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()  # Переключаем модель в режим оценки
        
        return model
    
    def predict_weights(self, observation, last_action):
        """Прогнозирует веса портфеля на основе текущего наблюдения.
        
        Аргументы:
            observation: Наблюдение из среды.
            last_action: Последнее действие (распределение портфеля).
            
        Возвращает:
            Прогнозируемые веса портфеля.
        """
        # Переводим в батч, если не батч
        if isinstance(observation, np.ndarray) and len(observation.shape) == 3:
            obs_batch = np.expand_dims(observation, axis=0)
        else:
            obs_batch = observation
            
        if isinstance(last_action, np.ndarray) and len(last_action.shape) == 1:
            last_action_batch = np.expand_dims(last_action, axis=0)
        else:
            last_action_batch = last_action
        
        # Получаем прогноз от модели
        with torch.no_grad():
            predicted_weights = self.model(obs_batch, last_action_batch)
            
        return predicted_weights
    
    def predict_future_weights(self, df, initial_amount, time_window, num_steps, tics_in_portfolio="all", 
                               features=None, normalize_df=None):
        """Прогнозирует распределение портфеля на несколько шагов вперед.
        
        Аргументы:
            df: Датафрейм с историческими данными.
            initial_amount: Начальная сумма для инвестирования.
            time_window: Размер временного окна для наблюдений.
            num_steps: Количество шагов для прогнозирования.
            tics_in_portfolio: Тикеры в портфеле ('all' или список).
            features: Признаки для использования. По умолчанию ['close', 'high', 'low'].
            normalize_df: Метод нормализации.
            
        Возвращает:
            DataFrame с прогнозируемыми весами для каждого шага.
        """
        # Настраиваем параметры по умолчанию
        if features is None:
            features = ['close','high','low','volume','open']
        
        # Создаем среду для получения наблюдений
        env = PortfolioOptimizationEnv(
            df=df,
            initial_amount=initial_amount,
            time_window=time_window,
            features=features,
            #normalize_df=normalize_df,
            tics_in_portfolio=tics_in_portfolio,
            comission_fee_pct=0.0025,
            reward_scaling=0.01,
            return_last_action=True,
        )
        
        # Получаем начальное состояние
        if env._new_gym_api:
            observation, info = env.reset()
        else:
            observation = env.reset()
        
        weights_history = []
        
        # Получаем информацию о тикерах
        tic_list = env._tic_list
        
        # Прогнозируем на num_steps шагов вперед
        for _ in range(num_steps):
            # Если наблюдение - словарь
            if isinstance(observation, dict):
                state = observation['state']
                last_action = observation['last_action']
            else:
                state = observation
                last_action = env._actions_memory[-1]
            
            # Получаем прогноз весов
            predicted_weights = self.predict_weights(state, last_action)
            
            # Сохраняем веса и временную метку
            step_info = {
                'date': env._info.get('end_time'),
                'cash': predicted_weights[0]
            }
            print(env._info.get('end_time'))
            # Добавляем веса для каждого тикера
            for i, tic in enumerate(tic_list):
                step_info[tic] = predicted_weights[i+1]
                
            weights_history.append(step_info)
            
            # Делаем шаг в среде
            observation, _, done, *_ = env.step(predicted_weights)
            
            if done:
                break
                
        # Создаем DataFrame с историей весов
        weights_df = pd.DataFrame(weights_history)
        if 'date' in weights_df.columns:
            weights_df.set_index('date', inplace=True)
            
        return weights_df
    
    def backtest(self, df, initial_amount, time_window, tics_in_portfolio="all", 
                features=None, normalize_df=None, online_learning=True,
                learning_rate=0.01, online_period=10, optimizer_class=torch.optim.AdamW):
        """Проводит бэктестинг модели на исторических данных.
        
        Аргументы:
            df: Датафрейм с историческими данными.
            initial_amount: Начальная сумма для инвестирования.
            time_window: Размер временного окна для наблюдений.
            tics_in_portfolio: Тикеры в портфеле ('all' или список).
            features: Признаки для использования. По умолчанию ['close', 'high', 'low'].
            normalize_df: Метод нормализации.
            online_learning: Включить онлайн-обучение во время бэктестинга.
            learning_rate: Скорость обучения при онлайн-обучении.
            online_period: Период, через который происходит обновление модели.
            optimizer_class: Класс оптимизатора для онлайн-обучения.
            
        Возвращает:
            Кортеж, содержащий:
            - last_action_final: Последний observation['last_action']
            - results_df: DataFrame с историей стоимости портфеля и распределением весов (для mlflow)
            - model_train: Обученная модель (если online_learning=True)
        """
        # Настраиваем параметры по умолчанию
        if features is None:
            features = ['close',	'high',	'low',	'volume',	'open']
            
        # Создаем среду для бэктестинга
        env = PortfolioOptimizationEnv(
            df=df,
            initial_amount=initial_amount,
            time_window=time_window,
            features=features,
            normalize_df=normalize_df,
            comission_fee_pct=0.0025,
            reward_scaling=0.01,
            tics_in_portfolio=tics_in_portfolio,
            return_last_action=True,
        )
        
        # Получаем начальное состояние
        if env._new_gym_api:
            observation, info = env.reset()
        else:
            observation = env.reset()
            
        done = False
        steps = 0
        last_action_final = None  # Будем хранить только последний last_action
        
        # Настройка компонентов для онлайн-обучения, если оно включено
        if online_learning:
            # Создаем копию модели для обучения
            model_train = copy.deepcopy(self.model)
            # Переключаем в режим обучения
            model_train.train()
            # Создаем оптимизатор
            optimizer = optimizer_class(model_train.parameters(), lr=learning_rate)
            # Создаем буфер опыта
            replay_buffer = ReplayBuffer(capacity=online_period*2)
            # Создаем объект для управления памятью портфеля
            pvm = PVM(env.episode_length, env.portfolio_size)
            # Инициализируем PVM
            pvm.reset()
        else:
            # Используем обычную модель в режиме оценки
            model_train = self.model
            # Создаем объект для управления памятью портфеля
            pvm = PVM(env.episode_length, env.portfolio_size)
            # Инициализируем PVM
            pvm.reset()
        
        # Прогоняем модель через всю историю
        while not done:
            steps += 1
            
            # Получаем последнее действие и обрабатываем наблюдение
            last_action = pvm.retrieve()
            
            # Если наблюдение - словарь
            if isinstance(observation, dict):
                state = observation['state']
                # Сохраняем последний last_action из observation
                if 'last_action' in observation:
                    last_action_final = observation['last_action']
            else:
                state = observation
                
            # Подготавливаем данные для модели
            obs_batch = np.expand_dims(state, axis=0)
            last_action_batch = np.expand_dims(last_action, axis=0)
            
            # Получаем прогноз весов
            if online_learning:
                with torch.no_grad():
                    predicted_weights = model_train(obs_batch, last_action_batch)
            else:
                predicted_weights = self.predict_weights(state, last_action)
            
            # Добавляем действие в память
            pvm.add(predicted_weights)
            
            # Делаем шаг в среде
            next_obs, reward, done, info = env.step(predicted_weights)
            
            # Онлайн обучение
            if online_learning:
                # Добавляем опыт в буфер
                exp = (state, last_action, info["price_variation"], info.get("trf_mu", 1.0))
                replay_buffer.append(exp)
                
                # Обновляем модель каждые online_period шагов
                if steps % online_period == 0 and len(replay_buffer) >= online_period:
                    # Создаем датасет и даталоадер
                    dataset = RLDataset(replay_buffer)
                    dataloader = torch.utils.data.DataLoader(
                        dataset=dataset, 
                        batch_size=min(len(replay_buffer), online_period),
                        shuffle=False
                    )
                    
                    # Получаем батч данных
                    batch = next(iter(dataloader))
                    obs, last_actions, price_variations, trf_mu = batch
                    
                    # Преобразуем в тензоры на нужном устройстве
                    obs = torch.tensor(obs, device=self.device).float()
                    last_actions = torch.tensor(last_actions, device=self.device).float()
                    price_variations = torch.tensor(price_variations, device=self.device).float()
                    trf_mu = torch.tensor(trf_mu, device=self.device).float().unsqueeze(1)
                    
                    # Вычисляем выход модели (веса портфеля)
                    mu = model_train.mu(obs, last_actions)
                    
                    # Вычисляем функцию потерь (максимизация доходности)
                    # Это эквивалентно минимизации отрицательной доходности
                    policy_loss = -torch.mean(torch.log(torch.sum(mu * price_variations * trf_mu, dim=1)))
                    
                    # Обратное распространение ошибки и оптимизация
                    optimizer.zero_grad()
                    policy_loss.backward()
                    optimizer.step()
            
            # Переходим к следующему состоянию
            observation = next_obs
            
        # Получаем результаты бэктестинга для mlflow
        results = {
            'portfolio_values': env._asset_memory['final'],
            'dates': env._date_memory,
            'returns': env._portfolio_return_memory,
            'weights': env._actions_memory
        }
        
        # Создаем DataFrame с историей весов
        weights_history = []
        tic_list = env._tic_list
        
        for i, date in enumerate(results['dates']):
            step_info = {
                'date': date,
                'portfolio_value': results['portfolio_values'][i] if i < len(results['portfolio_values']) else None,
                'return': results['returns'][i] if i < len(results['returns']) else None,
                'cash': results['weights'][i][0] if i < len(results['weights']) else None
            }
            
            # Добавляем веса для каждого тикера
            for j, tic in enumerate(tic_list):
                step_info[f'{tic}_weight'] = results['weights'][i][j+1] if i < len(results['weights']) else None
                
            weights_history.append(step_info)
            
        results_df = pd.DataFrame(weights_history)
        if 'date' in results_df.columns:
            results_df.set_index('date', inplace=True)
        
        # Если использовалось онлайн-обучение, возвращаем also обученную модель
        if online_learning:
            return last_action_final, results_df, model_train
        else:
            return last_action_final, results_df


