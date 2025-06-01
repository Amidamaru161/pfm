import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch_geometric.nn import RGCNConv, GATv2Conv
from torch_geometric.data import Data, Batch
import os
import json
import copy
from torch.utils.data import DataLoader
from environments.portfolio_memory import PVM, ReplayBuffer, RLDataset, apply_portfolio_noise
from tqdm import tqdm
from models.hyper_net import HyperNetPortfolioRL


class HyperNetPolicyGradient:
    """Расширенная реализация алгоритма Policy Gradient для обучения агентов оптимизации портфеля
    с поддержкой смешанных критериев оптимизации и многозадачного обучения.
    """
    
    def __init__(
        self,
        env,
        policy=HyperNetPortfolioRL,
        policy_kwargs=None,
        validation_env=None,
        batch_size=100,
        lr=1e-3,
        action_noise=0,
        risk_aversion=0.5,  # Параметр избегания риска (больше = меньше риска)
        entropy_reg=0.01,   # Коэффициент регуляризации энтропии (для исследования)
        device="cuda:0",
        mixed_precision=True,  # Использование смешанной точности для ускорения обучения
        gradient_clip=1.0,     # Ограничение градиентов для стабильности
        max_episodes=1000,     # Максимальное количество эпизодов
        target_sharpe=1.5,     # Целевой коэффициент Шарпа
        adaptive_risk=True,    # Адаптивная настройка параметра риска
        eval_interval=10,      # Интервал оценки
        early_stopping=True,   # Ранняя остановка
        patience=20,           # Количество эпизодов для ранней остановки
        ensemble_size=0,       # Размер ансамбля (0 - без ансамбля)
        save_path=None,        # Путь для сохранения модели
    ):
        """Инициализирует расширенный алгоритм Policy Gradient для оптимизации портфеля.
        
        Args:
          env: Среда обучения.
          policy: Архитектура политики, которая будет использоваться.
          policy_kwargs: Аргументы для сети политики.
          validation_env: Среда валидации.
          batch_size: Размер батча для обучения нейронной сети.
          lr: Скорость обучения нейронной сети политики.
          action_noise: Параметр шума (от 0 до 1), применяемый во время обучения.
          risk_aversion: Параметр избегания риска (от 0 до 1).
          entropy_reg: Коэффициент регуляризации энтропии для поощрения исследования.
          device: Устройство, на котором запускается нейронная сеть.
          mixed_precision: Использовать ли смешанную точность для ускорения обучения.
          gradient_clip: Значение для ограничения градиентов.
          max_episodes: Максимальное количество эпизодов для обучения.
          target_sharpe: Целевой коэффициент Шарпа для ранней остановки.
          adaptive_risk: Адаптивно настраивать параметр избегания риска.
          eval_interval: Интервал эпизодов для оценки на валидационной среде.
          early_stopping: Использовать ли раннюю остановку.
          patience: Количество эпизодов без улучшения для ранней остановки.
          ensemble_size: Размер ансамбля моделей (0 - без ансамбля).
          save_path: Путь для сохранения модели.
        """
        # Инициализация параметров обучения
        self.policy = policy
        self.policy_kwargs = {} if policy_kwargs is None else policy_kwargs
        self.validation_env = validation_env
        self.batch_size = batch_size
        self.lr = lr
        self.action_noise = action_noise
        self.risk_aversion = risk_aversion
        self.entropy_reg = entropy_reg
        self.device = device
        self.mixed_precision = mixed_precision
        self.gradient_clip = gradient_clip
        self.max_episodes = max_episodes
        self.target_sharpe = target_sharpe
        self.adaptive_risk = adaptive_risk
        self.eval_interval = eval_interval
        self.early_stopping = early_stopping
        self.patience = patience
        self.ensemble_size = ensemble_size
        self.save_path = save_path
        
        # Настройка обучения и среды
        self.train_env = env
        self._setup_train(env, self.policy, self.batch_size, self.lr)
        
        # Дополнительно отслеживаем метрики обучения
        self.train_metrics = {
            'policy_loss': [],
            'return': [],
            'sharpe_ratio': [],
            'max_drawdown': [],
            'win_rate': [],
            'consistency': [],
            'turnover': []
        }
        
        # Для ранней остановки
        self.best_sharpe = -float('inf')
        self.epochs_no_improve = 0
        self.best_policy = None
        
        # Для ансамбля моделей
        self.ensemble = []
        if self.ensemble_size > 0:
            for _ in range(self.ensemble_size):
                self.ensemble.append(copy.deepcopy(self.train_policy))
                
        # Настройка scaler для смешанной точности
        self.scaler = torch.cuda.amp.GradScaler() if self.mixed_precision and torch.cuda.is_available() else None
        
    def _setup_train(self, env, policy, batch_size, lr):
        """Настраивает алгоритм перед обучением с расширенными возможностями.
        
        Args:
            env: Среда обучения.
            policy: Архитектура политики.
            batch_size: Размер батча.
            lr: Скорость обучения.
        """
        # Инициализация политики и оптимизатора
        self.train_policy = policy(**self.policy_kwargs).to(self.device)
        
        # Оптимизатор с регуляризацией L2 (weight decay)
        self.train_optimizer = torch.optim.AdamW(
            self.train_policy.parameters(), 
            lr=lr,
            weight_decay=1e-5,
            betas=(0.9, 0.999)
        )
        
        # Планировщик скорости обучения
        self.lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.train_optimizer, 
            T_0=20,
            T_mult=2,
            eta_min=lr/10
        )
        
        # Инициализация буфера воспроизведения и памяти векторов портфеля
        self.train_batch_size = batch_size
        self.train_buffer = ReplayBuffer(capacity=batch_size * 2)  # Увеличиваем емкость для более стабильного обучения
        self.train_pvm = PVM(env.episode_length, env.portfolio_size)
        
        # Настройка dataset и dataloader
        dataset = RLDataset(self.train_buffer)
        self.train_dataloader = DataLoader(
            dataset=dataset, 
            batch_size=batch_size, 
            shuffle=None,  # Для IterableDataset shuffle должен быть None
            pin_memory=True,
            num_workers=0  # При необходимости можно увеличить
        )
        
        # Настройка тестовых компонентов
        if self.validation_env is not None:
            self.test_policy = copy.deepcopy(self.train_policy)
            self.test_optimizer = torch.optim.AdamW(
                self.test_policy.parameters(), 
                lr=lr/2,  # Меньшая скорость обучения для тестирования
                weight_decay=1e-5
            )
            self.test_buffer = ReplayBuffer(capacity=batch_size)
            self.test_pvm = PVM(self.validation_env.episode_length, self.validation_env.portfolio_size)
            test_dataset = RLDataset(self.test_buffer)
            self.test_dataloader = DataLoader(
                dataset=test_dataset, 
                batch_size=batch_size, 
                shuffle=None,  # Для IterableDataset shuffle должен быть None
                pin_memory=True
            )
    
    def train(self, episodes=None):
        """Последовательность обучения с расширенными функциями.
        
        Args:
            episodes: Количество эпизодов для симуляции (если None, используется self.max_episodes).
        
        Returns:
            Словарь с метриками обучения.
        """
        if episodes is None:
            episodes = self.max_episodes
            
        # Инициализация метрик для отслеживания прогресса
        episode_returns = []
        episode_lengths = []
        best_return = -float('inf')
        
        for episode in tqdm(range(1, episodes + 1), desc="Training"):
            # Сбрасываем среду и память портфеля
            obs = self.train_env.reset()
            self.train_pvm.reset()
            done = False
            episode_reward = 0
            step_count = 0
            turnover = 0  # Для отслеживания оборота портфеля
            
            # Собираем траекторию для эпизода
            while not done:
                step_count += 1
                
                # Определяем последнее действие и новое действие
                last_action = self.train_pvm.retrieve()
                obs_batch = np.expand_dims(obs, axis=0)
                last_action_batch = np.expand_dims(last_action, axis=0)
                
                # Получаем действие от политики с шумом для исследования
                action = apply_portfolio_noise(
                    self.train_policy(obs_batch, last_action_batch), 
                    self.action_noise * (1 - episode / episodes)  # Постепенно уменьшаем шум
                )
                
                # Рассчитываем оборот портфеля (L1 разница между весами)
                if step_count > 1:
                    turnover += np.sum(np.abs(action - last_action))
                
                # Сохраняем действие в памяти портфеля
                self.train_pvm.add(action)
                
                # Выполняем шаг в среде
                next_obs, reward, done, info = self.train_env.step(action)
                episode_reward += reward
                
                # Добавляем опыт в буфер воспроизведения
                exp = (obs, last_action, info["price_variation"], info["trf_mu"])
                self.train_buffer.append(exp)
                
                # Обновляем сети политики через градиентный подъем
                if len(self.train_buffer) >= self.train_batch_size:
                    loss_info = self._gradient_ascent()
                    for k, v in loss_info.items():
                        if k not in self.train_metrics:
                            self.train_metrics[k] = []
                        self.train_metrics[k].append(v)
                
                # Сохраняем следующее наблюдение для следующего шага
                obs = next_obs
            
            # Градиентный подъем с оставшимися данными буфера после эпизода
            if len(self.train_buffer) > 0:
                self._gradient_ascent()
                
            # Обновляем планировщик скорости обучения
            self.lr_scheduler.step()
            
            # Сохраняем метрики эпизода
            episode_returns.append(episode_reward)
            episode_lengths.append(step_count)
            
            # Адаптивная настройка параметра риска
            if self.adaptive_risk and episode > 10:
                returns = np.array(episode_returns[-10:])
                sharpe = returns.mean() / (returns.std() + 1e-6)
                
                # Настраиваем параметр риска в зависимости от текущего коэффициента Шарпа
                if sharpe < 0.8:  # Низкий коэффициент Шарпа - увеличиваем избегание риска
                    self.risk_aversion = min(0.9, self.risk_aversion + 0.05)
                elif sharpe > 1.5:  # Высокий коэффициент Шарпа - можем снизить избегание риска
                    self.risk_aversion = max(0.1, self.risk_aversion - 0.05)
            
            # Валидация на каждые eval_interval эпизодов
            if self.validation_env is not None and episode % self.eval_interval == 0:
                val_metrics = self.test(self.validation_env)
                
                # Ранняя остановка, если необходимо
                if self.early_stopping:
                    current_sharpe = val_metrics.get('sharpe_ratio', -float('inf'))
                    if current_sharpe > self.best_sharpe:
                        self.best_sharpe = current_sharpe
                        self.epochs_no_improve = 0
                        # Сохраняем лучшую модель
                        self.best_policy = copy.deepcopy(self.train_policy)
                        
                        # Сохраняем модель, если указан путь
                        if self.save_path:
                            self._save_model(self.save_path, episode, val_metrics)
                    else:
                        self.epochs_no_improve += 1
                        
                    # Проверяем условие ранней остановки
                    if self.epochs_no_improve >= self.patience:
                        print(f"Early stopping triggered after {episode} episodes")
                        break
                
                # Проверяем, достигли ли целевого коэффициента Шарпа
                if current_sharpe >= self.target_sharpe:
                    print(f"Target Sharpe ratio {self.target_sharpe} reached after {episode} episodes")
                    break
                
                # Выводим прогресс обучения
                print(f"Episode {episode}/{episodes}, Return: {episode_reward:.4f}, "
                      f"Sharpe: {current_sharpe:.4f}, Risk Aversion: {self.risk_aversion:.4f}")
                
            # Обновляем ансамбль моделей
            if self.ensemble_size > 0 and episode % 10 == 0:
                self._update_ensemble()
        
        # После обучения используем лучшую модель, если есть
        if self.early_stopping and self.best_policy is not None:
            self.train_policy = self.best_policy
            
        # Возвращаем метрики обучения
        return self.train_metrics
    
    def _gradient_ascent(self, test=False):
        """Выполняет шаг градиентного подъема в алгоритме policy gradient с расширенными критериями.
        
        Args:
            test: Если True, используются тестовые датлоадер и политика.
            
        Returns:
            Словарь с информацией о потерях.
        """
        # Выбираем соответствующие компоненты
        policy = self.test_policy if test else self.train_policy
        optimizer = self.test_optimizer if test else self.train_optimizer
        dataloader = self.test_dataloader if test else self.train_dataloader
        
        # Получаем данные батча из датлоадера
        try:
            batch = next(iter(dataloader))
        except StopIteration:
            return {}  # Возвращаем пустой словарь, если датлоадер пуст
            
        obs, last_actions, price_variations, trf_mu = batch
        
        # Переносим данные на устройство
        obs = obs.to(self.device)
        last_actions = last_actions.to(self.device)
        price_variations = price_variations.to(self.device)
        trf_mu = trf_mu.unsqueeze(1).to(self.device)
        
        # Выполняем вычисления с использованием смешанной точности, если включено
        if self.mixed_precision and self.scaler is not None:
            with torch.cuda.amp.autocast():
                # Получаем распределение весов от политики
                mu = policy.mu(obs, last_actions)
                
                # Основная цель PG: максимизация ожидаемой доходности
                expected_return = torch.sum(mu * price_variations * trf_mu, dim=1)
                log_returns = torch.log(expected_return + 1e-10)
                policy_return_loss = -torch.mean(log_returns)
                
                # Оценка риска: минимизация волатильности
                if self.risk_aversion > 0:
                    returns = torch.sum(mu * price_variations, dim=1)
                    portfolio_variance = torch.var(returns)
                    risk_loss = self.risk_aversion * portfolio_variance
                else:
                    risk_loss = torch.tensor(0.0, device=self.device)
                    
                # Регуляризация энтропии для поощрения исследования
                entropy = -torch.sum(mu * torch.log(mu + 1e-10), dim=1).mean()
                entropy_loss = -self.entropy_reg * entropy
                
                # Основной компонент предсказания доходности (для самодистилляции)
                if hasattr(policy, 'returns_predictor') and hasattr(policy, 'temporal_features'):
                    # Используем сохраненные предсказания доходности
                    returns_pred = policy.returns_pred  # [batch, assets]
                    # Убедимся, что размерности совпадают
                    target_returns = price_variations  # [batch, assets]
                    # Проверяем и выравниваем размерности
                    if returns_pred.size() != target_returns.size():
                        # Если размерности не совпадают, берем среднее по активам
                        returns_pred = returns_pred.mean(dim=-1, keepdim=True).expand_as(target_returns)
                    pred_loss = F.mse_loss(returns_pred, target_returns)
                else:
                    pred_loss = torch.tensor(0.0, device=self.device)
                
                # Объединяем все компоненты потерь
                total_loss = policy_return_loss + risk_loss + entropy_loss + 0.1 * pred_loss
            
            # Обновляем сеть политики с использованием scaler
            optimizer.zero_grad()
            self.scaler.scale(total_loss).backward()
            self.scaler.unscale_(optimizer)
            
            # Ограничиваем градиенты для стабильности
            if self.gradient_clip > 0:
                torch.nn.utils.clip_grad_norm_(policy.parameters(), self.gradient_clip)
                
            self.scaler.step(optimizer)
            self.scaler.update()
        else:
            # Стандартное вычисление без смешанной точности
            # Получаем распределение весов от политики
            mu = policy.mu(obs, last_actions)
            
            # Основная цель PG: максимизация ожидаемой доходности
            expected_return = torch.sum(mu * price_variations * trf_mu, dim=1)
            log_returns = torch.log(expected_return + 1e-10)
            policy_return_loss = -torch.mean(log_returns)
            
            # Оценка риска: минимизация волатильности
            if self.risk_aversion > 0:
                returns = torch.sum(mu * price_variations, dim=1)
                portfolio_variance = torch.var(returns)
                risk_loss = self.risk_aversion * portfolio_variance
            else:
                risk_loss = torch.tensor(0.0, device=self.device)
                
            # Регуляризация энтропии для поощрения исследования
            entropy = -torch.sum(mu * torch.log(mu + 1e-10), dim=1).mean()
            entropy_loss = -self.entropy_reg * entropy
            
            # Основной компонент предсказания доходности (для самодистилляции)
            if hasattr(policy, 'returns_predictor') and hasattr(policy, 'temporal_features'):
                # Используем сохраненные предсказания доходности
                returns_pred = policy.returns_pred  # [batch, assets]
                # Убедимся, что размерности совпадают
                target_returns = price_variations  # [batch, assets]
                # Проверяем и выравниваем размерности
                if returns_pred.size() != target_returns.size():
                    # Если размерности не совпадают, берем среднее по активам
                    returns_pred = returns_pred.mean(dim=-1, keepdim=True).expand_as(target_returns)
                pred_loss = F.mse_loss(returns_pred, target_returns)
            else:
                pred_loss = torch.tensor(0.0, device=self.device)
            
            # Объединяем все компоненты потерь
            total_loss = policy_return_loss + risk_loss + entropy_loss + 0.1 * pred_loss
            
            # Обновляем сеть политики
            optimizer.zero_grad()
            total_loss.backward()
            
            # Ограничиваем градиенты для стабильности
            if self.gradient_clip > 0:
                torch.nn.utils.clip_grad_norm_(policy.parameters(), self.gradient_clip)
                
            optimizer.step()
        
        # Собираем и возвращаем информацию о потерях
        loss_info = {
            'policy_loss': policy_return_loss.item(),
            'risk_loss': risk_loss.item(),
            'entropy_loss': entropy_loss.item(),
            'pred_loss': pred_loss.item() if isinstance(pred_loss, torch.Tensor) else 0.0,
            'total_loss': total_loss.item()
        }
        
        return loss_info
    
    def test(self, test_env, policy=None, online_training_period=10, learning_rate=None, optimizer=None):
        """Тестирует модель в тестовом окружении.
        
        Args:
            test_env: Среда для тестирования.
            policy: Архитектура политики (не используется, так как у нас своя архитектура).
            online_training_period: Период, в котором будет происходить онлайн-обучение.
            learning_rate: Скорость обучения (если None, используется половина скорости обучения при тренировке).
            optimizer: Оптимизатор (не используется, так как у нас свой оптимизатор).
            
        Returns:
            Словарь с метриками тестирования.
        """
        # Инициализируем тестовую политику, если она еще не создана
        if not hasattr(self, 'test_policy'):
            self.test_policy = copy.deepcopy(self.train_policy)
            self.test_optimizer = torch.optim.AdamW(
                self.test_policy.parameters(),
                lr=self.lr/2,
                weight_decay=1e-5
            )
            
        # Если указана новая скорость обучения, обновляем оптимизатор
        if learning_rate is not None:
            self.test_optimizer = torch.optim.AdamW(
                self.test_policy.parameters(),
                lr=learning_rate,
                weight_decay=1e-5
            )
            
        # Выполняем тестирование
        metrics = self._test_impl(
            env=test_env,
            online_training_period=online_training_period,
            lr=learning_rate,
            reset_policy=False
        )
        
        return metrics
    
    def _test_impl(self, env, online_training_period=10, lr=None, reset_policy=False):
        """Внутренняя реализация тестирования модели.
        
        Args:
            env: Среда для тестирования.
            online_training_period: Период, в котором будет происходить онлайн-обучение.
            lr: Скорость обучения нейронной сети политики. Если None, используется половина
                скорости обучения при тренировке.
            reset_policy: Сбрасывать ли политику перед тестированием (для изолированной оценки).
            
        Returns:
            Словарь с метриками тестирования.
        """
        # Настраиваем тестовую среду и политику
        self._setup_test(env, reset_policy)
        
        # Инициализация параметров тестирования
        obs = env.reset()
        self.test_pvm.reset()
        done = False
        steps = 0
        test_return = 0
        returns = []
        actions_history = []
        
        # Цикл тестирования
        while not done:
            steps += 1
            
            # Определяем действие
            last_action = self.test_pvm.retrieve()
            obs_batch = np.expand_dims(obs, axis=0)
            last_action_batch = np.expand_dims(last_action, axis=0)
            
            # Используем ансамбль, если он включен
            if self.ensemble_size > 0 and len(self.ensemble) > 0:
                # Получаем предсказания от всех моделей в ансамбле
                actions = []
                for model in self.ensemble:
                    action = model(obs_batch, last_action_batch)
                    actions.append(action)
                # Усредняем действия
                action = np.mean(np.array(actions), axis=0)
            else:
                # Используем основную тестовую политику
                action = self.test_policy(obs_batch, last_action_batch)
            
            self.test_pvm.add(action)
            actions_history.append(action)
            
            # Выполняем шаг в среде
            next_obs, reward, done, info = env.step(action)
            test_return += reward
            returns.append(reward)
            
            # Добавляем опыт в тестовый буфер воспроизведения
            exp = (obs, last_action, info["price_variation"], info["trf_mu"])
            self.test_buffer.append(exp)
            
            # Обновляем политику через онлайн-обучение
            if steps <= online_training_period and len(self.test_buffer) >= self.batch_size:
                self._gradient_ascent(test=True)
            
            # Сохраняем следующее наблюдение
            obs = next_obs
        
        # Рассчитываем метрики тестирования
        returns = np.array(returns)
        sharpe_ratio = returns.mean() / (returns.std() + 1e-6)
        
        # Рассчитываем максимальную просадку
        cumulative_returns = np.cumsum(returns)
        max_drawdown = 0
        peak = cumulative_returns[0]
        for value in cumulative_returns:
            if value > peak:
                peak = value
            drawdown = (peak - value) / peak
            max_drawdown = max(max_drawdown, drawdown)
        
        # Рассчитываем процент выигрышных сделок
        win_rate = np.mean(returns > 0)
        
        # Рассчитываем консистентность (отношение положительных дней к общему числу дней)
        consistency = np.mean(returns > 0)
        
        # Рассчитываем оборот портфеля
        actions_array = np.array(actions_history)
        turnover = np.mean([np.sum(np.abs(actions_array[i] - actions_array[i-1])) 
                          for i in range(1, len(actions_array))])
        
        # Собираем все метрики
        metrics = {
            'return': test_return,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'win_rate': win_rate,
            'consistency': consistency,
            'turnover': turnover,
            'steps': steps
        }
        
        return metrics
    
    def _setup_test(self, env, reset_policy=False):
        """Настраивает тестовую среду и политику.
        
        Args:
            env: Среда для тестирования.
            reset_policy: Сбрасывать ли политику перед тестированием.
        """
        if reset_policy:
            self.test_policy = self.policy(**self.policy_kwargs).to(self.device)
            self.test_optimizer = torch.optim.AdamW(
                self.test_policy.parameters(),
                lr=self.lr/2,
                weight_decay=1e-5
            )
        else:
            self.test_policy = copy.deepcopy(self.train_policy)
            self.test_optimizer = torch.optim.AdamW(
                self.test_policy.parameters(),
                lr=self.lr/2,
                weight_decay=1e-5
            )
        
        self.test_buffer = ReplayBuffer(capacity=self.batch_size)
        self.test_pvm = PVM(env.episode_length, env.portfolio_size)
        
        test_dataset = RLDataset(self.test_buffer)
        self.test_dataloader = DataLoader(
            dataset=test_dataset,
            batch_size=self.batch_size,
            shuffle=None,  # Для IterableDataset shuffle должен быть None
            pin_memory=True
        )
    
    def _update_ensemble(self):
        """Обновляет ансамбль моделей."""
        if self.ensemble_size == 0:
            return
            
        # Создаем новый ансамбль
        new_ensemble = []
        for _ in range(self.ensemble_size):
            # Создаем копию текущей модели
            model_copy = copy.deepcopy(self.train_policy)
            # Добавляем небольшой шум к весам для разнообразия
            for param in model_copy.parameters():
                if param.requires_grad:
                    param.data += torch.randn_like(param) * 0.01
            new_ensemble.append(model_copy)
        
        # Обновляем ансамбль
        self.ensemble = new_ensemble
    
    def _save_model(self, path, episode, metrics):
        """Сохраняет модель и метрики.
        
        Args:
            path: Путь для сохранения.
            episode: Номер эпизода.
            metrics: Словарь с метриками.
        """
        save_dict = {
            'episode': episode,
            'model_state_dict': self.train_policy.state_dict(),
            'optimizer_state_dict': self.train_optimizer.state_dict(),
            'metrics': metrics,
            'risk_aversion': self.risk_aversion,
            'best_sharpe': self.best_sharpe
        }
        
        # Создаем директорию, если она не существует
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        # Сохраняем модель
        torch.save(save_dict, path)
        
        # Сохраняем метрики в отдельный файл
        metrics_path = path.replace('.pt', '_metrics.json')
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=4)
    
    def load_model(self, path):
        """Загружает сохраненную модель.
        
        Args:
            path: Путь к сохраненной модели.
        """
        checkpoint = torch.load(path)
        
        self.train_policy.load_state_dict(checkpoint['model_state_dict'])
        self.train_optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        self.risk_aversion = checkpoint.get('risk_aversion', self.risk_aversion)
        self.best_sharpe = checkpoint.get('best_sharpe', self.best_sharpe)
        
        # Обновляем тестовую политику
        self.test_policy = copy.deepcopy(self.train_policy)
        
        return checkpoint.get('metrics', {})