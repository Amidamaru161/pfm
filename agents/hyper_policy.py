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
from typing import Dict, Optional
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents.policy_gradient import PolicyGradient
from models.models import EIIE


class HyperNet(nn.Module):
    """Гиперсеть для генерации параметров основной политики"""
    
    def __init__(self, context_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.context_encoder = nn.Sequential(
            nn.Linear(context_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        # Генераторы для разных слоёв основной сети
        self.weight_generators = nn.ModuleDict()
        self.bias_generators = nn.ModuleDict()
        
    def add_generator(self, layer_name: str, weight_shape: torch.Size, bias_shape: Optional[torch.Size] = None):
        """Добавление генератора для конкретного слоя"""
        weight_size = np.prod(weight_shape)
        self.weight_generators[layer_name] = nn.Linear(self.context_encoder[-2].out_features, weight_size)
        
        if bias_shape is not None:
            bias_size = np.prod(bias_shape)
            self.bias_generators[layer_name] = nn.Linear(self.context_encoder[-2].out_features, bias_size)
    
    def forward(self, context: torch.Tensor) -> Dict[str, Dict[str, torch.Tensor]]:
        """Генерация параметров на основе контекста"""
        encoded = self.context_encoder(context)
        
        params = {}
        for layer_name, generator in self.weight_generators.items():
            weight = generator(encoded)
            params[layer_name] = {'weight': weight}
            
            if layer_name in self.bias_generators:
                bias = self.bias_generators[layer_name](encoded)
                params[layer_name]['bias'] = bias
        
        return params


class AdaptivePolicy(nn.Module):
    """Адаптивная политика с параметрами от гиперсети"""
    
    def __init__(self, base_policy: nn.Module, hypernet: HyperNet):
        super().__init__()
        self.base_policy = base_policy
        self.hypernet = hypernet
        
        # Регистрируем генераторы для каждого слоя базовой политики
        for name, module in base_policy.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                self.hypernet.add_generator(
                    name, 
                    module.weight.shape,
                    module.bias.shape if module.bias is not None else None
                )
    
    def forward(self, observation: torch.Tensor, last_action: torch.Tensor, context: torch.Tensor):
        """Forward pass с адаптацией параметров"""
        # Генерируем параметры на основе контекста
        generated_params = self.hypernet(context)
        
        # Применяем сгенерированные параметры к базовой политике
        with torch.no_grad():
            for name, module in self.base_policy.named_modules():
                if name in generated_params:
                    # Модулируем существующие веса
                    weight_mod = generated_params[name]['weight'].reshape(module.weight.shape)
                    module.weight.data = module.weight.data + 0.1 * weight_mod
                    
                    if 'bias' in generated_params[name] and module.bias is not None:
                        bias_mod = generated_params[name]['bias'].reshape(module.bias.shape)
                        module.bias.data = module.bias.data + 0.1 * bias_mod
        
        return self.base_policy(observation, last_action)


class HyperNetPolicyGradient(PolicyGradient):
    """Policy Gradient с HyperNet для адаптивной оптимизации"""
    
    def __init__(
        self,
        env,
        policy=EIIE,
        policy_kwargs=None,
        context_dim=32,
        hypernet_hidden=256,
        validation_env=None,
        batch_size=100,
        lr=1e-3,
        hypernet_lr=1e-4,
        action_noise=0,
        optimizer=torch.optim.AdamW,
        device="cuda:0"
    ):
        """
        Args:
            env: Training Environment
            policy: Policy architecture
            policy_kwargs: Policy network arguments
            context_dim: Dimension of market context
            hypernet_hidden: Hidden dimension of hypernet
            validation_env: Validation environment
            batch_size: Batch size
            lr: Policy learning rate
            hypernet_lr: Hypernet learning rate
            action_noise: Noise for exploration
            optimizer: Optimizer class
            device: Device for computation
        """
        # Инициализируем базовый класс
        super().__init__(
            env, policy, policy_kwargs, validation_env,
            batch_size, lr, action_noise, optimizer, device
        )
        
        self.context_dim = context_dim
        
        # Создаём гиперсеть
        self.hypernet = HyperNet(context_dim, hypernet_hidden).to(device)
        self.hypernet_optimizer = optimizer(self.hypernet.parameters(), lr=hypernet_lr)
        
        # Заменяем политику на адаптивную
        self.train_policy = AdaptivePolicy(self.train_policy, self.hypernet).to(device)
        
        # Буфер для контекста
        self.context_buffer = []
    
    def _extract_market_context(self, env_data) -> torch.Tensor:
        """Извлечение рыночного контекста из данных среды"""
        # Простой пример: используем статистики последних цен
        if hasattr(self.train_env, '_data') and self.train_env._data is not None:
            prices = self.train_env._data[self.train_env._features].values
            
            context = []
            # Волатильность
            context.append(np.std(prices, axis=0).mean())
            # Тренд
            context.append(np.mean(np.diff(prices, axis=0)))
            # Корреляция между активами
            if prices.shape[1] > 1:
                corr_matrix = np.corrcoef(prices.T)
                context.append(np.mean(corr_matrix[np.triu_indices_from(corr_matrix, k=1)]))
            else:
                context.append(0.0)
            
            # Заполняем до нужной размерности
            while len(context) < self.context_dim:
                context.append(0.0)
            
            return torch.tensor(context[:self.context_dim], dtype=torch.float32)
        else:
            return torch.zeros(self.context_dim, dtype=torch.float32)
    
    def train(self, episodes=100):
        """Обучение с адаптацией через гиперсеть"""
        for i in range(1, episodes + 1):
            obs = self.train_env.reset()
            self.train_pvm.reset()
            done = False
            
            # Извлекаем контекст для эпизода
            context = self._extract_market_context(self.train_env).to(self.device)
            
            while not done:
                last_action = self.train_pvm.retrieve()
                obs_batch = np.expand_dims(obs, axis=0)
                last_action_batch = np.expand_dims(last_action, axis=0)
                context_batch = context.unsqueeze(0)
                
                # Получаем действие с учётом контекста
                action = self.train_policy(obs_batch, last_action_batch, context_batch)
                action = apply_portfolio_noise(action, self.action_noise)
                self.train_pvm.add(action)
                
                next_obs, reward, done, info = self.train_env.step(action)
                
                # Сохраняем опыт с контекстом
                exp = (obs, last_action, info["price_variation"], info["trf_mu"], context)
                self.train_buffer.append(exp)
                self.context_buffer.append(context)
                
                if len(self.train_buffer) == self.train_batch_size:
                    self._gradient_ascent_with_hypernet()
                
                obs = next_obs
            
            # Финальное обновление
            self._gradient_ascent_with_hypernet()
            
            # Валидация
            if self.validation_env and i % 10 == 0:
                self.test(self.validation_env)
                print(f"Episode {i} completed")
    
    def _gradient_ascent_with_hypernet(self):
        """Gradient ascent с обновлением гиперсети"""
        if len(self.train_buffer) == 0:
            return
        
        # Получаем батч
        batch = list(self.train_buffer.buffer)
        self.train_buffer.clear()
        
        obs, last_actions, price_variations, trf_mu, contexts = zip(*batch)
        
        obs = torch.tensor(np.array(obs), dtype=torch.float32).to(self.device)
        last_actions = torch.tensor(np.array(last_actions), dtype=torch.float32).to(self.device)
        price_variations = torch.tensor(np.array(price_variations), dtype=torch.float32).to(self.device)
        trf_mu = torch.tensor(np.array(trf_mu), dtype=torch.float32).unsqueeze(1).to(self.device)
        contexts = torch.stack(contexts).to(self.device)
        
        # Forward pass через адаптивную политику
        mu = self.train_policy.base_policy.mu(obs, last_actions)
        
        # Policy loss
        policy_loss = -torch.mean(
            torch.log(torch.sum(mu * price_variations * trf_mu, dim=1))
        )
        
        # Обновляем все параметры
        self.train_optimizer.zero_grad()
        self.hypernet_optimizer.zero_grad()
        
        policy_loss.backward()
        
        self.train_optimizer.step()
        self.hypernet_optimizer.step()
        
        # Очищаем буфер контекста
        self.context_buffer.clear()
    
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