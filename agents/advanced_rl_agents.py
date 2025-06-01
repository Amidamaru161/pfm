"""
Продвинутые RL агенты для оптимизации портфеля
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Dirichlet
import numpy as np
from collections import deque
import random
from typing import Dict, Tuple, Optional
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from environments.portfolio_memory import ReplayBuffer
from models.models import TransformerPM


class PPOAgent:
    """Proximal Policy Optimization для оптимизации портфеля"""
    
    def __init__(
        self,
        env,
        policy_network,
        policy_kwargs=None,
        lr_actor=3e-4,
        lr_critic=1e-3,
        gamma=0.99,
        eps_clip=0.2,
        k_epochs=4,
        entropy_coef=0.01,
        value_loss_coef=0.5,
        max_grad_norm=0.5,
        device="cuda:0"
    ):
        """
        Args:
            env: Среда для торговли
            policy_network: Класс сети политики
            policy_kwargs: Параметры для сети
            lr_actor: Learning rate для актора
            lr_critic: Learning rate для критика
            gamma: Коэффициент дисконтирования
            eps_clip: Параметр клиппинга PPO
            k_epochs: Количество эпох обновления
            entropy_coef: Коэффициент энтропии
            value_loss_coef: Коэффициент value loss
            max_grad_norm: Максимальная норма градиента
            device: Устройство для вычислений
        """
        self.env = env
        self.device = device
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.k_epochs = k_epochs
        self.entropy_coef = entropy_coef
        self.value_loss_coef = value_loss_coef
        self.max_grad_norm = max_grad_norm
        
        # Инициализация сетей
        policy_kwargs = policy_kwargs or {}
        self.actor = policy_network(**policy_kwargs).to(device)
        
        # Критик - отдельная сеть для оценки value function
        self.critic = self._build_critic(policy_kwargs).to(device)
        
        self.optimizer_actor = optim.Adam(self.actor.parameters(), lr=lr_actor)
        self.optimizer_critic = optim.Adam(self.critic.parameters(), lr=lr_critic)
        
        # Буферы для хранения траекторий
        self.states = []
        self.actions = []
        self.rewards = []
        self.log_probs = []
        self.values = []
        self.dones = []
        
    def _build_critic(self, policy_kwargs):
        """Построение сети критика"""
        class ValueNetwork(nn.Module):
            def __init__(self, input_shape, hidden_size=256):
                super().__init__()
                # Вычисляем размер входа
                n_features = input_shape[0]
                n_assets = input_shape[1]
                time_window = input_shape[2]
                input_size = n_features * n_assets * time_window
                
                self.flatten = nn.Flatten()
                self.fc1 = nn.Linear(input_size, hidden_size)
                self.fc2 = nn.Linear(hidden_size, hidden_size)
                self.fc3 = nn.Linear(hidden_size, 1)
                
            def forward(self, state):
                x = self.flatten(state)
                x = F.relu(self.fc1(x))
                x = F.relu(self.fc2(x))
                value = self.fc3(x)
                return value
        
        # Получаем размерность входа из среды
        obs_shape = self.env.observation_space.shape
        return ValueNetwork(obs_shape)
    
    def get_action(self, state, last_action):
        """Получение действия и его log probability"""
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        last_action_tensor = torch.FloatTensor(last_action).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            # Получаем параметры распределения от актора
            action_probs = self.actor.mu(state_tensor, last_action_tensor)
            
            # Используем Dirichlet для семплирования портфеля
            concentration = action_probs * 100  # Увеличиваем концентрацию
            dist = Dirichlet(concentration)
            action = dist.sample()
            log_prob = dist.log_prob(action)
            
            # Получаем value от критика
            value = self.critic(state_tensor)
        
        return action.cpu().numpy().squeeze(), log_prob.item(), value.item()
    
    def store_transition(self, state, action, reward, log_prob, value, done):
        """Сохранение перехода в буфер"""
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.log_probs.append(log_prob)
        self.values.append(value)
        self.dones.append(done)
    
    def compute_returns(self):
        """Вычисление дисконтированных returns и advantages"""
        returns = []
        advantages = []
        
        # Вычисляем returns с конца
        discounted_reward = 0
        for i in reversed(range(len(self.rewards))):
            if self.dones[i]:
                discounted_reward = 0
            discounted_reward = self.rewards[i] + self.gamma * discounted_reward
            returns.insert(0, discounted_reward)
        
        # Нормализация returns
        returns = torch.tensor(returns, dtype=torch.float32).to(self.device)
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)
        
        # Вычисляем advantages
        values = torch.tensor(self.values, dtype=torch.float32).to(self.device)
        advantages = returns - values
        
        return returns, advantages
    
    def update(self):
        """Обновление политики используя PPO"""
        # Преобразуем данные в тензоры
        states = torch.FloatTensor(np.array(self.states)).to(self.device)
        actions = torch.FloatTensor(np.array(self.actions)).to(self.device)
        old_log_probs = torch.FloatTensor(self.log_probs).to(self.device)
        
        # Вычисляем returns и advantages
        returns, advantages = self.compute_returns()
        
        # Обновляем политику k_epochs раз
        for _ in range(self.k_epochs):
            # Получаем текущие log probabilities
            # Для этого нужны last_actions - берём их со сдвигом
            last_actions = torch.cat([
                torch.ones(1, actions.shape[1]).to(self.device),
                actions[:-1]
            ])
            
            # Получаем параметры распределения
            action_probs = self.actor.mu(states, last_actions)
            concentration = action_probs * 100
            dist = Dirichlet(concentration)
            
            # Вычисляем log probabilities для взятых действий
            log_probs = dist.log_prob(actions)
            entropy = dist.entropy().mean()
            
            # Вычисляем ratio для PPO
            ratios = torch.exp(log_probs - old_log_probs)
            
            # Clipped surrogate loss
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages
            actor_loss = -torch.min(surr1, surr2).mean()
            
            # Value loss
            values = self.critic(states).squeeze()
            value_loss = F.mse_loss(values, returns)
            
            # Общий loss
            loss = actor_loss + self.value_loss_coef * value_loss - self.entropy_coef * entropy
            
            # Обновление актора
            self.optimizer_actor.zero_grad()
            self.optimizer_critic.zero_grad()
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
            torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
            
            self.optimizer_actor.step()
            self.optimizer_critic.step()
        
        # Очищаем буферы
        self.clear_buffers()
    
    def clear_buffers(self):
        """Очистка буферов траекторий"""
        self.states.clear()
        self.actions.clear()
        self.rewards.clear()
        self.log_probs.clear()
        self.values.clear()
        self.dones.clear()


class SACAgent:
    """Soft Actor-Critic для оптимизации портфеля"""
    
    def __init__(
        self,
        env,
        policy_network,
        policy_kwargs=None,
        lr=3e-4,
        gamma=0.99,
        tau=0.005,
        alpha=0.2,
        automatic_entropy_tuning=True,
        buffer_size=1000000,
        batch_size=256,
        device="cuda:0"
    ):
        """
        Args:
            env: Среда для торговли
            policy_network: Класс сети политики
            policy_kwargs: Параметры для сети
            lr: Learning rate
            gamma: Коэффициент дисконтирования
            tau: Коэффициент soft update
            alpha: Температура энтропии
            automatic_entropy_tuning: Автоматическая настройка альфы
            buffer_size: Размер replay buffer
            batch_size: Размер батча
            device: Устройство для вычислений
        """
        self.env = env
        self.device = device
        self.gamma = gamma
        self.tau = tau
        self.alpha = alpha
        self.batch_size = batch_size
        
        # Размерности
        self.action_dim = env.action_space.shape[0]
        
        # Q-сети
        self.q_net1 = self._build_q_network(env.observation_space.shape).to(device)
        self.q_net2 = self._build_q_network(env.observation_space.shape).to(device)
        self.target_q_net1 = self._build_q_network(env.observation_space.shape).to(device)
        self.target_q_net2 = self._build_q_network(env.observation_space.shape).to(device)
        
        # Копируем веса в target сети
        self.target_q_net1.load_state_dict(self.q_net1.state_dict())
        self.target_q_net2.load_state_dict(self.q_net2.state_dict())
        
        # Политика
        policy_kwargs = policy_kwargs or {}
        self.policy = policy_network(**policy_kwargs).to(device)
        
        # Оптимизаторы
        self.q_optimizer = optim.Adam(
            list(self.q_net1.parameters()) + list(self.q_net2.parameters()), 
            lr=lr
        )
        self.policy_optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        
        # Автоматическая настройка энтропии
        if automatic_entropy_tuning:
            self.target_entropy = -self.action_dim
            self.log_alpha = torch.zeros(1, requires_grad=True, device=device)
            self.alpha_optimizer = optim.Adam([self.log_alpha], lr=lr)
        
        # Replay buffer
        self.replay_buffer = deque(maxlen=buffer_size)
    
    def _build_q_network(self, obs_shape):
        """Построение Q-сети"""
        class QNetwork(nn.Module):
            def __init__(self, obs_shape, action_dim, hidden_size=256):
                super().__init__()
                n_features = obs_shape[0]
                n_assets = obs_shape[1]
                time_window = obs_shape[2]
                state_size = n_features * n_assets * time_window
                
                self.flatten = nn.Flatten()
                self.fc1 = nn.Linear(state_size + action_dim, hidden_size)
                self.fc2 = nn.Linear(hidden_size, hidden_size)
                self.fc3 = nn.Linear(hidden_size, 1)
                
            def forward(self, state, action):
                x = self.flatten(state)
                x = torch.cat([x, action], dim=1)
                x = F.relu(self.fc1(x))
                x = F.relu(self.fc2(x))
                q_value = self.fc3(x)
                return q_value
        
        return QNetwork(obs_shape, self.action_dim)
    
    def get_action(self, state, last_action, evaluate=False):
        """Получение действия от политики"""
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        last_action_tensor = torch.FloatTensor(last_action).unsqueeze(0).to(self.device)
        
        if evaluate:
            with torch.no_grad():
                action = self.policy.mu(state_tensor, last_action_tensor)
                return action.cpu().numpy().squeeze()
        else:
            # Добавляем шум для exploration
            action = self.policy.mu(state_tensor, last_action_tensor)
            
            # Используем Dirichlet для exploration
            if random.random() < 0.1:  # 10% exploration
                concentration = torch.ones_like(action) * 2
                dist = Dirichlet(concentration)
                action = dist.sample()
            
            return action.cpu().detach().numpy().squeeze()
    
    def update(self):
        """Обновление сетей"""
        if len(self.replay_buffer) < self.batch_size:
            return
        
        # Семплируем батч
        batch = random.sample(self.replay_buffer, self.batch_size)
        states, last_actions, actions, rewards, next_states, next_last_actions, dones = zip(*batch)
        
        states = torch.FloatTensor(states).to(self.device)
        last_actions = torch.FloatTensor(last_actions).to(self.device)
        actions = torch.FloatTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        next_last_actions = torch.FloatTensor(next_last_actions).to(self.device)
        dones = torch.FloatTensor(dones).unsqueeze(1).to(self.device)
        
        with torch.no_grad():
            # Получаем действия для следующих состояний
            next_actions = self.policy.mu(next_states, next_last_actions)
            
            # Вычисляем target Q values
            target_q1 = self.target_q_net1(next_states, next_actions)
            target_q2 = self.target_q_net2(next_states, next_actions)
            target_q = torch.min(target_q1, target_q2)
            
            # Добавляем энтропийный бонус
            next_concentration = next_actions * 100
            next_dist = Dirichlet(next_concentration)
            next_log_prob = next_dist.log_prob(next_actions)
            
            target_value = rewards + (1 - dones) * self.gamma * (target_q - self.alpha * next_log_prob.unsqueeze(1))
        
        # Обновляем Q-сети
        q1 = self.q_net1(states, actions)
        q2 = self.q_net2(states, actions)
        q_loss = F.mse_loss(q1, target_value) + F.mse_loss(q2, target_value)
        
        self.q_optimizer.zero_grad()
        q_loss.backward()
        self.q_optimizer.step()
        
        # Обновляем политику
        new_actions = self.policy.mu(states, last_actions)
        concentration = new_actions * 100
        dist = Dirichlet(concentration)
        log_prob = dist.log_prob(new_actions)
        
        q1_new = self.q_net1(states, new_actions)
        q2_new = self.q_net2(states, new_actions)
        q_new = torch.min(q1_new, q2_new)
        
        policy_loss = (self.alpha * log_prob.unsqueeze(1) - q_new).mean()
        
        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()
        
        # Обновляем alpha если нужно
        if hasattr(self, 'alpha_optimizer'):
            alpha_loss = -(self.log_alpha * (log_prob + self.target_entropy).detach()).mean()
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()
            self.alpha = self.log_alpha.exp().item()
        
        # Soft update target сетей
        self._soft_update()
    
    def _soft_update(self):
        """Soft update target сетей"""
        for target_param, param in zip(self.target_q_net1.parameters(), self.q_net1.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        
        for target_param, param in zip(self.target_q_net2.parameters(), self.q_net2.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
    
    def store_transition(self, state, last_action, action, reward, next_state, next_last_action, done):
        """Сохранение перехода в replay buffer"""
        self.replay_buffer.append((state, last_action, action, reward, next_state, next_last_action, done)) 