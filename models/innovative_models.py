"""
Инновационные архитектуры для оптимизации портфеля
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch_geometric.nn import GATConv, GraphConv, global_mean_pool
from torch_geometric.data import Data, Batch
import math
from typing import Optional, Tuple


class CrossAssetAttention(nn.Module):
    """Механизм кросс-внимания между активами для учёта взаимозависимостей"""
    
    def __init__(self, d_model: int, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.attention = nn.MultiheadAttention(d_model, num_heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch, assets, features]
        Returns:
            [batch, assets, features] с учётом взаимозависимостей
        """
        attn_out, _ = self.attention(x, x, x)
        x = self.norm(x + self.dropout(attn_out))
        return x


class TemporalConvBlock(nn.Module):
    """Временной сверточный блок с дилатацией для захвата долгосрочных паттернов"""
    
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3, dilation: int = 1):
        super().__init__()
        self.conv = nn.Conv1d(
            in_channels, out_channels, kernel_size,
            padding=(kernel_size - 1) * dilation // 2,
            dilation=dilation
        )
        self.norm = nn.BatchNorm1d(out_channels)
        self.activation = nn.GELU()
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.activation(self.norm(self.conv(x)))


class HybridTransformerGNN(nn.Module):
    """Гибридная архитектура, комбинирующая Transformer и Graph Neural Networks"""
    
    def __init__(
        self,
        initial_features: int = 3,
        d_model: int = 128,
        num_heads: int = 8,
        num_layers: int = 4,
        gnn_hidden: int = 64,
        time_window: int = 50,
        dropout: float = 0.1,
        device: str = "cpu"
    ):
        super().__init__()
        self.device = device
        self.d_model = d_model
        
        # Временная обработка с TCN
        self.temporal_encoder = nn.Sequential(
            TemporalConvBlock(initial_features, d_model // 4, kernel_size=3, dilation=1),
            TemporalConvBlock(d_model // 4, d_model // 2, kernel_size=3, dilation=2),
            TemporalConvBlock(d_model // 2, d_model, kernel_size=3, dilation=4),
        )
        
        # Позиционное кодирование для временных данных
        self.temporal_pos_encoding = nn.Parameter(torch.randn(1, time_window, d_model))
        
        # Transformer для временных зависимостей
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )
        self.temporal_transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers // 2)
        
        # Graph Neural Network для межактивных зависимостей
        self.gnn_layers = nn.ModuleList([
            GATConv(d_model, gnn_hidden, heads=4, concat=True, dropout=dropout),
            GATConv(gnn_hidden * 4, gnn_hidden, heads=4, concat=True, dropout=dropout),
            GATConv(gnn_hidden * 4, d_model, heads=1, concat=False, dropout=dropout)
        ])
        
        # Cross-asset attention
        self.cross_asset_attention = CrossAssetAttention(d_model, num_heads, dropout)
        
        # Адаптивный механизм взвешивания
        self.adaptive_weights = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 1),
            nn.Sigmoid()
        )
        
        # Финальная проекция
        self.output_projection = nn.Sequential(
            nn.Linear(d_model + 1, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1)
        )
        
        self.softmax = nn.Softmax(dim=-1)
        
    def _create_asset_graph(self, features: torch.Tensor) -> Data:
        """Создание графа активов на основе корреляций"""
        batch_size, num_assets, _ = features.shape
        
        # Вычисляем корреляционную матрицу
        features_flat = features.reshape(batch_size * num_assets, -1)
        corr_matrix = torch.corrcoef(features_flat)
        
        # Создаём рёбра для сильно коррелированных активов
        threshold = 0.3
        edge_index = (torch.abs(corr_matrix) > threshold).nonzero().t()
        
        # Убираем самосвязи
        mask = edge_index[0] != edge_index[1]
        edge_index = edge_index[:, mask]
        
        return Data(x=features_flat, edge_index=edge_index)
    
    def forward(self, observation: torch.Tensor, last_action: torch.Tensor) -> torch.Tensor:
        """
        Args:
            observation: [batch, features, assets, time]
            last_action: [batch, assets+1]
        Returns:
            [batch, assets+1] - веса портфеля
        """
        if isinstance(observation, np.ndarray):
            observation = torch.from_numpy(observation)
        observation = observation.to(self.device).float()
        
        if isinstance(last_action, np.ndarray):
            last_action = torch.from_numpy(last_action)
        last_action = last_action.to(self.device).float()
        
        batch_size = observation.shape[0]
        num_features = observation.shape[1]
        num_assets = observation.shape[2]
        time_steps = observation.shape[3]
        
        # 1. Временная обработка для каждого актива
        temporal_features = []
        for i in range(num_assets):
            asset_data = observation[:, :, i, :]  # [batch, features, time]
            encoded = self.temporal_encoder(asset_data)  # [batch, d_model, time]
            encoded = encoded.transpose(1, 2)  # [batch, time, d_model]
            encoded = encoded + self.temporal_pos_encoding[:, :time_steps, :]
            encoded = self.temporal_transformer(encoded)  # [batch, time, d_model]
            temporal_features.append(encoded[:, -1, :])  # Берём последний временной шаг
        
        temporal_features = torch.stack(temporal_features, dim=1)  # [batch, assets, d_model]
        
        # 2. Graph Neural Network обработка
        graph = self._create_asset_graph(temporal_features[0])  # Для простоты берём первый батч
        x = graph.x
        
        for gnn_layer in self.gnn_layers:
            x = F.relu(gnn_layer(x, graph.edge_index))
        
        gnn_features = x.reshape(batch_size, num_assets, -1)  # [batch, assets, d_model]
        
        # 3. Cross-asset attention
        attended_features = self.cross_asset_attention(gnn_features)  # [batch, assets, d_model]
        
        # 4. Адаптивное взвешивание temporal и graph features
        combined = torch.cat([temporal_features, attended_features], dim=-1)  # [batch, assets, d_model*2]
        alpha = self.adaptive_weights(combined)  # [batch, assets, 1]
        
        final_features = alpha * temporal_features + (1 - alpha) * attended_features  # [batch, assets, d_model]
        
        # 5. Включение информации о последнем действии
        last_stocks = last_action[:, 1:].unsqueeze(-1)  # [batch, assets, 1]
        features_with_last = torch.cat([final_features, last_stocks], dim=-1)  # [batch, assets, d_model+1]
        
        # 6. Финальная проекция
        weights = self.output_projection(features_with_last).squeeze(-1)  # [batch, assets]
        
        # 7. Добавляем cash weight
        cash_weight = torch.zeros((batch_size, 1), device=self.device)
        weights = torch.cat([cash_weight, weights], dim=1)  # [batch, assets+1]
        
        # 8. Применяем softmax для получения валидных весов
        weights = self.softmax(weights)
        
        return weights
    
    def mu(self, observation: torch.Tensor, last_action: torch.Tensor) -> torch.Tensor:
        """Алиас для forward для совместимости"""
        return self.forward(observation, last_action)


class NeuralODE_Portfolio(nn.Module):
    """Neural ODE для моделирования непрерывной динамики портфеля"""
    
    def __init__(
        self,
        initial_features: int = 3,
        hidden_dim: int = 128,
        time_window: int = 50,
        ode_steps: int = 10,
        device: str = "cpu"
    ):
        super().__init__()
        self.device = device
        self.ode_steps = ode_steps
        
        # Энкодер для начального состояния
        self.encoder = nn.Sequential(
            nn.Conv2d(initial_features, hidden_dim // 2, kernel_size=(1, 5), padding=(0, 2)),
            nn.GELU(),
            nn.Conv2d(hidden_dim // 2, hidden_dim, kernel_size=(1, 5), padding=(0, 2)),
            nn.GELU(),
        )
        
        # ODE функция
        self.ode_func = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Tanh()
        )
        
        # Декодер для весов портфеля
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1)
        )
        
        self.softmax = nn.Softmax(dim=-1)
        
    def ode_step(self, state: torch.Tensor, dt: float = 0.1) -> torch.Tensor:
        """Один шаг интегрирования ODE"""
        derivative = self.ode_func(state)
        return state + dt * derivative
    
    def forward(self, observation: torch.Tensor, last_action: torch.Tensor) -> torch.Tensor:
        if isinstance(observation, np.ndarray):
            observation = torch.from_numpy(observation)
        observation = observation.to(self.device).float()
        
        if isinstance(last_action, np.ndarray):
            last_action = torch.from_numpy(last_action)
        last_action = last_action.to(self.device).float()
        
        batch_size = observation.shape[0]
        num_assets = observation.shape[2]
        
        # Кодируем начальное состояние
        encoded = self.encoder(observation)  # [batch, hidden_dim, assets, time]
        state = encoded.mean(dim=-1).transpose(1, 2)  # [batch, assets, hidden_dim]
        
        # Интегрируем ODE
        for _ in range(self.ode_steps):
            state = self.ode_step(state)
        
        # Декодируем в веса
        weights = self.decoder(state).squeeze(-1)  # [batch, assets]
        
        # Добавляем cash
        cash_weight = torch.zeros((batch_size, 1), device=self.device)
        weights = torch.cat([cash_weight, weights], dim=1)
        
        return self.softmax(weights)
    
    def mu(self, observation: torch.Tensor, last_action: torch.Tensor) -> torch.Tensor:
        return self.forward(observation, last_action)


class MetaLearningPortfolio(nn.Module):
    """Мета-обучение для быстрой адаптации к новым рыночным режимам"""
    
    def __init__(
        self,
        base_model: nn.Module,
        adaptation_steps: int = 5,
        meta_lr: float = 0.01,
        device: str = "cpu"
    ):
        super().__init__()
        self.base_model = base_model
        self.adaptation_steps = adaptation_steps
        self.meta_lr = meta_lr
        self.device = device
        
        # Мета-параметры для быстрой адаптации
        self.meta_params = nn.ParameterList([
            nn.Parameter(torch.zeros_like(p)) 
            for p in base_model.parameters()
        ])
        
        # Контекстный энкодер для определения рыночного режима
        self.context_encoder = nn.LSTM(
            input_size=128,
            hidden_size=64,
            num_layers=2,
            batch_first=True
        )
        
        # Генератор адаптационных параметров
        self.adaptation_generator = nn.Sequential(
            nn.Linear(64, 128),
            nn.GELU(),
            nn.Linear(128, 64),
            nn.Tanh()
        )
    
    def adapt(self, support_data: Tuple[torch.Tensor, torch.Tensor], 
              support_labels: torch.Tensor) -> nn.Module:
        """Адаптация модели к новым данным"""
        adapted_model = self.base_model
        
        # Быстрая адаптация через градиентные шаги
        for _ in range(self.adaptation_steps):
            obs, last_action = support_data
            pred = adapted_model(obs, last_action)
            loss = F.mse_loss(pred, support_labels)
            
            # Вычисляем градиенты
            grads = torch.autograd.grad(loss, adapted_model.parameters(), create_graph=True)
            
            # Обновляем параметры
            for param, grad, meta_param in zip(adapted_model.parameters(), grads, self.meta_params):
                param.data = param.data - self.meta_lr * (grad + meta_param)
        
        return adapted_model
    
    def forward(self, observation: torch.Tensor, last_action: torch.Tensor,
                context_window: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            observation: Текущее наблюдение
            last_action: Последнее действие
            context_window: Окно контекста для определения режима рынка
        """
        if context_window is not None:
            # Определяем текущий рыночный режим
            context_features, _ = self.context_encoder(context_window)
            adaptation_params = self.adaptation_generator(context_features[:, -1, :])
            
            # Модулируем параметры базовой модели
            for param, meta_param, adapt_param in zip(
                self.base_model.parameters(), 
                self.meta_params,
                adaptation_params
            ):
                param.data = param.data + meta_param * adapt_param.mean()
        
        return self.base_model(observation, last_action)
    
    def mu(self, observation: torch.Tensor, last_action: torch.Tensor) -> torch.Tensor:
        return self.forward(observation, last_action) 