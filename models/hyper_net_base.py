import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch_geometric.nn import RGCNConv, GATv2Conv
from torch_geometric.data import Data, Batch


import numpy as np
import torch
from torch import nn
from torch_geometric.data import Batch
from torch_geometric.data import Data
from torch_geometric.nn import RGCNConv
from torch_geometric.nn import Sequential
from torch_geometric.utils import to_dense_batch
import sys 
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import math
class TemporalAttentionBlock(nn.Module):
    """Блок временного внимания для анализа финансовых временных рядов."""
    
    def __init__(self, embed_dim, num_heads, dropout=0.1):
        super().__init__()
        self.attention = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 4, embed_dim),
            nn.Dropout(dropout)
        )
        
    def forward(self, x):
        # x имеет размерность [batch_size, seq_len, embed_dim]
        attention_output, _ = self.attention(x, x, x)
        x = self.norm1(x + attention_output)
        ffn_output = self.ffn(x)
        x = self.norm2(x + ffn_output)
        return x

class AssetInteractionGraph(nn.Module):
    def __init__(self, in_features, hidden_features, num_relations=3, heads=4):
        super().__init__()
        self.rgcn = RGCNConv(in_features, hidden_features, num_relations=num_relations)
        self.gat = GATv2Conv(hidden_features, hidden_features // heads, heads=heads)
        self.norm = nn.LayerNorm(hidden_features)
        
    def forward(self, x, edge_index, edge_type):
        # x: [num_nodes, in_features]
        # edge_index: [2, num_edges]
        # edge_type: [num_edges]
        # Убедимся, что edge_type имеет тип long
        edge_type = edge_type.long()
        x = self.rgcn(x, edge_index, edge_type)
        x = F.relu(x)
        x = self.gat(x, edge_index)
        x = self.norm(x)
        return x


class RecurrentGatedBlock(nn.Module):
    """Рекуррентный блок с механизмом гейтов для долгосрочной памяти."""
    
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.gru = nn.GRU(input_size, hidden_size, batch_first=True)
        self.gate = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.Sigmoid()
        )
        self.transform = nn.Linear(hidden_size * 2, hidden_size)
        
    def forward(self, x, hidden=None):
        # x: [batch_size, seq_len, input_size]
        output, hidden = self.gru(x, hidden)
        
        # Применяем механизм гейтинга для объединения долгосрочной и краткосрочной информации
        if hidden is not None:
            hidden_expanded = hidden.transpose(0, 1).expand_as(output)
            combined = torch.cat([output, hidden_expanded], dim=-1)
            gate_values = self.gate(combined)
            transformed = self.transform(combined)
            output = gate_values * output + (1 - gate_values) * transformed
            
        return output, hidden


class HyperNetPortfolioRL(nn.Module):
    """
    HyperNetPortfolioRL: передовая архитектура нейронной сети для оптимизации портфеля 
    с интеграцией внимания, графовых сетей и гибридных рекуррентных механизмов.
    
    Архитектура сочетает несколько передовых подходов:
    1. Временные трансформеры для анализа паттернов временных рядов
    2. Графовые нейронные сети для моделирования корреляций между активами
    3. Рекуррентные блоки с гейтингом для сохранения долгосрочной памяти
    4. Многомасштабный анализ с разными временными окнами
    5. Механизм адаптивного скейлинга весов портфеля
    6. Механизм самодистилляции для стабильности решений
    """
    
    def __init__(
        self,
        initial_features=3,
        time_window=50,
        embed_dim=64,
        attention_heads=4,
        graph_features=32,
        recurrent_features=32,
        dropout=0.1,
        num_assets=None,  # Будет определено во время forward pass
        device="cpu",
        time_scales=[3, 10, 30]  # Разные временные масштабы для анализа
    ):
        super().__init__()
        self.device = device
        self.time_window = time_window
        self.time_scales = time_scales
        self.initial_features = initial_features
        
        # Проекция входных данных в пространство признаков более высокой размерности
        self.feature_projection = nn.Conv2d(
            in_channels=initial_features,
            out_channels=embed_dim,
            kernel_size=(1, 1)
        )
        
        # Позиционное кодирование для временного ряда
        self.pos_encoding = nn.Parameter(torch.zeros(1, time_window, embed_dim))
        nn.init.normal_(self.pos_encoding, mean=0, std=0.02)
        
        # Блоки внимания для разных временных масштабов
        self.attention_blocks = nn.ModuleList([
            TemporalAttentionBlock(embed_dim, attention_heads, dropout) 
            for _ in range(len(time_scales) + 1)  # +1 для полного временного окна
        ])
        
        # Графовый модуль для взаимодействия между активами
        self.graph_module = AssetInteractionGraph(
            in_features=embed_dim,
            hidden_features=graph_features
        )
        
        # Рекуррентный блок для долгосрочной памяти
        self.recurrent_block = RecurrentGatedBlock(
            input_size=embed_dim + graph_features,
            hidden_size=recurrent_features
        )
        
        # Финальные слои для формирования весов портфеля
        self.final_layers = nn.Sequential(
            nn.Linear(embed_dim + graph_features + recurrent_features + 1, 64),  # +1 для предыдущего веса актива
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, 32),
            nn.GELU(),
            nn.Linear(32, 1)
        )
        
        # Слой для адаптивного скейлинга
        self.adaptive_scaling = nn.Linear(embed_dim, 1)
        
        # Параметр температуры для softmax (обучаемый)
        self.temperature = nn.Parameter(torch.ones(1))
        
        # Слой для самодистилляции (предсказание будущих доходностей)
        self.returns_predictor = nn.Linear(embed_dim + graph_features + recurrent_features, 1)
        

    def _create_asset_graph(self, batch_size, num_assets, features):
        """Создаем граф взаимодействий между активами на основе признаков."""
        # features: [batch_size, embed_dim, num_assets, 1]
        features = features.squeeze(-1).transpose(1, 2)  # [batch_size, num_assets, embed_dim]
        
        all_graphs = []
        for b in range(batch_size):
            asset_features = features[b]  # [num_assets, embed_dim]
            
            # Вычисляем корреляции между активами
            corr_matrix = torch.corrcoef(asset_features)
            corr_matrix.fill_diagonal_(0)  # Убираем самокорреляции
            
            # Создаем рёбра на основе корреляций (положительные, отрицательные и топ-корреляции)
            pos_edges = (corr_matrix > 0.3).nonzero(as_tuple=False)  # Положительные корреляции
            neg_edges = (corr_matrix < -0.3).nonzero(as_tuple=False)  # Отрицательные корреляции
            
            # Топ-корреляции (берем топ-5 для каждого актива)
            top_edges = []
            for i in range(num_assets):
                top_k = min(5, num_assets - 1)
                _, indices = torch.topk(torch.abs(corr_matrix[i]), k=top_k)
                for j in indices:
                    top_edges.append([i, j.item()])
            top_edges = torch.tensor(top_edges, device=self.device, dtype=torch.long)
            
            # Объединяем все рёбра
            all_edges = []
            edge_types = []
            
            if len(pos_edges) > 0:
                all_edges.append(pos_edges.t())
                edge_types.append(torch.zeros(pos_edges.size(0), device=self.device, dtype=torch.long))
                
            if len(neg_edges) > 0:
                all_edges.append(neg_edges.t())
                edge_types.append(torch.ones(neg_edges.size(0), device=self.device, dtype=torch.long))
                
            if len(top_edges) > 0:
                all_edges.append(top_edges.t())
                edge_types.append(torch.full((top_edges.size(0),), 2, device=self.device, dtype=torch.long))
            
            if all_edges:  # Проверяем, что список не пустой
                edge_index = torch.cat(all_edges, dim=1)
                edge_type = torch.cat(edge_types)
            else:
                # Создаем фиктивный граф полного соединения, если не нашли значимых корреляций
                edge_index = torch.zeros((2, num_assets * (num_assets - 1)), device=self.device, dtype=torch.long)
                edge_counter = 0
                for i in range(num_assets):
                    for j in range(num_assets):
                        if i != j:
                            edge_index[0, edge_counter] = i
                            edge_index[1, edge_counter] = j
                            edge_counter += 1
                edge_type = torch.zeros(num_assets * (num_assets - 1), device=self.device, dtype=torch.long)
            
            # Создаем графовый объект
            graph = Data(
                x=asset_features,
                edge_index=edge_index,
                edge_type=edge_type
            )
            all_graphs.append(graph)
        
        # Объединяем графы в батч
        batch_graph = Batch.from_data_list(all_graphs)
        return batch_graph

            
    def multi_scale_attention(self, x):
        """Применяем внимание на разных временных масштабах."""
        # x: [batch_size, embed_dim, num_assets, time_window]
        batch_size, embed_dim, num_assets, _ = x.size()
        
        # Транспонируем для обработки временного измерения
        x = x.permute(0, 2, 3, 1)  # [batch_size, num_assets, time_window, embed_dim]
        
        # Добавляем позиционное кодирование
        x = x + self.pos_encoding
        
        # Обрабатываем полное временное окно
        full_window = x.reshape(batch_size * num_assets, self.time_window, embed_dim)
        full_context = self.attention_blocks[0](full_window)
        
        # Обрабатываем разные временные масштабы
        scale_contexts = []
        for i, scale in enumerate(self.time_scales):
            # Берем последние `scale` точек времени
            scale_window = x[:, :, -scale:, :].reshape(batch_size * num_assets, scale, embed_dim)
            scale_context = self.attention_blocks[i+1](scale_window)
            # Берем представление последнего момента времени
            scale_contexts.append(scale_context[:, -1:, :])
        
        # Конкатенируем контексты разных масштабов
        multi_scale_context = torch.cat([full_context[:, -1:, :]] + scale_contexts, dim=1)
        # Агрегируем информацию по временным масштабам
        multi_scale_context = torch.mean(multi_scale_context, dim=1)
        
        # Возвращаем к исходной форме
        multi_scale_context = multi_scale_context.reshape(batch_size, num_assets, embed_dim)
        
        return multi_scale_context
    
    def mu(self, observation, last_action):
        """Определяет наиболее благоприятное распределение портфеля.
        
        Аргументы:
          observation: наблюдение среды.
          last_action: последнее действие, выполненное агентом.
          
        Возвращает:
          Распределение весов портфеля.
        """
        if isinstance(observation, np.ndarray):
            observation = torch.from_numpy(observation)
        observation = observation.to(self.device).float()
        
        if isinstance(last_action, np.ndarray):
            last_action = torch.from_numpy(last_action)
        last_action = last_action.to(self.device).float()
        
        batch_size = observation.shape[0]
        num_assets = observation.shape[2]  # Предполагаем формат [batch, features, assets, time]
        
        # Извлекаем последние веса и кэш
        last_weights = last_action[:, 1:].unsqueeze(-1)  # [batch, assets, 1]
        cash_bias = last_action[:, 0]  # [batch]
        
        # Проекция входных данных
        x = self.feature_projection(observation)  # [batch, embed_dim, assets, time]
        
        # Многомасштабный анализ временных рядов с помощью блоков внимания
        temporal_features = self.multi_scale_attention(x)  # [batch, assets, embed_dim]
        
        # Создаем и обрабатываем граф взаимодействий активов
        asset_graph = self._create_asset_graph(batch_size, num_assets, x[:, :, :, -1:])
        graph_features = self.graph_module(
            asset_graph.x, 
            asset_graph.edge_index, 
            asset_graph.edge_type
        )  # [batch*assets, graph_features]
        
        # Возвращаем к форме батча
        graph_features = graph_features.reshape(batch_size, num_assets, -1)
        
        # Объединяем временные и графовые признаки
        combined_features = torch.cat([temporal_features, graph_features], dim=-1)  # [batch, assets, embed_dim+graph_f]
        
        # Применяем рекуррентный блок
        recurrent_in = combined_features.reshape(batch_size, num_assets, -1)
        recurrent_out, _ = self.recurrent_block(recurrent_in)  # [batch, assets, recurrent_f]
        
        # Сохраняем промежуточные значения для использования в _gradient_ascent
        self.temporal_features = temporal_features
        self.graph_features = graph_features
        self.recurrent_out = recurrent_out
        
        # Объединяем все признаки с предыдущими весами
        all_features = torch.cat(
            [temporal_features, graph_features, recurrent_out, last_weights], 
            dim=-1
        )  # [batch, assets, all_features]
        
        # Предсказываем логиты для весов активов
        logits = self.final_layers(all_features).squeeze(-1)  # [batch, assets]
        
        # Предсказываем будущие доходности (для самодистилляции)
        returns_pred = self.returns_predictor(
            torch.cat([temporal_features, graph_features, recurrent_out], dim=-1)
        ).squeeze(-1)  # [batch, assets]
        
        # Адаптивный скейлинг на основе временных признаков
        adaptive_scale = F.softplus(self.adaptive_scaling(temporal_features.mean(dim=1)))  # [batch, 1]
        
        # Применяем масштабирование температуры
        scaled_logits = logits / (self.temperature + 1e-6)
        
        # Добавляем bias для кэша
        cash_logits = torch.zeros(batch_size, 1, device=self.device)
        all_logits = torch.cat([cash_logits, scaled_logits], dim=1)  # [batch, assets+1]
        
        # Применяем softmax для получения весов портфеля
        portfolio_weights = F.softmax(all_logits, dim=-1)
        
        # Дополнительно регулируем веса с помощью предсказанных доходностей
        # (ставим больше веса на активы с более высокой ожидаемой доходностью)
        if self.training:
            asset_weights = portfolio_weights[:, 1:]
            # Нормализуем предсказанные доходности
            normalized_returns = F.softmax(returns_pred / 0.1, dim=-1)
            # Слегка смещаем веса в сторону предсказанных доходностей
            adjusted_weights = 0.9 * asset_weights + 0.1 * normalized_returns
            # Перенормируем с учетом кэша
            sum_weights = adjusted_weights.sum(dim=1, keepdim=True)
            cash_weights = portfolio_weights[:, 0:1]
            adjusted_weights = adjusted_weights * (1 - cash_weights) / (sum_weights + 1e-8)
            portfolio_weights = torch.cat([cash_weights, adjusted_weights], dim=1)
        
        return portfolio_weights
    
    def forward(self, observation, last_action):
        """Прямое распространение политики сети.
        
        Аргументы:
          observation: наблюдение среды.
          last_action: последнее действие, выполненное агентом.
          
        Возвращает:
          Действие, которое нужно выполнить (numpy array).
        """
        mu = self.mu(observation, last_action)
        action = mu.cpu().detach().numpy().squeeze()
        return action