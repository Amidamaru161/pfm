"""
Модуль для мониторинга обучения и торговли через TensorBoard
"""
import torch
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional
import pandas as pd
from pathlib import Path

class TensorBoardMonitor:
    """Мониторинг метрик обучения и торговли через TensorBoard"""
    
    def __init__(self, log_dir: str = "runs/portfolio_optimization"):
        """
        Args:
            log_dir: Директория для логов TensorBoard
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.writer = SummaryWriter(f"{log_dir}/{timestamp}")
        self.step = 0
        self.episode = 0
        
        # Буферы для расчёта скользящих средних
        self.reward_buffer = []
        self.return_buffer = []
        self.action_buffer = []
        
    def log_episode_metrics(self, metrics: Dict[str, float]):
        """Логирование метрик эпизода"""
        for name, value in metrics.items():
            self.writer.add_scalar(f"Episode/{name}", value, self.episode)
        self.episode += 1
    
    def log_step_metrics(self, state, action, reward, info):
        """Логирование метрик на каждом шаге"""
        # Базовые метрики
        self.writer.add_scalar("Step/reward", reward, self.step)
        
        # Распределение весов портфеля
        for i, weight in enumerate(action):
            asset_name = "cash" if i == 0 else f"asset_{i}"
            self.writer.add_scalar(f"Weights/{asset_name}", weight, self.step)
        
        # Логирование изменения цен
        if "price_variation" in info:
            for i, var in enumerate(info["price_variation"][1:]):  # Пропускаем cash
                self.writer.add_scalar(f"PriceVariation/asset_{i}", var - 1, self.step)
        
        # TRF mu (transaction remainder factor)
        if "trf_mu" in info:
            self.writer.add_scalar("Step/trf_mu", info["trf_mu"], self.step)
        
        self.step += 1
    
    def log_portfolio_distribution(self, weights: np.ndarray, names: List[str] = None):
        """Визуализация распределения портфеля"""
        fig, ax = plt.subplots(figsize=(10, 6))
        
        if names is None:
            names = ["Cash"] + [f"Asset_{i}" for i in range(len(weights) - 1)]
        
        # Создаём pie chart
        colors = plt.cm.Set3(np.linspace(0, 1, len(weights)))
        wedges, texts, autotexts = ax.pie(weights, labels=names, autopct='%1.1f%%',
                                          colors=colors, startangle=90)
        
        # Улучшаем визуализацию
        for text in texts:
            text.set_fontsize(10)
        for autotext in autotexts:
            autotext.set_color('white')
            autotext.set_weight('bold')
        
        ax.set_title(f'Portfolio Distribution at Step {self.step}')
        
        # Сохраняем в TensorBoard
        self.writer.add_figure('Portfolio/Distribution', fig, self.step)
        plt.close(fig)
    
    def log_correlation_matrix(self, returns_df: pd.DataFrame):
        """Логирование корреляционной матрицы доходностей"""
        fig, ax = plt.subplots(figsize=(12, 10))
        
        # Вычисляем корреляционную матрицу
        corr_matrix = returns_df.corr()
        
        # Создаём heatmap
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0,
                   square=True, linewidths=0.5, cbar_kws={"shrink": 0.8},
                   ax=ax)
        
        ax.set_title('Asset Returns Correlation Matrix')
        
        self.writer.add_figure('Analysis/Correlation', fig, self.episode)
        plt.close(fig)
    
    def log_performance_metrics(self, portfolio_values: List[float], 
                               returns: List[float], benchmark_returns: List[float] = None):
        """Логирование метрик производительности портфеля"""
        # Sharpe Ratio
        if len(returns) > 1:
            sharpe = np.mean(returns) / (np.std(returns) + 1e-8) * np.sqrt(252)
            self.writer.add_scalar("Performance/Sharpe_Ratio", sharpe, self.episode)
        
        # Maximum Drawdown
        cumulative = np.cumprod(1 + np.array(returns))
        running_max = np.maximum.accumulate(cumulative)
        drawdown = (cumulative - running_max) / running_max
        max_drawdown = np.min(drawdown)
        self.writer.add_scalar("Performance/Max_Drawdown", max_drawdown, self.episode)
        
        # Sortino Ratio
        negative_returns = [r for r in returns if r < 0]
        if negative_returns:
            downside_std = np.std(negative_returns)
            sortino = np.mean(returns) / (downside_std + 1e-8) * np.sqrt(252)
            self.writer.add_scalar("Performance/Sortino_Ratio", sortino, self.episode)
        
        # Alpha и Beta относительно benchmark
        if benchmark_returns:
            # Простой расчёт альфы и беты
            cov_matrix = np.cov(returns, benchmark_returns)
            beta = cov_matrix[0, 1] / (cov_matrix[1, 1] + 1e-8)
            alpha = np.mean(returns) - beta * np.mean(benchmark_returns)
            
            self.writer.add_scalar("Performance/Alpha", alpha * 252, self.episode)
            self.writer.add_scalar("Performance/Beta", beta, self.episode)
    
    def log_network_gradients(self, model: torch.nn.Module):
        """Логирование градиентов нейросети для отслеживания обучения"""
        for name, param in model.named_parameters():
            if param.grad is not None:
                # Логируем норму градиентов
                grad_norm = param.grad.data.norm(2).item()
                self.writer.add_scalar(f"Gradients/{name}_norm", grad_norm, self.step)
                
                # Логируем гистограмму градиентов
                self.writer.add_histogram(f"Gradients/{name}", param.grad.data, self.step)
    
    def log_action_entropy(self, action_probs: np.ndarray):
        """Логирование энтропии действий для оценки exploration"""
        # Избегаем log(0)
        action_probs = np.clip(action_probs, 1e-8, 1.0)
        entropy = -np.sum(action_probs * np.log(action_probs))
        self.writer.add_scalar("Exploration/Action_Entropy", entropy, self.step)
    
    def log_custom_metric(self, name: str, value: float, step: Optional[int] = None):
        """Логирование пользовательской метрики"""
        if step is None:
            step = self.step
        self.writer.add_scalar(f"Custom/{name}", value, step)
    
    def log_hyperparameters(self, hparams: Dict, metrics: Dict):
        """Логирование гиперпараметров и результатов"""
        self.writer.add_hparams(hparams, metrics)
    
    def create_comparison_plot(self, portfolio_values: List[float], 
                              benchmark_values: List[float], 
                              title: str = "Portfolio vs Benchmark"):
        """Создание графика сравнения портфеля с бенчмарком"""
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
        
        # График стоимости
        ax1.plot(portfolio_values, label='Portfolio', color='blue', linewidth=2)
        ax1.plot(benchmark_values, label='Benchmark', color='red', linewidth=1.5, alpha=0.7)
        ax1.set_title(f'{title} - Value Over Time')
        ax1.set_xlabel('Time Steps')
        ax1.set_ylabel('Portfolio Value')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # График относительной производительности
        relative_perf = np.array(portfolio_values) / np.array(benchmark_values)
        ax2.plot(relative_perf, color='green', linewidth=2)
        ax2.axhline(y=1.0, color='black', linestyle='--', alpha=0.5)
        ax2.set_title('Relative Performance (Portfolio / Benchmark)')
        ax2.set_xlabel('Time Steps')
        ax2.set_ylabel('Relative Value')
        ax2.grid(True, alpha=0.3)
        
        self.writer.add_figure('Comparison/Portfolio_vs_Benchmark', fig, self.episode)
        plt.close(fig)
    
    def close(self):
        """Закрытие writer'а TensorBoard"""
        self.writer.close()

class LiveTradingMonitor(TensorBoardMonitor):
    """Расширенный монитор для live торговли"""
    
    def __init__(self, log_dir: str = "runs/live_trading"):
        super().__init__(log_dir)
        self.trade_history = []
        
    def log_trade(self, trade_info: Dict):
        """Логирование информации о сделке"""
        self.trade_history.append(trade_info)
        
        # Логируем в TensorBoard
        self.writer.add_scalar("Trading/OrderSize", trade_info.get("size", 0), self.step)
        self.writer.add_scalar("Trading/ExecutionPrice", trade_info.get("price", 0), self.step)
        
        # Расчёт slippage если есть
        if "expected_price" in trade_info and "price" in trade_info:
            slippage = (trade_info["price"] - trade_info["expected_price"]) / trade_info["expected_price"]
            self.writer.add_scalar("Trading/Slippage", slippage, self.step)
    
    def log_market_conditions(self, market_data: Dict):
        """Логирование рыночных условий"""
        # Волатильность
        if "volatility" in market_data:
            self.writer.add_scalar("Market/Volatility", market_data["volatility"], self.step)
        
        # Объём торгов
        if "volume" in market_data:
            self.writer.add_scalar("Market/Volume", market_data["volume"], self.step)
        
        # Spread
        if "bid" in market_data and "ask" in market_data:
            spread = market_data["ask"] - market_data["bid"]
            self.writer.add_scalar("Market/Spread", spread, self.step) 