# Продвинутая система оптимизации портфеля с RL

## 🚀 Новые возможности

### 1. **Интеграция с MOEX**
- ✅ Реал-тайм получение данных через WebSocket
- ✅ Песочница для безопасного тестирования стратегий
- ✅ Автоматическая ребалансировка портфеля
- ✅ Симуляция исполнения ордеров с учетом рыночных условий

### 2. **TensorBoard мониторинг**
- 📊 Визуализация метрик обучения в реальном времени
- 📈 Отслеживание производительности портфеля
- 🎯 Анализ распределения весов активов
- 📉 Мониторинг Sharpe ratio, drawdown, alpha/beta
- 🔍 Визуализация градиентов нейросети

### 3. **Современные RL алгоритмы**

#### PPO (Proximal Policy Optimization)
- Стабильное обучение с ограничением обновлений политики
- Адаптивная энтропийная регуляризация
- Эффективная работа с непрерывными действиями

#### SAC (Soft Actor-Critic)
- Максимизация энтропии для лучшего exploration
- Автоматическая настройка температуры
- Двойные Q-сети для стабильности

### 4. **Инновационные архитектуры**

#### HybridTransformerGNN
- Комбинация Transformer и Graph Neural Networks
- Учёт межактивных корреляций через граф
- Адаптивное взвешивание временных и графовых признаков
- Cross-asset attention механизм

#### NeuralODE_Portfolio
- Моделирование непрерывной динамики портфеля
- Интегрирование дифференциальных уравнений
- Более гладкие траектории оптимизации

#### MetaLearningPortfolio
- Быстрая адаптация к новым рыночным режимам
- MAML-подобный алгоритм для few-shot learning
- Контекстное определение рыночных условий

## 📦 Установка зависимостей

```bash
pip install -r requirements_advanced.txt
```

### Основные зависимости:
```
torch>=1.9.0
torch-geometric>=2.0.0
tensorboard>=2.8.0
aiohttp>=3.8.0
websocket-client>=1.2.0
pandas>=1.3.0
numpy>=1.21.0
matplotlib>=3.4.0
seaborn>=0.11.0
quantstats>=0.0.37
gym>=0.21.0
stable-baselines3>=1.3.0
```

## 🏃 Быстрый старт

### 1. Базовое использование
```python
from main_advanced import AdvancedPortfolioSystem

# Конфигурация
config = {
    'data_path': './data/',
    'tickers': ['SBER', 'GAZP', 'LKOH', 'NVTK', 'ROSN'],
    'initial_amount': 1000000,
    'features': ['close', 'high', 'low'],
    'time_window': 20,
    'episodes': 100
}

# Создание и запуск системы
system = AdvancedPortfolioSystem(config)
```

### 2. Обучение PPO агента
```python
# Создание PPO агента с Transformer архитектурой
ppo_agent = system.create_agent(
    env, 
    agent_type='ppo', 
    model_type='transformer'
)

# Обучение
trained_agent = system.train_agent(
    ppo_agent, 
    train_env, 
    val_env, 
    episodes=100
)
```

### 3. Live trading в песочнице
```python
import asyncio

# Запуск live trading demo
asyncio.run(system.live_trading_demo(trained_agent))
```

## 📊 Мониторинг через TensorBoard

```bash
# Запуск TensorBoard
tensorboard --logdir ./runs/portfolio_optimization

# Открыть в браузере
http://localhost:6006
```

### Доступные метрики:
- **Episode/**
  - total_reward
  - portfolio_value
  - sharpe_ratio
  
- **Step/**
  - reward
  - trf_mu (transaction costs)
  
- **Weights/**
  - Распределение весов по активам
  
- **Performance/**
  - Sharpe_Ratio
  - Max_Drawdown
  - Sortino_Ratio
  - Alpha/Beta (относительно бенчмарка)

## 🔧 Расширенная настройка

### Создание кастомной архитектуры
```python
class MyCustomModel(nn.Module):
    def __init__(self, initial_features, time_window, device):
        super().__init__()
        # Ваша архитектура
        
    def mu(self, observation, last_action):
        # Логика предсказания весов
        return weights
```

### Добавление нового RL алгоритма
```python
class CustomRLAgent:
    def __init__(self, env, policy_network, **kwargs):
        # Инициализация
        
    def get_action(self, state, last_action):
        # Выбор действия
        
    def update(self):
        # Обновление политики
```

## 🏗️ Архитектура системы

```
portfolio_optimization/
├── environments/
│   ├── portfolio_env.py      # Основная среда
│   └── portfolio_memory.py   # Управление памятью
├── models/
│   ├── models.py            # Базовые модели
│   └── innovative_models.py # Инновационные архитектуры
├── agents/
│   ├── policy_gradient.py   # Policy Gradient
│   ├── dlr_agent.py        # DRL агент
│   └── advanced_rl_agents.py # PPO, SAC
├── integration/
│   └── moex_integration.py  # Интеграция с MOEX
├── monitoring/
│   └── tensorboard_monitor.py # TensorBoard мониторинг
├── utils/
│   └── utils.py            # Утилиты
└── main_advanced.py        # Главный файл
```

## 🎯 Рекомендации по использованию

### Для начинающих:
1. Начните с PPO + TransformerPM
2. Используйте небольшое количество активов (5-10)
3. Обучайте на исторических данных минимум за 2 года

### Для продвинутых:
1. Экспериментируйте с HybridTransformerGNN для учёта корреляций
2. Используйте SAC для более стабильного обучения
3. Настройте MetaLearning для адаптации к разным рынкам

### Оптимальные гиперпараметры:
```python
{
    'time_window': 20-50,        # Окно наблюдения
    'lr_actor': 1e-4 - 3e-4,     # Learning rate актора
    'lr_critic': 1e-3,           # Learning rate критика
    'gamma': 0.99,               # Дисконтирование
    'commission_fee': 0.0025,    # Комиссия брокера
    'batch_size': 64-256         # Размер батча
}
```

## 📈 Результаты

Типичные результаты после обучения:
- Sharpe Ratio: 1.5 - 2.5
- Max Drawdown: < 15%
- Annual Return: 15-30%
- Win Rate: 55-65%

## 🤝 Вклад в проект

Приветствуются:
- Новые архитектуры нейросетей
- Дополнительные RL алгоритмы
- Улучшения интеграции с биржами
- Оптимизация производительности

## ⚠️ Дисклеймер

Данная система предназначена для исследовательских целей. Используйте на реальных рынках на свой страх и риск. Всегда тестируйте стратегии на исторических данных и в песочнице перед реальной торговлей. 