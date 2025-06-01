"""
Быстрый тест системы оптимизации портфеля
"""
import pandas as pd
import numpy as np
import torch
import warnings
warnings.filterwarnings('ignore')

from environments.portfolio_env import PortfolioOptimizationEnv
from agents.advanced_rl_agents import PPOAgent
from models.models import EIIE
from monitoring.tensorboard_monitor import TensorBoardMonitor

def create_sample_data():
    """Создание примера данных для тестирования"""
    dates = pd.date_range(start='2022-01-01', end='2023-12-31', freq='D')
    tickers = ['STOCK_A', 'STOCK_B', 'STOCK_C']
    
    data = []
    for date in dates:
        for ticker in tickers:
            # Генерируем синтетические данные
            base_price = 100 + np.random.randn() * 10
            data.append({
                'date': date,
                'tic': ticker,
                'close': base_price,
                'high': base_price * (1 + np.random.uniform(0, 0.02)),
                'low': base_price * (1 - np.random.uniform(0, 0.02))
            })
    
    df = pd.DataFrame(data)
    return df

def test_basic_functionality():
    """Тест базовой функциональности"""
    print("=== Тест базовой функциональности ===")
    
    # Создаем тестовые данные
    df = create_sample_data()
    print(f"Создан датафрейм: {df.shape}")
    print(f"Тикеры: {df['tic'].unique()}")
    
    # Создаем среду
    env = PortfolioOptimizationEnv(
        df=df,
        initial_amount=100000,
        features=['close', 'high', 'low'],
        time_window=10,
        normalize_df="by_previous_time",
        comission_fee_pct=0.001,
        reward_scaling=1,
        new_gym_api=True
    )
    print(f"Среда создана. Размер портфеля: {env.portfolio_size}")
    print(f"Пространство наблюдений: {env.observation_space}")
    print(f"Пространство действий: {env.action_space}")
    
    # Тестируем reset и step
    obs, info = env.reset()
    print(f"Начальное наблюдение: {obs.shape}")
    
    # Делаем несколько шагов
    for i in range(5):
        # Случайное действие (веса портфеля)
        action = np.random.dirichlet(np.ones(env.portfolio_size + 1))
        next_obs, reward, done, _, info = env.step(action)
        print(f"Шаг {i+1}: reward={reward:.4f}, done={done}")
        
        if done:
            break
    
    print("✅ Базовая функциональность работает!")
    return env

def test_ppo_agent():
    """Тест PPO агента"""
    print("\n=== Тест PPO агента ===")
    
    # Создаем среду
    df = create_sample_data()
    env = PortfolioOptimizationEnv(
        df=df,
        initial_amount=100000,
        features=['close', 'high', 'low'],
        time_window=10,
        normalize_df="by_previous_time",
        comission_fee_pct=0.001,
        reward_scaling=1,
        new_gym_api=True
    )
    
    # Создаем агента
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Используется устройство: {device}")
    
    agent = PPOAgent(
        env=env,
        policy_network=EIIE,
        policy_kwargs={
            'initial_features': 3,
            'time_window': 10,
            'conv_mid_features': 2,
            'conv_final_features': 10,
            'device': str(device)
        },
        lr_actor=3e-4,
        lr_critic=1e-3,
        gamma=0.99,
        device=str(device)
    )
    print("PPO агент создан")
    
    # Тестируем обучение
    obs, _ = env.reset()
    last_action = np.array([1.0] + [0.0] * env.portfolio_size)
    
    # Один эпизод обучения
    episode_reward = 0
    steps = 0
    done = False
    
    while not done and steps < 100:
        action, log_prob, value = agent.get_action(obs, last_action)
        next_obs, reward, done, _, info = env.step(action)
        
        agent.store_transition(obs, action, reward, log_prob, value, done)
        
        episode_reward += reward
        obs = next_obs
        last_action = action
        steps += 1
    
    print(f"Эпизод завершен: шагов={steps}, общая награда={episode_reward:.4f}")
    
    # Обновление агента
    if len(agent.states) > 0:
        agent.update()
        print("Агент обновлен")
    
    print("✅ PPO агент работает!")

def test_tensorboard_monitor():
    """Тест TensorBoard монитора"""
    print("\n=== Тест TensorBoard монитора ===")
    
    monitor = TensorBoardMonitor(log_dir="runs/test")
    
    # Тестируем логирование метрик
    for i in range(10):
        monitor.log_episode_metrics({
            'total_reward': np.random.randn(),
            'portfolio_value': 100000 + np.random.randn() * 1000,
            'sharpe_ratio': 1.5 + np.random.randn() * 0.5
        })
        
        # Логируем шаги
        action = np.random.dirichlet(np.ones(4))  # 3 акции + cash
        monitor.log_step_metrics(
            state=np.random.randn(3, 3, 10),
            action=action,
            reward=np.random.randn(),
            info={'price_variation': np.array([1.0, 1.01, 0.99, 1.02])}
        )
    
    # Тестируем визуализацию портфеля
    weights = np.array([0.2, 0.3, 0.3, 0.2])
    monitor.log_portfolio_distribution(weights, names=['Cash', 'Stock A', 'Stock B', 'Stock C'])
    
    monitor.close()
    print("✅ TensorBoard монитор работает!")
    print(f"Для просмотра результатов выполните: tensorboard --logdir runs/test")

def main():
    """Главная функция тестирования"""
    print("🚀 Запуск быстрого теста системы оптимизации портфеля\n")
    
    try:
        # Тест 1: Базовая функциональность
        test_basic_functionality()
        
        # Тест 2: PPO агент
        test_ppo_agent()
        
        # Тест 3: TensorBoard монитор
        test_tensorboard_monitor()
        
        print("\n✅ Все тесты пройдены успешно!")
        print("\nТеперь вы можете запустить полную систему командой:")
        print("python main_advanced.py")
        
    except Exception as e:
        print(f"\n❌ Ошибка при тестировании: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 