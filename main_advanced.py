"""
Продвинутая система оптимизации портфеля с интеграцией MOEX и современными RL методами
"""
import asyncio
import pandas as pd
import numpy as np
import torch
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Импорт компонентов системы
from environments.portfolio_env import PortfolioOptimizationEnv
from integration.moex_integration import MOEXIntegration, RealTimeEnvironmentWrapper
from monitoring.tensorboard_monitor import TensorBoardMonitor, LiveTradingMonitor
from agents.advanced_rl_agents import PPOAgent, SACAgent
from models.innovative_models import HybridTransformerGNN, NeuralODE_Portfolio, MetaLearningPortfolio
from models.models import TransformerPM, EIIE, EI3
from utils.utils import download_moex_candles, process_folder_data, TICKER_LIST
from agents.dlr_agent import DRLAgent


class AdvancedPortfolioSystem:
    """Продвинутая система для оптимизации портфеля"""
    
    def __init__(self, config):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Используется устройство: {self.device}")
        
        # Инициализация компонентов
        self.monitor = TensorBoardMonitor(log_dir=config['log_dir'])
        self.live_monitor = LiveTradingMonitor(log_dir=config['live_log_dir'])
        
    def prepare_data(self):
        """Подготовка данных для обучения"""
        print("Загрузка данных с MOEX...")
        
        # Используем существующие данные или загружаем новые
        try:
            # Попробуем загрузить существующие данные
            df = process_folder_data(
                self.config['data_path'],
                start_date=self.config['start_date'],
                threshold=0.01
            )
            print(f"Загружены данные: {df.shape}")
        except:
            # Если данных нет, загружаем с MOEX
            print("Загрузка новых данных с MOEX...")
            download_moex_candles(
                self.config['tickers'][:10],  # Ограничиваем для демо
                interval=60,
                save_path=self.config['data_path']
            )
            df = process_folder_data(
                self.config['data_path'],
                start_date=self.config['start_date']
            )
        
        # Преобразуем в формат для среды
        env_df = self._prepare_env_dataframe(df)
        return env_df
    
    def _prepare_env_dataframe(self, df):
        """Преобразование данных в формат для среды"""
        # Получаем список всех доступных тикеров
        available_tickers = []
        for col in df.columns:
            if isinstance(col, tuple) and len(col) > 1:
                ticker = col[1]
                if ticker not in available_tickers:
                    available_tickers.append(ticker)
        
        # Фильтруем только те тикеры, которые есть в конфигурации и в данных
        valid_tickers = [tic for tic in self.config['tickers'] if tic in available_tickers]
        
        if not valid_tickers:
            raise ValueError("Не найдено ни одного валидного тикера в данных!")
        
        print(f"Используются тикеры: {valid_tickers}")
        
        # Создаем список для хранения данных
        data_list = []
        
        for date in df.index:
            date_data = {}
            date_data['date'] = date
            
            # Собираем данные для каждого тикера
            for ticker in valid_tickers:
                has_data = True
                ticker_data = {}
                
                # Проверяем наличие всех необходимых признаков
                for feature in ['close', 'high', 'low']:
                    if (feature, ticker) in df.columns:
                        value = df.loc[date, (feature, ticker)]
                        if pd.notna(value):
                            ticker_data[feature] = value
                        else:
                            has_data = False
                            break
                    else:
                        has_data = False
                        break
                
                # Добавляем данные только если все признаки присутствуют
                if has_data:
                    data_entry = {
                        'date': date,
                        'tic': ticker,
                        **ticker_data
                    }
                    data_list.append(data_entry)
        
        if not data_list:
            raise ValueError("Нет валидных данных после фильтрации!")
        
        # Преобразуем в DataFrame
        final_df = pd.DataFrame(data_list)
        
        # Проверяем, что у каждого тикера достаточно данных
        ticker_counts = final_df['tic'].value_counts()
        min_required = self.config['time_window'] + 10  # Минимум для обучения
        
        valid_tickers_final = []
        for ticker in valid_tickers:
            if ticker in ticker_counts and ticker_counts[ticker] >= min_required:
                valid_tickers_final.append(ticker)
            else:
                print(f"Тикер {ticker} исключен: недостаточно данных ({ticker_counts.get(ticker, 0)} < {min_required})")
        
        # Фильтруем финальный датафрейм
        final_df = final_df[final_df['tic'].isin(valid_tickers_final)]
        
        # Обновляем список тикеров в конфигурации
        self.config['tickers'] = valid_tickers_final
        
        print(f"Финальный набор тикеров: {valid_tickers_final}")
        print(f"Размер данных: {final_df.shape}")
        
        # Сортируем по дате и тикеру
        final_df = final_df.sort_values(['date', 'tic']).reset_index(drop=True)
        
        return final_df
    
    def create_environments(self, df):
        """Создание сред для обучения и валидации"""
        # Разделение данных
        train_size = int(len(df) * 0.8)
        train_df = df[:train_size].copy()
        val_df = df[train_size:].copy()
        
        # Создание сред
        train_env = PortfolioOptimizationEnv(
            df=train_df,
            initial_amount=self.config['initial_amount'],
            features=self.config['features'],
            time_window=self.config['time_window'],
            normalize_df="by_previous_time",
            comission_fee_pct=self.config['commission_fee'],
            reward_scaling=self.config['reward_scaling'],
            new_gym_api=True
        )
        
        val_env = PortfolioOptimizationEnv(
            df=val_df,
            initial_amount=self.config['initial_amount'],
            features=self.config['features'],
            time_window=self.config['time_window'],
            normalize_df="by_previous_time",
            comission_fee_pct=self.config['commission_fee'],
            reward_scaling=self.config['reward_scaling'],
            new_gym_api=True
        )
        
        return train_env, val_env
    
    def create_agent(self, env, agent_type='ppo', model_type='transformer'):
        """Создание агента с выбранной архитектурой"""
        # Выбор модели
        model_kwargs = {
            'initial_features': len(self.config['features']),
            'time_window': self.config['time_window'],
            'device': str(self.device)
        }
        
        if model_type == 'transformer':
            policy_class = TransformerPM
            model_kwargs.update({
                'd_model': 128,
                'nhead': 8,
                'num_encoder_layers': 4,
                'dropout': 0.1
            })
        elif model_type == 'hybrid_gnn':
            policy_class = HybridTransformerGNN
            model_kwargs.update({
                'd_model': 128,
                'num_heads': 8,
                'num_layers': 4,
                'gnn_hidden': 64,
                'dropout': 0.1
            })
        elif model_type == 'neural_ode':
            policy_class = NeuralODE_Portfolio
            model_kwargs.update({
                'hidden_dim': 128,
                'ode_steps': 10
            })
        else:
            policy_class = EIIE
            model_kwargs.update({
                'conv_mid_features': 3,
                'conv_final_features': 20
            })
        
        # Создание агента
        if agent_type == 'ppo':
            agent = PPOAgent(
                env=env,
                policy_network=policy_class,
                policy_kwargs=model_kwargs,
                lr_actor=self.config['lr_actor'],
                lr_critic=self.config['lr_critic'],
                gamma=self.config['gamma'],
                eps_clip=0.2,
                k_epochs=4,
                entropy_coef=0.01,
                device=str(self.device)
            )
        elif agent_type == 'sac':
            agent = SACAgent(
                env=env,
                policy_network=policy_class,
                policy_kwargs=model_kwargs,
                lr=self.config['lr_actor'],
                gamma=self.config['gamma'],
                tau=0.005,
                alpha=0.2,
                batch_size=256,
                device=str(self.device)
            )
        else:  # policy gradient
            drl_agent = DRLAgent(env)
            model = drl_agent.get_model(
                model_name="pg",
                device=str(self.device),
                model_kwargs={
                    'policy': policy_class,
                    'lr': self.config['lr_actor'],
                    'batch_size': 100
                },
                policy_kwargs=model_kwargs
            )
            return model
        
        return agent
    
    def train_agent(self, agent, train_env, val_env, episodes=100):
        """Обучение агента с мониторингом"""
        print(f"\nНачало обучения агента {type(agent).__name__}...")
        
        for episode in range(episodes):
            # Training episode
            if hasattr(agent, 'train_pvm'):  # Policy Gradient
                obs = train_env.reset()
                agent.train_pvm.reset()
                done = False
                episode_reward = 0
                
                while not done:
                    last_action = agent.train_pvm.retrieve()
                    obs_batch = np.expand_dims(obs, axis=0)
                    last_action_batch = np.expand_dims(last_action, axis=0)
                    action = agent.train_policy(obs_batch, last_action_batch)
                    agent.train_pvm.add(action)
                    
                    next_obs, reward, done, _, info = train_env.step(action)
                    episode_reward += reward
                    
                    # Логирование шага
                    self.monitor.log_step_metrics(obs, action, reward, info)
                    
                    obs = next_obs
            
            else:  # PPO или SAC
                obs, _ = train_env.reset()
                if isinstance(agent, PPOAgent):
                    pvm = agent.env._actions_memory[-1]
                else:
                    pvm = np.array([1] + [0] * train_env.portfolio_size)
                
                done = False
                episode_reward = 0
                
                while not done:
                    if isinstance(agent, PPOAgent):
                        action, log_prob, value = agent.get_action(obs, pvm)
                        next_obs, reward, done, _, info = train_env.step(action)
                        agent.store_transition(obs, action, reward, log_prob, value, done)
                    else:  # SAC
                        action = agent.get_action(obs, pvm)
                        next_obs, reward, done, _, info = train_env.step(action)
                        next_pvm = train_env._actions_memory[-1]
                        agent.store_transition(obs, pvm, action, reward, next_obs, next_pvm, done)
                    
                    episode_reward += reward
                    self.monitor.log_step_metrics(obs, action, reward, info)
                    
                    obs = next_obs
                    pvm = train_env._actions_memory[-1]
                
                # Обновление агента
                if isinstance(agent, PPOAgent):
                    agent.update()
                elif isinstance(agent, SACAgent):
                    for _ in range(50):  # Multiple updates per episode
                        agent.update()
            
            # Логирование эпизода
            metrics = {
                'total_reward': episode_reward,
                'portfolio_value': train_env._portfolio_value,
                'sharpe_ratio': self._calculate_sharpe(train_env._portfolio_return_memory)
            }
            self.monitor.log_episode_metrics(metrics)
            
            # Валидация каждые 10 эпизодов
            if episode % 10 == 0:
                val_metrics = self.validate_agent(agent, val_env)
                print(f"Episode {episode}: Train Reward: {episode_reward:.4f}, "
                      f"Val Sharpe: {val_metrics['sharpe_ratio']:.4f}")
                
                # Логирование распределения портфеля
                if hasattr(train_env, '_actions_memory') and train_env._actions_memory:
                    last_weights = train_env._actions_memory[-1]
                    self.monitor.log_portfolio_distribution(
                        last_weights,
                        names=['Cash'] + train_env._tic_list.tolist()
                    )
        
        print("Обучение завершено!")
        return agent
    
    def validate_agent(self, agent, val_env):
        """Валидация агента"""
        obs, _ = val_env.reset()
        if hasattr(agent, 'test_pvm'):  # Policy Gradient
            agent.test_pvm.reset()
            pvm = agent.test_pvm.retrieve()
        else:
            pvm = np.array([1] + [0] * val_env.portfolio_size)
        
        done = False
        episode_reward = 0
        
        while not done:
            if hasattr(agent, 'test_policy'):  # Policy Gradient
                obs_batch = np.expand_dims(obs, axis=0)
                pvm_batch = np.expand_dims(pvm, axis=0)
                action = agent.test_policy(obs_batch, pvm_batch)
                agent.test_pvm.add(action)
            elif isinstance(agent, PPOAgent):
                action, _, _ = agent.get_action(obs, pvm)
            else:  # SAC
                action = agent.get_action(obs, pvm, evaluate=True)
            
            next_obs, reward, done, _, info = val_env.step(action)
            episode_reward += reward
            
            obs = next_obs
            if hasattr(val_env, '_actions_memory'):
                pvm = val_env._actions_memory[-1]
        
        return {
            'total_reward': episode_reward,
            'portfolio_value': val_env._portfolio_value,
            'sharpe_ratio': self._calculate_sharpe(val_env._portfolio_return_memory)
        }
    
    def _calculate_sharpe(self, returns):
        """Расчёт коэффициента Шарпа"""
        if len(returns) < 2:
            return 0
        return np.mean(returns) / (np.std(returns) + 1e-8) * np.sqrt(252)
    
    async def live_trading_demo(self, agent):
        """Демонстрация live торговли в песочнице"""
        print("\nЗапуск демо live торговли...")
        
        # Инициализация MOEX интеграции
        moex = MOEXIntegration(sandbox=True)
        moex.balance = self.config['initial_amount']
        
        # Создание real-time обёртки
        # Для демо используем историческую среду
        demo_env = self.create_environments(self.prepare_data())[1]
        rt_env = RealTimeEnvironmentWrapper(demo_env, moex, update_interval=60)
        
        # Симуляция live торговли
        print("Симуляция live торговли в песочнице...")
        obs, _ = demo_env.reset()
        pvm = np.array([1] + [0] * demo_env.portfolio_size)
        
        for step in range(50):  # 50 шагов для демо
            # Получение действия от агента
            if hasattr(agent, 'test_policy'):
                action = agent.test_policy(np.expand_dims(obs, 0), np.expand_dims(pvm, 0))
            elif isinstance(agent, PPOAgent):
                action, _, _ = agent.get_action(obs, pvm)
            else:
                action = agent.get_action(obs, pvm, evaluate=True)
            
            # Исполнение через real-time обёртку
            next_obs, reward, done, _, info = await rt_env.step(action)
            
            # Логирование live метрик
            portfolio_state = moex.get_portfolio_state()
            self.live_monitor.log_trade({
                'step': step,
                'action': action.tolist(),
                'portfolio_value': portfolio_state['balance'],
                'timestamp': portfolio_state['timestamp']
            })
            
            # Симуляция рыночных условий
            market_data = {
                'volatility': np.random.uniform(0.1, 0.3),
                'volume': np.random.uniform(1000000, 5000000)
            }
            self.live_monitor.log_market_conditions(market_data)
            
            obs = next_obs
            pvm = action
            
            if done:
                break
            
            # Небольшая задержка для реалистичности
            await asyncio.sleep(0.1)
        
        print("Live торговля завершена!")
        print(f"Финальное состояние портфеля: {moex.get_portfolio_state()}")


def main():
    """Главная функция для запуска системы"""
    # Конфигурация
    config = {
        'data_path': './data/',
        'log_dir': './runs/portfolio_optimization',
        'live_log_dir': './runs/live_trading',
        'start_date': '2020-01-01',
        'tickers': TICKER_LIST[:10],  # Используем первые 10 тикеров
        'initial_amount': 1000000,
        'features': ['close', 'high', 'low'],
        'time_window': 20,
        'commission_fee': 0.0025,
        'reward_scaling': 100,
        'lr_actor': 3e-4,
        'lr_critic': 1e-3,
        'gamma': 0.99,
        'episodes': 50
    }
    
    # Создание системы
    system = AdvancedPortfolioSystem(config)
    
    # Подготовка данных
    df = system.prepare_data()
    
    # Создание сред
    train_env, val_env = system.create_environments(df)
    
    print("\n=== Демонстрация различных агентов и архитектур ===")
    
    # 1. PPO с Transformer
    print("\n1. PPO агент с TransformerPM архитектурой")
    ppo_agent = system.create_agent(train_env, agent_type='ppo', model_type='transformer')
    ppo_agent = system.train_agent(ppo_agent, train_env, val_env, episodes=20)
    
    # 2. SAC с Hybrid GNN
    print("\n2. SAC агент с HybridTransformerGNN архитектурой")
    sac_agent = system.create_agent(train_env, agent_type='sac', model_type='hybrid_gnn')
    sac_agent = system.train_agent(sac_agent, train_env, val_env, episodes=20)
    
    # 3. Policy Gradient с Neural ODE
    print("\n3. Policy Gradient с NeuralODE архитектурой")
    pg_agent = system.create_agent(train_env, agent_type='pg', model_type='neural_ode')
    pg_agent.train(episodes=20)
    
    # Live trading демо
    print("\n=== Live Trading Demo ===")
    asyncio.run(system.live_trading_demo(ppo_agent))
    
    # Закрытие мониторов
    system.monitor.close()
    system.live_monitor.close()
    
    print("\n✅ Все демонстрации завершены! Проверьте TensorBoard для визуализации:")
    print(f"tensorboard --logdir {config['log_dir']}")


if __name__ == "__main__":
    main() 