"""
Модуль интеграции с MOEX для реальной торговли и песочницы
"""
import asyncio
import aiohttp
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import json
import websocket
from threading import Thread
import queue
import logging

class MOEXIntegration:
    """Интеграция с MOEX для реальной торговли и песочницы"""
    
    def __init__(self, api_key: str = None, sandbox: bool = True):
        """
        Инициализация интеграции с MOEX
        
        Args:
            api_key: API ключ для реальной торговли
            sandbox: Использовать песочницу (True) или реальную торговлю (False)
        """
        self.api_key = api_key
        self.sandbox = sandbox
        self.base_url = "https://iss.moex.com/iss"
        self.ws_url = "wss://iss.moex.com/ws" if not sandbox else "wss://demo.moex.com/ws"
        
        # Очереди для real-time данных
        self.price_queue = queue.Queue()
        self.order_queue = queue.Queue()
        
        # Кэш текущих позиций
        self.positions = {}
        self.balance = 0.0
        
        # Логирование
        self.logger = logging.getLogger(__name__)
        
    async def get_realtime_prices(self, tickers: List[str]) -> Dict[str, float]:
        """Получение цен в реальном времени"""
        async with aiohttp.ClientSession() as session:
            prices = {}
            for ticker in tickers:
                url = f"{self.base_url}/engines/stock/markets/shares/securities/{ticker}.json"
                async with session.get(url) as response:
                    data = await response.json()
                    if data['marketdata']['data']:
                        last_price = data['marketdata']['data'][0][12]  # LAST price
                        prices[ticker] = float(last_price) if last_price else 0.0
            return prices
    
    def start_websocket_stream(self, tickers: List[str]):
        """Запуск WebSocket стрима для real-time данных"""
        def on_message(ws, message):
            data = json.loads(message)
            self.price_queue.put(data)
            
        def on_error(ws, error):
            self.logger.error(f"WebSocket error: {error}")
            
        def on_open(ws):
            # Подписка на тикеры
            for ticker in tickers:
                ws.send(json.dumps({
                    "cmd": "subscribe",
                    "params": {
                        "type": "candles",
                        "security": ticker,
                        "interval": "1min"
                    }
                }))
        
        self.ws = websocket.WebSocketApp(self.ws_url,
                                         on_message=on_message,
                                         on_error=on_error,
                                         on_open=on_open)
        
        # Запуск в отдельном потоке
        ws_thread = Thread(target=self.ws.run_forever)
        ws_thread.daemon = True
        ws_thread.start()
    
    async def execute_order(self, ticker: str, amount: float, order_type: str = "market"):
        """
        Исполнение ордера
        
        Args:
            ticker: Тикер
            amount: Количество (положительное - покупка, отрицательное - продажа)
            order_type: Тип ордера (market/limit)
        """
        if self.sandbox:
            # Симуляция исполнения в песочнице
            price = await self._get_current_price(ticker)
            cost = abs(amount) * price
            
            if amount > 0:  # Покупка
                if self.balance >= cost:
                    self.balance -= cost
                    self.positions[ticker] = self.positions.get(ticker, 0) + amount
                    return {"status": "executed", "price": price, "amount": amount}
                else:
                    return {"status": "rejected", "reason": "insufficient_funds"}
            else:  # Продажа
                if self.positions.get(ticker, 0) >= abs(amount):
                    self.balance += cost
                    self.positions[ticker] -= abs(amount)
                    return {"status": "executed", "price": price, "amount": amount}
                else:
                    return {"status": "rejected", "reason": "insufficient_shares"}
        else:
            # Реальное исполнение через API
            # TODO: Implement real API call
            raise NotImplementedError("Real trading not implemented yet")
    
    async def _get_current_price(self, ticker: str) -> float:
        """Получение текущей цены"""
        prices = await self.get_realtime_prices([ticker])
        return prices.get(ticker, 0.0)
    
    def get_portfolio_state(self) -> Dict:
        """Получение текущего состояния портфеля"""
        return {
            "positions": self.positions.copy(),
            "balance": self.balance,
            "timestamp": datetime.now()
        }
    
    async def rebalance_portfolio(self, target_weights: Dict[str, float], total_value: float):
        """
        Ребалансировка портфеля до целевых весов
        
        Args:
            target_weights: Целевые веса {ticker: weight}
            total_value: Общая стоимость портфеля
        """
        current_prices = await self.get_realtime_prices(list(target_weights.keys()))
        orders = []
        
        for ticker, target_weight in target_weights.items():
            target_value = total_value * target_weight
            current_price = current_prices[ticker]
            target_shares = target_value / current_price
            current_shares = self.positions.get(ticker, 0)
            
            shares_diff = target_shares - current_shares
            
            if abs(shares_diff) > 0.01:  # Минимальный порог
                orders.append({
                    "ticker": ticker,
                    "amount": shares_diff,
                    "estimated_price": current_price
                })
        
        # Исполнение ордеров
        results = []
        for order in orders:
            result = await self.execute_order(order["ticker"], order["amount"])
            results.append(result)
        
        return results

class RealTimeEnvironmentWrapper:
    """Обёртка для среды с real-time данными"""
    
    def __init__(self, env, moex_integration: MOEXIntegration, update_interval: int = 60):
        """
        Args:
            env: Базовая среда PortfolioOptimizationEnv
            moex_integration: Интеграция с MOEX
            update_interval: Интервал обновления данных (секунды)
        """
        self.env = env
        self.moex = moex_integration
        self.update_interval = update_interval
        self.is_live = False
        
    async def start_live_trading(self):
        """Запуск live торговли"""
        self.is_live = True
        tickers = self.env._tic_list
        
        # Запуск WebSocket стрима
        self.moex.start_websocket_stream(tickers)
        
        while self.is_live:
            # Обновление данных среды из real-time потока
            await self._update_environment_data()
            await asyncio.sleep(self.update_interval)
    
    async def _update_environment_data(self):
        """Обновление данных среды из real-time источника"""
        # Получение последних данных из очереди
        latest_data = []
        while not self.moex.price_queue.empty():
            latest_data.append(self.moex.price_queue.get())
        
        if latest_data:
            # Преобразование в формат среды
            # TODO: Implement data transformation
            pass
    
    async def step(self, action):
        """Шаг с real-time исполнением"""
        if self.is_live:
            # Преобразование действий в ордера
            weights = action
            portfolio_value = self.moex.balance + sum(
                self.moex.positions.get(ticker, 0) * price 
                for ticker, price in (await self.moex.get_realtime_prices(self.env._tic_list)).items()
            )
            
            # Ребалансировка портфеля
            target_weights = {
                ticker: weight 
                for ticker, weight in zip(self.env._tic_list, weights[1:])
            }
            
            results = await self.moex.rebalance_portfolio(target_weights, portfolio_value)
            
            # Обновление среды с результатами
            # TODO: Update environment state based on execution results
            
        return self.env.step(action) 