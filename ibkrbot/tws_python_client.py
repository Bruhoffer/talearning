from ibapi.client import *
from ibapi.wrapper import *
from ibapi.contract import Contract
from ibapi.order import Order
from decimal import Decimal
from time import sleep
from config import host, port, client_id, ib_account
from threading import Thread
from datetime import datetime

contract_request_dictionary = {
    1 : 'AAPL'
}

history_request_dictionary = {
    4001: 'AAPL'
}

class TradeApp(EWrapper, EClient):
    def __init__(self):
        EClient.__init__(self, self)
        
        #custom attributes
        self.next_order_id = None
        self.account_balance = None
        self.account_equity = None
        self.portfolio = {}
        
        self.marketdata = {}
        for key, val in contract_request_dictionary.items():
            self.marketdata[val] = {}
            
        self.ohlc_data = {}
        for key, val in history_request_dictionary.items():
            self.ohlc_data[val] = {}
            
    def nextValidId(self, orderId):
        print("NextValidId:", orderId)
        self.next_order_id = orderId
        
    def get_next_valid_id(self):
        valid_order_id = self.next_order_id
        self.next_order_id += 1
        return valid_order_id
        
        
    def updateAccountValue(self, key: str, val: str, currency: str, accountName: str):
        
        if key == 'TotalCashBalance' and currency == 'BASE':
            self.account_balance = val
            
        elif key == 'NetLiquidationByCurrency' and currency == 'BASE':
            self.account_equity = None
            
    def updatePortfolio(self, contract: Contract, position: Decimal,marketPrice: float, marketValue: float, averageCost: float, unrealizedPNL: float, realizedPNL: float, accountName: str):
        print('test')
        print("UpdatePortfolio.", "Symbol:", contract.symbol, "SecType:", contract.secType, "Exchange:",contract.exchange, "Position:", decimalMaxString(position), "MarketPrice:", floatMaxString(marketPrice),
              "MarketValue:", floatMaxString(marketValue), "AverageCost:", floatMaxString(averageCost), 
              "UnrealizedPNL:", floatMaxString(unrealizedPNL), "RealizedPNL:", floatMaxString(realizedPNL), "AccountName:", 
              accountName)
        print(contract)
        
        self.portfolio[contract.localSymbol] = {
            'position': decimalMaxString(position),
            'marketPrice': floatMaxString(marketPrice),
            'marketValue': floatMaxString(marketValue),
            'averageCost': floatMaxString(averageCost),
            'unrealizedPNL': floatMaxString(unrealizedPNL),
            'realizedPNL': floatMaxString(realizedPNL)
        }
        
    def tickPrice(self, reqId: TickerId, tickType: TickType, price: float, attrib: TickAttrib): 
        if tickType == 1:
            self.marketdata[contract_request_dictionary[reqId]]['bid'] = price
        elif tickType == 2:
            self.marketdata[contract_request_dictionary[reqId]]['ask'] = price
        elif tickType == 4:
            self.marketdata[contract_request_dictionary[reqId]]['last'] = price
            
    def historicalData(self, reqId: int, bar: BarData):
        #print("HistoricalData. ReqId:", reqId, "BarData.", bar)
        
        time = bar.date
        self.ohlc_data[history_request_dictionary[reqId]][time] = {'open': bar.open, 'high': bar.high, 'low': bar.low, 'close': bar.close, 'volume': decimalMaxString(bar.volume)}

    def openOrder(self, orderId: OrderId, contract: Contract, order: Order, orderState: OrderState):
        print(orderId, contract, order, orderState)
    
    def orderStatus(self, orderId: OrderId, status: str, filled: Decimal, remaining: Decimal, avgFillPrice: float, permId: int, parentId: int, lastFillPrice: float, clientId: int, whyHeld: str, mktCapPrice: float):
        super().orderStatus(orderId, status, filled, remaining, avgFillPrice, permId, parentId, lastFillPrice, clientId, whyHeld, mktCapPrice)
    
if __name__ == '__main__':
    app = TradeApp()
    

    app.connect(host, port, client_id)
    sleep(1)
    app_thread = Thread(target=app.run, daemon = True)
    app_thread.start()
    
    app.reqAccountUpdates(True, ib_account)
    
    # defining Apple contract
    contract = Contract()
    contract.conId =  265598
    contract.exchange = 'NASDAQ'
    
    #subscribing to market data
    reqId = 1
    app.reqMktData(reqId, contract, "", False, False, [])
    
    sleep(3)
    
    # Open a market order
    market_order = Order()
    market_order.action = 'BUY'
    market_order.orderType = 'MKT'
    market_order.totalQuantity = 100
    market_order.tif = 'DAY'
    next_valid_id = app.get_next_valid_id()
    
    app.placeOrder(next_valid_id, contract, market_order)
    #subscribing to historical data
    req_id_historical = 4001
    app.reqHistoricalData(req_id_historical, contract, "", "1 D", "8 hours", "TRADES", 1, 2, True, [])
    while True:
        current_time = datetime.now()
        
        # print('Current Time: ', current_time)
        # print("Balance: ", app.account_balance)
        # print("Equity:", app.account_equity)
        # print('portfolio', app.portfolio)
        # print('---\n')
        print('OHLC Data', app.ohlc_data)
        print('Market Data', app.marketdata)
        sleep(5)
        
        
  