from ibapi.client import EClient
from ibapi.wrapper import EWrapper
from time import sleep
class TradeApp(EWrapper, EClient):
    def __init__(self):
        EClient.__init__(self, self)
        
        
if __name__ == '__main__':
    app = TradeApp()
    
    host = '127.0.0.1'
    port = 7497
    client_id = 0
    
    app.connect(host, port, client_id)
    sleep(1)
    app.run()