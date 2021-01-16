from websocket import create_connection
import json

ws = create_connection("wss://ws.kraken.com/")

subscription_dict = {
	"event":"subscribe",
	"subscription":{
		"name":"trade"
	},
	"pair":["BTC/USD"]
}

ws.send(json.dumps(subscription_dict))

while True:
	print(ws.recv_data())
