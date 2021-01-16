from websocket import create_connection
import urllib.request
import time, base64, hashlib, hmac, os
import json

def getToken():
    api_nonce = bytes(str(int(time.time()*1000)), "utf-8")

    api_request = urllib.request.Request("https://api.kraken.com/0/private/GetWebSocketsToken", b"nonce=%s" % api_nonce)

    api_request.add_header("API-Key",os.getenv("KRAKEN_KEY"))

    api_request.add_header("API-Sign", base64.b64encode(hmac.new(base64.b64decode(os.getenv("KRAKEN_SECRET")), b"/0/private/GetWebSocketsToken" + hashlib.sha256(api_nonce + b"nonce=%s" % api_nonce).digest(), hashlib.sha512).digest()))

    return json.loads(urllib.request.urlopen(api_request).read())['result']['token']


def init_kraken():
    token = getToken()
    public = create_connection("wss://ws.kraken.com/")
    private = create_connection("wss://ws-auth.kraken.com/")
    
    public_subscription = {
        "event":"subscribe",
        "subscription":{
            "name":"trade",
        },
        "pair":["XBT/USD","ETH/USD","XDG/USD"]
    }

    private_subscription = {
        "event": "subscribe",
        "subscription":{
            "name": "ownTrades",
            "token":token
                }
    }

    public.send(json.dumps(public_subscription))
    private.send(json.dumps(private_subscription))
    return public, private, token

