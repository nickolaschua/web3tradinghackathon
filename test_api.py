import hmac, hashlib, time, requests

lines = open('.env').readlines()
api_key = [l.split('=',1)[1].strip() for l in lines if l.startswith('ROOSTOO_API_KEY=')][0]
secret = [l.split('=',1)[1].strip() for l in lines if l.startswith('ROOSTOO_SECRET=')][0]

print('key:', repr(api_key))
print('secret:', repr(secret))

ts = str(int(time.time() * 1000))
msg = 'timestamp=' + ts
sig = hmac.new(secret.encode(), msg.encode(), hashlib.sha256).hexdigest()

r = requests.get('https://mock-api.roostoo.com/v3/balance',
    params={'timestamp': ts},
    headers={'RST-API-KEY': api_key, 'MSG-SIGNATURE': sig})

print('status:', r.status_code)
print('body:', r.text[:300])
