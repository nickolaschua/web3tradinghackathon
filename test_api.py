import hmac, hashlib, time, requests

lines = open('.env').readlines()
api_key = [l.split('=',1)[1].strip() for l in lines if l.startswith('ROOSTOO_API_KEY=')][0]
secret = [l.split('=',1)[1].strip() for l in lines if l.startswith('ROOSTOO_SECRET=')][0]

print('key:', repr(api_key))
print('secret:', repr(secret))

payload = {'timestamp': str(int(time.time() * 1000))}
total_params = "&".join(f"{k}={payload[k]}" for k in sorted(payload.keys()))
sig = hmac.new(secret.encode('utf-8'), total_params.encode('utf-8'), hashlib.sha256).hexdigest()
headers = {'RST-API-KEY': api_key, 'MSG-SIGNATURE': sig}

print('total_params:', total_params)
print('sig:', sig)

# Print the prepared URL so we can verify no mangling
req = requests.Request('GET', 'https://mock-api.roostoo.com/v3/balance',
    params=total_params, headers=headers)
prepared = req.prepare()
print('prepared_url:', prepared.url)

r = requests.Session().send(prepared)
print('status:', r.status_code)
print('body:', r.text[:300])
