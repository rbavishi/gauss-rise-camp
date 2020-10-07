from urllib.request import urlopen, Request

GAUSS_MAGIC_PREFIX = "__GAUSS__0xdeadbeef"

with urlopen(Request("https://cdnjs.cloudflare.com/ajax/libs/notify/0.3.4/notify.min.js",
                     headers={'User-Agent': 'Mozilla/5.0'})) as data:
    NOTIFY_JS = data.read().decode("utf-8")
