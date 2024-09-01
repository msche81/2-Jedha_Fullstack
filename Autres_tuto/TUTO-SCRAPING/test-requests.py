import requests

url = "https://codeavecjonathan.com/scraping/techsport/"
# url = "https://amazon.fr"

HEADERS = { "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/127.0.0.0 Safari/537.36"}

response = requests.get(url, headers=HEADERS)
response.encoding = response.apparent_encoding

if response.status_code == 200:
    html = response.text
    # print(html)

    f = open("test-request.html", "w")
    f.write(html)
    f.close()

else:
    print("ERREUR", response.status_code)


print("FIN")