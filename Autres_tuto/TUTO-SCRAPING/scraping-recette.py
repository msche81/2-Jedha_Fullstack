import requests
from bs4 import BeautifulSoup

url = "https://www.marmiton.org/"

response = requests.get(url)

if response.status_code == 200:
    html = response.text
    # print(html)

    f = open("recette.html", "w")
    f.write(html)
    f.close()

    soup = BeautifulSoup(html, "html5lib")

else:
    print("ERREUR", response.status_code)


print("FIN")