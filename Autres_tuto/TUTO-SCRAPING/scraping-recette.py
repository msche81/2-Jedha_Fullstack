import requests
from bs4 import BeautifulSoup

url = "https://www.galaxus.ch/en"

response = requests.get(url)

if response.status_code == 200:
    html = response.text
    # print(html)

    f = open("recette.html", "w")
    f.write(html)
    f.close()

    soup = BeautifulSoup(html, "html5lib")

    titre = soup.find("h4")

    print(titre)

else:
    print("ERREUR", response.status_code)


print("FIN")