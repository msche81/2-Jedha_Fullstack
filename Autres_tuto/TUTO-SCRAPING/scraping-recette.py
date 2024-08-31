import requests
from bs4 import BeautifulSoup

url = "https://www.galaxus.ch/en"

def get_text_if_not_none(e):
    if e:
        return e.text
    return None

response = requests.get(url)

if response.status_code == 200:
    html = response.text
    # print(html)

    f = open("recette.html", "w")
    f.write(html)
    f.close()

    soup = BeautifulSoup(html, "html5lib")

    titre = soup.find("h2", class_="sc-f93c62dd-1 emlFda").text
    print(titre)

    description = get_text_if_not_none(soup.find("p", class_="sc-2e43a11b-3 ezoffT").text)
    print(description)

else:
    print("ERREUR", response.status_code)


print("FIN")