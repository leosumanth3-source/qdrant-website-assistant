# import requests
# from bs4 import BeautifulSoup

# url = "https://qdrant.tech/documentation/"

# response = requests.get(url)

# print(response.status_code)

# soup = BeautifulSoup(response.text, "html.parser")

# print(soup.title.text)

#adding links here

# import requests
# from bs4 import BeautifulSoup

# url = "https://qdrant.tech/documentation/"

# response = requests.get(url)

# print(response.status_code)

# soup = BeautifulSoup(response.text, "html.parser")

# print(soup.title.text)

# for link in soup.find_all("a"):
#     href = link.get("href")

#     if href and href.startswith("https://qdrant.tech/documentation/"):
#         print(href)


#to complete the url path
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin

url = "https://qdrant.tech/documentation/"

response = requests.get(url)

print(response.status_code)

soup = BeautifulSoup(response.text, "html.parser")

print(soup.title.text)

for link in soup.find_all("a"):
    href = link.get("href")

    if href:
        full_url = urljoin(url, href)

        if full_url.startswith("https://qdrant.tech/documentation/"):
            print(full_url)