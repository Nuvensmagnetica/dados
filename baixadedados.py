from urllib.request import urlopen, urlretrieve, Request
from urllib.error import URLError, HTTPError
from bs4 import BeautifulSoup
import pandas as pd
from time import sleep
from selenium import webdriver
from selenium.webdriver.common.by import By
import pyautogui
navegador = webdriver.Chrome()
#navegador = webdriver.Opera()

navegador.get("https://izw1.caltech.edu/ACE/ASC/level2/index.html")

navegador.find_element(by= By.XPATH, value='/html/body/div[2]/center[2]/table/tbody/tr[1]/td[2]').click()

navegador.find_element(by= By.XPATH, value='/html/body/div[2]/p[1]/table[4]/tbody/tr/td[1]/p/table/tbody/tr/td[1]/form/select/option[1]').click()

navegador.find_element(by= By.XPATH, value='/html/body/div[2]/p[1]/table[4]/tbody/tr/td[1]/p/table/tbody/tr/td[2]/font/input').click()

navegador.find_element(by= By.XPATH, value='/html/body/pre/a/b/font').click()
#
navegador.find_element(by= By.XPATH, value='/html/body/form/p[3]/table/tbody/tr[1]/td[1]/input').click()

navegador.find_element(by= By.XPATH, value='/html/body/form/p[3]/table/tbody/tr[1]/td[2]/input').click()

navegador.find_element(by= By.XPATH, value='/html/body/form/p[3]/table/tbody/tr[1]/td[3]/input').click()

navegador.find_element(by= By.XPATH, value='/html/body/form/p[3]/table/tbody/tr[1]/td[4]/input').click()

navegador.find_element(by= By.XPATH, value='/html/body/form/p[3]/table/tbody/tr[1]/td[5]/input').click()

navegador.find_element(by= By.XPATH, value='/html/body/form/p[3]/table/tbody/tr[5]/td[1]/input').click()

navegador.find_element(by= By.XPATH, value='/html/body/form/p[3]/table/tbody/tr[6]/td[1]/input').click()

navegador.find_element(by= By.XPATH, value='/html/body/form/p[3]/table/tbody/tr[6]/td[2]/input').click()

navegador.find_element(by= By.XPATH, value='/html/body/form/p[3]/table/tbody/tr[6]/td[3]/input').click()

navegador.find_element(by= By.XPATH, value='/html/body/form/p[3]/table/tbody/tr[11]/td[1]/input').click()

navegador.find_element(by= By.XPATH, value='/html/body/form/p[3]/table/tbody/tr[11]/td[2]/input').click()

navegador.find_element(by= By.XPATH, value='/html/body/form/p[3]/table/tbody/tr[11]/td[3]/input').click()

navegador.find_element(by= By.XPATH, value='/html/body/a').click()

navegador.find_element(by= By.XPATH, value='/html/body/form/b[2]/font/input').click()
sleep(15)
navegador.close()
sleep(1)

#obtendo o HTML
url = "https://izw1.caltech.edu/ACE/ASC/DATA/level3/icmetable2.htm"
headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/94.0.4606.61 Safari/537.36"}

req = Request(url, headers=headers)
response = urlopen(req)
html = response.read()
soup = BeautifulSoup(html, 'html.parser')

# Obtendo as TAGs de interesse ( topo das informações)
table = soup.find('table').get_text()


print(soup.find('table').get_text())
print(soup.find('p'))

event = []

for obj in table:
    obj={
        table: event
    }
    event.append(obj)

event = pd.DataFrame(event)
event.to_csv("name1.csv", index=True, encoding='utf-8')


