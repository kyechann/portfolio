from time import strftime
from bs4 import BeautifulSoup
import datetime
import requests

class ScrapConsent():
    def __init__(self, url, info, res, soup, frame, frameaddr):     
        self.url = url
        self.info = info
        self.res = res
        self.soup = soup
        self.frame = frame
        self.frameaddr = frameaddr
        
        
    def return_value(self, url, info, frameaddr):
        url = 'https://finance.naver.com'
        info = '/marketindex/?tabSel=exchange#tab_section'
        res = requests.get(url + info)
        soup = BeautifulSoup(res.content, 'html.parser')
        frame = soup.find('iframe', id="frame_ex1")
        frameaddr = url +frame['src'] #frame내의 연결된 주소 확인    
        res1 = requests.get(frameaddr) # frame내의 연결된 주소를 읽어오기 
        frame_soup = BeautifulSoup(res1.content, 'html.parser')
        items = frame_soup.select('body > div > table > tbody > tr')
        
        date = datetime.date.today()
        datename = date.strftime('%Y-%m-%d') + '.txt'
        for item in items:
            name = item.select('td')[0].text.replace("\n","")
            name = name.replace("\t", "")
            print(name + " " + item.select('td')[1].text)
