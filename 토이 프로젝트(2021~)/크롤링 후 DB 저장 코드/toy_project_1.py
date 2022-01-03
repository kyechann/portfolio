from selenium import webdriver as driver
from bs4 import BeautifulSoup as bs
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from db_insert import DBHelper as Db
import time
import re
import sys
from basic import TourInfo
from selenium import webdriver as driver
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager

db = Db()
keyword  = input('어디를 가고 싶으세요? ->  ')
# 상품 정보를 담는 리스트 (TourInfo 리스트)
tour_list = []


driver = driver.Chrome(ChromeDriverManager().install())

tour_list = []  # 상품 정보 넣는 리스트 생성

driver.get('https://tour.interpark.com/') 
# 검색창을 찾아서 검색어 입력
driver.find_element_by_id('SearchGNBText').send_keys(keyword)  # id값 찾는법
# 검색 버튼 클릭
driver.find_element_by_css_selector('.search-btn').click()


try:
    element = WebDriverWait(driver, 10).until(
        EC.presence_of_element_located( (By.CLASS_NAME, 'oTravelBox') )
    )
except Exception as e:
    print( '오류 발생', e)
    
driver.implicitly_wait( 10 )
# 더보기 눌러서 => 게시판 진입 
driver.find_element_by_css_selector('.oTravelBox>.boxList>.moreBtnWrap>.moreBtn').click()

for page in range(1, 13):
    try:
        # 자바스크립트 구동하기
        driver.execute_script("searchModule.SetCategoryList(%s, '')" % page)
        time.sleep(2)
        print("%s 페이지 이동" % page)
        boxItems = driver.find_elements_by_css_selector('.oTravelBox>.boxList>li')
        # 상품 하나 하나 접근

        for li in boxItems:
            print( '썸네임', li.find_element_by_css_selector('img').get_attribute('src') )
            print( '링크', li.find_element_by_css_selector('a').get_attribute('onclick') )
            print( '상품명', li.find_element_by_css_selector('h5.proTit').text )
            print( '코멘트', li.find_element_by_css_selector('.proSub').text )
            print( '가격',   li.find_element_by_css_selector('.proPrice').text )

            area = ''
            for info in li.find_elements_by_css_selector('.info-row .proInfo'):
                print(  info.text )
            print('=' * 100)
            # 데이터 모음
            obj = TourInfo(  
                li.find_element_by_css_selector('h5.proTit').text,
                li.find_element_by_css_selector('.proPrice').text,
                li.find_elements_by_css_selector('.info-row .proInfo')[1].text,
                li.find_element_by_css_selector('a').get_attribute('onclick'),
                li.find_element_by_css_selector('img').get_attribute('src')
            )
            tour_list.append( obj )
    except Exception as e1:
        print( '오류', e1 )

print( tour_list, len(tour_list) )


# 수집한 정보 개수를 루프 => 데이터 추출 => 디비
for tour in tour_list:
    # tour => TourInfo
    print(type(tour))
    arr = tour.link.split(',')
    if arr:
        # 대체
        link = arr[0].replace('searchModule.OnClickDetail(','')
        url = link[1:-1]
        driver.get(url)
        time.sleep(2)


        soup = bs(driver.page_source, 'html.parser')
        # 현제 상세 정보 페이지에서 스케줄 정보 획득
        data = soup.select('.tip-cover')

        content_final = ''
        for ii in data[0].contents:
            content_final += str(ii)
        
        # html 콘첸츠 데이터 전처리
        content_final   = re.sub("'", '"', content_final)
        content_final   = re.sub(re.compile(r'\r\n|\r|\n|\n\r+'), '', content_final)

        print( content_final )
        db.db_insertCrawlingData(
            tour.title,
            tour.price[:-1],
            tour.area.replace('출발 가능 기간 : ',''),
            content_final,
            keyword
        )

# 종료
driver.close()
driver.quit()
sys.exit()