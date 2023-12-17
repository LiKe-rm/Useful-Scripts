# utf-8
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import time, datetime


# options =webdriver.ChromeOptions()
# driver = webdriver.Chrome(options=options)

# edge 通过添加环境变量的方式指定对应的driver.exe, ，如本机的D:\software\edgedriver_win64
options = webdriver.EdgeOptions()
driver = webdriver.Edge()

def test0():
    driver.get('https://bing.com')
    element = driver.find_element(By.ID, 'sb_form_q')
    element.send_keys('WebDriver')
    element.submit()
    time.sleep(5)
    driver.quit()


def test():
    url = "https://docs.qq.com/form/page/DSXJMWUVYWnl3UEZs?u=3c7b87fb66f7484e8549216d8fd08aa0#/fill" 
    driver.get(url)

    elmet = driver.find_element(By.ID,"header-login-btn")
    elmet.click()
    driver.implicitly_wait(2)
    elmet=driver.find_element(By.CSS_SELECTOR,'span.qq')
    elmet.click()
    while True:    # 等待通过手机扫码或者其他方式登录，之后输入y即可开始等待开始抢填
        ch = input("Are you logined ok?(y/n)")
        if(ch == 'y'):
            break
    
    # 填空题
    timeout = 10  # 设置超时时间，单位为秒
    locator = (By.XPATH, "//textarea[@placeholder='请输入']")
    elements = []
    while not elements:
        try:
            elements = WebDriverWait(driver, timeout).until(EC.presence_of_all_elements_located(locator))
        except:
            pass
    elements[0].send_keys("11") 

    # 选择题
    radio_buttons = WebDriverWait(driver, 10).until(
        EC.presence_of_all_elements_located((By.CLASS_NAME, "form-choice-radio-option"))
    )
    desired_class = "form-choice-radio-option-text-content"

    time.sleep(5)
    # 设定选项对应的key
    key = "一班"

    for radio_button in radio_buttons:
        text_content = radio_button.find_element(By.CLASS_NAME, desired_class).text
        if text_content == key:
            radio_button.click()
            break
    
    button = driver.find_element(By.XPATH,"//button[text()='提交']")
    driver.execute_script("arguments[0].click();", button)
    locator = (By.XPATH, "//button[contains(.,'确认')]")
    button = WebDriverWait(driver, timeout).until(EC.presence_of_element_located(locator))
    button.click()

if __name__ == "__main__":
    test()

