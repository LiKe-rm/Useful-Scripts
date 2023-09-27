# utf-8
from selenium import webdriver
from selenium.webdriver.edge.options import Options
from selenium.webdriver.common.by import By
import time,datetime

# start=time.time()
options =webdriver.ChromeOptions()
# 设置EdgeOptions
driver = webdriver.Chrome(options=options)
# driver = webdriver.Edge()
# 使用EdgeOptions初始化EdgeDriver

url = "https//docs.qq.com/form/page/..."     #在线收集表的url地址
driver.get(url)


# 打开网址
# time.sleep(1)
content=''
elmet = driver.find_element(By.ID,"header-login-btn")
elmet.click()
driver.implicitly_wait(2)
# driver.execute_script("window.location.reload()")
elmet=driver.find_element(By.CSS_SELECTOR,'span.qq')
elmet.click()
while True:    # 等待通过手机扫码或者其他方式登录，之后输入y即可开始等待开始抢填
    ch = input("Are you logined ok?(y/n)")
    if(ch == 'y'):
        break
# elmet = driver.find_element(By.ID,"img_out_qqnum")
# elmet.click()
# print("快捷登录成功")
# time.sleep(1)

# 设定执行时间2023年9月20日18点
execute_time = datetime.datetime(2023, 9, 25, 18, 0, 0)

# 等待到指定时间再执行
print("Waiting for the begining...")
while datetime.datetime.now() < execute_time:
    time.sleep(1)
print("Begining!")
start = time.time()
driver.execute_script("window.location.reload()")   # 刷新网页
elmet = driver.find_elements(By.XPATH,"//textarea[@placeholder='请输入']")
# elmet[0].send_keys("11")
# elmet[1].send_keys("22")
# elmet[2].send_keys("33")
# elmet[3].send_keys("44")

# 由于不同的网页收集表提交和确认按钮似乎会变化，不推荐使用css_selector选择
# button = driver.find_element(By.CSS_SELECTOR,'#root > div.form-root.fill-form-root > div > div > div.form-fill-container > div.form-with-history-record.fill-area > div.form-body.form-fill-body > div.question-commit > button')
# driver.execute_script("arguments[0].click();", button)
# elmet.click()
# button = driver.find_element(By.CSS_SELECTOR,'body > div.dui-modal-mask.dui-modal-mask-visible.dui-modal-mask-display > div > div.dui-modal-footer > button.dui-button.dui-modal-footer-ok.dui-button-type-primary.dui-button-size-default > div')
# button.click()
# elmet = driver.find_element(By.CSS_SELECTOR,'textarea[placeholder="请输入"]')
# elmet.send_keys("15623738228")
# textareas = driver.find_elements_by_css_selector('textarea[placeholder="请输入"]')

button = driver.find_element(By.XPATH,"//button[text()='提交']")
driver.execute_script("arguments[0].click();", button)
time.sleep(1)
button = driver.find_element(By.XPATH,"//button[contains(.,'确认')]")
button.click()

print(time.time()-start)
print("The current date and time is", datetime.datetime.now())
