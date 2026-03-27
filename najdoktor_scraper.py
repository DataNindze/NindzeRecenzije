from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
import time
import pandas as pd
from bs4 import BeautifulSoup
import re

doctors = [
("https://najdoktor.com/Snje%C5%BEana-Belak/d16900","Snježana Belak"),
("https://najdoktor.com/valentina-hecimovic/d13837","Valentina Hečimović"),
("https://najdoktor.com/blazenka-miskic/d11807","Blaženka Miškić"),
("https://najdoktor.com/docdrsc-dinko-bagatin/d8248", "Dinko Bagatin"),
("https://najdoktor.com/alenka-gagro/d8192", "Alenka Gagro"),
("https://najdoktor.com/lara-spalldi-barisic/d19233", "Lara Spalldi Barišić"),
("https://najdoktor.com/zeljko-sundov/d17304", "Željko Šundov"),
("https://najdoktor.com/tajana-marinovic-remetic/d8378", "Tajana Marinović-Remetić"),
("https://najdoktor.com/dr-blazenka-nekic/d8285", "Blaženka Nekić"),
("https://najdoktor.com/%C5%A0ime-Miji%C4%87/d11225", "Šime Mijić"),
("https://najdoktor.com/Marinko-Maru%C5%A1i%C4%87/d16585", "Marinko Marušić"),
("https://najdoktor.com/kresimir-grsic/d9999", "Krešimir Gršić"),
("https://najdoktor.com/mirela-grgic/d14044", "Mirela Grgić"),
("https://najdoktor.com/domagoj-gajski/d20128", "Domagoj Gajski"),
("https://najdoktor.com/marija-marusic/d10043", "Marija Marušić"),
("https://najdoktor.com/nada-aracic/d14181", "Nađa Aračić"),
("https://najdoktor.com/profdrsc-niksa-drinkovic/d8289", "Nikša Drinković")
]

groupid = 3

service = Service("chromedriver.exe")
driver = webdriver.Chrome(service=service)

all_rows = []
review_global_id = 0


for url, doctor_name in doctors:

    print("Otvaram:", doctor_name)

    driver.get(url)
    time.sleep(4)

    previous_count = 0

    while True:

        comments = driver.find_elements(By.CSS_SELECTOR, "div.commentItem")
        current_count = len(comments)

        if current_count == previous_count:
            break

        previous_count = current_count

        try:
            button = driver.find_element(By.ID,"load-more-comments")
            driver.execute_script("arguments[0].click();", button)
            time.sleep(2)

        except:
            break


    html = driver.page_source
    soup = BeautifulSoup(html,"html.parser")

    comment_items = soup.find_all("div",class_="commentItem")

    def split_sentences(text):
      
        text = re.sub(r'([.!?])(?=\S)', r'\1 ', text)
      
        patterns = [
            r'\bdr\s*\.',
            r'\bprof\s*\.',
            r'\bmr\s*\.',
            r'\bprim\s*\.',
            r'\bdoc\s*\.',
            r'\bitd\s*\.',
            r'\bnpr\s*\.',
            r'\bdr\.sc\s*\.'
        ]

        for pattern in patterns:
            text = re.sub(pattern, lambda m: m.group().replace('.', '<DOT>'), text, flags=re.IGNORECASE)

        sentences = re.split(r'(?<=[.!?])\s+', text)

        sentences = [s.replace("<DOT>", ".") for s in sentences]

        return sentences
  


    for item in comment_items:

        comment_div = item.find("div",class_="comment")

        if not comment_div:
            continue

        p = comment_div.find("p")

        if not p:
            continue

        text = p.get_text(strip=True)

        if len(text) < 10:
            continue

        # godina
        date_li = item.find("li")

        year = ""

        if date_li:
            year_match = re.search(r"\d{4}",date_li.text)
            if year_match:
                year = year_match.group()

        review_global_id += 1

        sentences = split_sentences(text)

        sentence_id = 0

        for s in sentences:

            s = s.strip()

            if len(s) < 2:
                continue

            sentence_id += 1

            row = {
                "groupid": groupid,
                "url": url,
                "name": doctor_name,
                "review_id": review_global_id,
                "sentence_id": sentence_id,
                "text": s,
                "label": "",
                "metadata-year": year
            }

            all_rows.append(row)


driver.quit()

df = pd.DataFrame(all_rows)

df.to_excel("najdoktor_recenice.xlsx",index=False)

print("------------------------------------------------")
print("Gotovo.")
print("Ukupno rečenica:",len(df))
print("Datoteka: najdoktor_recenice.xlsx")
