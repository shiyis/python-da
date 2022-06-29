import os

# ===============global inputs===================
path = os.path.dirname(os.path.realpath("__file__"))
os.chdir(path)  # sets the directory
input_path = "input/candidates_out.csv"  # just relative path should work
chrome_drive_exc_path = (
    "/Users/shiyishen/.wdm/drivers/chromedriver/mac64/89.0.4389.23/chromedriver"
)
output_path = "output.txt/"  # just relative path should work
# ===============================================

from selenium import webdriver
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver import ActionChains
from selenium.common.exceptions import *
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.options import Options
from bs4 import BeautifulSoup
import pandas as pd
from tqdm import tqdm
import sys
import re
import numpy as np
from random import randint

tqdm.pandas()

import time

import requests
from collections import defaultdict


class SeleniumScraper:
    def __init__(self, filename, name, url, driver):
        self.url = url
        self.name = name
        self.driver = driver
        self.driver.set_page_load_timeout(30)
        try:
            self.driver.get(url)
        except:
            print("Page load time out")
        self._file = open(filename, "a+")
        self._tags = ["h1", "h2", "h3", "h4", "h5", "strong", "p", "li", "span"]

    def wait_for_several_elements(self, selector, min_amount, limit=20):
        """
        This function provides awaiting of <min_amount> of elements found by <selector> with
        time limit = <limit>
        """
        step = 1  # in seconds; sleep for 500ms
        current_wait = 0

        while current_wait < limit:
            try:
                print("Waiting... " + str(current_wait))
                query = self.driver.find_elements_by_css_selector(selector)
                if len(query) > min_amount:
                    print("Found!")
                    self._file.write("Number of query: " + str(len(query)))
                    links = []
                    for a in query:
                        if a.text and "".join(a.text.split()).isalpha():
                            try:
                                ActionChains(self.driver).key_down(Keys.COMMAND).click(
                                    a
                                ).key_up(Keys.COMMAND).perform()
                            except TimeoutException:
                                print("Page load time out!")
                        links.append(a.get_attribute("href"))
                    return links
                else:
                    time.sleep(step)
                    current_wait += step
            except:
                time.sleep(step)
                current_wait += step

        return "Not enough elements found!"

    def extract_data(self):
        url = [i for i in self.url.split("/") if i != ""][1]
        domain_filter = re.compile(f"w*.?({url})", re.IGNORECASE)
        image_filter = re.compile(
            "|".join([".php", ".jpg", ".jpeg", ".png"]), re.IGNORECASE
        )  # getting rid of image links
        checked = []

        checked = set()
        for window in driver.window_handles:
            try:
                self.driver.switch_to_window(window)
                self.driver.delete_all_cookies()

                url = driver.current_url
                if (
                    url not in checked
                    and domain_filter.search(url) != None
                    and image_filter.search(url) == None
                ):
                    checked.add(url)
                    self._file.write(f"\n\n=====LINK SOURCE: {str(url)}=====\n")

                    if "issue" in url:
                        key = ["toggle", "accordion", "panel"]
                        elems = []
                        for k in key:
                            elems.extend(
                                self.driver.find_elements_by_css_selector(
                                    f'div:not([class*="footer"]):not([class*="header"]):not([class*="nav"]):not([class*="menu"])[class*="{k}"] a'
                                )
                            )

                        texts = defaultdict(set)

                        for e in elems:
                            if e.text:
                                WebDriverWait(self.driver, 50).until(
                                    EC.element_to_be_clickable(
                                        (By.LINK_TEXT, e.text.rstrip())
                                    )
                                ).send_keys(Keys.COMMAND + Keys.ENTER)
                                for tag in self._tags:
                                    elems = self.driver.find_elements_by_tag_name(tag)
                                    for el in elems:
                                        try:
                                            texts[tag].add(
                                                " ".join(
                                                    [
                                                        tok.rstrip()
                                                        for tok in el.text.split()
                                                    ]
                                                )
                                            )
                                        except StaleElementReferenceException:
                                            print("Stale element!")
                        for k, v in texts.items():
                            if "".join(v) != "":
                                self._file.write(f"\n <tag> {k}\n")
                                self._file.write("\n\n" + str("\n\n".join(v)) + "\n")
                    else:
                        for tag_idx in range(1, len(self._tags)):
                            elems = self.driver.find_elements_by_tag_name(
                                self._tags[tag_idx]
                            )
                            texts = set()
                            for el in elems:
                                try:
                                    texts.add(
                                        " ".join(
                                            [tok.rstrip() for tok in el.text.split()]
                                        )
                                    )
                                except StaleElementReferenceException:
                                    print("Stale element!")
                            if "".join(texts) != "":
                                self._file.write(f"\n <tag> {self._tags[tag_idx]}\n")
                                self._file.write(
                                    "\n\n" + str("\n\n".join(texts)) + "\n"
                                )
                self._file.write(
                    "\n\nNumber of links (dup removed)"
                    + str(len(checked))
                    + "\n\nAll links:\n"
                    + str(checked)
                )
            except TimeoutException:
                print("Page loading time out!")


if __name__ == "__main__":
    df = pd.DataFrame(pd.read_csv(input_path))
    chrome_options = Options()
    for index, row in df.iterrows():
        if index >= 50:
            print(index)
            driver = webdriver.Chrome(
                chrome_options=chrome_options, executable_path=chrome_drive_exc_path
            )
            driver.maximize_window()
            time.sleep(3)
            if str(row["campaign_site"]) != "nan":
                print(row["campaign_site"], row["CAND_NAME"])

                filename = (
                    output_path
                    + str(index)
                    + "_"
                    + "_".join([i.strip() for i in row["CAND_NAME"].split(",")]).lower()
                    + ".txt"
                )
                file = open(filename, "w")
                file.write("Candidate: " + str(row["CAND_NAME"]).strip() + "\nResult:")
                scraper = SeleniumScraper(
                    filename, row["CAND_NAME"], row["campaign_site"], driver
                )
                key = ["nav", "ul"]
                for k in key:
                    elems = scraper.wait_for_several_elements(
                        selector=f"{k} a", min_amount=1, limit=randint(1, 2)
                    )
                print(elems)
                scraper.extract_data()
                driver.quit()
