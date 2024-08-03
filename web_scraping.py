import asyncio
import aiohttp
import logging
from bs4 import BeautifulSoup
import pandas as pd
import json
import sqlite3
from typing import Dict, List, Union
from lxml import html

logging.basicConfig(level=logging.INFO)

class WebScraper:
    def __init__(self, base_url: str, headers: Dict[str, str] = None, proxies: Dict[str, str] = None):
        self.base_url = base_url
        self.headers = headers
        self.proxies = proxies

    async def fetch_page(self, session: aiohttp.ClientSession, url: str) -> Union[str, None]:
        try:
            async with session.get(url) as response:
                response.raise_for_status()
                return await response.text()
        except aiohttp.ClientError as e:
            logging.error(f"Error fetching {url}: {e}")
            return None

    async def fetch_all_pages(self, endpoints: List[str]) -> List[Union[str, None]]:
        async with aiohttp.ClientSession(headers=self.headers) as session:
            tasks = [self.fetch_page(session, f"{self.base_url}{endpoint}") for endpoint in endpoints]
            return await asyncio.gather(*tasks)

    def parse_page(self, html_content: str, parser: str = 'html.parser') -> BeautifulSoup:
        try:
            return BeautifulSoup(html_content, parser)
        except Exception as e:
            logging.error(f"Error parsing HTML content: {e}")
            return None

    def extract_data(self, soup: BeautifulSoup, selectors: Dict[str, Union[str, List[str]]]) -> Dict[str, Union[str, List[str]]]:
        data = {}
        try:
            for key, selector in selectors.items():
                if isinstance(selector, list):
                    elements = [elem.text.strip() for elem in soup.select(selector[0])]
                    data[key] = elements
                else:
                    element = soup.select_one(selector)
                    data[key] = element.text.strip() if element else None
            return data
        except Exception as e:
            logging.error(f"Error extracting data: {e}")
            return None

    def extract_data_xpath(self, html_content: str, xpath_selectors: Dict[str, str]) -> Dict[str, str]:
        data = {}
        try:
            tree = html.fromstring(html_content)
            for key, xpath in xpath_selectors.items():
                elements = tree.xpath(xpath)
                data[key] = elements[0].strip() if elements else None
            return data
        except Exception as e:
            logging.error(f"Error extracting data using XPath: {e}")
            return None

    async def scrape(self, endpoints: List[str], selectors: Dict[str, Union[str, List[str]]], use_xpath: bool = False) -> pd.DataFrame:
        html_contents = await self.fetch_all_pages(endpoints)
        scraped_data = []
        tasks = []

        async def process_html_content(html_content):
            if html_content:
                if use_xpath:
                    data = self.extract_data_xpath(html_content, selectors)
                else:
                    soup = self.parse_page(html_content)
                    data = self.extract_data(soup, selectors)
                if data:
                    scraped_data.append(data)

        for html_content in html_contents:
            tasks.append(process_html_content(html_content))
        
        await asyncio.gather(*tasks)
        return pd.DataFrame(scraped_data)

    def save_data(self, df: pd.DataFrame, file_format: str = 'csv', file_path: str = 'data.csv'):
        try:
            if file_format == 'csv':
                df.to_csv(file_path, index=False)
            elif file_format == 'json':
                df.to_json(file_path, orient='records', lines=True)
            elif file_format == 'sqlite':
                conn = sqlite3.connect(file_path)
                df.to_sql('scraped_data', conn, if_exists='replace', index=False)
                conn.close()
            logging.info(f"Data saved to {file_path}")
        except Exception as e:
            logging.error(f"Error saving data: {e}")

if __name__ == "__main__":
    base_url = "https://example.com"
    endpoints = ["/page1", "/page2", "/page3"]
    selectors = {
        "title": "h1.title",
        "description": "div.description",
        "date": "span.date"
    }
    
    scraper = WebScraper(base_url)
    df = asyncio.run(scraper.scrape(endpoints, selectors))
    scraper.save_data(df, file_format='csv', file_path='scraped_data.csv')
    print(df)
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
