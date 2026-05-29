import requests
from bs4 import BeautifulSoup
import pandas as pd
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed

class AMISExtractor:
    """
    Handles the Extraction phase. Bypasses AMIS.pk security tokens and scrapes daily prices.
    """
    def __init__(self, max_workers: int = 5):
        # Added &commodityId=1 to prevent the ASP.NET backend from throwing a 500 error
        self.base_url = "http://www.amis.pk/ViewPrices.aspx?searchType=1&commodityId=1"
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
            'Content-Type': 'application/x-www-form-urlencoded',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8'
        }
        self.max_workers = max_workers
        self.session = requests.Session()
        self.session.headers.update(self.headers)

    def _fetch_security_tokens(self) -> dict:
        """Extracts ASP.NET hidden security tokens required for POST requests."""
        logging.info("Fetching AMIS server security tokens...")
        try:
            resp = self.session.get(self.base_url, timeout=30)
            resp.raise_for_status()
            soup = BeautifulSoup(resp.text, 'html.parser')
            
            viewstate = soup.find('input', id='__VIEWSTATE')
            generator = soup.find('input', id='__VIEWSTATEGENERATOR')
            validation = soup.find('input', id='__EVENTVALIDATION')
            
            return {
                '__VIEWSTATE': viewstate.get('value', '') if viewstate else '',
                '__VIEWSTATEGENERATOR': generator.get('value', '') if generator else '',
                '__EVENTVALIDATION': validation.get('value', '') if validation else ''
            }
        except Exception as e:
            raise RuntimeError(f"CRITICAL: AMIS Token Extraction Failed. {e}")

    def _scrape_city(self, city_id: int, date_str: str, tokens: dict) -> list:
        """Worker function to scrape a single market."""
        # Dynamic city routing matching your teammate's successful payload
        url = f"http://www.amis.pk/ViewPrices.aspx?searchType=1&commodityId={city_id}"
        payload = {
            '__EVENTTARGET': '', 
            '__EVENTARGUMENT': '',
            '__VIEWSTATE': tokens['__VIEWSTATE'],
            '__VIEWSTATEGENERATOR': tokens['__VIEWSTATEGENERATOR'],
            '__EVENTVALIDATION': tokens['__EVENTVALIDATION'],
            'ctl00$cphPage$DateTextBox': date_str,
            'ctl00$cphPage$height': '',
            'ctl00$cphPage$ReminderButton': 'Show prices',
            'Radio': 'on',
            'ctl00$cphPage$myInput': ''
        }
        try:
            resp = self.session.post(url, data=payload, timeout=15)
            soup = BeautifulSoup(resp.text, 'html.parser')
            
            city_label = soup.find('span', id='ctl00_cphPage_lblMsg')
            city_name = city_label.text.strip() if city_label else f"ID_{city_id}"
            
            td_container = soup.find('td', id='ctl00_cphPage_Grd')
            table = td_container.find('table') if td_container else None
            rows_data = []
            
            if table:
                rows = table.find_all('tr')
                for row in rows:
                    cols = row.find_all('td')
                    if len(cols) >= 6 and cols[0].find('a'):
                        rows_data.append({
                            'date': date_str,
                            'market': city_name,
                            'crop': cols[0].find('a').text.strip(),
                            'price': cols[3].text.replace('\xa0', '').strip()  # Standard Max Price
                        })
            return rows_data
        except Exception:
            return []

    def run_daily_scrape(self, target_date: str) -> pd.DataFrame:
        """Executes threaded scraping across standard Punjab market IDs."""
        tokens = self._fetch_security_tokens()
        all_data = []
        
        logging.info(f"Initiating concurrent daily scrape for {target_date}...")
        # Testing a small subset first (1 to 10) to confirm pipeline execution without network flooding
        market_ids = range(1, 11) 
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {executor.submit(self._scrape_city, cid, target_date, tokens): cid for cid in market_ids}
            for future in as_completed(futures):
                result = future.result()
                if result:
                    all_data.extend(result)
                    
        df = pd.DataFrame(all_data)
        logging.info(f"Extraction complete. Pulled {len(df)} raw records.")
        return df