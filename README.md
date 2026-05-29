# Punjab Agricultural Market Intelligence System 🌾📈

An end-to-end, production-ready Machine Learning and Data Engineering pipeline designed to extract, clean, and forecast agricultural commodity prices across the Punjab province using data from AMIS.pk.

## 🏗️ System Architecture
This project implements a strict ETL (Extract, Transform, Load) architecture:
1. **Extract:** Threaded web scraper bypassing ASP.NET security tokens.
2. **Transform:** Dynamic anomaly purging using a Rolling Median Absolute Deviation (MAD) filter, followed by Time-Aware Cubic Spline Interpolation to bridge missing dates.
3. **Load:** In-memory batch loading to a local PostgreSQL database.
4. **Analysis:** Automated generation of high-resolution Data Forensic Dashboards.

*Current Target Commodities:* Onion, Tomato, Potato Fresh, Green Chilli.

## ⚙️ Prerequisites
* Python 3.10+
* PostgreSQL 14+

## 🚀 Setup & Installation (For Collaborators)

**1. Clone the repository**
```bash
git clone <your_github_repo_url>
cd punjab_market_intelligence