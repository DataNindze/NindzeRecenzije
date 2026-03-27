# Project: Croatian Sentiment Reviews Corpus

## Team Members
- Lorena Čizmek
- Dora Posilovic
- Sara Henč
- Nika Haraminčić

## Project Description
This project focuses on collecting Croatian reviews and creating a manually annotated sentiment dataset. 
The data were collected from publicly available medical review website Najdoktor (https://najdoktor.com/). 

## Dataset
- Language: Croatian
- Domain: Medical reviews
- Format: CSV
- Minimum sentences: 3000

## Methodology
1. Reviews were collected using Python web scraping scripts.
2. The text was cleaned and split into individual sentences.
3. A pilot annotation round was conducted on a sample of sentences.
4. Inter-annotator agreement was measured using Fleiss' kappa.


## Repository Structure
- corpus/ → datasets
- python scripts/ → scraper and fleiss kapa
- README.md → description

## Methods and Tools
- Python
- Selenium
- BeautifulSoup
- Pandas
- Statsmodels
- GitHub

## Notes
- Only Croatian reviews
- Only full sentences
- The dataset was created for academic purposes
