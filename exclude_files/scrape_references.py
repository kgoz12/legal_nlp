#######################################################################################################################
#  Exploratory analysis of a particular executive orders and its relationship to legal or scholarly precedence 
#
# The EO we will focus on: https://www.whitehouse.gov/presidential-actions/2025/02/ensuring-accountability-for-all-agencies/ 
#
# Pertent sources to help us understand this EO (unitary executive theory?)
# 
#   1. federalist paper #70: https://avalon.law.yale.edu/18th_century/fed70.asp \
#   2. federalist paper #47: https://avalon.law.yale.edu/18th_century/fed47.asp \
#   3. definition of "agencies" mentioned in EO: https://www.law.cornell.edu/uscode/text/44/3502 (cornell law has good resources)
#   4. definition of "employee" mentioned in EO: https://www.law.cornell.edu/uscode/text/5/2105
#   5. EO from 1993 being referenced: https://www.archives.gov/files/federal-register/executive-orders/pdf/12866.pdf
#
## these are some notes on additional sources that might be useful:
# Specific example of good FOIA info: https://www.justice.gov/archives/oip/blog/foia-update-oip-guidance-congressional-access-under-foia



#######################################################################################################################
# Scrape the text of the two federalist papers
import scrapy 
from pathlib import Path

class MySpider(scrapy.Spider):
    name = 'federalist_papers'

    def start_requests(self):
        urls= [
            'https://avalon.law.yale.edu/18th_century/fed47.asp', 
            'https://avalon.law.yale.edu/18th_century/fed70.asp'
            ]
        for url in urls:
            yield scrapy.Request(url=url, callback=self.parse)

    def parse(self, response):
        page = response.url.split("/")[-1]
        filename=f"federalist-paper-{page}"
        Path(filename).write_bytes(response.body)
        self.log(f"Saved file {filename}")






# let's use the scrapy shell to see what we can get from the fed papers site:
# scrapy shell 'https://avalon.law.yale.edu/18th_century/fed47.asp' --nolog
# response.xpath('//body').get()

# scrapy runspider scrape_references.py