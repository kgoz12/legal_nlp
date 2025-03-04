import html2text
import re
import os
import glob


#########################################################################################################################################################
### FEDERALIST PAPERS ###################################################################################################################################
#########################################################################################################################################################
# loop through the federalist papers
# we know the papers all start w/ the names of one of the founders, so start the text body there:
# #### MADISON; #### HAMILTON
# we know the papers end w/
# PUBLIUS. 
#########################################################################################################################################################
#########################################################################################################################################################

# path to the federalist paper files
papers_files_names = f"/Users/katherineGoznikar/Desktop/legal_nlp/exclude_files/federalist-paper-fed*.asp"
papers_files_list = glob.glob(papers_files_names)

for paper in papers_files_list:
    # open the web-scraped .asp files & read them w/ html2text function
    html = open(paper)
    f = html.read()
    h = html2text.HTML2Text()
    h.ignore_links = True
    h.body_width = 0
    markdown_string = h.handle(f)

    # read the text after the founder's name 
    markdown_string_lookbehind = re.search(r'((?<=#### MADISON)[\s\S]*)|((?<=#### HAMILTON)[\s\S]*)', markdown_string).group(0)
    # print(markdown_string_lookbehind.group(0))

    # read the text before PUBLIUS.
    markdown_string_lookahead = re.search(r'[\s\S]*(?=PUBLIUS.)', markdown_string_lookbehind).group(0)
    # print(markdown_string_lookahead.group(0))

    # write the txt file locally
    text_file_name = os.path.basename(paper).replace(".asp", "")
    text_file_path = "/Users/katherineGoznikar/Desktop/legal_nlp/exclude_files/"+text_file_name+".txt"
    w = open(text_file_path, "w")
    w.write(markdown_string_lookahead)
    html.close()
    w.close()

# some cleaning of the above text needs to be done. KMG: TODO: tackle this later


#########################################################################################################################################################
### EXECUTIVE ORDER  ###################################################################################################################################
#########################################################################################################################################################
# Order #12866, from 1993
#########################################################################################################################################################
#########################################################################################################################################################