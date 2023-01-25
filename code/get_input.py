import logging
import sys
sys.path.insert(1, '/service/code')
from common_utils import setup_logger
logger = logging.getLogger("PATENT")
setup_logger(logger)

class load_input():
    
    def __init__(self):
        pass
    def pre_process(self, data):
        data = str(data)
        if type(data) == type('a'):
            data = data.replace('\t', '')
            data = data.replace('\n', '')
            data = " ".join(data.split())
        return data   

    def list_input(self, lst):
        appl, documents, ntn_cd = [],[],[]
        
        logger.info(f"Input number: {len(lst)}")
        
        for i in lst:
            try:
                appl.append(i['apply_no'])
                ntn_cd.append(i['nation_code'].replace('\n', ''))
                doc = self.pre_process(i['title']) +' [SEP] '+ self.pre_process(i['abstract']) +' [SEP] '+ self.pre_process(i['claim']) 
                documents.append(doc) 
            except:
                logger.info(f"ERROR: apply number: {i['apply_no']}")
                pass
        
        return appl, documents, ntn_cd
    
    def file_input(self, file_path):
        
        appl, documents, ntn_cd = [],[],[]
        f_all = open(file_path,'rt')
   
        lines_all = f_all.readlines()
        lines_all2 = [i.split('\t') for i in lines_all]
        logger.info(f"Input number: {len(lines_all2)}")
        for i in lines_all2:
            try:
                appl.append(i[0])
                ntn_cd.append(i[4].replace('\n', ''))
                doc = self.pre_process(i[1]) +' [SEP] '+ self.pre_process(i[2]) +' [SEP] '+ self.pre_process(i[3])  
                documents.append(doc)
            except:
                logger.info(f"ERROR: apply number: {i[0]}")
                pass

        return appl, documents, ntn_cd
