import os
from fastapi import FastAPI, APIRouter, File, UploadFile
from pydantic import BaseModel
import uvicorn
import setproctitle
import pandas as pd
import shutil
import logging
import argparse
import json
import sys
import os
sys.path.insert(1, '/service/code')
from get_input import load_input
from infer_models import HybridHelper, ClassificationHelper, GenerationHelper
from common_utils import setup_logger
# FastAPI setting
description = """
Patent Document Classifier Server
"""
tags_metadata = [
    {
        "name": "1. File based classification",
        "descrption": "Patent Document Classification using csv file",
    },
    {
        "name": "2. Content based classification",
        "descrption": "Patent text Classification using input data",
    }
]
    
logger = logging.getLogger("PATENT")
setup_logger(logger)
WORKDIR = "/share"
parser = argparse.ArgumentParser()
parser.add_argument("--top_n", type=int, default=3)
parser.add_argument("--decode_seq_len", type=int, default=5)
parser.add_argument("--gen_model_type", type=str, default="bert_generation")
parser.add_argument("--gen_model_name_or_path", type=str, default=os.path.join(WORKDIR, "gen_model"))
parser.add_argument("--cls_model_type", type=str, default="saltluxbert")
parser.add_argument("--cls_model_name_or_path", type=str, default=os.path.join(WORKDIR, "cls_model"))
parser.add_argument("--max_seq_len", type=int, default=512)
parser.add_argument("--eval_batch_size", type=int, default=int(os.environ.get('BATCH_SIZE')))
parser.add_argument("--do_lower_case", action='store_true', help="")
parser.add_argument("--no_cuda", action='store_true', help="")
args = parser.parse_args()

input_loader = load_input()
gen_model = GenerationHelper(args)
cls_model = ClassificationHelper(args)
hyb_model = HybridHelper()

router = APIRouter(prefix = '/api/v1')

app = FastAPI(
    title = "Patent Document API APP",
    description = description,
    version = "1.0.0",
    openapi_tags = tags_metadata
)

setproctitle.setproctitle('Patent Classifier Server')

@router.post('/file/upload', tags = ["1. File based classification"])
async def file_uploader(file: UploadFile = File(...)):
    try:
        save_path = os.path.join(WORKDIR, file.filename)
        with open(save_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
    except:
        return {"code":0, 'message' : 'File Upload Fail', "filename" : file.filename}
    
    return {"code":1, 'message' : 'File Upload Successful', "filename" : file.filename}

class file_inference(BaseModel):
    file_name: str

@router.post('/file/inference', tags = ["1. File based classification"])
async def file_inference(fi: file_inference):
    file_name = fi.file_name
    logger.info("File Inference")
    logger.info(f"File Name: {file_name}")
    try:
        appn, docs, ntn_cd = input_loader.file_input(os.path.join(WORKDIR, file_name))
        gen_temp = gen_model.classifyList_decoding(docs, args.top_n)
        cls_result = cls_model.classifyList(docs, args.top_n)
        gen_result = hyb_model.gen_result_proc(gen_temp)
        hyb_result = hyb_model.get_hybrid_result(cls_result, gen_result)

        hyb_lst = []
        for a, c, res in zip(appn, ntn_cd, hyb_result):
            hyb_lst.append([a, '""', c, res['1_pred'],res['2_pred'],res['3_pred'], str(res['1_prob']),str(res['2_prob']), str(res['3_prob'])])
        hyb_f = ['|'.join(i) for i in hyb_lst]

        save_dir = WORKDIR
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        
        f_save = f"{save_dir}/{file_name.replace('.csv', '_output.csv')}"

        fw= open(f_save, 'w', encoding='utf-8')
        fw.write('applno|ltrtno|ntn_cd|class1|class2|class3|score1|score2|score3'+'\n')

        for item in hyb_f:
            fw.write(item + '\n')
        fw.close()
        logger.info("Inference Done")
        logger.info(f"Output number: {len(hyb_f)}")
        logger.info(f"Output path: {os.path.join(save_dir, file_name.replace('.csv', '_output.csv'))}")
        
    except:
        logger.error("ERROR: File Inference")
        return {'code' : 0, 'message' : 'Inference Fail', 'file_name' : file_name }
    
    return {'code' : 1, 'message' : 'Inference Successful', 'file_name' : file_name.replace('.csv', '_output.csv'), 'file_path' : save_dir}
    
class content_inference(BaseModel):
    inputs: list
    
@router.post('/content/inference', tags = ["2. Content based classification"])
async def content_inference(fi: content_inference):
    items = fi.inputs
    logger.info("Content Inference")
    try:
        appn, docs, ntn_cd = input_loader.list_input(items)
        gen_temp = gen_model.classifyList_decoding(docs, args.top_n)
        cls_result = cls_model.classifyList(docs, args.top_n)
        gen_result = hyb_model.gen_result_proc(gen_temp)
        hyb_result = hyb_model.get_hybrid_result(cls_result, gen_result)

        res_dict = {}
        for a,c, res in zip(appn, ntn_cd, hyb_result):
            res_dict[a] = {"nation_code": c,
                      "inference_results" : []}
            for id_ in range(1,4):
                res_dict[a]["inference_results"].append({"class_code": res[f"{id_}_pred"], 'score':float(res[f"{id_}_prob"])})
            
        logger.info("Inference Done")
        logger.info(f"Output number: {len(hyb_result)}")
    except:
        logger.error("ERROR: Content Inference")
        return {'code' : 0, 'message' : 'Inference Fail'}

    return {'code' : 1, 'message' : 'Inference Successful', 'results': res_dict}

app.include_router(router)
if __name__ == '__main__':
       
    uvicorn.run(app, host=os.environ.get('SERVER_IP'), port = os.environ.get('SERVER_PORT'))