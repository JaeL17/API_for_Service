from infer_models import ClassificationHelper, GenerationHelper
import argparse
import json

def pre_process(data):
    data = str(data)
    if type(data) == type('a'):
        data= data.replace('\t', '')
        data = " ".join(data.split())
    return data

def get_generation_result(data):
    gen_result = []
    for i in data:
        mid_res = {}
        for topn_idx in range(1,4):
            pred = ''
            score = 1
            for digit_idx in range(1,6):
                pred += i[f"{digit_idx}_{topn_idx}_pred"][-1]
                score = score * float(i[f"{digit_idx}_{topn_idx}_prob"])
            mid_res[f'{topn_idx}_pred']=pred
            mid_res[f'{topn_idx}_prob']=score
        gen_result.append(mid_res)
    return gen_result

def get_hybrid_result(cls_result, gen_result, lab_keys):
    hybrid_result =[]
    for cls, gen in zip(cls_result, gen_result):
        save_dict = {}
        for tn in range(1,4):
            if float(cls[f'{tn}_prob'])>= float(gen[f'{tn}_prob']):
                save_dict[f'{tn}_pred']=cls[f'{tn}_pred']
                save_dict[f'{tn}_prob']=cls[f'{tn}_prob']
            elif gen[f'{tn}_pred'] in lab_keys:
                save_dict[f'{tn}_pred']=gen[f'{tn}_pred']
                save_dict[f'{tn}_prob']=gen[f'{tn}_prob']
            elif gen[f'{tn}_pred'] not in lab_keys:
                save_dict[f'{tn}_pred']=cls[f'{tn}_pred']
                save_dict[f'{tn}_prob']=cls[f'{tn}_prob']
            else:
                print('??')        
        hybrid_result.append(save_dict)
    return hybrid_result

def main(args):
    cls_model = ClassificationHelper(args)   
    gen_model = GenerationHelper(args)
    
    print(f'loading file: {args.input_file}')
    f_all = open(args.input_file,'rt')
    lines_all = f_all.readlines()
    lines_all2 = [i.split('\t') for i in lines_all]
    print('file_length: ', len(lines_all2))
    
    m_cl, title, abst, numf = [],[],[],[]
    for i in lines_all2:
        try:
            numf_val = i[0]
            title_val = pre_process(i[1])
            abst_val = pre_process(i[2])
            m_cl_val = pre_process(i[3])
            cn_code = i[4]

            numf.append([numf_val, '""', cn_code.strip('\n')])
            m_cl.append(m_cl_val.strip())
            title.append(title_val.strip())
            abst.append(abst_val.strip())

        except:
            pass
    numf2 = ['|'.join(i) for i in numf]
    texts = []
    for a,b,c in zip(title, abst, m_cl):
        doc = a+' [SEP] '+b +' [SEP] '+c
        texts.append(doc)
    
    cls_result = cls_model.classifyList(texts, args.top_n)
    print("classification model done")
    gen_result = gen_model.classifyList_decoding(texts, args.top_n)
    gen_result2 = get_generation_result(gen_result)
    print("generation model done")
    
    with open('labels.json', 'r', encoding ='utf8') as fp:
        lab_data =json.load(fp)
    lab_keys = lab_data['labels']
    hybrid_result = get_hybrid_result(cls_result, gen_result2, lab_keys)
    
    whle =[]
    for i in hybrid_result:
        whle.append([i['1_pred'], i['2_pred'], i['3_pred'], str(i['1_prob']),str(i['2_prob']),str(i['3_prob'])])
    whle_f = ['|'.join(i) for i in whle]

    fw = open(args.output_name, 'w', encoding='utf-8')
    fw.write('applno|ltrtno|ntn_cd|class1|class2|class3|score1|score2|score3'+'\n')
    
    for i, j in zip(numf2, whle_f):
        fw.write(i+'|'+j+'\n')
    fw.close()
        
if __name__ == "__main__":    
    parser = argparse.ArgumentParser()
    parser.add_argument("--top_n", type=int, default=3)
    parser.add_argument("--decode_seq_len", type=int, default=5)
    parser.add_argument("--gen_model_type", type=str, default="bert_generation")
    parser.add_argument("--gen_model_name_or_path", type=str, default="/service/docker_img/models/gen_model")
    parser.add_argument("--cls_model_type", type=str, default="saltluxbert")
    parser.add_argument("--cls_model_name_or_path", type=str, default="/service/docker_img/models/cls_model")
    parser.add_argument("--input_file", type=str, required=True)
    parser.add_argument("--output_name", type=str, required=True)
    parser.add_argument("--max_seq_len", type=int, default=512)
    parser.add_argument("--eval_batch_size", type=int, default=16)
    parser.add_argument("--do_lower_case", action='store_true', help="")
    parser.add_argument("--no_cuda", action='store_true', help="")

    args = parser.parse_args()
    main(args)