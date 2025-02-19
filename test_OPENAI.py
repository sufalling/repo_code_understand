#coding=utf-8
import time
from openai import OpenAI
from datasets import load_from_disk, Dataset, concatenate_datasets, load_dataset
# 句子处理
import nltk
from nltk.tokenize import word_tokenize
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.translate.meteor_score import meteor_score
from rouge import Rouge
from sentence_transformers import SentenceTransformer, util
import string
import json
import pandas as pd
import os
import copy
from torch.cuda import temperature

print('import complete')


# 下载词典，用于计算meteor分数,无法下载，见https://blog.csdn.net/mjj1024/article/details/105618784
# nltk.data.path.append('D:\\anaconda3\envs\\repoagent_test\\share\\nltk_data')
# nltk.download('wordnet')


# 加载数据
def upload(code_ds: str, split: str = 'test', father_dir:str= './data/',begin_num: int = 1, end_num:int=0):
    """

    :param code_ds: 数据集路径
    :param ds: 数据集名称,默认路径在data下
    :param num: 加载几条
    :param split: 加载数据集下的train或test
    :return: 选择后的数据集
    """
    if 'csv' in code_ds:
        datas = load_dataset("csv", data_files=f"{father_dir}{code_ds}")
        # 读进来默认自动分成一个train
        ds_tests = datas['train']
    elif 'tsv' in code_ds:
        # datas = pd.read_csv(f"./data/{code_ds}", sep='\t')
        # ds_tests = Dataset.from_pandas(datas)
        datas = load_dataset("csv", data_files=f"{father_dir}{code_ds}", sep='\t')
        # 读进来默认自动分成一个train
        ds_tests = datas['train']
    else:
        try:
            datas = load_from_disk(f"{father_dir}{code_ds}")
            ds_tests = datas[split]
        except:
            print("加载Arrow文件失败")
    if end_num != 0:
        ds_tests = ds_tests.select(range(begin_num-1,end_num-1))
    return ds_tests


# 加载模型
def read_models(path:str = "models.json"):
    with open(path) as f:
        models_json = json.load(f)
    return models_json


# 从外部文件读取配置信息
def get_api_and_url(configurepath: str):
    dict_api_url = {}
    with open(configurepath) as f:
        text = f.readlines()
        for item in text:
            item = item.strip("\n").split(":")
            if 'url' in item[0]:
                dict_api_url[item[0]] = "https:" + item[2]
            elif "xunfei" in item[0]:
                dict_api_url[item[0]] = item[1] + ":" + item[2]
            else:
                dict_api_url[item[0]] = item[1]
    return dict_api_url


api_and_url = get_api_and_url("configure.txt")


# 检索语义相似样本
def retrieval_top_5(example):
    corpus = ds_full['stripped_code']
    # Use "convert_to_tensor=True" to keep the tensors on GPU (if available)

    corpus_embeddings = sentence_model.encode(corpus, convert_to_tensor=True)
    # corpus_embeddings = corpus_embeddings.to("cuda")
    # 归一化，方便后续使用点积计算相似度
    corpus_embeddings = util.normalize_embeddings(corpus_embeddings)
    # Query sentences:
    queries = [
        example['stripped_code']
    ]
    queries_embeddings = sentence_model.encode(queries, convert_to_tensor=True)
    # query_embeddings = queries_embeddings.to("cuda")
    query_embeddings = util.normalize_embeddings(queries_embeddings)

    hits = util.semantic_search(query_embeddings, corpus_embeddings, score_function=util.dot_score, top_k=6)
    return hits[0]

# 生成包含语义相似性的提示模板
def generate_retrieval_semantic_prompt(example, sample_num: int = 5, CoT: int = 0):
    top5 = retrieval_top_5(example)
    begin_text = "f\"Good! Here are some examples,They have similar semantics and can be used as a reference:\n\n"
    full_text = begin_text
    if CoT == 0:
        end_text = "**YOUR MISSION**:\nNow output the functionality of the following code snippet:\n{code}\nJust output final answer and less than 50 words. Please answer in English. \""
    else:
        end_text = "**YOUR MISSION**:\nNow output the functionality of the following code snippet:\n{code}\n. Let's think it step by step. Please answer in English, and just output final answer and less than 50 words. \""

    for i in range(1, sample_num + 1):
        corpus_id = top5[i]['corpus_id']
        text = f"**CODE{i}**: {ds_full['stripped_code'][corpus_id]}\n **CORRESPONDING COMMENT{i}**: \n{ds_full['comment'][corpus_id]}\n\n"
        full_text = full_text + text
    full_text = full_text + end_text
    prompt = [
        {
            "role": "system",
            "content": "You are Frederic, a veteran code expert with a deep understanding of what code does in a project. Your task is to introduce new members of the project to the role and functionality of the project code by using short introduction."
        },
        {
            "role": "user",
            "content": "I will give you some snippets of code, please tell me their function by using short introduction. "
        },
        {
            "role": "assistant",
            "content": "Well. I'm Frederick, and I'm going to explain what code does by using short introduction.."
        },
        {
            "role": "user",
            "content": full_text
        }]
    return prompt

# 将生成的提示加入prompt
def add_retrieval__prompt_to_json(example):
    new_prompt = []
    p = generate_retrieval_semantic_prompt(example, sample_num=1, CoT=0)
    p2 = generate_retrieval_semantic_prompt(example, sample_num=5, CoT=0)
    p3 = generate_retrieval_semantic_prompt(example, sample_num=5, CoT=1)
    new_prompt = new_prompt + [p, p2, p3]
    # new_prompt = new_prompt + [p]
    with open("prompts.json", "r+", encoding='utf-8') as f:
        old_prompt_dict = json.load(f)
        new_prompt_index = [5,6,7]
        # new_prompt_index = [5]
        for index in new_prompt_index:
            old_prompt_dict[f"prompt_{index}"] = new_prompt[index-5]
        # json.dump(old_prompt_dict, f)
    return old_prompt_dict

def add_retrieval_prompt_to_json(example)
# 导入提示模板
# def get_prompt(path:str = "prompts.json"):  # 输出json格式
#     with open(path) as f:
#         prompts_json = json.load(f)
#     return prompts_json


# 获取模型输出
def get_response(tags, code: str = None, model="Qwen/QwQ-32B-Preview", path=None,
                 prompt=None,
                 temperature: float = 1, top_p: float = 1):
    """
    :param prompt: 导入提示模板
    :param tags: 确定调用的API
    :param model: 选择模型
    :param prompt_i: 选择使用哪个模板
    :param code: str
    :param temperature: 0~2
    :param top_p: -1~1
    :return: output content
    """
    # prompt = [
    #     # 角色设定
    #     {
    #         "role": "system",
    #         "content": "You are Frederic, a veteran code expert with a deep understanding \
    #         of what code does in a project. Your task is to introduce new members of the project \
    #         to the role and functionality of the project code."
    #      },
    #     # 大模型回应
    #     {
    #         "role": "assistant",
    #         "content": "you are a code copilot,"
    #     },
    #     # 用户
    #     {
    #         "role": "user",
    #         "content": f"The function of the {code},just one sentence,less than 50 words. Please answer in English. "
    #     }
    #     ]
    try:
        ## settings
        if tags == "aihubmix_model":
            client = OpenAI(
                api_key=api_and_url['aihubmax_API_KEY'],
                base_url=api_and_url['aihubmax_base_url']
            )
        elif tags == "mindcraft_model":
            client = OpenAI(
                api_key=api_and_url['mindcraft_API_KEY'],
                base_url=api_and_url['mindcraft_base_url']
            )
        elif tags == "xunfei":
            client = OpenAI(
                api_key=api_and_url[f'{tags}_{model}_API_KEY'],
                base_url=api_and_url[f'{tags}_base_url']
            )
        else:
            client = OpenAI(
                api_key=api_and_url[f'{tags}_API_KEY'],
                base_url=api_and_url[f'{tags}_base_url']
            )
        time.sleep(0.5)
        default_prompt = [
      {
          "role": "user",
          "content": f"The function of the **{code}**, just ues one sentence and less than 50 words. Please answer in English. "
      }]
        chat_completion = client.chat.completions.create(
            model=model,
            messages=prompt if not prompt else default_prompt,
            # 介于 0 和 2 之间。较高的值（如 0.8）将使输出更加随机，而较低的值（如 0.2）将使其更加集中和确定,默认为1
            temperature=temperature,
            # 默认为1，范围-1~1
            top_p=top_p
        )
        return chat_completion.choices[0].message.content
    except:
        return 'None_output'

# 用来合并两个字典
def add_dict1_to_dict2(dict2,dict1):
    for key in dict1.keys():
        dict2[key] = dict1[key]
    return dict2
# 把提示加入模型得到输出
def add_prompt_to_model(code,tags,model,prompt,temperature_i=1.0):
    model_out = get_response(tags, code, model=model, prompt=prompt)
    return {'model': model, 'model_output': model_out}
# 对ds的每个数据使用不同的提示模板，返回包含输出结果的ds
def recycle_prompt(ds,tags,model,temperature_i=1.0):
    temp_list = []
    for row_id in range(len(ds)):
        temp_ds = ds[row_id]
        prompt_dict = add_retrieval_prompt_to_json(temp_ds)
        for i in range(len(prompt_dict)):
            print(f'prompt{i}')
            temp_model_output = add_prompt_to_model(temp_ds['stripped_code'],tags,model,prompt_dict[f'prompt_{i}'],temperature_i)
            temp_ds1 = add_dict1_to_dict2(temp_ds,temp_model_output)
            temp_ds2 = add_dict1_to_dict2(temp_ds1,{'prompt':i})
            temp_ds3 = copy.copy(temp_ds2)
            temp_list.append(temp_ds3)
            temp_ds = ds[row_id]
    temp_df = (pd.DataFrame(temp_list))
    temp_df.to_csv('./csv_output/has_model_out_not_compare.tsv',sep='\t')
    return Dataset.from_dict(temp_df.to_dict(orient='list'))

# # 把输出文本结果放到新的列
# def add_response(example, tags, model, prompt,temperature_i):# datasets放最前面，否则后面map函数报错
#     # 输入dataset，配合map函数
#     # prompt = get_prompt(example)
#     print('add_response')
#     return {'model': model, 'model_output': get_response(tags, example['stripped_code'], model=model, prompt=prompt)}



# 去掉标点符号
def remove_punctuation(sentence: str):
    for i in string.punctuation:
        sentence = sentence.replace(i, "")
    return sentence


def add_bleu_4(references, candidate_list):
    # BLEU-4,0-1
    # print('bleu')
    try:
        chencherry = SmoothingFunction()
        bleu_score = sentence_bleu(references, candidate_list, weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=chencherry.method1)
        return float(bleu_score)
    except:
        return 'None_output'
    # chencherry = SmoothingFunction()
    # bleu_score = sentence_bleu(references, candidate_list, weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=chencherry.method1)
    # return float(bleu_score)


def add_meteor(references, candidate_list):
    try:
        # bleu_score = sentence_bleu(reference, candidate, weights=(1, 0, 0, 0))
        # Meteor,https://github.com/zembrodt/pymeteor
        # 用nltk实现
        # print('meteor')
        meteor_score_out = meteor_score(references, candidate_list)  # 输入分词后的列表
        return float(meteor_score_out)
    except:
        return 'None_output'


def add_rouge(references, candidate, rouge_function):
    try:
        # 输入句子
        rouge_scores = rouge_function.get_scores(candidate, references, avg=True)
        return rouge_scores
    except:
        return {"rouge-1": 'None_output', "rouge-2": 'None_output', "rouge-l": 'None_output'}


def add_metrics(example, rouge_function):
    # time.sleep(4)
    print("add-metrics")
    hypothesis = remove_punctuation(example['model_output'])
    # hypothesis = remove_punctuation('ghs sjsh sjjs')
    references = remove_punctuation(example['comment'])

    reference = [word_tokenize(references)]  ## 分词,reference已经是一个套了两层的列表
    candidate = word_tokenize(hypothesis)

    bleu_score = add_bleu_4(reference, candidate)
    meteor_score_out = add_meteor(reference, candidate)
    rouge_scores = add_rouge(hypothesis, references, rouge_function)

    return {'meteor_score': meteor_score_out, 'bleu_4': bleu_score, 'rouge_1': rouge_scores["rouge-1"], 'rouge_2': rouge_scores["rouge-2"], 'rouge_L': rouge_scores["rouge-l"]}


def set_llm_as_judge(example, model="gpt-4o-2024-11-20", path=None, temperature: float = 1, top_p: float = 1):
    client = OpenAI(
        api_key=api_and_url['aihubmax_API_KEY'],
        base_url=api_and_url['aihubmax_base_url']
    )
    time.sleep(0.5)
    print('set-llm-as-judge')
    try:
        chat_completion = client.chat.completions.create(
            model=model,
            messages=[
                # 角色设定
                {
                    "role": "system",
                    "content": "You are Frederic, a veteran code expert with a deep understanding of what code does in a project. Your task is to Compare the similarity of two code comments."
                },
                # 用户
                {
                    "role": "user",
                    "content": f"Compare the similarity score between {example['model_output']} and {example['comment']}, \
                                 give a score of 0-100, where 0 means the contents are not similar at all and \
                                100 means the contents are completely similar. \
                                **note**: Just output the score. Don't explain."
                }
            ],
            # 介于 0 和 2 之间。较高的值（如 0.8）将使输出更加随机，而较低的值（如 0.2）将使其更加集中和确定,默认为1
            temperature=temperature,
            # 默认为1，范围-1~1
            top_p=top_p
        )
        return {'LLM_as_judge': int(chat_completion.choices[0].message.content)/100}
    except:
        return {'LLM_as_judge': 0}


def generate_embedding(example, sent_model):

    # Download model
    # model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
    print('embedding')
    comment = remove_punctuation(example['comment'])
    model_output = remove_punctuation(example['model_output'])
    try:
        # Get embeddings of sentences
        embeddings_comment = sent_model.encode(comment)
        embeddings_model_output = sent_model.encode(model_output)

        # calculate similarity
        cosine_scores = util.cos_sim(embeddings_comment, embeddings_model_output)
        return {"sentenceBERT_similarity": float(cosine_scores)}
    except:
        return {"sentenceBERT_similarity": 'None_output'}


# 按模型保存数据集到单个文件
def save(past_ds,corporation,model):
    if corporation != model:
        past_ds.to_csv(path_or_buf=f'./csv_output/{corporation}/{model}_output.tsv', sep='\t')
    else:
        past_ds.to_csv(path_or_buf=f'./csv_output/save_all_output/all_out/{corporation}_output.tsv', sep='\t')


#保存到一个文件all_out中
def save_all_output(father_dir:str='./csv_output/save_all_output'):
    files = os.listdir(father_dir)
    file_list = []
    for file in files:
        if file == 'all_out':
            continue
        files_ds = upload(file,father_dir=father_dir + '/')
        files_df = pd.DataFrame(files_ds)
        file_list.append(files_df)
    all_output_ds = pd.concat(file_list)
    save(all_output_ds,"all_output","all_output")


def process(model_list, tags, dset, sentence_model, rouge_model, test_temperature_list=None, runtime=1):
    if test_temperature_list is None:
        test_temperature_list = [1.0]
    for model in model_list:
        corporation = list(key for key in model)[0]
        tags = tags
        # rouge = Rouge()
        # print('begin-sentence-model')
        # sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
        # print('finish-sentence-model')
        corporation_output_list = []
        for called_model in model[corporation]:
            print(called_model)
            model_prompts_list = []
            for runtime_i in range(runtime):
                for test_temperature_i in test_temperature_list:
                    new_ds_test = recycle_prompt(dset,tags,called_model,test_temperature_i)
                    # new_ds_test = dset.map(add_response, fn_kwargs={"tags": tags, "model": called_model, "prompt": prompts[f"prompt_{prompt_i}"],"temperature_i":test_temperature_i})
                    new_ds_test = new_ds_test.map(lambda example: {"runtime": runtime_i+1})
                    # new_ds_test = new_ds_test.map(lambda example: {"temperature": test_temperature_i})
                    # print(add_metrics)
                    new_ds_test = new_ds_test.map(add_metrics, fn_kwargs={'rouge_function': rouge_model})  ## map后的数据集要保存
                    # print(set_llm_as_judge)
                    new_ds_test = new_ds_test.map(set_llm_as_judge)  ## map后的数据集要保存
                    # print(generate_embedding)
                    new_ds_test = new_ds_test.map(generate_embedding, fn_kwargs={'sent_model': sentence_model})  ## map后的数据集要保存
                    corporation_output_list.append(new_ds_test)
                    model_prompts_list.append(new_ds_test)
            model_prompts_ds = concatenate_datasets(model_prompts_list)##合并的值类型要一样
            save(model_prompts_ds, corporation, called_model)
        corporation_model_prompt_ds = concatenate_datasets(corporation_output_list)
        save(corporation_model_prompt_ds, corporation="save_all_output", model=corporation)



def final_process(models, dset, sentence_model, rouge_model,test_temperature_list:list,runtime=1):
    first_level_model = list(key for key in models)
    for first_level_model_item in first_level_model:
        process(models[first_level_model_item], tags=first_level_model_item, dset=dset, sentence_model=sentence_model, rouge_model=rouge_model, test_temperature_list=test_temperature_list,runtime=runtime)
    save_all_output(father_dir='./csv_output/save_all_output')


if __name__ == '__main__':
    # ds_test = upload("code_cat_full",split="test",num = 2)
    ds = upload("function_no_duplicated.tsv",begin_num=10,end_num=11)
    ds_full = upload("function_no_duplicated.tsv")
    print('data-load-complete')

    models = read_models("models.json")
    print('model-complete')

    # {prompt_0:[content]}
    # prompts = get_prompt("./prompts.json")
    print('prompts-complete')
    rouge = Rouge()
    print('begin-sentence-model')
    sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
    # sentence_model = None
    print('finish-sentence-model')

    # process
    temp_list = [1.0]## 不要为空
    final_process(models=models, dset=ds, sentence_model=sentence_model, rouge_model=rouge, test_temperature_list=temp_list)




