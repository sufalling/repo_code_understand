from test_OPENAI import api_and_url
from openai import OpenAI
import time
from datasets import load_from_disk,load_dataset
def get_comment_response(example, tags,model="Qwen/QwQ-32B-Preview", path=None, temperature: float = 1,
                 top_p: float = 1):
    """
    :param example: dataset
    :param temperature: 0~2
    :param top_p: -1~1
    :return: output content
    """
    # prompt = get_prompt()
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
    else:
        client = OpenAI(
          api_key=api_and_url[f'{tags}_API_KEY'],
          base_url=api_and_url[f'{tags}_base_url']
        )
    try:
        chat_completion = client.chat.completions.create(
        model=model,
        messages=[
          ## 角色设定
          # {"role": "system", },
          ## 用户
          {
            "role": "user",
            "content": f"""Please determine whether the following sentence contains a natural language representation, if yes, print 1, otherwise print 0.Don't explain.\
                       sentence:{example['comment']}\
                       output:0 OR 1"""
          }
        ],
        # 介于 0 和 2 之间。较高的值（如 0.8）将使输出更加随机，而较低的值（如 0.2）将使其更加集中和确定,默认为1
        temperature=temperature,
        # 默认为1，范围-1~1
        top_p=top_p
        )
        time.sleep(0.5)
        return {"is_NL":chat_completion.choices[0].message.content}
    except:
        return {"is_NL": '--'}
    time.sleep(0.1)


csv_comment = load_dataset("csv",data_files="comment.csv")
csv_comment_new = csv_comment['train']
csv_comment_new_test = csv_comment_new.map(get_comment_response,fn_kwargs={"tags":"deepseek","model":"deepseek-chat"})
print(csv_comment_new_test['is_NL'])
csv_comment_new_test.to_csv('is_NL.csv')