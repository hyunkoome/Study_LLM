import openai
import base64 # 추가한다\
import json
from openai.embeddings_utils import cosine_similarity


api_key = "YOUR_OPENAI_API_KEY_HERE"

def read_prompt(fname):
  f = open(fname, encoding='utf-8')
  content = f.read()
  f.close()
  return content

def access_openai(prompt_value):
  openai.api_key = api_key

  jsonf = open("sample_data.json", encoding='utf-8')
  json_data = json.loads(jsonf.read(),)
  jsonf.close()

  response = openai.Embedding.create(
    model="text-embedding-ada-002",
    input=prompt_value
  )
  emb_data = response.data[0]["embedding"]

  result = []
  for ob in json_data:
    res = {
      'title': ob['title'],
      'sim': cosine_similarity(ob['embedding'], emb_data)
    }
    result.append(res)

  result = sorted(result, key=lambda ob:ob['sim'], reverse=True)
  for ob in result:
    print(ob['title'] + ' (' + str(ob['sim']) + ')')


if __name__ == "__main__":
  input_text = input("텍스트를 입력: ")
  access_openai(input_text)
