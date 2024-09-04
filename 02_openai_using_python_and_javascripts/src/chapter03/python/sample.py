#-*- coding: utf-8 -*-

import openai

api_key = "YOUR_OPENAI_API_KEY_HERE"  # ☆

suffix_prompt = "\n출력: "

def read_prompt(fname):
  f = open(fname, encoding="utf-8")
  content = f.read()
  f.close()
  return content

def access_openai(prompt_value):
  openai.api_key = api_key
  prompt = read_prompt("prompt.txt")
  response = openai.Completion.create(
    model="gpt-3.5-turbo-instruct",
    prompt=prompt + prompt_value + suffix_prompt,
    max_tokens=200)
  result = response.choices[0].text.strip()
  print("\n" + result)

if __name__ == "__main__":
  print("※ 실행할 수 있는 명령: ")
  print("문의(대상자, 안건 이름)")
  print("회의(대상자, 안건 이름)")
  input_text = input("\n명령을 선택: ")
  access_openai(input_text)

