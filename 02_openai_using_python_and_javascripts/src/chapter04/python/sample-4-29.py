import openai

api_key = "YOUR_OPENAI_API_KEY_HERE"

suffix_prompt = "\n\n1. "

def read_prompt(fname):
  f = open(fname, encoding='utf-8')
  content = f.read()
  f.close()
  return content

def access_openai(prompt_value):
  openai.api_key = api_key
  prompt = read_prompt("prompt.txt")

  response = openai.Completion.create(
    model="gpt-3.5-turbo-instruct",
    prompt=prompt + prompt_value,
    n=2,
    max_tokens=500,
    stream=True, #☆
  )

  for i,chunk in enumerate(response):
    if chunk.choices[0].index == 0:
      print("A:" + chunk.choices[0].text.strip())
    if chunk.choices[0].index == 1:
      print("B:" + chunk.choices[0].text.strip())

if __name__ == "__main__":
    input_text = input("텍스트를 입력하십시오: ")
    access_openai(input_text)

