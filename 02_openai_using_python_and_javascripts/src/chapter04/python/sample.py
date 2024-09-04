import openai

api_key = "YOUR_OPENAI_API_KEY_HERE"

suffix_prompt = "\n출력: "

def read_prompt(fname):
  f = open(fname, encoding='utf-8')
  content = f.read()
  f.close()
  return content

def access_openai(prompt_value):
  openai.api_key = api_key
  prompt = read_prompt("prompt.txt")
  try:
    response = openai.Completion.create(
      model="gpt-3.5-turbo-instruct",
      prompt=prompt + suffix_prompt,
      max_tokens=200)
    result = response.choices[0].text.strip()
    print("\n결과: " + result)
  except openai.error.APIError as e:
    print(f"API 에러가 발생했습니다: {e}")
    pass
  except openai.error.AuthenticationError as e:
    print(f"API 인증에 실패했습니다: {e}")
    pass
  except openai.error.APIConnectionError as e:
    print(f"API로의 접속에 실패했습니다: {e}")
    pass
  except openai.error.InvalidRequestError as e:
    print(f"유효하지 않은 요청을 전송했습니다: {e}")
    pass
  except openai.error.RateLimitError as e:
    print(f"API 이용 상한에 도달했습니다: {e}")
    pass
  except:
    print("에러가 발생했습니다.")
    pass

if __name__ == "__main__":
    input_text = input("텍스트를 입력하십시오: ")
    access_openai(input_text)

