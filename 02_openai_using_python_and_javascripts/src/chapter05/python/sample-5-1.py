import openai

api_key = "YOUR_OPENAI_API_KEY_HERE"

# messages=[
  # {"role": "system", "content": "당신은 베이킹 어시스턴트입니다."}
# ]

messages=[
  {"role": "system", "content": "당신은 베이킹 어시스턴트입니다. 베이킹에 관한 질문에 대답하고, 베이킹 만드는 방법을 생각합니다. 베이킹 이외의 화제에 관해서는 '모르겠습니다'라고 대답합니다."}
]

def access_openai():
  openai.api_key = api_key
  response = openai.ChatCompletion.create(
    model="gpt-3.5-turbo",
    messages=messages,
  )
  message = response.choices[0].message
  messages.append(message)
  print("AI: " + message.content.strip())

if __name__ == "__main__":
  print("AI: 질문을 환영합니다.")
  while True:
    input_text = input("입력: ")
    if input_text == "":
      print("close")
      break
    message = {
      "role": "user",
      "content": input_text
    }
    messages.append(message)
    access_openai()


