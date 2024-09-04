import openai

api_key = "YOUR_OPENAI_API_KEY_HERE"  # ☆

def access_openai(prompt_value):
    openai.api_key = api_key
    response = openai.Completion.create(model="gpt-3.5-turbo-instruct",
        prompt=prompt_value, max_tokens=100)
    result = response.choices[0].text.strip()
    print(result)

if __name__ == "__main__":
    input_text = input("텍스트를 입력하십시오: ")
    access_openai(input_text)
