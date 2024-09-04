const api_key = "YOUR_OPENAI_API_KEY_HERE"; //☆

function doAction() {
  const value = document.querySelector('#prompt').value;
  access_openai(value);
}

function setQnA(question, result) {
  document.querySelector('#question').textContent = question;
  document.querySelector('#result').innerHTML = result;
}

function access_openai(prompt) {
  fetch("https://api.openai.com/v1/edits", {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      "Authorization": "Bearer " + api_key
    },
    body: JSON.stringify({
      model: "text-davinci-edit-001",
      input: prompt,
      instruction: "문장을 존댓말 표현으로 만든다.\n\n",
    })
  })
  .then(response => response.json())
  .then(json_data => {
    const result = json_data.choices[0].text.trim();
    setQnA(prompt, result);
  });
}

