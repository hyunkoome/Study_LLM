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
  fetch("https://api.openai.com/v1/models", {
    method: "GET",
    headers: {
      "Content-Type": "application/json",
      "Authorization": "Bearer " + api_key
    }
  })
  .then(response => response.json())
  .then(json_data => {
    let res = "<ul>";
    for(let ob of json_data.data) {
      res += "<li>" + ob.id + "</li>";
    }
    res += "</ul>";
    setQnA("AI Model List", res);
  });
}

