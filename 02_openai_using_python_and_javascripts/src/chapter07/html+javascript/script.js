const api_key = "YOUR_OPENAI_API_KEY_HERE"; //☆

var prompt;
var result;

function init() {
  prompt = document.querySelector('#prompt');
  result = document.querySelector('#result');
}

function doAction(e) {
  access_openai(prompt.value);
}

function access_openai(prompt_value) {
  fetch("https://api.openai.com/v1/moderations", {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      "Authorization": "Bearer " + api_key
    },
    body: JSON.stringify({
      input: prompt_value,
    })
  })
  .then(response=>response.json())
  .then(data=>{
    const res = JSON.stringify(data,null,2);
    result.textContent = res;
  });
}
