const api_key = "YOUR_OPENAI_API_KEY_HERE"; //☆

var container;
var file;

function init() {
  container = document.querySelector('#result');
  file = document.querySelector('#file');
}

function doChange(e) {
  const file = e.target.files[0];
  const data = new FormData();
  data.append("file", file);
  data.append("model", "whisper-1");
  access_openai(data);
}

function access_openai(content) {
  fetch("https://api.openai.com/v1/audio/translations", { //☆
    method: "POST",
    headers: {
      "Authorization": "Bearer " + api_key,
    },
    body: content,
  })
  .then(response => response.json())
  .then(json_data => {
    container.textContent = json_data.text;
  });
}

