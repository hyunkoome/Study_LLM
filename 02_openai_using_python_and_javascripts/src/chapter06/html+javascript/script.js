const api_key = "YOUR_OPENAI_API_KEY_HERE"; //☆

var prompt;
var question;
var result;
// 추가한다
var original;
var mask;

function doAction(e) {
  // const value = prompt.value;
  access_openai();
}

// 추가한다
function createFormData() {
  let form_data = new FormData();
  form_data.append("prompt", prompt.value);
  form_data.append("image", image.files[0]);
  form_data.append("mask", mask.files[0]);
  form_data.append("n", 1);
  form_data.append("size", "256x256");
  return form_data;
}

// 수정한다
function init() {
  prompt = document.querySelector('#prompt');
  result = document.querySelector('#result');
  image = document.querySelector('#original');
  mask = document.querySelector('#mask');
}

function show_img(arg) {
  const div = document.createElement('div');
  const p = document.createElement('p');
  p.textContent = prompt.value;
  const img = document.createElement('img');
  img.src = arg;
  div.appendChild(p);
  div.appendChild(img);
  result.appendChild(div);
  process.value = "";
}

// 수정한다
async function access_openai() {
  const fd = createFormData();
  console.log("fd: " + fd);
  const response = await fetch("https://api.openai.com/v1/images/edits", {
    method: "POST",
    headers: {
      "Content-Type": "multipart/form-data",
      "Authorization": "Bearer " + api_key
    },
    // headers: {
    //   "Content-Type": "application/json",
    //   "Authorization": "Bearer " + api_key
    // },
    body: fd
  });
  const data = await response.json();
  console.log(data);
  show_img(data["data"][0]["url"]);
}
 