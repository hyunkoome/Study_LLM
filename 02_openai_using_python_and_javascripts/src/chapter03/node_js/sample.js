const readline = require('readline');
const { Configuration, OpenAIApi } = require('openai');

const rl = readline.createInterface({
  input: process.stdin,
  output: process.stdout
});

const api_key = "YOUR_OPENAI_API_KEY_HERE"; //☆
const config = new Configuration({
  apiKey: api_key,
});


// ☆ 메인 처리
(function(){
  console.log("※ 사용할 수 있는 명령: ");
  console.log("코드(내용)");
  console.log("영어 번역(내용)");
  input_prompt("\n명령을 입력: ");
})();


// input_prompt("텍스트를 입력하십시오: ");
input_prompt();

function input_prompt(msg) {
  rl.question(msg, (inputText) => {
    rl.close();
    access_openai(inputText);
  });
}

const fs = require('fs')

function read_prompt(fname) {
  return fs.readFileSync(fname, 'utf-8');
}

const suffix_prompt = "\n출력: ";

function access_openai(prompt_value) {
  const openai = new OpenAIApi(config);
  const prompt = read_prompt("prompt.txt");
  openai.createCompletion({
    model: "gpt-3.5-turbo-instruct",
    prompt: prompt + prompt_value,
    max_tokens: 2000,
  }).then(response=>{
    const result = response.data.choices[0].text.trim();
    console.log("\n" + result);
  });
}

// function access_openai(prompt_value) {
//   const openai = new OpenAIApi(config);
//   let prompt = read_prompt("prompt.txt");
  
//   // 패턴 체크
//   input_value = prompt_value.match(/코드\((.*\))|영어 번역\((.*)\)/);
//   if (input_value == null) {
//     console.log("명령이 아닙니다.");
//     return;
//   }

//   openai.createCompletion({
//     model: "gpt-3.5-turbo-instruct",
//     prompt: prompt + prompt_value + suffix_prompt,
//     max_tokens: 200,
//   }).then(response=>{
//     const result = response.data.choices[0].text.trim();
//     console.log("\n※ 다음 명령을 실행했습니다: \n" + input_value + "\n\n결과: \n" + result);
//   });
// }