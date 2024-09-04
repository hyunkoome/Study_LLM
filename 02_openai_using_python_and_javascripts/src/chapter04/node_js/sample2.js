const { Configuration, OpenAIApi } = require('openai');

const api_key = "YOUR_OPENAI_API_KEY_HERE"; //☆

const config = new Configuration({
  apiKey: api_key,
});

(function(){
  const openai = new OpenAIApi(config);
  openai.listModels().then(value=>{
    for(let ob of value.data.data){
      console.log(ob.id);
    }
    rl.close();
  });
})();
