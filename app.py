import streamlit as st

from intel_extension_for_transformers.neural_chat import PipelineConfig
from intel_extension_for_transformers.neural_chat import build_chatbot
from intel_extension_for_transformers.neural_chat import plugins
from intel_extension_for_transformers.transformers import RtnConfig

#模型下载
from modelscope import snapshot_download
embedding_model_dir = snapshot_download('AI-ModelScope/bge-base-zh-v1.5')
llm_dir = snapshot_download('ZhipuAI/chatglm3-6b')

@st.cache_resource
def init():
    plugins.retrieval.enable=True
    plugins.retrieval.args['embedding_model'] = embedding_model_dir
    # plugins.retrieval.args['embedding_model'] = "AI-ModelScope/bge-base-zh-v1.5"
    plugins.retrieval.args['process'] = False
    plugins.retrieval.args["input_path"]="./mental_health.txt"

    config = PipelineConfig(model_name_or_path=llm_dir,
    plugins=plugins,
    optimization_config=RtnConfig(compute_dtype="int8",
    weight_dtype="int4_fullrange"))
    # config = PipelineConfig(model_name_or_path=llm_dir, plugins=plugins)
    # config = PipelineConfig(model_name_or_path='ZhipuAI/chatglm3-6b', plugins=plugins)
    print("Config finish!")
    
    chatbot = build_chatbot(config)
    print("Chatbot init!")
    return chatbot

cb = init()

def response(question):
    answer = cb.predict(query=question)
    print('infer finished!')
    return answer

# if __name__ == '__main__':
col1, col2 = st.columns(2)
with col1:
    input = st.text_input('text input', 'Ask Me Anything', key='word_seg_input')

with col2:
    input2 = st.text_input('text output', placeholder=response(input), disabled=True)