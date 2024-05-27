import streamlit as st

from intel_extension_for_transformers.neural_chat import PipelineConfig
from intel_extension_for_transformers.neural_chat import build_chatbot
from intel_extension_for_transformers.neural_chat import plugins
from intel_extension_for_transformers.transformers import RtnConfig

plugins.retrieval.enable=True
plugins.retrieval.args['embedding_model'] = "ZhipuAI/chatglm3-6b"
plugins.retrieval.args["input_path"]="./mental_health.txt"

config = PipelineConfig(model_name_or_path='AI-ModelScope/bge-base-zh-v1.5',
plugins=plugins,
optimization_config=RtnConfig(compute_dtype="int8",
weight_dtype="int4_fullrange"))
print("Config finish!")

@st.experimental_singleton
chatbot = build_chatbot(config)
print("Chatbot init!")

def response(question):
    answer = chatbot.predict(query=question)
    print('infer finished!')
    return answer

col1, col2 = st.columns(2)
with col1:
    input = st.text_input('text input', 'Ask Me Anything', key='word_seg_input')

with col2:
    input2 = st.text_input('text output', placeholder=response(question), disabled=True)