"""
example17 - 自定义聊天机器人
"""
import streamlit as st
from openai import OpenAI

from common import get_llm_response

"""
common - 工具函数模块

函数：封装功能上相对独立而且会被重复使用的代码。
装饰器：用一个函数去装饰另外一个函数或类并为其提供额外的能力。

lru - 缓存置换策略 - least recently used - 最近最少使用
cache - 缓存 - 空间换时间 - 优化性能


"""
import json
from functools import lru_cache


@lru_cache(maxsize=64)
def get_llm_response(client, *, system_prompt='', few_shot_prompt='',
                     user_prompt='', model='deepseek-chat', temperature=0.2,
                     top_p=0.1, frequency_penalty=0, presence_penalty=0,
                     max_tokens=1024, stream=True):
    """
    获取大模型响应
    :param client: OpenAI对象
    :param system_prompt: 系统提示词
    :param few_shot_prompt: 小样本提示（JSON字符串）
    :param user_prompt: 用户提示词
    :param model: 模型名称
    :param temperature: 温度参数
    :param top_p: Top-P参数
    :param frequency_penalty: 频率惩罚参数
    :param presence_penalty: 出现惩罚参数
    :param max_tokens: 最大token数量
    :param stream: 是否开启流模型
    :return: 大模型的响应内容或Stream对象
    """
    messages = []

    if system_prompt:
        messages.append({'role': 'system', 'content': system_prompt})
    if few_shot_prompt:
        messages += json.loads(few_shot_prompt)
    if user_prompt:
        messages.append({'role': 'user', 'content': user_prompt})

    resp = client.chat.completions.create(
        model=model,
        temperature=temperature,
        top_p=top_p,
        frequency_penalty=frequency_penalty,
        presence_penalty=presence_penalty,
        max_tokens=max_tokens,
        messages=messages,
        stream=stream,
    )
    if not stream:
        return resp.choices[0].message.content
    return resp


def get_answer(question: str):
    """
    从大模型获取答案
    :param question: 用户的问题
    :return: 迭代器对象
    """
    try:
        client = OpenAI(base_url=base_url, api_key=api_key)
        stream = get_llm_response(client, model=model_name, user_prompt=question, stream=True)
        for chunk in stream:
            yield chunk.choices[0].delta.content or ''
    except Exception as e:
        # print(e)
        yield from '暂时无法提供回复，请检查你的配置是否正确'


with st.sidebar:
    api_vendor = st.radio(label='请选择服务提供商：', options=['OpenAI', 'DeepSeek'])
    if api_vendor == 'OpenAI':
        base_url = 'https://api.deepseek.com'
        model_options = ['deepseek-reasoner', 'gpt-3.5-turbo', 'gpt-4o', 'gpt-4.1-mini', 'gpt-4.1']
    elif api_vendor == 'DeepSeek':
        base_url = 'https://api.deepseek.com'
        model_options = ['deepseek-chat', 'deep-reasoner']
    model_name = st.selectbox(label='请选择要使用的模型：', options=model_options)
    api_key = st.text_input(label='请输入你的Key：', type='password')

if 'messages' not in st.session_state:
    st.session_state['messages'] = [('ai', '你好，我是你的AI助手，我叫小美。')]

st.write('## 骆昊的聊天机器人')

if not api_key:
    st.error('请提供访问大模型需要的API Key！！！')
    st.stop()

for role, content in st.session_state['messages']:
    st.chat_message(role).write(content)

user_input = st.chat_input(placeholder='请输入')
if user_input:
    _, history = st.session_state['messages'][-1]
    st.chat_message('human').write(user_input)
    st.session_state['messages'].append(('human', user_input))
    with st.spinner('AI正在思考，请耐心等待……'):
        answer = get_answer(f'{history}, {user_input}')
        result = st.chat_message('ai').write_stream(answer)
        st.session_state['messages'].append(('ai', result))
