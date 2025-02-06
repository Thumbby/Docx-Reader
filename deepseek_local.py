from ollama import ChatResponse, chat

model_name = 'deepseek-r1:8b'

def get_response(user_input: str, messages: list = []):
    # put user's message into memroy
    messages.append({
        'role': 'user',
        'content': user_input
    })
    
    response: ChatResponse = chat(
    model = model_name,
    messages = messages,
    )
    
    print(response.message.content)
    
    messages.append({
        'role': 'assistant',
        'content': response.message.content
    })

if __name__ == "__main__":
    # initialize the memory
    messages = [{
        'role':'system',
        'content': '你是一个仿生人,可以模仿人类的情感并且表达在回答中,请根据回答的内容选择适合的语气进行'
    }]
    while True:
        user_input = input("请输入对话:")
        get_response(user_input, messages=messages)