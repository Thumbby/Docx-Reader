from ollama import ChatResponse, chat

model_name = 'qwen2.5:1.5b'

def get_bill(year:int, month:int) -> str:
  return f'{year}年{month}月的账单支出为114514元'

def get_profit(year:int, month:int) -> str:
  return f'{year}年{month}月的账单收入为1919810元'

def get_login_member() -> str:
  return 'admin'

get_bill_tool = {
  'type': 'function',
  'function': {
    'name': 'get_bill',
    'description': '获取时间为{year}年{month}月的账单支出',
    'parameters': {
      'type': 'object',
      'required': ['year', 'month'],
      'properties': {
        'year': {'type': 'integer', 'description': '年份'},
        'month': {'type': 'integer', 'description': '月份'},
      },
    },
  }
}

get_profit_tool = {
  'type': 'function',
  'function': {
    'name': 'get_profit',
    'description': '获取时间为{year}年{month}月的收入',
    'parameters': {
      'type': 'object',
      'required': ['year', 'month'],
      'properties': {
        'year': {'type': 'integer', 'description': '年份'},
        'month': {'type': 'integer', 'description': '月份'},
      },
    },
  }
}

get_login_member_tool = {
  'type': 'function',
  'function': {
    'name': 'get_login_member',
    'description': '获取当前登录的用户',
    'parameters': {
      'type': 'object',
      'required': [],
      'properties': {},
    },
  }
}

def get_api_result(message: str):
    messages = [{
      'role': 'system',
      'content': '你是一个接口调用机器人,请根据用户的输入选择你在tools中的函数进行调用,并获取函数调用的结果.若你的messages中存在已经调用的tools返回输出,你需要将返回的输出告知用户,且函数的返回结果优先级高于你的已知知识'
    }]
    messages.append({
        'role': 'user',
        'content': message
    })
    print('Prompt:', messages[0]['content'])
 
    available_functions = {
    'get_bill': get_bill,
    'get_profit': get_profit,
    'get_login_member': get_login_member
    }

    response: ChatResponse = chat(
    model = model_name,
    messages = messages,
    tools = [get_bill_tool, get_profit_tool, get_login_member_tool],
    )

    if response.message.tool_calls:
        for tool in response.message.tool_calls:

            try:
                if function_to_call := available_functions.get(tool.function.name):
                    print('Calling function:', tool.function.name)
                    print('Arguments:', tool.function.arguments)
                    output = function_to_call(**tool.function.arguments)
                    print('Function output:', output)
                    messages.append(response.message)
                    messages.append({'role': 'tool', 'content': str(output), 'name': tool.function.name})
                else:
                    print('Function', tool.function.name, 'not found')
            except Exception as e:
                print('Function error:', e)                 

    print(f"最终输入为{messages}")    
    final_response = chat(model = model_name, messages=messages)
    print('最终输出:', final_response.message.content)    

if __name__ == "__main__":
    while True:
        user_input = input("请输入命令:")
        get_api_result(user_input)