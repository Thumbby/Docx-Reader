import re

def remove_think_chain(s:str)->str:
    pattern = r'<think>.*?</think>'
    return re.sub(pattern, '', s)   