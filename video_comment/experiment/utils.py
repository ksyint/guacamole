import re
import emoji

def remove_useless(text):
    text = text.replace("ㅋ", "")
    text = text.replace("<br>", "")
    text = re.sub(r'<a.*?>|</a>', '', text)
    text = emoji.replace_emoji(text, "")
    text = re.sub(r'\.{2,}', '...', text)
    text = text.replace("&#39;", "")
    text = text.replace("&quot;", "")
    text = text.replace("&gt;", "")
    text = re.sub(r'\,{2,}', '...', text)
    
    return text