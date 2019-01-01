import json

import requests

from hanziconv import HanziConv
# 翻譯函數，word 需要翻譯的內容
def translate(word):
    # 有道詞典 api
    url = 'http://fanyi.youdao.com/translate?smartresult=dict&smartresult=rule&smartresult=ugc&sessionFrom=null'
    # 傳輸的參數，其中 i 為需要翻譯的內容
    key = {
        'type': "AUTO",
        'i': word,
        "doctype": "json",
        "version": "2.1",
        "keyfrom": "fanyi.web",
        "ue": "UTF-8",
        "action": "FY_BY_CLICKBUTTON",
        "typoResult": "true"
    }
    # key 這個字典為發送給有道詞典服務器的內容
    response = requests.post(url, data=key)
    # 判斷服務器是否相應成功
    if response.status_code == 200:
        # 然後相應的結果
        return response.text
    else:
        print("有道詞典調用失敗")
        # 相應失敗就返回空
        return None

def main():
    print("本進程調用有道詞典的API進行翻譯，可達到以下效果：")
    print("外文-->中文")
    print("中文-->英文")
    word = input('請輸入你想要翻譯的詞或句：')
    list_trans = translate(word)
    result = json.loads(list_trans)
    print ("輸入的詞為：%s" % result['translateResult'][0][0]['src'])
    print ("翻譯結果為：%s" % result['translateResult'][0][0]['tgt'])
    print (HanziConv.toTraditional("翻譯結果為：%s" % result['translateResult'][0][0]['tgt']))

if __name__ == '__main__':
    main()