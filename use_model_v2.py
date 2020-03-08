import torch
import torch.nn as nn
from transformers import *
import numpy as np
import argparse


parser = argparse.ArgumentParser()
parser.add_argument('--model_name', default='bert-base-multilingual-cased', type=str)
parser.add_argument('--model_path', default=None, type=str, required=True)
args = parser.parse_args()

context = input('please input context\n> ')

tokenizer = BertTokenizer.from_pretrained(args.model_name)
context_ids = tokenizer.encode(context, add_special_tokens=True)
input_ids = torch.tensor([context_ids])
context_ids = torch.tensor([context_ids])


print('loading ...')
model = BertForMaskedLM.from_pretrained(args.model_path)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
model.eval()
print('loaded success')

mask = torch.tensor([[103]])
sep = torch.tensor([[102]])

reply = []
with torch.no_grad():
    for i in range(10):
        if i >= 2 :
            input_ids = torch.cat((context_ids, reply[-2]), dim=1)
            input_ids = torch.cat((input_ids, reply[-1]), dim=1)
        input_ids = torch.cat((input_ids, mask), dim=1)
        tmp = []

        for j in range(50):
            predict = model(input_ids=input_ids.cuda())
            x = torch.argmax(predict[0][0][-1]).item()
            input_ids[0][-1] = x
            tmp.append(x)
            input_ids = torch.cat((input_ids, mask), dim=1)
            if x == 102 : # '[SEP]' : 102
                break
            elif j == 20 :
                input_ids = torch.cat((input_ids, sep), dim=1)
        reply.append(torch.tensor([tmp]))
for i in range(len(reply)):
    output = ''.join(tokenizer.convert_ids_to_tokens(reply[i][0]))
    print(output)
    print()
# python use_model_v2.py --model_path ./NEW/ccc/

# 高雄市民好久不見的韓國瑜出現啦 台下歡呼聲很大 噓聲也很大 就這樣 開心的陪我們度過最後的三分鐘 真羨慕幸福的高雄人啊
# 求有沒有中天新聞說 高雄人不投韓國瑜，是因為不捨得 他離開高雄的新聞影片? 小弟找了好久都沒看到.... 
# 韓國瑜在幾個月前到桃園必到中壢辦造勢 中壢的立委候選人魯明哲也都合體出擊 但好像在11月初那一次以後，感覺就沒看到韓國瑜來中壢了 甚至魯明哲12月的競選總部開幕也只有張善政來 取而代之的是12月以後馬英九一直狂來中壢陪魯明哲拜票  2019.12.15馬英九來訪中壢新明夜市 2019.12.20 馬總統青埔見面會 2020.01.03馬英九陪逛忠貞夜市 2020.01.05預告馬英九陪同車隊遊行(同時間還是政見發表會)  奇怪，記得有人曾經說過中壢的韓粉最多 難道韓國瑜在中壢已經沒市場了，所以魯蛋才選擇找英九不找國瑜呢？ 是韓國瑜已經在中壢開始沒票房，不如馬英九形象比較穩了？ 還是其實馬英九才是魯蛋比較信任的吸票機呢？  從今天韓國瑜在桃園的造勢，選擇到蘆竹繞過中壢 