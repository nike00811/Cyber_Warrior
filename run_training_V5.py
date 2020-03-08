import torch
import torch.nn as nn
from transformers import *
from torch.utils import data
import numpy as np
import random
import argparse
import logging
import json

logging.basicConfig(level=logging.DEBUG,
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    format='%(asctime)s %(message)s',
                    filename='losslog.txt')
logger = logging.getLogger()

parser = argparse.ArgumentParser()
parser.add_argument('--model_name', default='bert-base-multilingual-cased', type=str)
parser.add_argument('--train_file', default=None, type=str, required=True)
parser.add_argument('--epoch_time', default=2, type=int)
parser.add_argument('--learning_rate',default=3e-5, type=float)
parser.add_argument('--output_name',default=None, type=str, required=True)
parser.add_argument('--save_steps', type=int, default=-1)
parser.add_argument('--batch_size', type=int, default=2)
parser.add_argument('--seed', type=int, default=20)
args = parser.parse_args()

with open(file=args.train_file, mode='r', encoding='utf-8') as reader :
  dataset = json.loads(reader.read())

tokenizer = BertTokenizer.from_pretrained(args.model_name)

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
set_seed(args.seed)

def choose(content, title):
    dk = ['韓國瑜', '韓總', '韓市長', '高雄市長', '韓草包', '賣菜郎']
    for i in dk:
        if i in content or i in title:
            return True
    return False

def find_url(content):
    start = content.find('http')
    end = len(content) - 1
    if content.find('.gif') != -1 :
        end = content.find('.gif') + 3
    elif content.find('.jpg') != -1 :
        end = content.find('.jpg') + 3
    return start, end


def build(content, reply, sample):
    content_ids = tokenizer.encode(content, add_special_tokens=False)
    reply_ids = tokenizer.encode(reply, add_special_tokens=False)
    for i in range(len(reply_ids)):
        mask_ids = [0]*(len(content_ids) + i)
        
        context_ids = content_ids + reply_ids[:i]
        context_ids.append(103) # id of '[MASK]' is 103
        
        mask_ids.append(reply_ids[i])

        context_ids += [0] * (350 - len(context_ids))
        mask_ids += [0] * (350 - len(mask_ids))
        row_dict = {}
        row_dict['context_ids'] = torch.tensor(context_ids)
        row_dict['mask_ids'] = torch.tensor(mask_ids)
        sample.append(row_dict)

        # return context_ids, mask_ids


class loadDataset(data.Dataset):
    def __init__(self,):
        sample = []
        korea_fish = []
        #八卦板板規
        gossip_remind = ' 是否有專板 ， 本板並非萬能問板 。 兩則 問卦， 自刪及被刪也算兩篇之內 ， 本看板嚴格禁止政治問卦 ， 發文問卦前請先仔細閱讀相關板規 。 未滿30繁體中文字水桶3個月，嚴重者以鬧板論 ，請注意'
        for article in dataset:
            try :
                if article['article_title'][:3] in ['Re:', 'Fw:']:
                    continue
                elif article['article_title'][:4] in ['[公告]', '[新聞]', '[臉書]'] :
                    continue
                elif article['article_title'] == '':
                    continue
                
                article['content'] = article['content'].replace(gossip_remind, '')
                if 'http' in article['content']:
                    split = article['content'].split()
                    split = [article for article in split if 'http' not in article]
                    article['content'] = ' '.join(split)
                
                if len(article['content']) > 350:
                    continue
                if article['message_count']['boo'] < 10:
                    continue
                if choose(article['content'], article['article_title']):
                    korea_fish.append(article)
            except Exception as e:
                print(e)
        for i in korea_fish:
            for j in i['messages']:
                while 'http' in j['push_content']:
                    start, end = find_url(j['push_content'])
                    j['push_content'] = j['push_content'].replace( j['push_content'][ start : end+1 ], '' )
        
        for i in korea_fish:
            boo, push = i['message_count']['boo'], i['message_count']['push']
            print('{} ({:3}, {:3}) {} {}'.format(i['score'], boo, push, i['article_title'], i['url']))
            
            context = '[CLS]' + i['content'] + '[SEP]'
            
            for index, j in enumerate(i['messages']):
                if j['push_tag'] == '噓':
                    article_content = context 
                    if index == 0:
                        pass
                    elif index == 1:
                        article_content += i['messages'][0]['push_content'] + '[SEP]'
                    else:
                        article_content += i['messages'][index-2]['push_content'] + '[SEP]'
                        article_content += i['messages'][index-1]['push_content'] + '[SEP]'
                    if len(tokenizer.encode(article_content+j['push_content']+'[SEP]', add_special_tokens=False)) < 350:
                        build( article_content, j['push_content']+'[SEP]', sample )
                        # context_ids, mask_ids = build( article_content, j['push_content']+'[SEP]' )
                        # row_dict = {}
                        # row_dict['context_ids'] = torch.tensor(context_ids)
                        # row_dict['mask_ids'] = torch.tensor(mask_ids)
                        # sample.append(row_dict)
        self.sample = sample

    def __len__(self):
        return len(self.sample)

    def __getitem__(self, idx):
        return self.sample[idx]


train_dataset = loadDataset()

train_iter = data.DataLoader(dataset=train_dataset,
                             batch_size=args.batch_size,
                             shuffle=True)
print('len(dataset) : ', len(train_iter))

print('loading model')
model = BertForMaskedLM.from_pretrained(args.model_name)
model = nn.DataParallel(model.cuda())
device = torch.device( 'cuda' if torch.cuda.is_available() else 'cpu' )
model.to(device)
print('loading successfully')

optimizer = torch.optim.Adamax(model.parameters(), lr=args.learning_rate)
t_total = len(train_dataset) // args.epoch_time
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=t_total)

print('start training')
step = 0
model.train()
for epoch in range(args.epoch_time):
    loss_sum = 0
    local_loss = 0
    for i, batch in enumerate(train_iter):
        try :
            input_ids = batch['context_ids'].cuda()
            masked_lm_labels = batch['mask_ids'].cuda()
            loss, prediction_scores = model(input_ids=input_ids, masked_lm_labels=masked_lm_labels)

            if (i+1) % 100 == 0:
                print('epoch: {} {}/{}, loss = {}'.format(epoch+1, i+1, len(train_iter), local_loss ))
                local_loss = 0
                # print('input_ids :', input_ids[0])
                # print('masked_lm_labels', masked_lm_labels[0])
                # print(tokenizer.decode(input_ids[0]))
                # print(tokenizer.decode(masked_lm_labels[0]))
            loss_sum += loss.mean().item()
            local_loss += loss.mean().item()

            optimizer.zero_grad()
            loss.mean().backward()
            optimizer.step()
            step += 1
            if args.save_steps != -1 and step % args.save_steps == 0 :
                print("start saving check_point-{} ".format(step))
                torch.save(model, args.output_name+'_step_'+str(step)+'.pkl')
        except Exception as e:
            print(e)
            # print(batch)
    print('epoch: {}, loss = {}'.format(epoch+1, loss_sum))
    logger.info('epoch: {}, loss = {}'.format(epoch+1, loss_sum))
    if (epoch+1) % 20 == 0 :
        print("start saving ...")
        torch.save(model, args.output_name+'_epoch_'+str(epoch+1)+'.pkl')


# python run_training_batch.py \
# --model_name bert-base-multilingual-cased \
# --train_file ../dataset/WI_v1.json \
# --epoch_time 100 \
# --learning_rate 3e-5 \
# --output_name ./training \
# --batch_size 2 \
# --seed 20 \