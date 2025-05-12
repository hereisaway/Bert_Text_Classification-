from huggingface_hub import snapshot_download
import torch
from torch import nn,optim
import os
from torch.utils.data import Dataset,DataLoader
from transformers import BertTokenizer,BertModel


model_name="google-bert/bert-base-chinese"
cache_dir="./pretrained_bert"

# 下载预训练模型
# snapshot_download(repo_id=model_name, cache_dir=cache_dir)



# 测试分词器和模型的输出
# model_path="E:\编程\深度学习\Bert\pretrained_bert\models--google-bert--bert-base-chinese\snapshots\c30a6ed22ab4564dc1e3b2ecbf6e766b0611a33f"
# tokenizer=BertTokenizer.from_pretrained(model_path)
# model=BertModel.from_pretrained(model_path)
# text = [
#     "你好，世界",
#     "二十四岁，是学生"
# ]
# inputs = tokenizer(text, return_tensors='pt',add_special_tokens=True,max_length=128,
#                    padding=True,truncation=True)
#
# input_ids = inputs['input_ids']
# attention_mask = inputs['attention_mask']
#
# print(input_ids.shape)
# print(attention_mask.shape)
#
# model=BertModel.from_pretrained(model_path)
# features=model(input_ids=input_ids,attention_mask=attention_mask)
#
# pooler_output=features.pooler_output
#
# print(pooler_output.shape)

# 定义模型
class BertClassifier(nn.Module):
    def __init__(self,model_path,class_num):
        super(BertClassifier, self).__init__()

        self.Bert = BertModel.from_pretrained(model_path)

        self.classifier=nn.Linear(self.Bert.config.hidden_size,class_num)

    def forward(self,input_ids,attention_mask):
            features = self.Bert(input_ids=input_ids,attention_mask=attention_mask)

            logits = self.classifier(features.pooler_output)

            return logits

#定义数据集
class BertDataset(Dataset):
    def __init__(self,path):
        self.examples = list()
        file=open(path,'r',encoding='utf-8')

        for line in file:
            text,label=line.strip().split('\t')
            self.examples.append((text,int(label)))
        file.close()

    def __len__(self):
        return len(self.examples)
    def __getitem__(self, idx):
        return self.examples[idx]

#定义dataloader
def collate_fn(batch,tokenizer):
    texts=[item[0] for item in batch]
    labels=[item[1] for item in batch]

    labels=torch.tensor(labels,dtype=torch.long)
    tokens=tokenizer(
        texts,
        max_length=512,
        padding=True,
        truncation=True,
        return_tensors="pt",
        add_special_tokens=True
    )

    return tokens['input_ids'],tokens['attention_mask'],labels

if __name__=="__main__":
    dataset=BertDataset("data/train.txt")
    model_path="E:\编程\深度学习\Bert\pretrained_bert\models--google-bert--bert-base-chinese\snapshots\c30a6ed22ab4564dc1e3b2ecbf6e766b0611a33f"
    tokenizer = BertTokenizer.from_pretrained(model_path)
    dataloader=DataLoader(
        dataset=dataset,
        batch_size=128,
        collate_fn=lambda x: collate_fn(x,tokenizer),
        shuffle=True
    )
    device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = BertClassifier(model_path=model_path,class_num=10).to(device)
    model.train()

    optimizer = optim.Adam(model.parameters(),lr=5e-5)
    criterion = nn.CrossEntropyLoss()

    os.makedirs("output_models",exist_ok=True)

    epochs=10
    for epoch in range(epochs):
        for batch ,data in enumerate(dataloader):
            input_ids=data[0].to(device)
            attention_mask=data[1].to(device)
            label=data[2].to(device)

            optimizer.zero_grad()
            output=model(input_ids,attention_mask)
            loss=criterion(output,label)
            loss.backward()
            optimizer.step()

            pred=torch.argmax(output,dim=1)
            correct=(pred==label).sum().item()
            acc=correct/output.size(0)

            print(
                f"Epoch{epoch+1}/{epochs}"
                f"|Batch{batch+1}/{len(dataloader)}"
                f"|Loss:{loss.item():.4f}"
                f"|Acc:{acc:.4f}"
            )
        model_name=f"./output_models/model{epoch}.pt"
        print(f"save model to {model_name}")
        torch.save(model.state_dict(),model_name)



