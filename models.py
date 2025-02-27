import torch
import torch.nn as nn
import torch.nn.functional as F
from text_embedding import TextEmbeddingModel

class ClassificationHead(nn.Module):
    """Head for sentence-level classification tasks."""

    def __init__(self, in_dim, out_dim):
        super(ClassificationHead, self).__init__()
        self.dense1 = nn.Linear(in_dim, in_dim//4)
        self.dense2 = nn.Linear(in_dim//4, in_dim//16)
        self.out_proj = nn.Linear(in_dim//16, out_dim)
        #很经典的操作！初始化连接层权重和偏置，使网络在训练初期更稳定
        nn.init.xavier_uniform_(self.dense1.weight)
        nn.init.xavier_uniform_(self.dense2.weight)
        nn.init.xavier_uniform_(self.out_proj.weight)
        nn.init.normal_(self.dense1.bias, std=1e-6)
        nn.init.normal_(self.dense2.bias, std=1e-6)
        nn.init.normal_(self.out_proj.bias, std=1e-6)

    def forward(self, features):
        x = features
        x = self.dense1(x)
        x = torch.tanh(x)
        x = self.dense2(x)
        x = torch.tanh(x)
        x = self.out_proj(x)
        return x
    

class SimCLR_Classifier_SCL(nn.Module):
    '''复杂的伪装方式才是：监督对比学习 + 交叉熵（混合损失）'''
    def __init__(self, opt,fabric):
        super(SimCLR_Classifier_SCL, self).__init__()
        
        self.temperature = opt.temperature
        self.opt=opt
        #自动检测并配置可用的硬件资源并优化加速
        self.fabric = fabric

        self.model = TextEmbeddingModel(opt.model_name)
        self.device=self.model.model.device
        if opt.resum:
            state_dict = torch.load(opt.pth_path, map_location=self.device)
            self.model.load_state_dict(state_dict)
        self.esp=torch.tensor(1e-6,device=self.device)

        self.classifier = ClassificationHead(opt.projection_size, opt.classifier_dim)

    def get_encoder(self):
        return self.model

    def forward(self, batch, labels):
        """
        输入:
            batch: 包含文本输入的字典，通常为 {'input_ids', 'attention_mask'}
            labels: 真实标签（0=Human, 1=GPT）
        输出:
            loss: 交叉熵损失
            logits: 模型输出的原始预测值（未归一化）
        """
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        embeddings = self.model(input_ids, attention_mask)
        logits = self.classifier(embeddings)

        loss = F.cross_entropy(logits, labels)
        
        return loss, logits
    
    











  