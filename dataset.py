from torch.utils.data import Dataset
import json
class MyDataset(Dataset):
    def __init__(self, jsonl_file, need_details=False):
        """
        初始化数据集
        :param jsonl_file: jsonl格式的数据集文件路径
        :param need_details: 是否返回伪装方式字段
        """
        self.need_details = need_details
        self.dataset = self.load_jsonl(jsonl_file)  # 加载jsonl数据
        self.classes = ['gpt', 'human']  # label字段的类别
        print(f'There are {len(self.dataset)} samples in the dataset')
    
    def load_jsonl(self, jsonl_file):
        """
        从jsonl文件加载数据
        :param jsonl_file: jsonl文件路径
        :return: 返回所有的样本数据
        """
        dataset = []
        with open(jsonl_file, 'r', encoding='utf-8') as f:
            for line in f:
                dataset.append(json.loads(line.strip()))  # 每行解析为json对象
        return dataset
    
    def get_class(self):
        """
        返回所有的类别标签
        :return: 类别标签列表
        """
        return self.classes

    def __len__(self):
        """
        返回数据集的大小
        :return: 数据集大小
        """
        return len(self.dataset)

    def __getitem__(self, idx):
        """
        获取指定索引的数据
        :param idx: 数据索引
        :return: (article, label, detail) 或 (article, label)
        """
        item = self.dataset[idx]
        article, label, detail = item['article'], item['label'], item['detail']
        
        label_idx = self.classes.index(label)  # 将label转换为索引0或1
        
        if self.need_details:
            return article, label_idx, detail  # 如果需要返回detail
        else:
            return article, label_idx 