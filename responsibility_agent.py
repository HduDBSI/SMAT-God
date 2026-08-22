import javalang
import re
from transformers import AutoConfig, AutoTokenizer, AutoModel
import torch
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import json
import os
from tqdm import tqdm

def embedding_model_init():
    device = "cuda:0" # 依然在你指定的显卡上
    # path = '/model/bd_zjx/model/codebert'  # 你的模型路径
    # config = AutoConfig.from_pretrained(path, trust_remote_code=True)
    # tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
    # state_dict = torch.load(f"{path}/pytorch_model.bin", map_location="cpu", weights_only=True)
    # model = AutoModel.from_config(config, trust_remote_code=True)
    # model.load_state_dict(state_dict)
    # model.save_pretrained(path, safe_serialization=True)
    model_path = '/model/bd_zjx/model/codebert'

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModel.from_pretrained(model_path).to(device)
    return device, tokenizer, model.eval()

class ResponsibilityAgent:
    def __init__(self, model, tokenizer, device, stats_path="./statistics/responsibility_stats_result.json"):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        # 加载之前 1000 条样本的统计基准
        with open(stats_path, 'r') as f:
            self.stats = json.load(f)

    def analyze_class(self, code, method_names=None):
        """
        输入: 代码
        输出: 结构化 JSON 报告
        """
        class_name, extracted_method_names = self._extract_method_names(code)
        # 优先使用外部提取的主要方法名；没有提供时保持原有 AST 提取行为。
        class_method_names = method_names if method_names is not None else extracted_method_names
        class_method_names = [name for name in class_method_names if isinstance(name, str) and name.strip()]
        method_names_tokens = [self._split_camel_case(method_name) for method_name in class_method_names]
        if len(method_names_tokens) < 2:
            return self._empty_report(class_name)
        max_pair, min_pair, distance_stats = self._get_gap(method_names_tokens)
        # 4. 构建 JSON 结果
        report = {
            "agent_id": "Responsibility_Expert",
            "target_class": class_name,
            "results": {
                "method_embedding_cosine_distance": {
                    "max": round(float(distance_stats["max"]), 4),
                    "min": round(float(distance_stats["min"]), 4),
                    "mean": round(float(distance_stats["mean"]), 4)
                },
                "max_distance_method_pair": [
                    class_method_names[max_pair[0]],
                    class_method_names[max_pair[1]]
                ],
                "min_distance_method_pair": [
                    class_method_names[min_pair[0]],
                    class_method_names[min_pair[1]]
                ],
                "statistics_report": self.stats
            }
        }
        return report

    def _get_gap(self, method_names_tokens):
        # 1. 计算所有方法的 Embeddings
        embeddings = [self._get_embedding(tokens) for tokens in method_names_tokens]
        matrix = np.vstack(embeddings)

        # 2. 计算语义间隙 (Gap)
        sim_matrix = cosine_similarity(matrix)
        # 只取上三角，保证每个无序方法对只统计一次。
        distance_matrix = 1.0 - sim_matrix
        pair_indices = np.triu_indices(len(method_names_tokens), k=1)
        distances = distance_matrix[pair_indices]
        max_distance = float(np.max(distances))
        min_distance = float(np.min(distances))
        mean_distance = float(np.mean(distances))
        max_position = int(np.argmax(distances))
        min_position = int(np.argmin(distances))
        max_pair = (int(pair_indices[0][max_position]), int(pair_indices[1][max_position]))
        min_pair = (int(pair_indices[0][min_position]), int(pair_indices[1][min_position]))

        return max_pair, min_pair, {
            "max": max_distance,
            "min": min_distance,
            "mean": mean_distance,
        }

    def _get_embedding(self, token_list):
        # 拼接并使用 Mean Pooling (如之前讨论，比 [CLS] 更敏感)
        sentence = " ".join(token_list)
        inputs = self.tokenizer(sentence, return_tensors="pt", padding=True, truncation=True).to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
        # Mean Pooling 逻辑
        mask = inputs['attention_mask'].unsqueeze(-1).expand(outputs.last_hidden_state.size()).float()
        return (torch.sum(outputs.last_hidden_state * mask, 1) / torch.clamp(mask.sum(1), min=1e-9)).cpu().numpy()

    def _empty_report(self, name):
        return {
            "target_class": name,
            "results": {
                "method_embedding_cosine_distance": {"max": 0, "min": 0, "mean": 0},
                "max_distance_method_pair": [],
                "min_distance_method_pair": [],
                "statistics_report": self.stats
            }
        }


    def _extract_method_names(self, java_code):
        try:
            # 解析 Java 代码
            tree = javalang.parse.parse(java_code)
            methods = []
            for path, node in tree.filter(javalang.tree.ClassDeclaration):
                class_name = node.name
                break
            # 遍历所有方法声明
            for path, node in tree.filter(javalang.tree.MethodDeclaration):
                methods.append(node.name)
            return class_name, methods
        except Exception as e:
            return None, []

    def _split_camel_case(self, name):
        # 将 camelCase 转换为 space separated words
        s1 = re.sub('(.)([A-Z][a-z]+)', r'\1 \2', name)
        return re.sub('([a-z0-0])([A-Z])', r'\1 \2', s1).lower().split()

class NpEncoder(json.JSONEncoder):
    """自定义编码器，处理 NumPy 导致的所有序列化错误"""

    def default(self, obj):
        if isinstance(obj, (np.bool_, np.true_divide)):
            return bool(obj)
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)

def save_current_progress(data_list, path):
    """
    安全保存逻辑：先写临时文件再重命名，防止保存时崩溃导致原文件损坏
    """
    temp_path = path + ".tmp"
    with open(temp_path, 'w', encoding='utf-8') as f:
        for entry in data_list:
            f.write(json.dumps(entry, ensure_ascii=False, cls=NpEncoder) + '\n')
    # 原子替换：将 tmp 文件替换为正式文件
    os.replace(temp_path, path)

def get_statistic(responsibility_agent):
    save_path = './statistics/responsibility_results.jsonl'
    if os.path.exists(save_path):
        with open(save_path, 'r', encoding='utf-8') as f:
            data = [json.loads(line) for line in f]
    else:
        data = []
        method_names_path = './method_names/statistics_method_names.jsonl'
        method_names_by_index = []
        if os.path.exists(method_names_path):
            with open(method_names_path, 'r', encoding='utf-8') as methods_file:
                for line in methods_file:
                    if line.strip():
                        method_record = json.loads(line)
                        method_names_by_index.append(method_record)
        for index, row in tqdm(enumerate(method_names_by_index), total=len(method_names_by_index)):
            result = responsibility_agent.analyze_class(row['code'], row['method_names'])
            data.append(result)
            if index%100==0:
                save_current_progress(data, save_path)
        save_current_progress(data, save_path)
    df = pd.DataFrame(data)
    percentiles = [i for i in range(0, 101, 10)]
    p_names = [f'p{p}' for p in percentiles]
    result = {}
    for stat in df.columns:  # stat 就是 'max', 'min', 'mean'
        # 计算该统计量的百分位数
        percentiles_values = df[stat].quantile([p/100 for p in percentiles])
        # 转换为字典，键为 p0, p10, ...
        result[stat] = {p_name: percentiles_values[p/100] for p_name, p in zip(p_names, percentiles)}
    with open('statistics/responsibility_stats_result.json', 'w') as f:
        json.dump(result, f, ensure_ascii=False, indent=4)
if __name__ == "__main__":
    import pandas as pd
    embedding_device, embedding_tokenizer, embedding_model = embedding_model_init()
    responsibility_agent = ResponsibilityAgent(embedding_model, embedding_tokenizer, embedding_device)

    file_path = './method_names/test_primary_methods.jsonl'
    save_path = 'test_responsibility_result.jsonl'
    with open(file_path, 'r') as f:
        data = [json.loads(line) for line in f]
    save_data = []
    for index, row in tqdm(enumerate(data), total=len(data)):
        result = responsibility_agent.analyze_class(row['ai_code'], row['method_names'])
        save_data.append(result)
        if index%100 == 0:
            save_current_progress(save_data, save_path)
    save_current_progress(save_data, save_path)
    # get_statistic(responsibility_agent)



'''
8
4,3,2
3,2,47
3,9,66
11,9,88
2,8,84
12,7,64
5,9,1
8,9,18
6
3,11,93
2,8,38
3,12,95
11,12,25
8,10,60
10,2,37


4,3,2
3,2,47
3,9,88
11,9,88
2,8,84
12,7,64
5,9,88
8,9,88
false

4,3,2
3,2,47
3,9,88
11,9,88
2,8,84
12,7,64
5,9,88
8,9,88
3,11,93
2,8,38
3,12,95
11,12,95
8,10,60
10,2,37
'''
