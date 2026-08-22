import pandas as pd
import json
import subprocess
import os
import networkx as nx
from itertools import combinations
import javalang
import tempfile
import numpy as np
from tqdm import tqdm
class Lcom4Calculator:
    def __init__(self):
        self.last_graph = None  # 存储最近一次生成的图，供调试或可视化
        self.last_class_name = None
    def calculate(self, code):
        """
        外部调用的主入口
        """
        # 局部变量：确保并发安全和数据隔离
        result = self._extract_class_metrics(code)
        class_name, fields, field_access, calls = result

        # 解析失败的情况
        if class_name is None:
            return 0, ''  # 或者返回 None，代表计算失败

        self.last_class_name = class_name
        lcom4_val, G = self._compute_graph_logic(field_access, calls)

        self.last_graph = G  # 存入成员变量供后续提取
        return lcom4_val, class_name

    def _compute_graph_logic(self, field_access, calls):
        """
        将图计算逻辑剥离，保持纯粹
        """
        methods = list(field_access.keys())
        if len(methods) <= 1:
            return 1, nx.Graph()

        G = nx.Graph()
        G.add_nodes_from(methods)

        # 基于字段共享连边
        for m1, m2 in combinations(methods, 2):
            if field_access[m1].intersection(field_access[m2]):
                G.add_edge(m1, m2)

        # 基于方法调用连边
        for caller, callees in calls.items():
            for callee in callees:
                if callee in G:
                    G.add_edge(caller, callee)

        return nx.number_connected_components(G), G

    def _traverse_ast_manually(self, node):
        # 1. 如果当前节点是列表，直接遍历其中的元素
        if isinstance(node, list):
            for item in node:
                yield from self._traverse_ast_manually(item)
            return  # 必须 return，防止继续向下执行 __dict__

        # 2. 只有是 javalang 节点对象时，才遍历其属性
        if isinstance(node, javalang.ast.Node):
            for attr_name, attr_value in node.__dict__.items():
                if attr_name.startswith('_'):
                    continue

                # 如果属性值是单个节点
                if isinstance(attr_value, javalang.ast.Node):
                    yield (attr_name, attr_value)
                    yield from self._traverse_ast_manually(attr_value)

                # 如果属性值是节点列表（这通常是 Block 里的语句列表）
                elif isinstance(attr_value, list):
                    for item in attr_value:
                        if isinstance(item, javalang.ast.Node):
                            yield (attr_name, item)
                            yield from self._traverse_ast_manually(item)

    def _extract_class_metrics(self, java_code):
        instance_fields = set()
        methods_field_access = {}
        method_calls_raw = {}
        instance_method_names = set()

        try:
            tree = javalang.parse.parse(java_code)
            # 获取类名用于度量报告匹配
            class_node = next(iter(tree.types), None)
            if not class_node or not isinstance(class_node, javalang.tree.ClassDeclaration):
                return None, instance_fields, {}, {}

            target_class_name = class_node.name
        except:
            return None, set(), {}, {}

        # 1. 提取实例字段和实例方法名 (同你之前的逻辑)
        for member in class_node.body:
            if isinstance(member, javalang.tree.FieldDeclaration) and 'static' not in member.modifiers:
                for declarator in member.declarators:
                    instance_fields.add(declarator.name)
            elif isinstance(member, javalang.tree.MethodDeclaration) and 'static' not in member.modifiers:
                instance_method_names.add(member.name)

        # 2. 遍历方法提取关系
        for member in class_node.body:
            if isinstance(member, javalang.tree.MethodDeclaration) and 'static' not in member.modifiers:
                method_name = member.name
                accessed_fields = set()
                calls = set()

                # 直接遍历整个方法体
                if member.body:
                    for _, node in self._traverse_ast_manually(member.body):
                        # 字段访问：只看 MemberReference
                        if isinstance(node, javalang.tree.MemberReference):
                            # 确保不是在调用方法，且在字段表里
                            if node.member in instance_fields:
                                accessed_fields.add(node.member)

                        # 方法调用
                        if isinstance(node, javalang.tree.Invocation):
                            if node.member in instance_method_names and node.member != method_name:
                                calls.add(node.member)

                methods_field_access[method_name] = accessed_fields
                method_calls_raw[method_name] = calls

        return target_class_name, instance_fields, methods_field_access, method_calls_raw

class MetricsExpert:
    def __init__(self, ck_jar_path, lcom4_calculator, stats_path='./statistics/metrics_stats_result.json'):
        self.ck_jar_path = ck_jar_path
        self.lcom4_calculator = lcom4_calculator
        self.class_name = ''
        self.base_temp_dir = "/dev/shm" if os.path.exists("/dev/shm") else None
        with open(stats_path, 'r') as f:
            self.stats = json.load(f)
        f.close()
    def analyze_class(self, code):
        """
        整合所有度量指标
        """
        # 1. 提取 CK 指标 (WMC, LOC, CBO, RFC 等)

        lcom4_value, self.class_name = self.lcom4_calculator.calculate(code)

        ck_metrics = self._extract_ck_metrics(code)

        # 3. 合并报告
        report = {
            "agent_id": "Metrics_Expert",
            "target_class": self.class_name,
            "metrics": {
                "size_metrics": {
                    "LOC": ck_metrics.get("loc", 0),
                    "WMC": ck_metrics.get("wmc", 0),
                    "WMC/LOC":ck_metrics.get("wmc", 0)/ck_metrics.get("loc", 0) if ck_metrics.get("loc", 0) != 0 else 0,
                },
                "cohesion_metrics": {
                    "LCOM4": lcom4_value,
                    "LCOM_CK": ck_metrics.get("lcom", 0)  # CK 默认也带一个 LCOM
                },
                "coupling_metrics": {
                    "CBO": ck_metrics.get("cbo", 0),
                    "RFC": ck_metrics.get("rfc", 0)
                }
            },
            "risk_signals": self._pre_check_risks(ck_metrics, lcom4_value)
        }
        return report

    def _extract_ck_metrics(self, java_code):
        """
        核心方法：将代码存入内存，运行 CK 并解析结果
        """
        # 1. 使用 TemporaryDirectory 确保离开作用域时自动清理内存中的残留文件
        with tempfile.TemporaryDirectory(dir=self.base_temp_dir, prefix="ck_eval_") as workspace:

            # 2. 准备目录结构
            # CK 要求输入是一个目录，即使里面只有一个文件
            src_dir = os.path.join(workspace, "src")
            out_dir = os.path.join(workspace, "out")
            os.makedirs(src_dir)
            os.makedirs(out_dir)

            # 3. 按照 Java 规范命名文件（类名.java）
            # 注意：如果 target_class 包含包名，只需取最后的部分
            simple_class_name = self.class_name
            java_file_path = os.path.join(src_dir, f"{simple_class_name}.java")

            try:
                # 4. 将代码写入内存文件系统
                with open(java_file_path, "w", encoding="utf-8") as f:
                    f.write(java_code)

                # 5. 调用 CK 工具 (子进程)
                # 参数含义：[jar, 源码目录, 变量开关, 方法开关, 字段开关, 输出目录/]
                result = subprocess.run(
                    [
                        "java", "-jar", self.ck_jar_path,
                        src_dir, "true", "0", "false", f"{out_dir}/", 'class'
                    ],
                    stdout=subprocess.DEVNULL,  # 屏蔽冗余日志
                    stderr=subprocess.PIPE,  # 捕获错误信息
                    timeout=10  # 防止 JVM 卡死导致主程序挂起
                )
                '''
                subprocess.run([
                "java", "-jar", ck_jar_path,
                group,
                "true",
                "0",
                "false",
                "./",
                "class"
            ], check=True, capture_output=True, cwd="/model/bd_zjx/AI_code_with_Human_written_code/Smell_LCOM")

                '''
                if result.returncode != 0:
                    print(f"CK Error: {result.stderr.decode()}")
                    return {}

                # 6. 解析结果 (由于是单文件，class.csv 中理论上只有一行或少量几行)
                csv_path = os.path.join(out_dir, "class.csv")
                if os.path.exists(csv_path):
                    df = pd.read_csv(csv_path)
                    if not df.empty:
                        # 尝试匹配目标类名，匹配不到则取首行
                        match = df[df['class'].str.contains(simple_class_name, na=False)]
                        target_row = match.iloc[0] if not match.empty else df.iloc[0]

                        # 转换为标准的字典格式返回
                        return self._format_metrics(target_row)

            except subprocess.TimeoutExpired:
                print(f"CK Analysis timeout for {self.class_name}")
            except Exception as e:
                print(f"Metrics Expert internal error: {e}")

        return {}

    def _pre_check_risks(self, ck, lcom4):
        # 度量专家可以先根据经验公式给出一个“离群”警告
        signals = []
        if ck.get("wmc", 0) > 40: signals.append("High Complexity (WMC)")
        if lcom4 > 1: signals.append("Potential Responsibility Split (LCOM4 > 1)")
        return signals

    def _format_metrics(self, row):
        """
        精简并格式化 CK 输出，只保留决策代理关心的核心指标
        """
        return {
            "class_name": row['class'],
            "loc": int(row['loc']),  # 代码行数
            "wmc": int(row['wmc']),  # 圈复杂度
            "cbo": int(row['cbo']),  # 类间耦合度
            "rfc": int(row['rfc']),  # 响应类数
            "lcom_ck": int(row['lcom']),  # CK 版本的 LCOM
            "is_valid": True
        }
    def generate_metrics_statistics_report(self, metrics_agent):
        metrics = metrics_agent.get("metrics", metrics_agent)
        size_metrics = metrics.get("size_metrics", {})
        cohesion_metrics = metrics.get("cohesion_metrics", {})
        coupling_metrics = metrics.get("coupling_metrics", {})

        return {
            "agent_id":'Metrics_Expert',
            "target_class":metrics_agent.get('target_class'),
            "metric_values": {
                "LOC": size_metrics.get("LOC", 0),
                "WMC": size_metrics.get("WMC", 0),
                "WMC/LOC": size_metrics.get("WMC/LOC", 0),
                "LCOM4": cohesion_metrics.get("LCOM4", 0),
                "LCOM_CK": cohesion_metrics.get("LCOM_CK", 0),
                "CBO": coupling_metrics.get("CBO", 0),
                "RFC": coupling_metrics.get("RFC", 0)
            },
            "statistics_report": self.stats
        }

def get_metrics_stats(datapath):
    short_names = ['NOM', 'WMC', 'NOF', 'CLOC', 'LCOM', 'LCOM_HS', 'LCOM_4', 'WMC/LOC']
    full_names = ['totalMethodsQty', 'wmc', 'totalFieldsQty', 'loc', 'lcom', 'lcom*', 'lcom4', 'wmc/loc']
    column_mapping = dict(zip(full_names, short_names))
    df = pd.read_csv(datapath)
    df['wmc/loc'] = np.where(df['loc'] != 0, df['wmc'] / df['loc'], 0)
    df.rename(columns=column_mapping, inplace=True)
    percentiles = [i for i in range(0, 101, 10)]
    p_names = [f'p{p}' for p in percentiles]
    result = {}
    for stat in df.columns: 
        percentiles_values = df[stat].quantile([p/100 for p in percentiles])
        result[stat] = {p_name: percentiles_values[p/100] for p_name, p in zip(p_names, percentiles)}
    with open('./statistics/metrics_stats_result.json', 'w') as f:
        json.dump(result, f, ensure_ascii=False, indent=4)

def save_current_progress(data_list, path):
    """
    安全保存逻辑：先写临时文件再重命名，防止保存时崩溃导致原文件损坏
    """
    temp_path = path + ".tmp"
    with open(temp_path, 'w', encoding='utf-8') as f:
        for entry in data_list:
            f.write(json.dumps(entry, ensure_ascii=False) + '\n')
    # 原子替换：将 tmp 文件替换为正式文件
    os.replace(temp_path, path)
if __name__ == "__main__":

    # get_metrics_stats('/model/bd_zjx/AI_code_with_Human_written_code/Empirical_Study/ck_metrics_output/generated_metrics.csv')
    calculator = Lcom4Calculator()
    metrics_expert = MetricsExpert('/model/bd_zjx/AI_code_with_Human_written_code/ck-0.7.0-jar-with-dependencies.jar', calculator)
    data_path = '../Fine_Tuning/dataset/decision_agent_train_data.jsonl'
    save_path = './sub_agent_result/train_metrics_result.jsonl'
    save_data = []
    with open(data_path, 'r') as f:
        data = [json.loads(line) for line in f] 
    for i, row in tqdm(enumerate(data), total=len(data)):
        result = metrics_expert.generate_metrics_statistics_report(row['Metrics'])
        save_data.append(result)
        if i%100==0:
            save_current_progress(save_data, save_path)
    save_current_progress(save_data, save_path)



 # import json
    # from tqdm import tqdm
    # data = []
    # data_path = './dataset/decision_agent_test.jsonl'
    # with open(data_path, 'r', encoding="utf-8") as f:
    #     for line in f:
    #         data.append(json.loads(line))


    # for index, row in tqdm(enumerate(data), total=len(data)):
    #     result = metrics_expert.analyze_class(row['code'])
    #     data[index]['Metrics'] = result
    #     if index%100==0:
    #         save_current_progress(data, data_path)

    # save_current_progress(data, data_path)


    # df = pd.read_csv('./dataset/train.csv')
    # locm4s = []
    # from time import time
    # start = time()
    # for index, row in df[94:95].iterrows():
    #     report = metrics_expert.analyze_class(row['code'])
    #     print(report)
    # print((time() - start)/1000)
    # from collections import Counter
    # counter = Counter(locm4s)
    # print(counter)


