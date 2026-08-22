import os.path

import torch
import pandas as pd
from datasets import Dataset, load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq,
    BitsAndBytesConfig
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, PeftModel
import json
from tqdm import tqdm
from openai import OpenAI
from dotenv import load_dotenv
load_dotenv()
API_KEY = os.getenv("API_KEY")
BASE_URL = os.getenv("BASE_URL")
MODEL_NAME = "gpt-5.4"
client = OpenAI(
    api_key=API_KEY,
    base_url=BASE_URL
)
class SemanticAgent:
    # def __init__(self):
    #     self.system_prompt = (
    #         """Role: Expert Software Architect
    #         Task: Perform a deep semantic analysis of the provided class source code to evaluate its functional intent and responsibility structure.
    #
    #         Instructions:
    #         1. Core Intent: Define the primary high-level purpose of this class in a single, clear sentence.
    #         2. Responsibility Mapping: Detailed breakdown of the distinct functional tasks found within the code. Identify if the class is mixing different business domains or abstraction layers.
    #         3. Logical Depth & Flow: Analyze the complexity of the control flow. Describe why the code is dense or difficult to follow (e.g., complex state transitions, heavily branched business rules).
    #         4. Design Critique: Provide a qualitative assessment of the class's design. Is it focused, or does it show signs of becoming a "catch-all" container for unrelated logic?
    #
    #         Output Constraints:
    #         - Output MUST be a valid JSON object.
    #         - Use precise, professional English.
    #         - Focus entirely on the meaning and intent of the code, not on structural connectivity or static metrics.
    #
    #         JSON Schema:
    #         {
    #           "class_name": "string",
    #           "core_intent": "string",
    #           "functional_responsibilities": ["Responsibility 1", "Responsibility 2"],
    #           "logic_complexity_analysis": "Description of the logic's depth and cognitive load",
    #           "semantic_design_assessment": "Critique focusing on responsibility balance"
    #         }
    #         Do not include any text outside the JSON object."""
    #     )
    def __init__(self, model_id, device_map={"": 0}, is_training=False):
        self.model_id = model_id
        self.device_map = device_map
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        self.model = None
        self.system_prompt = (
            """Role: Expert Software Architect
            Task: Perform a deep semantic analysis of the provided class source code to evaluate its functional intent and responsibility structure.

            Instructions:
            1. Core Intent: Define the primary high-level purpose of this class in a single, clear sentence.
            2. Responsibility Mapping: Detailed breakdown of the distinct functional tasks found within the code. Identify if the class is mixing different business domains or abstraction layers.
            3. Logical Depth & Flow: Analyze the complexity of the control flow. Describe why the code is dense or difficult to follow (e.g., complex state transitions, heavily branched business rules).
            4. Design Critique: Provide a qualitative assessment of the class's design. Is it focused, or does it show signs of becoming a "catch-all" container for unrelated logic?

            Output Constraints:
            - Output MUST be a valid JSON object.
            - Use precise, professional English.
            - Focus entirely on the meaning and intent of the code, not on structural connectivity or static metrics.

            JSON Schema:
            {
              "class_name": "string",
              "core_intent": "string",
              "functional_responsibilities": ["Responsibility 1", "Responsibility 2"],
              "logic_complexity_analysis": "Description of the logic's depth and cognitive load",
              "semantic_design_assessment": "Critique focusing on responsibility balance"
            }
            Do not include any text outside the JSON object."""
        )

        if not is_training:
            self._load_base_model()

    def _load_base_model(self):
        """加载用于推理的 4-bit 模型"""
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            quantization_config=bnb_config,
            device_map=self.device_map,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True
        )


    def train(self, jsonl_path, output_dir="./qwen_semantic_agent_final"):
        """执行 LoRA 微调训练"""
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            quantization_config=bnb_config,
            device_map=self.device_map,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True
        )
        self.model = prepare_model_for_kbit_training(self.model)

        # 2. LoRA 配置
        lora_config = LoraConfig(
            r=16,
            lora_alpha=32,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
        )
        self.model = get_peft_model(self.model, lora_config)

        # 3. 处理数据
        dataset = self._prepare_dataset(jsonl_path)

        # 4. 训练参数
        train_args = TrainingArguments(
            output_dir="./checkpoints",
            per_device_train_batch_size=1,
            gradient_accumulation_steps=16,
            logging_steps=5,
            num_train_epochs=3,
            learning_rate=5e-5,
            bf16=True,
            gradient_checkpointing=True,
            optim="paged_adamw_32bit",
            save_total_limit=1,
            lr_scheduler_type="cosine",
            warmup_ratio=0.1
        )

        # 5. 开始训练
        trainer = Trainer(
            model=self.model,
            args=train_args,
            train_dataset=dataset,
            data_collator=DataCollatorForSeq2Seq(self.tokenizer, padding=True),
        )
        trainer.train()
        self.model.save_pretrained(output_dir)
        print(f"Model saved to {output_dir}")

    def _prepare_dataset(self, jsonl_path):
        data = []
        with open(jsonl_path, 'r', encoding='utf-8') as f:
            for line in f:
                data.append(json.loads(line))

        def tokenize_batch(examples):
            input_ids, labels = [], []
            for ex in examples:
                msgs = [
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": ex["input"]},
                    {"role": "assistant", "content": ex["output"]},
                ]
                # 构造 Prompt 与 Full Text
                prompt_text = self.tokenizer.apply_chat_template(msgs[:2], tokenize=False, add_generation_prompt=True)
                full_text = self.tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False)

                p_tokens = self.tokenizer(prompt_text, truncation=True, max_length=4096, add_special_tokens=False)[
                    "input_ids"]
                f_tokens = self.tokenizer(full_text, truncation=True, max_length=4096, add_special_tokens=False)[
                    "input_ids"]

                input_ids.append(f_tokens)
                labels.append([-100] * len(p_tokens) + f_tokens[len(p_tokens):])
            return {"input_ids": input_ids, "labels": labels}

        raw_ds = Dataset.from_list(data)
        return raw_ds.map(lambda x: tokenize_batch(x), batched=True, remove_columns=raw_ds.column_names)

    # ================= 推理部分 =================

    def load_adapter(self, adapter_path):
        """加载训练好的 LoRA 权重"""
        if self.model is None:
            self._load_base_model()
        self.model = PeftModel.from_pretrained(self.model, adapter_path)
        self.model.eval()
        print("Adapter loaded.")

    def generate_report(self, code):
        """传入判定结果和代码，生成专家报告"""
        user_input = f"Code to analyze:\n{code}"

        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user_input}
        ]

        model_inputs = self.tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True, return_tensors="pt"
        )

        if hasattr(model_inputs, "to"):
            model_inputs = model_inputs.to(self.model.device)

        if isinstance(model_inputs, torch.Tensor):
            generation_inputs = {"input_ids": model_inputs}
        else:
            generation_inputs = {k: v.to(self.model.device) for k, v in model_inputs.items()}

        input_len = generation_inputs["input_ids"].shape[-1]

        with torch.no_grad():
            output_ids = self.model.generate(
                **generation_inputs,
                max_new_tokens=512,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                eos_token_id=self.tokenizer.eos_token_id,
                pad_token_id=self.tokenizer.pad_token_id
            )

        # 只取生成的部分
        response = self.tokenizer.decode(output_ids[0][input_len:], skip_special_tokens=True)
        return response

    def generate_report_via_api(self, code, client, model_name):
        """
        使用 OpenAI 兼容 API 生成语义分析报告（不依赖本地模型推理）。
        返回模型原始文本输出。
        """
        try:
            response = client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": f"Analyze this code:\n---\n{code}\n---"}
                ],
                response_format={"type": "json_object"},
                temperature=0.3
            )
            report_json = response.choices[0].message.content
            print(f"结果: {report_json}")
            return report_json
        except Exception as e:
            print(f"错误: {e}")
            if hasattr(e, 'response') and hasattr(e.response, 'text'):
                print(f"响应内容: {e.response.text}")
            return None


    def generate_report_batch_via_api(self, csv_path, output_jsonl_path, client, model_name,
                                      timeout=60, extra_headers=None):
        """
        从 CSV 文件批量生成语义报告并增量写入 JSONL，支持断点续跑。

        CSV 必须包含列：
        - primary_code
        - label

        输出 JSONL 每行字段：
        - code
        - Semantic Audit
        - Label
        - Metrics
        - Reasponsibilit Gap
        """
        df = pd.read_csv(csv_path)
        # if 'primary_code' not in df.columns or 'label' not in df.columns:
        #     raise ValueError("CSV must contain 'primary_code' and 'label' columns")

        done_count = 0
        if os.path.exists(output_jsonl_path):
            with open(output_jsonl_path, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        done_count += 1

        total = len(df)
        if done_count >= total:
            return {
                'total': total,
                'processed': 0,
                'already_done': done_count,
                'output_jsonl_path': output_jsonl_path
            }
        processed = 0
        with open(output_jsonl_path, 'a', encoding='utf-8') as out_f:
            for i in range(5):
            # for i in tqdm(range(done_count, total), total=total - done_count):
                row = df.iloc[i]
                code = row['primary_code']
                label = row['manual_label']

                semantic_audit = self.generate_report_via_api(
                    code=code,
                    client=client,
                    model_name=model_name
                )
                if not semantic_audit:
                    return None
                record = {
                    'code': code,
                    'Semantic Audit': semantic_audit,
                    'Label': label,
                    'Metrics': '',
                    'Reasponsibilit Gap': ''
                }
                out_f.write(json.dumps(record, ensure_ascii=False) + '\n')
                out_f.flush()
                processed += 1

        return {
            'total': total,
            'processed': processed,
            'already_done': done_count,
            'output_jsonl_path': output_jsonl_path
        }


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
import gc
def cleanup_model(model_obj):
    # 1. 将模型移动到 CPU (可选，但有助于某些环境下的释放)
    if hasattr(model_obj, 'model') and model_obj.model is not None:
        model_obj.model.cpu()

    # 2. 删除引用
    del model_obj

    # 3. 强制触发 Python 垃圾回收
    gc.collect()

    # 4. 清理 PyTorch 缓存 (这是释放显存的关键)
    torch.cuda.empty_cache()

def ablation_model_experiment(model_name):
    model_path = '/model/bd_zjx/model/'+model_name
    SM = SemanticAgent(model_path)
    train_path = f'./dataset/{model_name}_decision_agent_train.jsonl'
    if os.path.exists(train_path):
        with open(train_path, 'r') as f:
            train_data = [json.loads(line) for line in f]
    else:
        with open('./dataset/decision_agent_train.jsonl', 'r') as f:
            base_data = [json.loads(line) for line in f]
            train_data = []
            for index, item in enumerate(base_data):
                item['Semantic Audit'] = {}
                train_data.append(item)
    for index, item in tqdm(enumerate(train_data), total=len(train_data)):
        if item['Semantic Audit']!={}:
            continue
        train_data[index]['Semantic Audit'] = SM.generate_report(item['code'])
        if index % 10 == 0:
            save_current_progress(train_data, train_path)

    test_path = f'./dataset/{model_name}_decision_agent_test.jsonl'
    if os.path.exists(test_path):
        with open(test_path, 'r') as f:
            test_data = [json.loads(line) for line in f]
    else:
        with open('./dataset/decision_agent_test.jsonl', 'r') as f:
            base_data = [json.loads(line) for line in f]
            test_data = []
            for index, item in enumerate(base_data):
                item['Semantic Audit'] = {}
                test_data.append(item)
    for index, item in tqdm(enumerate(test_data), total=len(test_data)):
        if item['Semantic Audit'] != {}:
            continue
        test_data[index]['Semantic Audit'] = SM.generate_report(item['code'])
        if index % 10 == 0:
            save_current_progress(test_data, test_path)

    cleanup_model(SM)
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
    with open('../Fine_Tuning/dataset/decision_agent_test_data.jsonl', 'r') as f:
        data = [json.loads(line) for line in f]
    save_data = []
    for i, row in enumerate(data):
        save_data.append({
            'code':row['code'],
            'label':row['Teacher Verdict']
        })
    save_current_progress(save_data, './dataset/test_data.jsonl')
    # print(data[1450]['Semantic Audit'])
    # e = json.loads(data[1449]['Semantic Audit'])
    # save_data = []
    # for i, row in tqdm(enumerate(data), total=len(data)):
    #     try:
    #         save_data.append(json.loads(row['Semantic Audit']))
    #     except Exception as e:
    #         print(i, row['Semantic Audit'])
    # save_current_progress(save_data, './sub_agent_result/train_semantic_result.jsonl')
        



