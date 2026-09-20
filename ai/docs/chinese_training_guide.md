# OPS 中文訓練指南
## 建立具備強大中文能力的開源 AI Agent

本指南將引導您訓練一個專門優化中文理解和生成能力的 OPS 模型，使其在中文任務上的表現能與或超越 Claude Opus。

## 🎯 為何選擇中文優化？

1. **市場需求**：中文是世界上使用人數最多的語言之一
2. **資源不足**：高品質中文 LLMs 相對較少，特別是開源且可訓練的
3. **特殊挑戰**：中文有其獨特的語法、語義和文化背景
4. **應用場景**：中文環境下的程式設計、文書處理、資訊檢索等有廣泛需求

## 📋 訓練流程概覽

```
基礎模型選擇 → 中文數據準備 → 階段式訓練 → 模型優化 → Ollama 部署
```

## 🏗️ 第一步：選擇合適的基礎模型

為中文任務優化選擇以下模型之一（均為完全開放下載）：

| 模型名稱 | 參數量 | 中文能力 | 推薦理由 |
|----------|--------|----------|----------|
| **Qwen-1.5-7B-Chat** | 7B | ⭐⭐⭐⭐⭐ | 阿里巴巴開發，中文優秀，支援 32K 上下文 |
| **Qwen-2-7B-Instruct** | 7B | ⭐⭐⭐⭐⭐ | Qwen-1.5 的改進版，中文與英語能力均衡且強大 |
| **Baichuan2-7B-Base** | 7B | ⭐⭐⭐⭐☆ | 百川智能開源，中文基礎扎實 |
| **InternLM2-Chat-7B** | 7B | ⭐⭐⭐⭐☆ | 上海AI實驗室，中文推理能力強 |

**推薦起點**：Qwen-1.5-7B-Chat 或 Qwen-2-7B-Instruct
- 這些模型在中文基準測試中表現優秀
- 完全開放下載，無需申請權限
- 有豐富的社區支持和微調範例

## 📚 第二步：準備中文訓練數據

我們將採用階段式訓練策略，每個階段專注於不同的中文能力：

### 階段 1：通用中文指令跟隨 (SFT-General-Chinese)

**數據來源**：
- **中文 Alpaca 變種**：
  - `Chinese-Alpaca-Plus` (52K 中文指令-回應對)
  - `Belle-Group` 系列 (多個中文指令數據集)
  - `Firefly` 系列 (中英雙語指令數據)

- **中文問答數據**：
  - `WebCPM` (中文網頁問答)
  - `CMeKG` (醫療領域中文知識圖譜問答)
  - `LCSTS` (中文微博情感分析數據，可轉為指令格式)

- **中文百科知識**：
  - 基於維基百科中文版構建的知識問答對
  - 中文古詩詞、成語典故相關數據

**目標**：建立穩固的中文語言理解和基本指令遵循能力

### 階段 2：中文程式設計與技術文檔 (SFT-Code-Chinese)

**數據來源**：
- **中文程式教育數據**：
  - 中文版的 CodeAlpaca (程式指令+代碼)
  - 中文程式設計教程數據（如菜鳥教程、廖雪峰官方網站）
  - LeetCode 中文題解數據集

- **技術文檔與說明書**：
  - 中文開源項目的 README、API 文檔
  - 中文技術部落格文章
  - 程式語言中文官方文檔摘錄

- **代碼註解優化**：
  - 從開源項目中提取高質量中文註解的代碼片段
  - 強化「用中文解釋代碼功能」的能力

**目標**：提升模型在中文環境下理解、生成和解釋程式碼的能力

### 階段 3：中文工具使用與實務應用 (SFT-Tools-Chinese)

**數據來源**：
- **中文工具使用範例**：
  - 基於 OPS 工具系統設計的中文指令模板
  - 中文系統管理腳本範例（PowerShell, Bash）
  - 中文網路爬蟲與資料處理範例

- **中文多輪對話與任務規劃**：
  - 中文客服對話數據
  - 中文專案管理與任務分解範例
  - 中文決策過程與推理鏈數據

- **實際應用場景**：
  - 中文郵件寫作與修改
  - 中文報告摘要生成
  - 中文翻譯與語言轉換任務

**目標**：讓模型能在中文環境下有效使用工具解決實際問題

### 階段 4：中文對齊與人類偏好 (RLHF-Chinese - 可選)

如果有人工標註的中文偏好數據：
- 中文問答對的好壞回應排序
- 中文創意寫作的品質評分
- 中文邏輯推理的正誤判斷

## ⚙️ 第三步：訓練配置調整

編輯 `C:\Users\jksdf\ops\configs\training\train_config.yaml` 以適合中文訓練：

```yaml
# OPS 中文優化訓練配置

## 模型設定 - 選擇其中一個
base_model: "Qwen/Qwen-1.5-7B-Chat"  # 或 "Qwen/Qwen-2-7B-Instruct"
output_dir: "./output/ops-chinese-7b"

## 訓練超參數 - 調整為中文優化
epochs: 3                          # 中文訓練可能需要更多輪數
batch_size: 2                      # 根據顯存調整
gradient_accumulation_steps: 16    # 增加梯度累積以補償小 batch
learning_rate: 5.0e-6              # 中文訓練常用較小學習率
warmup_steps: 100                  # 適當的熱身步數
weight_decay: 0.01
max_grad_norm: 1.0
max_seq_length: 4096               # 充分利用 Qwen 的長上下文能力

## 數據設定
dataset_name: "shibing624/Chinese-Alpaca-Plus"  # 主要中文數據集
val_set_size: 0.03                    # 小一点的驗證集

## LoRA 設定 - 針對 Qwen 架構優化
use_lora: true
lora_r: 16                          # 可以適當增加以捕捉更多中文特徵
lora_alpha: 32
lora_dropout: 0.05
lora_target_modules:                # Qwen 特定的目標模組
  - "q_proj"
  - "k_proj" 
  - "v_proj"
  - "o_proj"
  - "gate_proj"
  - "up_proj"
  - "down_proj"

## 量化設定 (必須)
use_4bit: true
bnb_4bit_quant_type: "nf4"
bnb_4bit_compute_dtype: "bfloat16"

## 日誌與保存
logging_steps: 5
save_steps: 50                     # 更頻繁的保存以監控中文訓練進度
save_total_limit: 3
report_to: "none"

## 評估設定
evaluation_strategy: "steps"
eval_steps: 50                     # 更頻繁評估中文能力
```

## 🏃 第四步：執行訓練

### 階段 1：通用中文指令跟隨
```powershell
python -m ops.training.pipeline `
    --base-model Qwen/Qwen-1.5-7B-Chat `
    --dataset shibing624/Chinese-Alpaca-Plus `
    --epochs 3 `
    --output-dir ./output/ops-chinese/stage1_general `
    --dataset_name shibing624/Chinese-Alpaca-Plus
```

### 階段 2：中文程式設計與技術文檔
```powershell
python -m ops.training.pipeline `
    --base-model ./output/ops-chinese/stage1_general/final_model `
    --dataset ./training_data/chinese_code `
    --epochs 2 `
    --output-dir ./output/ops-chinese/stage2_code `
    --dataset_name json
```

### 階段 3：中文工具使用與實務應用
```powershell
python -m ops.training.pipeline `
    --base-model ./output/ops-chinese/stage2_code/final_model `
    --dataset ./training_data/chinese_tools `
    --epochs 2 `
    --output-dir ./output/ops-chinese/stage3_tools `
    --dataset_name json
```

## 📊 第五步：中文能力評估

建立中文專項評估基準：

### 建立中文測試題組 (`C:\Users\jksdf\ops\benchmark\chinese_test_set.json`)
```json
[
  {
    "category": "中文知識理解",
    "question": "請用白話文解釋《論語》中『學而時習之，不亦說乎』的意思，並舉出現代生活中的例子。",
    "reference_keywords": ["學習", "複習", "快樂", "持續"]
  },
  {
    "category": "中文程式設計",
    "question": "用 Python 寫一個函式，判斷一個字串是否為回文（忽略大小寫和標點符號），並提供詳細的中文註解說明每一步的作用。",
    "expected_features": ["中文註解", "正確演算法", "邊界條件處理"]
  },
  {
    "category": "中文工具使用",
    "question": "請幫我：1) 建立一個名為 chinese_notes 的資料夾 2) 在裡面建立一個文件 today.txt，內容是今天的日期和星期 3) 顯示這個文件的完整路徑",
    "expected_tools": ["shell", "file"],
    "validation": "檢查資料夾和文件是否正確建立"
  },
  {
    "category": "中文長文理解與摘要",
    "question": "請閱讀以下中文段落並用三句話 summarise 其主要觀點：[在此插入一段約200字的中文科技新聞]",
    "expected_length": "3 sentences",
    "expected_content": ["主要事件", "時間地點", "意義影響"]
  },
  {
    "category": "中文創意與表達",
    "question": "以『春江花月夜』為題，寫一首五言絕句，要求嚴格平仄和押韻。",
    "format": "五言絕句",
    "constraints": ["平仄規則", "押韻要求"]
  }
]
```

### 執行中文評估
```powershell
python -m ops.benchmark `
    --model ./output/ops-chinese/stage3_tools/final_model `
    --test-set C:\Users\jksdf\ops\benchmark\chinese_test_set.json `
    --output ./results/chinese_eval_results.json `
    --language chinese
```

## 🦙 第六步：準備中文優化版 Ollama 部署

### 步驟 1：合併並準備最終模型
```powershell
# 合併 LoRA 權重
python -c "
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

base_model = AutoModelForCausalLM.from_pretrained(
    'Qwen/Qwen-1.5-7B-Chat', 
    torch_dtype=torch.bfloat16,
    trust_remote_code=True
)
model = PeftModel.from_pretrained(base_model, './output/ops-chinese/stage3_tools/final_model')
model = model.merge_and_unload()
model.save_pretrained('./output/ops-chinese/final_model')
tokenizer = AutoTokenizer.from_pretrained('Qwen/Qwen-1.5-7B-Chat')
tokenizer.save_pretrained('./output/ops-chinese/final_model')
"
```

### 步驟 2：建立中文優化的 Modelfile
```powershell
cd C:\Users\jksdf\ops\output\ops-chinese\final_model
@"
FROM ./
# 中文優化的聊天模板 - 適用於 Qwen 系列
TEMPLATE \"\"\"<|im_start|>system
你是 OPS 中文版，一個專門為中文理解和生成優化的開源 AI 助手。你具備：
1. 深厚的中文語言理解能力，包括文言文、網路用語、專業術語
2. 優秀的中文程式設計能力，能寫出帶有詳細中文註解的高質量代碼
3. 良好的中文工具使用能力，能在中文環境下有效使用檔案、網路、程式執行等工具
4. 豐富的中文文化知識，能理解和生成符合中文表達習慣的內容
<|im_end|>
<|im_start|>user
{{ .Prompt }}<|im_end|>
<|im_start|>assistant
{{ .Response }}<|im_end|>
\"\"\"
PARAMETER temperature 0.7
PARAMETER top_p 0.9
PARAMETER repeat_penalty 1.05
PARAMETER stop \"<|im_end|>\"
PARAMETER stop \"<|im_start|>\"
SYSTEM \"\"\"你是 OPS 中文版，一個專門為中文理解和生成優化的開源 AI 助手。你具備：
1. 深厚的中文語言理解能力，包括文言文、網路用語、專業術語
2. 優秀的中文程式設計能力，能寫出帶有詳細中文註解的高質量代碼
3. 良好的中文工具使用能力，能在中文環境下有效使用檔案、網路、程式執行等工具
4. 豐富的中文文化知識，能理解和生成符合中文表達習慣的內容。\"\"\"
"@ | Set-Content -Encoding utf8 Modelfile
```

### 步驟 3：建立並測試中文模型
```powershell
ollama create ops-chinese -f Modelfile
ollama run ops-chinese "請用中文解釋什麼是遞歸函式，並用 Python 寫個計算階乘的例子，請務必包括詳細的中文註解"
```

### 步驟 4：推送到 Ollama 模型庫（供他人使用）
```powershell
ollama login  # 使用您的 Ollama 帳號登入
ollama push your-username/ops-chinese:latest
# 或標註為特定版本
ollama push your-username/ops-chinese:v1.0-chinese-optimized
```

## 📈 第七步：持續改進與社區貢獻

### 建議的未來方向：
1. **多方言支持**：加入粵語、閩南語等方言理解能力
2. **專業領域深化**：醫療、法律、金融等專業中文能力
3. **多模態擴展**：結合中文圖像理解（如圖片描述、OCR）
4. **工具鏈本土化**：開發更多適合中文環境的專用工具
5. **社區協作**：建立中文開發者社區共同改進模型

### 評估基準建議：
定期在以下中文基準上評估您的模型：
- **C-Eval**：中國各學科水平考試基準
- **CMMLU**：中國大規模多任務語言理解基準
- **MMLU-Chinese**：中文版的巨大多任務語言理解基準
- **CEval-Geo**：中國地理知識基準
- **CMath**：中文數學問題解決基準
- **ChatBot Arena 中文版**：中文對話品質人間評估

## 💡 專業技巧與最佳實踐

### 訓練期間監控：
1. **中文 perplexity**：監控訓練和驗證集上的中文 perplexity 下降趨勢
2. **代碼生成品質**：定期抽樣檢查生成代碼的中文註準確性
3. **工具使用正確性**：檢查工具調用格式是否正確
4. **長文連貫性**：測試模型在較長中文對話中的上下文保持能力

### 常見問題與解決方案：
- **中文亂碼**：確認所有訓練數據和模型都使用 UTF-8 編碼
- **混合語言問題**：如果出現中英混亂，增加純中文訓練數據比例
- **文言文理解不足**：加入古文典籍數據如《論語》、《孟子》選段
- **網路用語貧弱**：加入微博、知乎、貼吧等社交媒體數據樣本

### 資源優化建議：
1. **如果顯存受限**：
   - 減少 batch_size 到 1
   - 增加 gradient_accumulation_steps 到 32-64
   - 使用 8-bit 量化代替 4-bit (雖然較慢但更穩定)

2. **如果訓練過慢**：
   - 確認使用了 GPU (nvidia-smi)
   - 檢查數據載入是否成為瓶頸 (考慮使用資料緩存)
   - 減少 tokenizer 的 padding 開銷

## 📚 參考資源與數據來源

### 中文訓練數據集：
- **shibing624/Chinese-Alpaca-Plus**：https://huggingface.co/datasets/shibing624/Chinese-Alpaca-Plus
- **PKU-Alignment/Beauty-Chinese-Instruction-Data**：https://huggingface.co/datasets/PKU-Alignment/Beauty-Chinese-Instruction-Data
- **augmxnt/gooseai-dev-cn**：中文寫作提示數據
- **lisijie/chinese-lora-alpaca**：中文 LoRA Alpaca 數據

### 中文基準測試：
- **C-Eval**：https://github.com/SUDA-EVL/C-Eval
- **CMMLU**：https://github.com/haonan-li/CMMLU
- **MMLU-Chinese**：各社區翻譯版本
- **AGIEval**：包含中文部分的通用人工智能評估

### 中文模型社區：
- **Qwen 社區**：https://github.com/QwenLM/Qwen
- **Chinese LLaMA & Alpaca 社區**：https://github.com/ymcui/Chinese-LLaMA-Alpaca
- **OpenChineseLLM**：https://github.com/imoneoi/openchat

## 🎉 結論

通過遵循本指南，您將能夠訓練出一個：
- ✅ 具備深厚中文理解和生成能力的開源 AI 模型
- ✅ 能在中文環境下有效使用工具完成實際任務
- ✅ 完全開放源碼，可自由使用、修改和分發
- ✅ 可透過 Ollama 輕鬆部署和分享給他人使用
- ✅ 有潛力在中文任務上達到或超越閉源模型如 Claude Opus 的表現

這個模型不僅是一個技術成就，更是推動中文 AI 普及和發展的重要貢獻。讓我們一起讓開源 AI 在中文世界裡發光發熱！

--- 

**準備開始？** 您可以：
1. 首先安裝依賴：`pip install -e "C:/Users/jksdf/ops[training]"`
2. 準備中文訓練數據（參考上述數據來源）
3. 根據您的硬體選擇合適的基礎模型開始訓練
4. 隨時回顧本指南以獲取具體的執行步驟

祝您訓練順序，創造出優秀的中文 OPS 模型！ 🚀