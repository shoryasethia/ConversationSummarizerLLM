**Clone Repo**
```
git clone https://github.com/shoryasethia/ConversationSummarizerLLM
```

# AIM : Conversation Summarization for Context Window Optimization in RAG-LLMs
Models like FLAN-T5, Pegasus when fine-tuned on specific datasets like DialogSum, are specialized to handle summaries of conversations. This approach can indeed be highly useful in scenarios where managing the context window size is crucial, such as with Retrieval-Augmented Generation (RAG) models and other language models with limited context windows.
> **This repo is simpler execution of the ideas/methods mentioned in [this](https://community.openai.com/t/how-to-construct-the-prompt-for-a-standalone-question/177048) OpenAI's blog**

**Context Window Optimization with Summarization**
By summarizing conversations, one can effectively reduce the amount of text that needs to be fed into the context window of the language model. This not only helps in managing the limited context size but also ensures that the most relevant information is retained. Here’s how this approach can be beneficial and implemented:

- **Context Window Constraints:** Most large language models, including GPTs, Gemini, BERT etc;, have a maximum token limit of few 1000 tokens for the context window. Summarizing previous conversations allows you to fit more relevant content within this limit.

- **Enhanced Focus:** Summarization distills the key points of a conversation, enabling the model to focus on the most important information without being overwhelmed by less relevant details.

- **Memory Efficiency:** By reducing the amount of text, we can make more efficient use of the model's memory, which can lead to faster processing and reduced computational load.

**Implementation Strategy**
- **Summarization Model:** Fine-tune a summarization model like Flan-T5 on the conversation or dialogue based datasets like DialogSum dataset.

**Integrate Summarization with RAG:**
- **Summarize Historical Conversations:** Before feeding past conversations into the context window, use the summarization model to generate concise summaries.
- **Feed Summarized Conversations:** Integrate these summaries into the context window along with the most recent interaction to maintain continuity.

> This method can be particularly beneficial for applications involving long-term user interactions, such as customer support, personal assistants, and other conversational AI systems. By focusing on summarization, you can effectively manage and enhance the context provided to the model, leading to more coherent and contextually appropriate responses.


# For this I Fine Tuned `google/pegasus-xsum` and `google/flan-t5-base` pre-trained LM on `knkarthick/dialogsum` dataset

**Load Dataset**
```
huggingface_dataset_name = "knkarthick/dialogsum"

dataset = load_dataset(huggingface_dataset_name)
```
or run
```
git clone https://huggingface.co/datasets/knkarthick/dialogsum
```

**Load Pre-Trained Model**
```
model_name = 'google/pegasus-xsum'

original_model = AutoModelForSeq2SeqLM.from_pretrained(model_name, torch_dtype = torch.bfloat16)
```
```
model_name = 'google/flan-t5-base'

original_model = AutoModelForSeq2SeqLM.from_pretrained(model_name, torch_dtype = torch.bfloat16)
```
# PEFT (Parameter Efficient Fine Tuning) - LoRA
## **LoRA Cofigurations I used for Fine Tuning**

**To get target modules, run following**
```
print(original_model)
```
Output for `google/pegasus-xsum`
```
PegasusForConditionalGeneration(
  (model): PegasusModel(
    (shared): Embedding(96103, 1024, padding_idx=0)
    (encoder): PegasusEncoder(
      (embed_tokens): Embedding(96103, 1024, padding_idx=0)
      (embed_positions): PegasusSinusoidalPositionalEmbedding(512, 1024)
      (layers): ModuleList(
        (0-15): 16 x PegasusEncoderLayer(
          (self_attn): PegasusAttention(
            (k_proj): Linear(in_features=1024, out_features=1024, bias=True)
            (v_proj): Linear(in_features=1024, out_features=1024, bias=True)
            (q_proj): Linear(in_features=1024, out_features=1024, bias=True)
            (out_proj): Linear(in_features=1024, out_features=1024, bias=True)
          )
          (self_attn_layer_norm): LayerNorm((1024,), eps=1e-05, elementwise_affine=True)
          (activation_fn): ReLU()
          (fc1): Linear(in_features=1024, out_features=4096, bias=True)
          (fc2): Linear(in_features=4096, out_features=1024, bias=True)
          (final_layer_norm): LayerNorm((1024,), eps=1e-05, elementwise_affine=True)
        )
      )
      (layer_norm): LayerNorm((1024,), eps=1e-05, elementwise_affine=True)
    )
    (decoder): PegasusDecoder(
      (embed_tokens): Embedding(96103, 1024, padding_idx=0)
      (embed_positions): PegasusSinusoidalPositionalEmbedding(512, 1024)
      (layers): ModuleList(
        (0-15): 16 x PegasusDecoderLayer(
          (self_attn): PegasusAttention(
            (k_proj): Linear(in_features=1024, out_features=1024, bias=True)
            (v_proj): Linear(in_features=1024, out_features=1024, bias=True)
            (q_proj): Linear(in_features=1024, out_features=1024, bias=True)
            (out_proj): Linear(in_features=1024, out_features=1024, bias=True)
          )
          (activation_fn): ReLU()
          (self_attn_layer_norm): LayerNorm((1024,), eps=1e-05, elementwise_affine=True)
          (encoder_attn): PegasusAttention(
            (k_proj): Linear(in_features=1024, out_features=1024, bias=True)
            (v_proj): Linear(in_features=1024, out_features=1024, bias=True)
            (q_proj): Linear(in_features=1024, out_features=1024, bias=True)
            (out_proj): Linear(in_features=1024, out_features=1024, bias=True)
          )
          (encoder_attn_layer_norm): LayerNorm((1024,), eps=1e-05, elementwise_affine=True)
          (fc1): Linear(in_features=1024, out_features=4096, bias=True)
          (fc2): Linear(in_features=4096, out_features=1024, bias=True)
          (final_layer_norm): LayerNorm((1024,), eps=1e-05, elementwise_affine=True)
        )
      )
      (layer_norm): LayerNorm((1024,), eps=1e-05, elementwise_affine=True)
    )
  )
  (lm_head): Linear(in_features=1024, out_features=96103, bias=False)
)
```
> I have just used q_proj and v_proj, and paramters of k_proj, other dense layer or any other parameters are kept freezed
> 
```
lora_config = LoraConfig(
    r=32, #Rank
    lora_alpha=32,
    target_modules=['q_proj',
                    'v_proj',],
    bias="none",
    lora_dropout=0.05,  # Conventional
    task_type=TaskType.SEQ_2_SEQ_LM   #pegasus
)
```
## Results

| Model    | Rouge1   | Bleu    | Notebook    | FT Checkpoints    |
|-------------|-------------|-------------|-------------|-------------|
| pegasus-xsum|0.1960458975298376| 0.39602403633577834 | - | - |
| pegasus-peft-lora-dialogsum| 0.25098017644738956| 0.43713359476346771| [.ipynb](https://github.com/shoryasethia/ConversationSummarizerLLM/blob/main/Pegasus_PEFT_LoRA_DialogSum.ipynb) , [Google Colab](https://colab.research.google.com/drive/1F_8nRmp_b7bHFSZh7c7G04Fg3WrQtrFa)| [🔗](https://github.com/shoryasethia/ConversationSummarizerLLM/blob/main/pegasus-peft-lora-dialogsum-checkpoints.zip)|
| flan-t5-base| 0.37651698036006545| 0.1550404559570942| - | - |
| flan-t5-peft-lora-dialogsum| 0.42925850100353405| 0.19056854347856564| [.ipynb](https://github.com/shoryasethia/ConversationSummarizerLLM/blob/main/FLAN-T5-PEFT-LoRA-DialogSum.ipynb) , [Google Colab](https://colab.research.google.com/drive/1YRYPBU4kvRzCXuDh8VKvGIlSQ3nqxQA_?authuser=1)| [🔗](https://github.com/shoryasethia/ConversationSummarizerLLM/blob/main/flan-t5-peft-lora-dialogsum-checkpoints.zip)|

> **Fine tuning Pegasus :**
> Absolute percentage improvement of PEFT MODEL over ORIGINAL MODEL
> - rouge1: 28.5065721082448025%
> - rouge2: 1.3583623289828225%
> - rougeL: 2.7046748437383954%
> - rougeLsum: 9.6404848848455602%

> **Fine tuning Flan-T5 :**
> Absolute percentage improvement of PEFT MODEL over ORIGINAL MODEL
> - rouge1: 4.72584793565314%
> - rouge2: 1.13980578970623214%
> - rougeL: 1.631214528504942%
> - rougeLsum: 2.5820780787278435%

> Reference/Proof of these values : https://github.com/shoryasethia/ConversationSummarizerLLM/tree/main/output

If there was anything that helped you, then do hit a like.  
 
