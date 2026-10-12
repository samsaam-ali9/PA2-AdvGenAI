\---



base\_model: HuggingFaceTB/SmolLM2-360M-Instruct

library\_name: peft

model\_name: persona\_lora

tags:



\* dpo

\* lora

\* peft

\* transformers

\* trl

&#x20; pipeline\_tag: text-generation



\---



\# Persona LoRA-DPO Adapter



This directory contains a LoRA adapter trained on persona-style preference data using Direct Preference Optimization (DPO). The base model is \[HuggingFaceTB/SmolLM2-360M-Instruct](https://huggingface.co/HuggingFaceTB/SmolLM2-360M-Instruct).



The experiment explores whether preference optimization can steer a model toward a target response style while preserving helpfulness and appropriate behavior.



\## Quick start



The following example loads the base model and this trained LoRA adapter, then generates a response to a sample question.



Run the code from the \*\*root of the PA2 repository\*\*, where `outputs/persona\_lora/` is the adapter directory.



```python

import torch

from transformers import AutoModelForCausalLM, AutoTokenizer

from peft import PeftModel



base\_model\_id = "HuggingFaceTB/SmolLM2-360M-Instruct"

adapter\_path = "outputs/persona\_lora"



tokenizer = AutoTokenizer.from\_pretrained(base\_model\_id)

base\_model = AutoModelForCausalLM.from\_pretrained(base\_model\_id)



model = PeftModel.from\_pretrained(base\_model, adapter\_path)



device = "cuda" if torch.cuda.is\_available() else "cpu"

model = model.to(device)

model.eval()



question = "If you had a time machine, but could only go to the past or the future once and never return, which would you choose and why?"



messages = \[{"role": "user", "content": question}]



prompt = tokenizer.apply\_chat\_template(

&#x20;   messages,

&#x20;   tokenize=False,

&#x20;   add\_generation\_prompt=True,

)



inputs = tokenizer(prompt, return\_tensors="pt").to(device)



with torch.no\_grad():

&#x20;   output = model.generate(

&#x20;       \*\*inputs,

&#x20;       max\_new\_tokens=128,

&#x20;       do\_sample=False,

&#x20;   )



response = output\[0]\[inputs\["input\_ids"].shape\[1]:]

print(tokenizer.decode(response, skip\_special\_tokens=True))

```



The base model must be available locally or downloadable from Hugging Face. Install PyTorch, Transformers, and PEFT in your Python environment before running the example.



\## Training procedure



The adapter was trained using TRL's `DPOTrainer` with PEFT/LoRA. DPO optimizes preferences between chosen and rejected responses without requiring a separately trained reward model.



\### Dataset and configuration



| Setting                        | Value                                 |

| ------------------------------ | ------------------------------------- |

| Base model                     | `HuggingFaceTB/SmolLM2-360M-Instruct` |

| Training examples              | 198                                   |

| Validation examples            | 22                                    |

| Epochs                         | 3                                     |

| Optimizer                      | AdamW (`adamw\_torch`)                 |

| Learning rate                  | `5e-5`                                |

| Scheduler                      | Linear                                |

| Warmup ratio                   | 0.0                                   |

| Weight decay                   | 0.0                                   |

| DPO beta                       | 0.1                                   |

| Per-device training batch size | 2                                     |

| Gradient accumulation steps    | 4                                     |

| Effective batch size           | 8                                     |

| Maximum sequence length        | 512                                   |

| Maximum prompt length          | 256                                   |

| Random seed                    | 42                                    |



\### LoRA configuration



| Setting        | Value                                  |

| -------------- | -------------------------------------- |

| Rank (`r`)     | 8                                      |

| Alpha          | 16                                     |

| Dropout        | 0.05                                   |

| Target modules | `q\_proj`, `k\_proj`, `v\_proj`, `o\_proj` |



\### Recorded training results



The saved training result records a final training loss of approximately `0.5320` and a best metric of approximately `0.4635`. The saved Trainer history records evaluation reward accuracy of `1.0` at the three evaluation checkpoints.



These results should be interpreted cautiously because the validation split contained only 22 examples. The separate persona pairwise evaluation used four prompts, which is too small to establish a reliable overall preference.



\## Files



This directory contains the trained adapter and its supporting configuration and tokenizer files.



\* `adapter\_config.json`: LoRA adapter configuration.

\* `adapter\_model.safetensors`: trained adapter weights.

\* `run\_config.json`: experiment configuration.

\* `result.json`: saved training summary.

\* Tokenizer and chat-template files: supporting files for loading the adapter.



The full base model is not included in this directory.



\## Framework versions



The recorded environment used:



\* PyTorch: `2.11.0+cu128`

\* Transformers: `4.57.6`

\* TRL: `0.26.2`

\* PEFT: `0.20.0`

\* Datasets: `4.8.5`



\## References



\*\*DPO\*\*



Rafailov, R. et al. (2023). \*Direct Preference Optimization: Your Language Model is Secretly a Reward Model\*. NeurIPS 2023. \[Paper](https://huggingface.co/papers/2305.18290)



\*\*TRL\*\*



Hugging Face. \[TRL documentation and repository](https://github.com/huggingface/trl)



\*\*Base model\*\*



Hugging Face. \[SmolLM2-360M-Instruct model card](https://huggingface.co/HuggingFaceTB/SmolLM2-360M-Instruct)



