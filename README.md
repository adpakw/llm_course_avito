# HW 1 LLM Course 


## 1 experiment (baseline)

```
TRAINING_CONFIG = {
    'output_dir': f'{OUTPUT_DIR}/gpt2-1b-russian',
    'optim': 'adamw_torch_fused',
    'num_train_epochs': 1,
    'per_device_train_batch_size': 4,
    'save_steps': 100,
    'save_total_limit': 2,
    'learning_rate': 5e-5,
    'weight_decay': 0.01,
    'warmup_steps': 200,
    'logging_steps': 1,
    'eval_steps': 100,
    'eval_strategy': 'steps',
    'load_best_model_at_end': True,
    'metric_for_best_model': 'eval_loss',
    'bf16': True,
    'tf32': True,
    'gradient_checkpointing': False,
    'gradient_accumulation_steps': 1,
    'dataloader_num_workers': 4,
    'torch_compile': False,
    'report_to': 'none',
}
```

**Final Eval Loss: 7.0030 | Perplexity: 1099.93**
график лосса в `1 exp/loss_curve.html`


```
Prompt: Столица Франции -
Generation: Столица Франции -  —,-У.
 : 


 Официаль,8 — 4. Д. — М.: В. — американский. — 7-7-2-8 человек. В. А. В. — М.:
 А. — № А
--------------------------------------------------
Prompt: LLM работают на архитектуре 
Generation: LLM работают на архитектуре.
Биография 
Описание 

Известные в 2011 году.

Заёл на 2000 года — 15 сентября.

Входит также 

В 2011 г. — за этом в 15 мая 1943 года года. — за «Сюская»
 
--------------------------------------------------
Prompt: Самая большая страна в мире 
Generation: Самая большая страна в мире  «Бна» в «Бол»)» и к в США, в 1998-Н России. После этого «Л-дан в честь-Т и в матче, при 3-У-е в этом и до 1, входит в составе 14 лет.

В 2008 году она из 2014 года,
--------------------------------------------------
Prompt: Первый закон Ньютона гласит 
Generation: Первый закон Ньютона гласит  

Сверные 


Гот на 2020 года

Аль апреля 

 

В 2018 года в 1-
  () 
  в 1998 по Европы в

В сборной 
 
Пка 

Ф. также 

Ссылки 

Жп на
--------------------------------------------------
```

## 2 experiment

```
TRAINING_CONFIG = {
    'output_dir': f'{OUTPUT_DIR}/gpt2-1b-russian',
    'optim': 'adamw_torch_fused',
    'num_train_epochs': 1,
    'per_device_train_batch_size': 16,
    'save_steps': 100,
    'save_total_limit': 2,
    'learning_rate': 5e-5,
    'weight_decay': 0.01,
    'warmup_steps': 200,
    'logging_steps': 1,
    'eval_steps': 100,
    'eval_strategy': 'steps',
    'load_best_model_at_end': True,
    'metric_for_best_model': 'eval_loss',
    'bf16': True,
    'tf32': True,
    'gradient_checkpointing': False,
    'gradient_accumulation_steps': 1,
    'dataloader_num_workers': 4,
    'torch_compile': False,
    'report_to': 'none',
}
```

** **
график лосса в ``

```

```