# HW 2

## Краткий результат

Для модели из первого задания исследованы настройки batch size и gradient
accumulation, FlashAttention2, padding-free batching, sequence packing и Liger
kernels. Baseline и итоговая конфигурация обучались с нуля по 300 секунд на
одной и той же NVIDIA A100 80 GB.

Итоговая конфигурация достигла **21 533.91 токенов/с** против
**6 809.22 токенов/с** у baseline:

```text
S = 21 533.9059 / 6 809.2214 = 3.1625
```

Итоговое ускорение — **3.16×**, или **+216.2%**. При этом validation loss
улучшился с **6.6664** до **5.5326**, а peak allocated GPU memory снизилась с
**41.24** до **38.46 GiB**.

## Сетап

### Оборудование и окружение

- GPU: NVIDIA A100 80 GB PCIe;
- одна GPU для всех запусков;
- PyTorch 2.6.0, CUDA 11.8;
- Transformers 4.56.2;
- TRL 0.23.1;
- FlashAttention 2.7.3;
- Liger Kernel 0.6.2;
- BF16 и TF32;

## Конфигурации экспериментов

Все короткие эксперименты запускались на 90 секунд. Для каждого запуска
использовался отдельный `output_dir`. 

| Конфигурация | `output_dir` | Attention | Batch | GA | Packing | Padding-free | Время, с |
|---|---|---|---:|---:|---|---|---:|
| Baseline | `results/baseline` | SDPA | 32 | 2 | False | False | 300 |
| Batch 32, GA1 | `results/bs32_ga1` | SDPA | 32 | 1 | False | False | 90 |
| Batch 40, GA1 | `results/bs40_ga1` | SDPA | 40 | 1 | False | False | 90 |
| FlashAttention2 | `results/flash` | FlashAttention2 | 32 | 2 | False | False | 90 |
| FlashAttention2 + GA1 | `results/flash_bs32_ga1` | FlashAttention2 | 32 | 1 | False | False | 90 |
| FlashAttention2 + padding-free | `results/flash_padding_free` | FlashAttention2 | 32 | 2 | False | True | 90 |
| FlashAttention2 + packing | `results/flash_packing` | FlashAttention2 | 32 | 2 | True | False | 90 |
| FlashAttention2 + packing + GA1 | `results/flash_packing_ga1` | FlashAttention2 | 32 | 1 | True | False | 90 |
| Final | `results/final_best` | FlashAttention2 | 32 | 1 | True | False | 300 |

## Результаты

`Speedup` для всех строк рассчитан относительно 300-секундного baseline.
Validation loss коротких и длинных запусков приведён для полноты, но напрямую
сравнивать его корректнее между экспериментами одинаковой длительности.

| Конфигурация | Факт. время, с | Tokens/s | Speedup | Peak alloc., GiB | Peak reserv., GiB | Val loss | Optimizer steps |
|---|---:|---:|---:|---:|---:|---:|---:|
| Baseline | 300.38 | 6 809.22 | 1.000× | 41.24 | 43.44 | 6.6664 | 357 |
| Batch 32, GA1 | 90.08 | 7 176.86 | 1.054× | 39.43 | 40.37 | 7.8012 | 223 |
| Batch 40, GA1 | 90.42 | 7 066.05 | 1.038× | 47.94 | 50.08 | 7.9184 | 176 |
| FlashAttention2 | 90.22 | 7 698.38 | 1.131× | 39.30 | 41.60 | 8.1010 | 119 |
| FlashAttention2 + GA1 | 90.40 | 7 520.35 | 1.104× | 37.51 | 38.53 | 7.7650 | 234 |
| FlashAttention2 + padding-free | 90.11 | 19 384.95 | 2.847× | **15.11** | **21.24** | 6.8772 | 304 |
| FlashAttention2 + packing | 90.29 | **22 650.83** | **3.327×** | 40.15 | 76.20 | 8.2160 | 63 |
| FlashAttention2 + packing + GA1 | 90.17 | 22 408.68 | 3.291× | 38.46 | 76.20 | 7.8731 | 125 |
| Final | 300.54 | **21 533.91** | **3.162×** | **38.46** | 76.20 | **5.5326** | **404** |

## Наблюдения

### 1. Уменьшение gradient accumulation полезно не во всех режимах

На SDPA переход с GA2 на GA1 увеличил throughput на 5.4% и снизил peak
allocated memory с 41.24 до 39.43 GiB. Более частые optimizer steps при этом
дали 223 шага за 90 секунд. Однако увеличение batch size с 32 до 40 оказалось
невыгодным: throughput снизился до 7 066 токенов/с, а память выросла до
47.94 GiB. Больший microbatch не компенсировал рост стоимости одного шага.

### 2. FlashAttention2 дал отдельный, но умеренный выигрыш

При неизменных batch 32 и GA2 замена SDPA на FlashAttention2 повысила скорость
до 7 698 токенов/с, то есть на 13.1% относительно baseline, и немного снизила
allocated memory. Сочетание FlashAttention2 с GA1 не сложило ускорения
линейно: скорость составила 7 520 токенов/с, зато память снизилась до
37.51 GiB и число optimizer steps почти удвоилось. Это показывает, что
стоимость более частого шага оптимизатора становится заметной после ускорения
forward/backward.

### 3. Исключение padding оказалось главным источником ускорения

Padding-free режим в сочетании с FlashAttention2 достиг 19 385 токенов/с —
2.85× baseline. Одновременно peak allocated memory упала до 15.11 GiB. Причина
в том, что модель перестала выполнять вычисления над pad-позициями коротких
абзацев, а метрика учитывает только полезные target-токены.

### 4. Packing дал максимальный throughput, но повысил reserved memory

Sequence packing заполняет последовательности длины 512 полезными токенами и
дал лучший короткий результат — 22 651 токен/с, или 3.33× baseline. Цена этого
ускорения — рост peak reserved memory до 76.20 GiB. При GA2 за короткий запуск
произошло только 63 optimizer steps, поэтому модель оставалась внутри
200-шагового LR warmup и показала более высокий validation loss.

Сочетание packing с GA1 сохранило почти всю скорость — 22 409 токенов/с — но
удвоило число optimizer steps, снизило allocated memory и улучшило validation
loss. Поэтому для финального запуска выбрана именно эта конфигурация: она
уступает варианту GA2 примерно 1%, но даёт существенно более подходящую
динамику обучения.

