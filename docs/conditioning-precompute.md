# Precompute prompts: границы и сопровождение

`Precompute prompts` — узкий opt-in режим скрипта **Prompts from File or
Textbox**. Он предназначен для нескольких текстовых txt2img заданий на моделях,
где text encoder и generator не помещаются одновременно в VRAM. Это не новый
интерфейс scripts и не общий кэш Forge.

Этот документ описывает факты, проверенные в этой ветке, инварианты реализации
и известные ограничения. При расширении режима нельзя считать непокрытые ниже
случаи безопасными только потому, что они «похожи» на разрешённый.

## Pipeline и commit point

`scripts/prompts_from_file.py` сначала строит request-local `PromptJob` для
каждой строки. Старые аргументы скрипта сохраняют совместимость: новый checkbox
идёт последним и имеет default `False`.

Для первого реального задания обычный `process_images()` выполняет загрузку
модели, нормализацию размеров, `setup_prompts()`, `scripts.process()`,
`before_process_batch()`, extra-network parsing/activation и `process_batch()`.
Непосредственно **после `process_batch()` и перед `p.setup_conds()`**
`modules.processing.process_images_inner()` вызывает
`conditioning_precompute.maybe_precompute(p)`.

Это commit point. До кодирования сравниваются materialized positive/negative
prompts первого `PromptJob` и реальные `p.prompts`/`p.negative_prompts`. Если
они не совпали, cache очищается, выводится одна причина fallback и текущий
`process_images()` продолжает штатно.

При успехе `_materialise_prompts()` применяет штатные функции
`StyleDatabase.apply_*_styles_to_prompt()`, `processing_scripts.comments.strip_comments()`
и extra-network parser к каждой строке. Только затем временные shallow-копии
`p` последовательно вызывают штатный `setup_conds()` для всех строк и `n_iter`; он создаёт
`SdConditioning` и обрабатывает emphasis, `BREAK`, schedules (`[A:B:0.5]`,
`[A|B]`), `AND`, CFG=1 и engine-specific metadata. Sampling этих временных
объектов не запускается.

Порядок materialization **Styles → comments → extra-network parsing → `setup_conds()`** — обязательный
инвариант. Не удаляйте `_materialise_prompts()` и не переносите Styles/comments
в `setup_conds()`: в текущем Forge этот метод получает уже materialized batch и
сам их не применяет.
После подготовки text encoder освобождается, а первый настоящий batch продолжает
свою уже начатую обычную обработку; последующие jobs также проходят обычный
pipeline, но `get_conds_with_caching()` сначала получает request-local entry.

Нельзя «поставить на паузу» или переписать `process_images()`: именно его hooks
и порядок mutations определяют фактический текст/conditioning. Реализация
поэтому использует единственную необязательную точку перед `setup_conds`, не
перезапускает выполненные hooks и не вводит новый ScriptRunner protocol.

## Инварианты дизайна

### Scope и allowlist

Режим разрешён только для обычного txt2img без Hires, refiner, PiD,
image/reference/video passes. Разрешённые engines задаются строгими парами
`(module, class)`, а не только module:

- `sd15.StableDiffusion`, `sdxl.StableDiffusionXL` (не
  `sdxl.StableDiffusionXLRefiner`), `flux.Flux`, `flux2.Flux2`, `chroma.Chroma`,
  `zimage.ZImage`, `krea.Krea2`.

Также отклоняются inpaint/video/reference state. Krea2 технически несёт
`is_wan=True` из-за WanVAE tensor layout, но это текстовый T2I engine и
разрешён. Реальный Wan остаётся запрещённым.

Для Flux2/Klein `dynamic_args.klein` сам по себе не является причиной запрета:
он также обозначает выбранное семейство. Проверяются реальные `ref_latents`,
references/initial latents и edit/kontext/anima state; ImageStitch должен быть
выключен и не иметь cached state. Аналогично Krea2 разрешён только без
reference latents.

Из построчных аргументов допускаются только `prompt`, `negative_prompt`,
`seed`, `subseed`, `batch_size`, `n_iter`. Любой иной аргумент переключает весь
request на normal path. Один checkpoint/TE configuration должен оставаться
общим.

`forge_additional_modules` требует особой осторожности: Forge использует этот
список и для standalone text encoder. Допускаются только известный VAE или файл
под `models/text_encoder`; любой другой module считается LoRA/неизвестной
модификацией. Разрешено любое количество встроенных Forge `<lora:...>` только в
общем основном positive prompt: порядок, имена, positional/named arguments и
TE/UNet multipliers должны быть одинаковыми для всех jobs. LoRA в построчном
prompt, negative prompt, materialized Style или глобальной `opts.sd_lora`, а
также любой другой extra-network tag вызывают fallback.

Always-on scripts сверяются с path+class allowlist. Известные special scripts
(Refiner, Extra Options, ControlNet, ImageStitch, NeverOOM, Spectrum,
RadialAttention, MultiDiffusion, Soft Inpainting, PiD) допустимы только в
явно безопасном выключенном состоянии. Незнакомый script, callback generation
или extra-network registry entry означает fallback. Нельзя определять
«выключенность» неизвестного script по его первому аргументу. Это compatibility
allowlist проверенной Forge-конфигурации, а не защита от произвольного
monkey-patching.

### Изоляция, ключи и metadata

`copy.copy(p)` намеренно не копирует модели и ScriptRunner, но он разделяет
mutable state. Каждый PromptJob получает отдельные conditioning caches,
`override_settings`, `styles`, `extra_generation_params`, `comments`,
`extra_result_images`, `latents_after_sampling`, `pixels_after_sampling` и
color-correction state. При добавлении нового mutable field в processing class
нужно решить, должен ли он быть изолирован здесь.

Prepared conditioning хранится только на CPU. Рекурсивный snapshot клонирует
tensors, dict/list/namedtuple/object metadata и запоминает исходное устройство
каждого tensor. Cache hit создаёт новую независимую копию на исходных devices;
sampling hooks не могут модифицировать CPU entry.

Keys являются immutable value snapshots, а не ссылками на `p` lists/dicts. В
них входят positive/negative type, prompts/metadata, steps, dimensions,
extra-network data, loaded model+CLIP identity, relevant conditioning options,
additional modules, фактически активный `current_lora_hash`/online-mode и
textual-inversion embedding filesystem snapshot. `ExtraNetworkParams`
сериализуются структурно, а не через object `repr`, поэтому повторное штатное
parsing создаёт тот же key, но смена порядка или multiplier — другой. Mismatch
в lookup, смена embeddings или поздняя смена prompt/model/options очищает все
entries только данного request и вызывает обычное TE encoding. Дедупликация
работает для полностью одинаковых conditioning requests, включая negative
prompt.

Snapshot embeddings, созданный вместе с request context, является только
предварительным. Forge присваивает `dynamic_args.embedding_dir` при загрузке
первой реальной модели, когда context уже существует. Поэтому authoritative
snapshot повторно снимается в conditioning commit point: после загрузки модели
и `process_batch()`, непосредственно перед массовым TE. Без этого переход от
неустановленного dynamic path к штатному каталогу embeddings ошибочно выглядит
как изменение textual inversion и вызывает fallback при первом cache lookup.

`sd_model.extra_generation_params` принадлежит shared engine. Временный
`setup_conds()` очищает его для каждого prepared entry, snapshot сохраняет
сгенерированную metadata, а исходная metadata первого реального batch
восстанавливается в `finally`. При cache hit сохранённая metadata объединяется
так же, как при обычном cache hit. Не заменяйте это простым assignment, иначе
metadata может утечь между jobs.

После successful prepare нельзя пользоваться `unload_model()`: в данной Forge
ветке он может лишь удалить bookkeeping entry. Используется
`memory_management.free_memory(1e30, clip.patcher.load_device)`, который
вызывает реальный `LoadedModel.model_unload`; следующие sampling passes сами
загружают generator. Это не обещает устранить остальные VAE/model swaps.

Interrupt прекращает prepare и не запускает sampling; Skip не поглощается и
возвращает request на штатный путь. MemoryError/CUDA OOM — fallback. Другие
ошибки text encoder намеренно не скрываются. Context очищается в `finally`.

## Наблюдённые тестовые факты

Unit test расположен в `tests/test_conditioning_precompute.py`. Он проверяет:

- несколько общих LoRA из main prompt и запрет тегов из других источников;
- стабильный структурный key для повторно распарсенных `ExtraNetworkParams`;
- invalidation при изменении фактически активного LoRA state;
- CPU snapshot остаётся неизменным после mutation returned object;
- изменение conditioning input отключает только request-local cache;
- parser schedules/`AND` сохраняет ожидаемую структуру;
- strict engine allowlist разрешает SDXL base и отвергает SDXL refiner и
  неизвестный class из того же module.

GPU smoke был выполнен на RTX 4060 Ti 16 GB с установленными Krea2 Turbo int4,
Qwen3VL FP8 text encoder и Qwen2D VAE. Для трёх обычных строк, fixed seed 123,
512x512, Euler/Krea2 и 1 step baseline и precompute дали 3/3 идентичных RGB
pixel SHA-256, identical infotexts и `all_prompts`. Log показал загрузку
`JointTextEncoder` до `Precomputed prompts: 1/3..3/3`, затем загрузку `KModel`;
новой TE load во время трёх sampling passes не наблюдалось.

После добавления common-LoRA support отдельный GPU smoke на той же конфигурации
использовал три LoRA из основного prompt с weight 0.1 и три строки. Forge
успешно загрузил все три LoRA, выполнил `Precomputed prompts: 1/3..3/3` между
единственной загрузкой `JointTextEncoder` и загрузкой `KModel`, а затем завершил
все три sampling passes без повторной TE load. Все infotexts содержали одинаковые
три LoRA hashes. Этот smoke проверял pipeline и metadata с `send_images=false`;
отдельное pixel-сравнение LoRA baseline/precompute не выполнялось.

Это smoke, а не доказательство совместимости с каждым script/extension/model.
Полная live matrix сторонних extensions не выполнялась; неизвестные участники
поэтому должны остаться fallback, пока не добавлен целевой integration test.

## Known limitations и test environment

Forge runtime нельзя корректно поднимать простым import processing в чистом
venv: `modules_forge.initialization.initialize_forge()` добавляет vendored
`modules_forge/packages`, включая `k_diffusion` и `gguf`. Отсутствие pip package
`k_diffusion` не доказывает, что Forge runtime недоступен.

В данной config `samples_format=avif`; generation завершается, но API serializer
может вернуть `Invalid image format` после неё, потому что API encoder AVIF не
поддерживает. Для smoke не меняйте `config.json`: запускайте disposable
in-process API и заменяйте только `modules.api.api.encode_pil_to_base64` на
in-memory PNG encoder, либо используйте `send_images=false`, если pixels не
нужно сравнивать.

На этой Krea2 runtime обычный sampling prompt с `AND` дошёл до существующей
Forge ошибки `prompt_parser.reconstruct_multicond_batch`: 3-D tensor достигает
`repeat` с недостаточным числом dimensions. Это не было исправлено этой
фичей. Parser structure тестируется unit test; Krea2 pixel baseline для `AND`
не является подтверждённым supported scenario до отдельной Forge fix.

## Как распознать регрессию

- Checkbox `Precompute prompts` добавляет ровно один DEBUG result event на
  request. При успешном завершении он имеет вид
  `Prompts from File precompute result: success; batch precompute completed:
  prepared N conditioning entries across M jobs; sampling will use the
  request-local cache.` При initial incompatibility, commit-point mismatch,
  OOM, Skip, interrupt **во время подготовки** или позднем cache miss он имеет
  вид `Prompts from File precompute result: fallback; reason=...; legacy
  fallback enabled with TE encoding per job.` Эти DEBUG сообщения намеренно не
  являются новым
  INFO/WARNING marker и не выводятся, когда checkbox выключен.
- Result event эмитится в cleanup request-local context: это позволяет позднему
  cache miss заменить потенциальный success единым fallback result, а не
  создавать success/fallback spam. Когда context нельзя создать из-за initial
  compatibility check, тот же fallback event эмитится сразу.
- В precompute log text encoder должен завершиться для всех entries до первого
  `Requested to load KModel`; повторный TE load во время ordinary sampling
  означает cache miss/fallback или неверный key.
- При fixed seed baseline и precompute должны иметь одинаковые prompts,
  infotexts и pixels для выбранной supported configuration.
- Изменение prompt, model, Styles, TE option или embedding между prepare и hit
  должно дать одно ясное fallback warning, не second generation и не stale hit.
- Для общей LoRA сверяются два независимых уровня. Parsed configuration
  доказывает одинаковый source contract всех jobs, а `current_lora_hash` после
  реальной Forge activation доказывает одинаковое фактическое состояние модели.
  Нельзя заменить одну из этих проверок другой: одинаковый текст может не
  загрузиться, а одинаковый runtime state не доказывает происхождение тегов.
- Не используйте `repr(ExtraNetworkParams)` в ключах: стандартный object repr
  содержит identity и даст ложный cache miss после повторного parsing.
- Проверяйте, что CPU stored tensor не изменился после mutation returned
  conditioning, и что returned tensor лежит на требуемом device.
- Новый script/callback/extra-network должен отказать безопасно, пока его
  conditions не описаны и не покрыты тестом.

## Безопасное расширение allowlist

1. Установите точные `(module, class)` и не расширяйте allowlist module-wide.
2. Проследите actual pipeline до `setup_conds()` для base, reference/edit,
   img2img, hires/refiner и batch modes; отдельно определите mutable state.
3. Добавьте preflight condition, commit-point revalidation и immutable key
   fields для каждого влияющего на TE параметра.
4. Не запускайте future standard callbacks заранее и не меняйте ScriptRunner
   contracts.
5. Добавьте positive и negative unit/integration tests, включая fallback без
   повторных hooks, metadata и CPU-to-device clone.
6. Выполните fixed-seed GPU baseline comparison и проверьте log порядка
   TE-offload/generator-load. Только после этого разрешайте class/script.

## Команды проверки

```powershell
venv\Scripts\python.exe tests\test_conditioning_precompute.py
venv\Scripts\python.exe -m py_compile modules\conditioning_precompute.py modules\processing.py scripts\prompts_from_file.py tests\test_conditioning_precompute.py
git diff --check
```

Для воспроизводимого GPU smoke запускайте штатный bootstrap (`launch.py` или
disposable process с `initialize_forge()`, затем `initialize.imports()` и
`initialize.initialize()`). В API запросе выберите selectable script `Prompts
from File or Textbox` и передайте пять `script_args`:

```python
[False, False, "start", three_lines, True]
```

Сделайте второй запрос с теми же prompt/settings/fixed seed, но последним
значением `False`. Не изменяя config, используйте `send_images=false` для
pipeline-only smoke или in-memory PNG serializer для decoding and pixel hashes.
Сравнивайте image count, RGB bytes, `all_prompts` и `infotexts`, а также log
order `JointTextEncoder` → all precompute progress → `KModel`.
