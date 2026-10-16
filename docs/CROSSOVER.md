# FSR4 вместо DLSS в CrossOver

Текущий вариант по умолчанию — **FSR 4.1.1 / Lossy**. В панели можно
переключать **Exact / Lossy**; новая настройка сбрасывает историю на следующем
кадре. Краткие команды: [GUIDE.md](GUIDE.md). Преобразование входов:
[FSR411_INPUT_CONTRACT.md](FSR411_INPUT_CONTRACT.md).

Сборка CLI и шейдеры описаны в [README.md](../README.md) и [STATE.md](STATE.md), пошаговая установка и откат — в [GUIDE.md](GUIDE.md). Здесь только то, как FSR4 встаёт на место DLSS внутри CrossOver Preview, и какие файлы при этом меняются.

DLSS в CrossOver Preview — это не отдельный `nvngx.so`. D3DMetal (GPTK, каталог `lib/apple_gptk`) сам реализует NGX и для временного апскейла вызывает MetalFX. Хук подменяет только `encodeToCommandBuffer:` у временного скейлера. Пространственный скейл, генерация кадров и денойз остаются на MetalFX.

`CrossOver.app` не изменяется. Бутылка указывает на копию фреймворка в `dist/crossover/`. Каталог `dist/` в git не входит.

## Почему не `DYLD_INSERT_LIBRARIES`

Wine в CrossOver — x86_64 под Rosetta, с hardened runtime. Флага `allow-dyld-environment-variables` нет, поэтому `DYLD_INSERT_LIBRARIES` до процесса не доходит. Флаг `disable-library-validation` есть, поэтому ad-hoc подпись копии D3DMetal и хука принимается.

`libd3dshared.dylib` читает `D3DMETAL_FRAMEWORK_PATH` и делает `dlopen` этого пути (`RTLD_NOW`). Если переменной нет, он открывает системный `D3DMetal.framework` или копию по rpath внутри приложения. Переменные секции `[EnvironmentVariables]` в `cxbottle.conf` CrossOver копирует в unix-процесс бутылки.

## Что создаёт установка

Источник фреймворка — найденный CrossOver, например:

```
/Applications/CrossOver Preview.app/Contents/SharedSupport/CrossOver/lib/apple_gptk/external/D3DMetal.framework
```

Если `apple_gptk` нет, берётся `lib/apple_gptk3/external/D3DMetal.framework`.

`python3 tools/crossover/crossover_fsr4.py install` делает следующее.

1. Собирает `dist/crossover/libfsr4hook.dylib` из `hook/fsr4_metalfx.m` и `src/fsr4_live.mm` (x86_64, ad-hoc подпись).
2. Копирует `D3DMetal.framework` в `dist/crossover/D3DMetal.framework`. Если каталог уже был, он удаляется и копируется заново.
3. В копию бинарника `Versions/A/D3DMetal` дописывает одну load-команду `LC_LOAD_DYLIB` с именем `@loader_path/libfsr4hook.dylib`. Команда ложится в нулевой зазор заголовка Mach-O между концом load-команд и `__text`. Код `EncodeTemporallyScaleMTLFX` не патчится.
4. Кладёт `libfsr4hook.dylib` рядом с бинарником и подписывает оба файла ad-hoc (`codesign -s -`).
5. Прогоняет smoke (свизл создаётся, FSR4 ещё не подменяет кадр), `dlopen` копии и `live_test` по `testcases/testcase_quality_720to1080`.
6. Если указана бутылка, дописывает переменные в конец `cxbottle.conf` и один раз сохраняет копию конфига.

По умолчанию бутылка:

```
~/Library/Application Support/CrossOver/Bottles/Игры/cxbottle.conf
```

Резервная копия конфига, снятая до первой записи:

```
dist/crossover/backups/Игры.cxbottle.conf
```

Ключи, которые пишет установщик (маркер `;; fsr4-mac`):

| Ключ | Значение |
| --- | --- |
| `D3DMETAL_FRAMEWORK_PATH` | `dist/crossover/D3DMetal.framework/Versions/A/D3DMetal` |
| `FSR4_HOOK_LOG` | `dist/crossover/hook.log` |
| `FSR4_ROOT` | корень этого репозитория |
| `FSR4_REPLACE` | `1` |
| `FSR4_VERSION` | `4.1.1` по умолчанию |
| `FSR4_PROFILE` | `lossy` по умолчанию, `exact-fast`, `exact` или `fsr4` (модель 4.0.2, четвёртый режим панели) |

Они должны остаться внутри `[EnvironmentVariables]`. Установщик дописывает их в конец файла; в бутылке «Игры» эта секция как раз последняя.

Другие ключи бутылки, ярлыки и само приложение не трогаются. Повторный `install` заново копирует фреймворк из CrossOver и заново накладывает только load-команду хука.

Команды:

```bash
python3 tools/crossover/crossover_fsr4.py install --panel --profile lossy
python3 tools/crossover/crossover_fsr4.py install --bottle "/path/to/cxbottle.conf"
python3 tools/crossover/crossover_fsr4.py install --no-bottle   # только dist/, конфиг бутылки не писать
python3 tools/crossover/crossover_fsr4.py restore
```

## DXMT and forced render size

DXMT also routes its DLSS/NGX temporal feature through
`MTLFXTemporalScaler`, so the same MetalFX encode hook can replace it. Unlike
D3DMetal, DXMT does not load a private framework that this installer can
patch. Build DXMT from source with the opt-in integration patch first:

```bash
git -C /path/to/dxmt checkout 50102f9
git -C /path/to/dxmt apply /path/to/fsr4-native-macos/patches/dxmt/fsr4-dxmt-integration.patch
# Build DXMT using its normal instructions, then install its build in the bottle.
python3 tools/crossover/crossover_fsr4.py install-dxmt --bottle "Game"
```

The DXMT patch loads `libfsr4hook.dylib` immediately before it creates a
temporal scaler. The install command builds the hook and adds its path and
FSR settings to the selected bottle. It does not replace DXMT files. Restart
the bottle after installing DXMT and configuring the hook. To restore, use the
normal `restore` command.

With a game that queries DXMT's NGX optimal-settings callback, force an exact
internal render size when installing:

```bash
python3 tools/crossover/crossover_fsr4.py install-dxmt --bottle "Game" \
  --render-width 1280 --render-height 720 --profile lossy
```

These dimensions are returned by DXMT before the game allocates its render
targets, so color, depth and motion vectors stay aligned. Width and height
must be supplied together, and each must be between one third of the requested
output size and that output size. The game must use DXMT's optimal-settings
callback and honor its returned size; titles that choose their own render
dimensions can ignore the override. Lower render sizes reduce game shading
work and increase the FSR scale ratio, so image quality can change. The FSR
4.1 Lite network, weights and temporal reuse cadence remain unchanged.

The integration patch is pinned to DXMT commit `50102f9`; apply it to a clean
checkout of that revision. DXMT's upstream project and build guidance:
[3Shain/dxmt](https://github.com/3Shain/dxmt). The patch is opt-in and has not
been runtime-validated in this project.

`restore` удаляет управляемые ключи установщика, включая версию/профиль и
панель, и строку `;; fsr4-mac`. Копия в `dist/` остаётся на диске. Следующий
запуск бутылки снова берёт D3D