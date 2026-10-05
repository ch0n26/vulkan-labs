# Vulkan Labs

Лабораторные работы по курсу «Компьютерная графика».

Стартовый код: [vulkan-starter-app](https://github.com/vladeemerr/vulkan-starter-app)

## Как посмотреть лабы

Каждая лаба — в отдельной ветке.

```bash
git clone https://github.com/ch0n26/vulkan-labs.git
cd vulkan-labs
git checkout lab1
```

## Как собрать и запустить

### Требования

- **Vulkan SDK** — [скачать](https://vulkan.lunarg.com/sdk/home)
- **CMake** 3.20+
- **Visual Studio 2022** с компонентом «Desktop development with C++» (Windows)
или **GCC 10+ / Clang 10+** (Linux)

Проверь, что `glslc` доступен из терминала:

```
glslc --version
```

Если команда не найдена — переустанови Vulkan SDK с галочкой «Add to PATH», потом перезапусти терминал.

### Сборка на Windows

Из корня проекта (там, где `CMakeLists.txt`):

```
cmake --preset msvc-debug
cmake --build build-debug --parallel
```

### Сборка на Linux

```
cmake --preset debug
cmake --build build-debug --parallel
```

### Запуск

**Важно:** запускать **из корня проекта** — приложение использует относительные пути к шейдерам. Если запустить двойным кликом из проводника, приложение не найдёт `.spv` файлы и упадёт.

**Windows:**

```
.\build-debug\Debug\vulkan-starter-app.exe
```

**Linux:**

```
./build-debug/vulkan-starter-app
```

### После переключения на другую ветку

Файлы на диске меняются, поэтому надо **пересобрать**:

```
git checkout lab2
cmake --build build-debug --parallel
.\build-debug\Debug\vulkan-starter-app.exe
```

`cmake --preset` можно пропустить, если `CMakeLists.txt` не менялся.

# Автор

Нгуен Шон, группа М8О-305БВ-24
