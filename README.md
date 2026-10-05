# Лабораторная работа 1: Основы 3D-графики

**Вариант 8.** Цилиндр (~50 вершин в основаниях) 

## Что сделано

- Генерация цилиндра (50 сегментов в основаниях)
- Переключение Perspective / Orthographic (ImGui)
- Управление позицией, поворотом, масштабом (ImGui)
- Анимация по круговой траектории (Play/Pause, скорость, радиус)
- Изменение цвета через ColorEdit (ImGui)
- Процедурные цвета вершин (градиент по высоте)
- Два объекта на сцене через отдельные VkDescriptorSet

## Сборка

**Требования:** Vulkan SDK, CMake 3.20+, Visual Studio 2022 (Windows) / GCC 10+ (Linux).

**Windows:**
```

cmake --preset msvc-debug
cmake --build build-debug --parallel
.\build-debug\Debug\vulkan-starter-app.exe

```

**Linux:**
```

cmake --preset debug
cmake --build build-debug --parallel
./build-debug/vulkan-starter-app

```

## Документация

Полное описание проекта, структура и инструкции — в [README основной ветки](../../tree/main).

## Автор

Нгуен Шон, группа М8О-305БВ-24
