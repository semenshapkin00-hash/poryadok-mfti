# Сторонние компоненты

Самописные ядро, API, интерфейс и тестовый корпус подготовлены командой SPN с ИИ-инструментами. Условия распространения собственного кода команда определяет отдельно; права на сторонние компоненты остаются у их авторов.

Прямые зависимости и закреплённые транзитивные версии перечислены в requirements-lock.txt. FastAPI, Pydantic, python-docx и ONNX Runtime используют MIT; Uvicorn и HTTPX — BSD; python-multipart и Tokenizers — Apache-2.0. Для pdfplumber, PDFium/pypdfium2, Pillow и NumPy сохраняются полные лицензии и notices поставщиков. Копии доступных текстов прямых зависимостей сохранены в licenses/. Полные условия поставщиков имеют приоритет перед этим кратким перечнем. PyMuPDF в навигаторе не используется.

Модель: [Sentence Transformers paraphrase-multilingual-MiniLM-L12-v2](https://huggingface.co/sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2), Apache-2.0, ревизия e8f8c211226b894fcb81acc59f3b34ba3efd5f42. Публичный ONNX int8 checkpoint, masked mean pooling 384 и L2-нормировка. Контрольные суммы — models/semantic-minilm/manifest.json, проверяются до загрузки. Лицензия модели не передаёт команде исключительные права на её веса.

Интерфейс использует системные шрифты, собственные SVG и CSS. Демо-документы вымышлены. Созданные для тестов PDF могут включать Ubuntu Regular/Bold под Ubuntu Font Licence 1.0; её текст сохранён в licenses/Ubuntu-Font-Licence-1.0.txt. Веса и пакеты не скачиваются во время ответов на вопросы.

OCR: [Tesseract.js](https://github.com/naptha/tesseract.js) 7.0.0 и Tesseract.js-core 7.0.0 (Apache-2.0), локальный WASM worker. Полное дерево закреплено в pnpm-lock.yaml. Веса [tessdata_fast](https://github.com/tesseract-ocr/tessdata_fast), Apache-2.0, ревизия 87416418657359cb625c412a48b6e1d6d41c29bd; SHA-256 в models/tessdata/manifest.json и коде проверки. Тексты лицензий worker/core сохранены в licenses/.

Русская морфология: [pymorphy3](https://github.com/no-plagiarism/pymorphy3) 2.0.6 и dawg2-python 0.9.0 — MIT. Код пакета pymorphy3-dicts-ru 2.4.417150.4580142 — MIT; **словарные данные OpenCorpora — [CC BY-SA 3.0](https://creativecommons.org/licenses/by-sa/3.0/)**. Источник OpenCorpora, ревизия 417150, компиляция 08.01.2022. Данные не изменены и устанавливаются отдельно как зависимость; права на словарь не переобозначаются лицензией проекта. Условия из официального описания поставляемого пакета сохранены в licenses/pymorphy3-dicts-ru-package-description.txt; лицензии и авторы анализатора также приложены.

Отдельный эксперимент (в рабочий API не включён): [MLX LM](https://github.com/ml-explore/mlx-lm), MIT, и [Qwen3-4B-Instruct-2507-4bit](https://huggingface.co/mlx-community/Qwen3-4B-Instruct-2507-4bit), Apache-2.0, преобразованный публичный Qwen checkpoint. Ревизия 50d427756c6b1b2fe0c0a10f67fbda1fc8e82c1b, контрольные суммы в tools/download_local_reasoner.py и локальном manifest. Экспериментальные веса исключены из текущего ZIP; зависимости изолированы в requirements-reasoner-lock.txt.
