# Установка OpenCV CUDA  
  
Оригинал статьи - [OpenCV-CUDA-installation](https://github.com/chrismeunier/OpenCV-CUDA-installation).  
Это перевод инструкции с английского по сборке OpenCV для Python 3 с CUDA на Windows 11/10.  
Работает только с **видеокартами Nvidia**.  
  
Если вы просто хотите собрать модуль OpenCV для Python 3, то пропустите параграфы, связанные с CUDA.  
  
## Источники и ссылки по устранению неполадок  
  
Данная инструкция основана на:
- [статье](https://web.archive.org/web/20240728214837/https://thinkinfi.com/install-opencv-gpu-with-cuda-for-windows-10/) от Anindya - пошаговая сборка с помощью CMake GUI
- [статье](https://www.jamesbowley.co.uk/qmd/opencv_cuda_python_windows.html) от James Bowley - сборка с помощью консоли CMake и решение некоторых проблем.
  
### Ошибка ImportError  
  
Статей выше будет достаточно для большинства случаев.
**Но** в каких-то, даже если модуль был успешно установлен в Python, вы все равно можете увидеть это сообщение,  
когда импортируете OpenCV через `import cv2`:  
  
```  
ImportError: DLL load failed while importing cv2: The specified module could not be found.  
```  
  
Этот случай подробно описан в разделе по [устранению неполадок](#Проверка-установки-и-устранение-неполадок) и почти полностью основан на [этом](https://github.com/opencv/opencv/issues/19972) большом GitHub issue.
В общем, вся твоя сборка, вероятно, полностью удачна, это просто может быть Python не смог прочесть переменные  
окружения.

### Используемый софт и конфигурация ПК  
  
Протестировано на `Windows 10 20H2` с процессором `i7-10700 2.90ГГц` и видеокартой `GeForce RTX 2080 Ti`  
  
Софт:  
  
- Python 3.8.10  
- OpenCV 4.5.5  
- NumPy 1.21.6  
- CUDA Toolkit v11.6  
- cuDNN v8.3.3  
- Visual Studio Community 2019 v16.11.13  
- CMake 3.19.1  
  
Все это делалось в апреле 2022 года  
  
P.S.: в сентябре 2022 года все повторно проделано без проблем на таком же ПК, но уже на Windows 11, и на ноутбуке с  
процессором i5 и старым ГПУ Quadro.  
  
P.P.S.: процесс успешно повторен в конце 2023 года (без CUDA) на ноутбуке (Windows 11, i7 8-го поколения, Intel UHD  
Graphics 620) с Python 3.10 и OpenCV 4.9.  
  
P.P.P.S: в конце 2024 года все работает (с CUDA 11.6, Windows 11, Ryzen 5 5600x, RTX 2060 Super, Python 3.10, OpenCV  
4.9.0)  
  
## Пошаговый процесс установки  
  
### Python, NumPy и pip  
  
Установите Python 3.x любым удобным для вас  
способом ([оф. сайт](https://www.python.org/downloads/), [Anaconda](https://www.anaconda.com/download/success), магазин  
Майкрософт или создать виртуальное окружение).  
  
Убедитесь, что у вас установлен модуль NumPy, иначе сделайте это в консоли `pip install numpy`. Удалите все версии  
OpenCV `pip uninstall opencv-python` и `pip uninstall opencv-contrib-python`. Удалите папку  
`ВАШ_ПУТЬ_К_ПАПКЕ_С_PYTHON\Lib\site-packages\cv2` (Python обычно находится в `AppData\Local\Programs`)  
  
### Visual Studio  
  
Скачайте Visual Studio (желательно 2019 года, [скачать](https://github.com/user-attachments/files/18280278/vs_Community.zip)), выставите галочки как на скриншоте ниже:
  
![image](https://github.com/user-attachments/assets/b6b5681f-77b3-48cc-9b49-0ef9a155013f)  

Рекомендуется использовать английский язык, так как далее настройки будут приведены на этом языке. После всего нажмите установить (Install).

### CUDA и cuDNN  
  
Убедитесь, что ваша видеокарта поддерживает CUDA, и узнайте соответственно версию CUDA  
Toolkit [здесь](https://en.wikipedia.org/wiki/CUDA#GPUs_supported). Сначала находите поддерживаемую архитектуру (вторая  
таблица, колонка "Compute capability"), потом по зеленым квадратикам смотрите версию CUDA Toolkit для данной  
архитектуры (первая таблица, колонка CUDA SDK version).  
  
Скачайте и установите [CUDA Toolkit](https://developer.nvidia.com/cuda-toolkit-archive) в соответствии с вашей  
видеокартой. Или же проверьте, что это уже установлено по пути `C:\Program Files\NVIDIA GPU Computing Toolkit`.  
  
Аналогично для [cuDNN](https://developer.nvidia.com/rdp/cudnn-archive) согласно установленной версии CUDA Toolkit (для  
скачивания нужно зарегистрироваться)  
  
Проверьте, что добавились переменные окружения `CUDA_PATH` и `CUDA_PATH_Vxx_x`. Они должны указывать на путь, где  
установилась CUDA.  
  
Скопируйте все файлы в подпапках cuDNN (bin, include, lib\x64) в одноименные папки CUDA (путь по умолчанию -  
`C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\vXX.X`)  
  
### OpenCV и OpenCV contrib  
  
Скачайте и распакуйте [OpenCV](https://github.com/opencv/opencv/releases) и [OpenCV-contrib](https://github.com/opencv/opencv_contrib/tags) (версии должны совпадать)  
  
### CMake конфигурация  

#### Подготовка  

Создайте папку `build` с вашими распакованными OpenCV папками:

![image](https://github.com/user-attachments/assets/ded80da1-8223-4aef-ad27-b21b7db6c659)

Измените конец файла `OpenCVDetectPython.cmake` в папке `opencv-x.x.x\cmake`. Переместите второй `elseif` выше над первым:

![image](https://github.com/user-attachments/assets/71c13929-c303-48bf-ae97-4c445bc75dbe)

Это сделает приоритет на использование Python 3 для сборки.

#### Конфигурация сборки CMake GUI

Укажите пути к папке OpenCV и папке, где будет выполнена сборка:

![image](https://github.com/user-attachments/assets/8a5d69a6-bf0a-4a6f-b8da-82d7942c769b)

Нажмите `Configure` и выберите `x64` в поле `Optional platform for generator`. Далее нажмите `Finish`, чтобы начать первый этап конфигурации.

Когда все готово, измените следующие параметры:

| Name (имя)                |                 Value (значение)                 |
| ------------------------- | :----------------------------------------------: |
| ENABLE_FAST_MATH          |                        ✅                         |
| OPENCV_DNN_CUDA           |                        ✅                         |
| OPENCV_EXTRA_MODULES_PATH | Путь до папки `modules` в `opencv-contrib-x.x.x` |
| OPENCV_PYTHON3_VERSION    |                        ✅                         |
| WITH_CUDA                 |                        ✅                         |

Заметьте, что любые пути, в том числе в параметре `OPENCV_EXTRA_MODULES_PATH`, должны иметь `/`, **не** `\`. Пример - `C:/opencv-contrib-x.x.x/modules`.
Проверьте параметры, начинающиеся с `PYTHON3_`, чтобы они соответствовали тем значениям, которым вы ожидаете.

Нажмите `Configure` снова.

Измените еще 2 параметра:

| Name (имя)     | Value (значение) |
| -------------- | :--------------: |
| CUDA_FAST_MATH |        ✅         |
| CUDA_ARCH_BIN  |       x.x        |

Параметр `CUDA_ARCH_BIN` должен быть в соответствии с вашей [видеокартой](https://en.wikipedia.org/wiki/CUDA#GPUs_supported). Например значение `7.5` будет для `RTX 2080 Ti`.

![image](https://github.com/user-attachments/assets/54b74990-fccd-4da5-a00c-876d9932b08f)

Нажмите `Configure` для финального этапа конфигурации.

Когда все выполнилось, убедитесь, что у вас нет ничего, выделенного красным слева.
Далее нажмите `Generate`. Дождитесь окончания процесса, потом можно выходить из CMake.

### Сборка проекта с помощью Visual Studio

Откройте файл `OpenCV.sln`, созданный в папке `build`.

В верхней панели перейдите `Tools ➔ Options`, затем `Projects and Solutions ⭢ Web Projects`.
Снимите галочку с последнего параметра и перезапустите Visual Studio (запуск также через двойное нажатие по `OpenCV.sln`).

![image](https://github.com/user-attachments/assets/8e31bc02-d592-48ef-a9b2-5f6a9f22536b)

Измените режим сверху с `Debug` на `Release`

![image](https://github.com/user-attachments/assets/297f5bfe-f22c-4c73-a387-848b4be2fa3c)

Справа в обозревателе решений (Solution Explorer) раскройте меню `CMakeTargets`, правой кнопкой мыши по `ALL_BUILD` и выберите `Build`. Это займет у вас около **30 мин**.

![image](https://github.com/user-attachments/assets/f9fd1b08-146a-4343-b8a6-1dfc25a98c8e)

Те же действия повторите для пункта `INSTALL` (ниже `ALL_BUILD`).
После проверьте, есть ли ошибки. Если их нет, то все готово 🥳.
  
## Проверка установки и устранение неполадок

Первым делом откройте привычным образом Python и выполните эти строки, чтобы убедиться, что все получилось:

```python
import cv2
print(cv2.__version__)
print(cv2.cuda.getCudaEnabledDeviceCount())
```
Если работает, мои поздравления 👐

Если же не сработало, то, возможно, у вас эта проблема
`ImportError: DLL load failed while importing cv2: The specified module could not be found.`. Ее решение описано в начале.

Другие ошибки и проблемы можно посмотреть [здесь](https://www.jamesbowley.co.uk/qmd/opencv_cuda_python_windows.html#troubleshooting-python-bindings-installation-issues).

### Все ли на месте?

У вас должна быть папка `cv2` в вашей папке с Python - `ВАШ_ПУТЬ_К_ПАПКЕ_С_PYTHON\Lib\site-packages`.

Если ее нет, то можете попробовать изменить 2 параметра в CMake конфигурации:

| Name                     | Value |
| ------------------------ | ----- |
| BUILD_SHARED_LIBS        | 🔳    |
| OPENCV_FORCE_PYTHON_LIBS | ✅     |

Потом нажмите `Generate` и проделайте заново сборку в Visual Studio.

### CV2 обнаруживается?

В IDE с подсказками (например PyCharm) попробуйте напечатать `import cv2`, затем `cv2.` и посмотрите подсказки. Если они появились, то все хорошо.
  
![image](https://github.com/user-attachments/assets/bac01c44-63f7-4ee0-b588-8bf01e730553)

### OpenCV библиотеки

Проверьте, что установленные библиотеки твоей сборкой не вызывают ошибку импорта. Чтобы это сделать, вы можете вручную добавить DLL файлы:

```python  
import os  
os.add_dll_directory('C:/path_to_opencv_build_folder/install/x64/vc16/bin')

import cv2  
```

Этот импорт должен быть выполнен по умолчанию в файле `config.py` в папке `cv2` и, возможно, сам по себе не решит проблему.

### Дополнительные библиотеки

The problem is most likely linked to other libraries not loaded by Python _even if they are in your PATH environment  
variables_. You can troubleshoot this by adding all the PATH variables to the script with `os.add_dll_directory()` until  
it works or use the [Dependency walker](https://www.dependencywalker.com/) to find which DLLs you are missing.  
  
##### Using the Dependency walker  
  
Opening the `cv2.cp38-win_amd64.pyd` (or the .pyd file corresponding to the python version you're using) with the  
dependency walker can get you a list of DLLs it is missing. However it will also list a ton of Microsoft DLLs (starting  
with API-MS-... or EXT-MS-...) that actually do not impact the import error. Then you can try to add manually the  
missing libraries and see if it solves the issue.  
  
##### Using Anaconda binaries  
  
A solution highlighted in the [github issue](https://github.com/opencv/opencv/issues/19972) mentioned in the intro of  
this README was that using an Anaconda Python install made it work, so having a Python 3.8 Anaconda install I added the  
`C:/Users/username/Anaconda3/Library/bin` path to my script and voilà, it worked.  
  
It turns out my only missing libraries were `hdf5.dll` and `zlib.dll` out of the >200 DLL files located there. So they  
are here in this repository if you do not want to needlessly install Anaconda.  
  
Once you have located the folders containing your missing DLLs you have a few options to permanently solve the import  
error:  
  
- copy the files to your `path_to_opencv_build_folder/install/x64/vc16/bin` folder (easy but not clean)  
- add the `import os` and `os.add_dll_directory('...')` to any script using OpenCV (ok but not convenient)  
- add all the needed `os.add_dll_directory()` in the `__init__.py` file of `cv2` right after the `__all__ = []` line (  
  cleanest but make it clear!)  
  
If some part of this solved your  
`ImportError: DLL load failed while importing cv2: The specified module could not be found.` then great! Otherwise I  
suggest going thoroughly through the [github issue](https://github.com/opencv/opencv/issues/19972) for more ideas. Feel  
free to make any remarks, I will update this page if need be.
