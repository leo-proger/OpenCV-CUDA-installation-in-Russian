# Установка OpenCV CUDA
Оригинал статьи - [OpenCV-CUDA-installation](https://github.com/chrismeunier/OpenCV-CUDA-installation). Это перевод инструкции с английского по сборке OpenCV для Python 3 с CUDA на Windows 11/10. Работает только с **видеокартами Nvidia**.

Также этот гайд подойдет, если вы просто хотите собрать модуль OpenCV для Python 3. Для этого пропустите параграфы, связанные с CUDA.

## Источники и ссылки по устранению неполадок

Данная инструкция основана на [статье](https://web.archive.org/web/20240728214837/https://thinkinfi.com/install-opencv-gpu-with-cuda-for-windows-10/) от Anindya о пошаговой сборке с CMake GUI, а также на [статье](https://www.jamesbowley.co.uk/qmd/opencv_cuda_python_windows.html) от James Bowley по сборке с помощью консоли CMake и решениям некоторых проблем.

### Ошибка ImportError

Статей выше будет достаточно для большинства людей. **Но** в каких-то случаях, даже если модуль был успешно установлен в Python, вы все равно можете увидеть это сообщение, когда импортируете OpenCV через `import cv2`:
```
ImportError: DLL load failed while importing cv2: The specified module could not be found.
```
Этот случай подробно описан в разделе по устранению неполадок и почти полностью основан на [этом](https://github.com/opencv/opencv/issues/19972) большом GitHub issue.
В общем, вся ваша сборка, вероятно, полностью удачна, может быть просто Python не смог прочесть переменные окружения.

### Используемый софт и конфигурация ПК

Протестировано на `Windows 10 20H2` с процессором `i7-10700 2.90ГГц` и видеокартой GeForce `RTX 2080 Ti`

Софт:
- Python 3.8.10
- OpenCV 4.5.5
- NumPy 1.21.6
- CUDA toolkit v11.6
- cuDNN v8.3.3
- Visual Studio Community 2019 v16.11.13
- CMake 3.19.1
Все это делалось в апреле 2022 года

P.S.: в сентябре 2022 года все повторно проделано без проблем на таком же ПК, но уже на Windows 11, и на ноутбуке с процессором i5 и старым ГПУ Quadro. 

P.P.S.: процесс успешно повторен в конце 2023 года (без CUDA) на ноутбуке (Windows 11, i7 8-го поколения, Intel UHD Graphics 620) с Python 3.10 и OpenCV 4.9.

P.P.P.S: в конце 2024 года все работает (с CUDA 11.6, Windows 11, Ryzen 5 5600x, RTX 2060 Super, Python 3.10, OpenCV 4.9.0)

## Пошаговый процесс установки

### Требуемое ПО

### Python, NumPy и pip

Установите Python 3.x любым удобным для вас способом ([оф. сайт](https://www.python.org/downloads/), [Anaconda](https://www.anaconda.com/download/success), магазин Майкрософт или создать виртуальное окружение).

Убедитесь, что у вас установлен модуль NumPy, иначе сделайте это в консоли `pip install numpy`. Удалите все версии OpenCV `pip uninstall opencv-python` и `pip uninstall opencv-contrib-python`. Удалите папку `ВАШ_ПУТЬ_К_ПАПКЕ_С_PYTHON\Lib\site-packages\cv2` (Python обычно находится в `AppData\Local\Programs`)

### Visual Studio

Скачайте Visual Studio (желательно 2019 года, [ссылка](https://github.com/user-attachments/files/18280278/vs_Community.zip)) и выставите галочки как на скриншоте ниже.

![image](https://github.com/user-attachments/assets/b6b5681f-77b3-48cc-9b49-0ef9a155013f)

### CUDA и cuDNN

Убедитесь, что ваша видеокарта поддерживает CUDA, и узнайте соответственно версию CUDA Toolkit [здесь](https://en.wikipedia.org/wiki/CUDA#GPUs_supported). Сначала находите поддерживаемую архитектуру (вторая таблица, колонка "Compute capability"), потом по зеленым квадратикам смотрите версию CUDA Toolkit для данной архитектуры (первая таблица, колонка CUDA SDK version).
 
Скачайте и установите [CUDA Toolkit](https://developer.nvidia.com/cuda-toolkit-archive) в соответствии с вашей видеокартой. Или же проверьте, что это уже установлено по пути `C:\Program Files\NVIDIA GPU Computing Toolkit`.

Аналогично для [cuDNN](https://developer.nvidia.com/rdp/cudnn-archive) согласно установленной версии CUDA Toolkit (для скачивания нужно зарегистрироваться)

Проверьте, что добавились переменные окружения `CUDA_PATH` и `CUDA_PATH_Vxx_x`. Они должны указывать на путь, где установилась CUDA.

Скопируйте все файлы в подпапках cuDNN (bin, include, lib/x64) в одноименные папки CUDA (путь по умолчанию - `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\vXX.X`)

### OpenCV и OpenCV contrib

Скачайте и распакуйте [OpenCV](https://github.com/opencv/opencv/releases) и [OpenCV-contrib](https://github.com/opencv/opencv_contrib/tags) (версии должны совпадать)

### CMake конфигурация

#### Preparation

Prepare a **"build"** folder with your OpenCV extracted folders.

![explorer_bSMon2LQY0](https://user-images.githubusercontent.com/28230243/166434918-458aa3ae-8696-4cee-bb8e-5d9713401147.png)

Edit the end of the _OpenCVDetectPython.cmake_ file in _opencv-x.x.x/cmake_. Move the second _elseif_ above the first to get this:

![notepad++_Zwz4Lsl2kZ](https://user-images.githubusercontent.com/28230243/166435763-da8d2429-ba15-45ab-aede-ec61a6715cbc.png)

This will prioritize the use of your Python 3 installation for the build.

#### CMake GUI build configuration

Provide the paths to the OpenCV and target build folders:

![cmake-gui_IBQybmF6kh](https://user-images.githubusercontent.com/28230243/166436165-126efd0b-43e5-4d1a-9e8a-1d46dbec7f35.png)

Hit _Configure_ and select _x64_ as the _Optional platform for generator_, then hit finish to start the first round of configuration.

Once this is done edit the following parameters:
| Name | Value |
|---|:---:|
| ENABLE_FAST_MATH | ✅ |
| OPENCV_DNN_CUDA | ✅ |
| OPENCV_EXTRA_MODULES_PATH | path of **modules** directory in extracted opencv_contrib-x.x.x |
| OPENCV_PYTHON3_VERSION | ✅ |
| WITH_CUDA | ✅ |
<!--| BUILD_SHARED_LIBS | 🔳 |-->

Check the PYTHON3_... parameters so that the paths correspond to what you expect.
Note that the path separator in OPENCV_EXTRA_MODULES_PATH (or any other parameter value) has to be "/" and _not "\\"_.

Hit _Configure_ again.

Edit two more parameters:
| Name | Value |
|---|:---:|
| CUDA_FAST_MATH | ✅ |
| CUDA_ARCH_BIN | x.x |

The CUDA_ARCH_BIN corresponding to your GPU is the value found in the left column of the [GPU support table](https://en.wikipedia.org/wiki/CUDA#GPUs_supported). For instance "7.5" for the RTX 2080 Ti.

![firefox_pEBLFW3y2g](https://user-images.githubusercontent.com/28230243/166440728-1e63a0ed-6340-4b85-b350-448274c3d077.png)

Hit _Configure_ for the final configuration round.
Once the configuration is done you should not have any parameter left in red.
Now hit _Generate_. When generation is finished we are done with CMake.

### Build the project with Visual Studio

Open the `OpenCV.sln` just created in the build folder.

Go in "Tools>Options...", then in "Projects and Solutions > Web Projects" uncheck the last parameter. Continue if it was already unchecked, otherwise close Visual Studio and reopen `OpenCV.sln`.

_N.B. If you are not using Visual Studio in english, this setting may be elsewhere or (from personal experience) somehow simply unfindable. If this is the case **change the language to english**. While I have no precise idea of why this setting is needed, [it actually has an impact](https://github.com/opencv/opencv/issues/19972#issuecomment-1119781901)._

![devenv_TVsR0HP4yc](https://user-images.githubusercontent.com/28230243/166442847-060bb8cc-2333-4fc9-8f73-24749f233e60.png)

Change the "Debug" mode to "Release".

![debug2release](https://user-images.githubusercontent.com/28230243/166443402-9e1cdd4b-245e-4d0e-a202-6e1b3b9edac1.png)

In the solution explorer expand **CMakeTargets**, right-click **ALL_BUILD** and select **Build**. This should take about half an hour.

![devenv_vwYMW4osmJ](https://user-images.githubusercontent.com/28230243/166444051-d75deecb-eb99-42ff-a184-b2709e285f7b.png)

Then repeat the step for **INSTALL** (right below **ALL_BUILD**).
Check for errors in the two building steps, if everything is fine you are done.

### Check install and troubleshooting

First thing to do open your preferred way of executing some Python code and try this:
```python
import cv2
print(cv2.__version__)
print(cv2.cuda.getCudaEnabledDeviceCount())
```

If it works, congratulations you are good to go!

If not let's tackle _the_ problem. The problem being `ImportError: DLL load failed while importing cv2: The specified module could not be found.`.

**For other bugs and problems I refer you to the** [**troubleshooting section of James Bowley's tutorial**](https://www.jamesbowley.co.uk/qmd/opencv_cuda_python_windows.html#troubleshooting-python-bindings-installation-issues).

#### Is everything in place ?

You should have a "cv2" folder in your python installation (under `your_python_path/Lib/site-packages`). If not check if you have a "binding" folder in the Visual Studio solution.
Otherwise I suggest trying to change two parameters in the CMake configuration: `BUILD_SHARED_LIBS 🔳` and `OPENCV_FORCE_PYTHON_LIBS ✅`. Then re-generate and re-build everything.

#### Is CV2 detected ?

In an IDE with code suggestion (VS Code for instance) try to type `import cv2`, then write `cv2.` and see if suggestions appear. If they do your Python installation can successfully access OpenCV.

![image](https://user-images.githubusercontent.com/28230243/166449328-477e9fb3-d192-4615-aa31-9b6c89231eec.png)

#### OpenCV libraries

Check that the libraries installed by your build are not causing the import error. To do this you can add manually the DLL files path to a script:
```python
import os
os.add_dll_directory('C:/path_to_opencv_build_folder/install/x64/vc16/bin')
import cv2
```
This import should be done by default in the `config.py` file in the `cv2` folder and should probably not solve the issue by itself.

#### External libraries

The problem is most likely linked to other libraries not loaded by Python _even if they are in your PATH environment variables_. You can troubleshoot this by adding all the PATH variables to the script with `os.add_dll_directory()` until it works or use the [Dependency walker](https://www.dependencywalker.com/) to find which DLLs you are missing.

##### Using the Dependency walker

Opening the `cv2.cp38-win_amd64.pyd` (or the .pyd file corresponding to the python version you're using) with the dependency walker can get you a list of DLLs it is missing. However it will also list a ton of Microsoft DLLs (starting with API-MS-... or EXT-MS-...) that actually do not impact the import error. Then you can try to add manually the missing libraries and see if it solves the issue.

##### Using Anaconda binaries

A solution highlighted in the [github issue](https://github.com/opencv/opencv/issues/19972) mentioned in the intro of this README was that using an Anaconda Python install made it work, so having a Python 3.8 Anaconda install I added the `C:/Users/username/Anaconda3/Library/bin` path to my script and voilà, it worked.

It turns out my only missing libraries were `hdf5.dll` and `zlib.dll` out of the >200 DLL files located there. So they are here in this repository if you do not want to needlessly install Anaconda.

Once you have located the folders containing your missing DLLs you have a few options to permanently solve the import error:

- copy the files to your `path_to_opencv_build_folder/install/x64/vc16/bin` folder (easy but not clean)
- add the `import os` and `os.add_dll_directory('...')` to any script using OpenCV (ok but not convenient)
- add all the needed `os.add_dll_directory()` in the `__init__.py` file of `cv2` right after the `__all__ = []` line (cleanest but make it clear!)

If some part of this solved your `ImportError: DLL load failed while importing cv2: The specified module could not be found.` then great! Otherwise I suggest going thoroughly through the [github issue](https://github.com/opencv/opencv/issues/19972) for more ideas. Feel free to make any remarks, I will update this page if need be.
