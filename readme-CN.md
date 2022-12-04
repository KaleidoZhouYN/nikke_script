# update 2022/12/04
- 移除pytorch依赖
- 添加手动防御，现在按下f键即可进入/退出 防御姿态

# v1.0 Beta version

## **配置需求**

**1.安装python3.10:**

https://www.python.org/ftp/python/3.10.0/python-3.10.0-amd64.exe

**2.安装pip**

在此路径下打开windows powershell，输入 python get_pip.py

**3.安装依赖包**

在windows powershell中输入  pip install -r requirements.txt

**4.下载yolov5模型文件**

链接：https://pan.baidu.com/s/10VPrJt5WUrawTaYsHsYZiQ?pwd=e6ek 
提取码：e6ek 

下载yolov5n_640.onnx, yolov5t2_384.onnx 到 utils/yolov5 目录下

## **模拟器需求**

**1.模拟器种类**

目前只支持MuMuX

**2.游戏设置**

- 关闭游戏内的自动瞄准
- 鼠标灵敏度调至150
- 编队中第3号位必须为霰弹枪角色

## **用法**
1.在任意位置打开windows powershell, 运行 start-process PowerShell -verb runas 打开新的管理员权限 powershell

2. 移至当前路径

3.打开游戏

4.进入拦截S

5.当boss完全现身之后，运行 python simulator_test.py

## **注意事项**
- 脚本运行是请不要在模拟器区域点击鼠标
- 目前不支持自动瞄准钻头
- 目前不支持自动躲技能
- 当boss要放技能时，请重来



