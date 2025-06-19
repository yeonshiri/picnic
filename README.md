# PMC

## 실행을 위한 설치 패키지
git에서 yolov5를 다운로드 받은 후 다음과 같은 directory를 만들면 실행이 가능합니다.
<pre>
#git에서 프로젝트 설치 
git clone https://github.com/yeonshiri/picnic.git
cd AGS
pip install -r requirements.txt
</pre>   
TensorRT, OpenCV, scipy가 실패시
<pre>
sudo apt update
sudo apt install -y \
    python3-opencv \                  
    python3-libnvinfer python3-libnvinfer-dev \  
    liblapack-dev libopenblas-dev libgeos-dev   
</pre>   

이후에 터미널에서 python3 project.py 를 씁니다.
만약 영상(defalt)을 실행하고 싶으면 영상을 VIDEO_PATH = "영상 경로"에 쓰고 실시간으로 하고 싶으면 0을 넣어 다음과 같이 수정합니다. VIDEO_PATH = 0 

경로 구조는 다음과 같습니다.
<pre>
├── workspace
│   ├── project.py
│   ├── input.mp4
│   ├── picnic_5n_fp16.engine
│   ├── modules
│   │   ├──clean_bbox.py
│   │   ├──detection.py
│   │   ├──detect.py
│   │   ├──fsm.py
│   │   ├──kalman_filter.py
│   │   ├──sort.py
│   │   ├──sort_jet.py
│   │   ├──sort_tracker.py
│   │   ├──state.py
│   │   ├──visualize.py
│   │   └──utils.py


### 실행 환경
- Python 3.6.9
- torch==2.4.1
- torchvision==0.19.1
- python3-opencv == 3.2.0
- cycler==0.10.0
- decorator==4.1.2
- filterpy==1.4.5
- matplotlib==2.1.1
- numpy==1.19.5
- pycuda==2019.1.2
- pyparsing==2.2.0
- scipy==1.5.4
- six==1.11.0
- tensorrt==8.2.1.8
- typing==3.7.4.3
- Pillow==5.1.0

혹시몰라 jetson nano에 설치된 모든 환경은 pip와 apt에 따라 all_requirement.txt에 저장해놨습니다.