# YOLOv5-RK3588-Python
Modify Code From rknn-toolkit2

# Getting Started
You can change source cam, resolution from capture in config.py

# Prerequisites
1. must compile opencv support gstreamer
2. opencv
3. gstreamer
4. rknnlite2 from rknn-toolkit2
5. usb webcam

# Running the program
python inference_npu.py

# This branch (singlethread)
1. Improve performance by transfer detect box to original stream
2. remove redundancy postprocess
3. add default support cv2.capture (cam2)
4. add file mp4 support (need gstreamer)

# Example on Youtube
https://www.youtube.com/watch?v=eD6L55MkDoo

# rknn-tookit-lite2 v2.0.0 까지 테스트 완료
1. 1.4.0은 python 3.9 환경에서 inference_npu.py 실행
2. 2.0.0은 python 3.11 환경에서 inference_npu2.py 실행