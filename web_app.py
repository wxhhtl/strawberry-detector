import streamlit as st
from ultralytics import YOLO
from PIL import Image
import cv2
import numpy as np

# 1. 页面基本设置
st.set_page_config(page_title="草莓病害检测专家", page_icon="🍓", layout="centered")
st.title("🍓 草莓叶片病害识别系统 (Web版)")
st.write("欢迎！请上传一张草莓叶片的照片，AI 将为您自动诊断病害。")

# 2. 加载“仙丹”模型 (使用缓存机制，避免每次传图都重新加载，提升速度)
@st.cache_resource
def load_model():
    # 注意：这里的路径请确保对准你的 best.pt
    return YOLO('best.pt') 

model = load_model()

# 3. 制作一个图片上传按钮
uploaded_file = st.file_uploader("点击或拖拽上传图片...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # 把用户上传的图片读进来
    image = Image.open(uploaded_file)
    st.image(image, caption='您上传的原始图片', use_container_width=True)

    # 制作一个“开始检测”按钮
    if st.button('一键开始检测 🚀'):
        with st.spinner('AI 正在发功照妖，请稍候...'):
            # 丢给模型去预测
            results = model.predict(image)
            
            # 把画好框的图片提取出来
            res_plotted = results[0].plot() 
            
            # YOLO 画图默认是 BGR 格式，网页显示需要 RGB 格式，转换一下
            res_rgb = cv2.cvtColor(res_plotted, cv2.COLOR_BGR2RGB)
            result_img = Image.fromarray(res_rgb)
            
            # 在网页上展示结果图
            st.success('检测完成！结果如下：')
            st.image(result_img, caption='AI 识别结果', use_container_width=True)