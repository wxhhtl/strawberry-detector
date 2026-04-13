import streamlit as st
from ultralytics import YOLO
from PIL import Image
import cv2
import numpy as np
import pandas as pd
from datetime import datetime
import io

# 1. 页面基本设置 (将版面改宽一点，方便展示图表)
st.set_page_config(page_title="草莓病害检测专家", page_icon="🍓", layout="wide")

# --- 核心升级 1：初始化网页备忘录 (Session State) ---
if 'history' not in st.session_state:
    st.session_state['history'] = []

st.title("🍓 草莓叶片病害识别系统 (Web版)")
st.write("欢迎！请上传一张草莓叶片的照片，AI 将为您自动诊断病害。")

# 2. 加载“仙丹”模型
@st.cache_resource
def load_model():
    return YOLO('best.pt') 

model = load_model()

# 3. 制作一个图片上传按钮
uploaded_file = st.file_uploader("点击或拖拽上传图片...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='您上传的原始图片', use_container_width=True)

    # 制作一个“开始检测”按钮
    if st.button('一键开始检测 🚀'):
        with st.spinner('AI 正在发功照妖，请稍候...'):
            # 丢给模型去预测
            results = model.predict(image)
            
            # 把画好框的图片提取出来
            res_plotted = results[0].plot() 
            res_rgb = cv2.cvtColor(res_plotted, cv2.COLOR_BGR2RGB)
            result_img = Image.fromarray(res_rgb)
            
            # 在网页上展示结果图
            st.success('检测完成！结果如下：')
            st.image(result_img, caption='AI 识别结果', use_container_width=True)

            # --- 核心升级 2：保存结果图并提供下载 ---
            buf = io.BytesIO()
            result_img.save(buf, format="PNG")
            byte_im = buf.getvalue()
            
            st.download_button(
                label="💾 点击下载识别结果图",
                data=byte_im,
                file_name=f"result_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png",
                mime="image/png",
            )

            # --- 核心升级 3：提取识别数据，写入历史记录 ---
            boxes = results[0].boxes
            names = model.names
            current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            if len(boxes) == 0:
                st.session_state.history.append({"时间": current_time, "病害类别": "健康 (未检测到病害)", "置信度": 0.0})
            else:
                for box in boxes:
                    cls_id = int(box.cls[0])
                    conf = float(box.conf[0])
                    label = names[cls_id]
                    # 把每一次检测到的框存入记录表
                    st.session_state.history.append({"时间": current_time, "病害类别": label, "置信度": round(conf, 4)})

# ---------------------------------------------------------
# --- 核心升级 4：在页面下方展示“历史记录”与“统计图表” ---
st.divider() # 画一条分割线
st.header("📊 检测历史与统计")

# 只有当备忘录里有数据时，才显示图表
if len(st.session_state.history) > 0:
    # 把字典列表转换成 pandas 的大表格
    df = pd.DataFrame(st.session_state.history)
    
    # 把页面分成左右两列
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📝 详细历史记录")
        st.dataframe(df, use_container_width=True)
        
        # --- 核心升级 5：导出 CSV 按钮 ---
        # 使用 utf-8-sig 防止用 Excel 打开时中文乱码
        csv = df.to_csv(index=False).encode('utf-8-sig')
        st.download_button(
            label="📥 导出为 CSV 报表",
            data=csv,
            file_name='detection_history.csv',
            mime='text/csv',
        )
        
    with col2:
        st.subheader("📈 识别统计图")
        # 自动统计各个病害出现了多少次，并画柱状图
        chart_data = df['病害类别'].value_counts()
        st.bar_chart(chart_data)
else:
    st.info("暂无数据，快去上面传张图试试吧！历史记录会在这里动态生成哦。")
