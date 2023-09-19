import streamlit as st
from core import Pix2TexModel
from streamlit_cropper import st_cropper
from PIL import Image
from urllib.parse import quote
import matplotlib.pyplot as plt
import matplotlib as mpl
import io


st.title("Formula Capture📸")

# 모델 초기화
model = Pix2TexModel()

# matplotlib의 폰트 설정 변경
# mpl.rcParams['font.family'] = 'serif'
# mpl.rcParams['font.serif'] = 'Computer Modern Roman'
mpl.rcParams['text.usetex'] = False

## 기능 함수 ##
# LaTeX문법 이미지로 변환
def latex_to_image(latex_str):
    # 일단 대략적인 크기로 그림과 축을 생성
    fig, ax = plt.subplots(figsize=(12, 3))
    
    # LaTeX 문자열로 텍스트를 생성
    txt = ax.text(0.5, 0.5, f'${latex_str}$', size=15, va='center', ha='center')
    
    # 텍스트의 바운딩 박스의 너비와 높이를 얻음
    fig.canvas.draw()  # 이를 호출해야 get_window_extent()가 정확한 값을 반환
    bbox = txt.get_window_extent(fig.canvas.get_renderer())
    width, height = bbox.width, bbox.height
    width /= fig.dpi  # 인치 단위로 변환
    height /= fig.dpi  # 인치 단위로 변환

    # 얻은 너비와 높이로 그림의 크기를 재조정
    fig.set_size_inches(width + 1, height + 1)  # 여백을 위해 약간 추가
    ax.axis('off')
    
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0.2)
    buf.seek(0)
    
    plt.close(fig)
    return buf
# 사용자로부터 이미지 입력
uploaded_file = st.file_uploader("", type=["png", "jpg"])
## 파일 업로드 이후 작업 ##
if uploaded_file is not None:
    img = Image.open(uploaded_file) # 이미지 열기
    
    # 이미지 크롭
    cropped_img = st_cropper(img_file=img, realtime_update=True, box_color="green")
    
    col1, col2 = st.columns([10, 1])
    use_full = st.toggle("전체 이미지 사용")
    
    if use_full:
        # 전체 이미지
        col1.image(uploaded_file, caption='최종 입력 이미지', use_column_width=True)
        final_img = uploaded_file
    else:
        # 이미지 자르기
        col1.image(cropped_img, caption='최종 입력 이미지', use_column_width=True)
        final_img = cropped_img
        
    # 전체 이미지 사용 토글
    
    
    ## 예측 부분 ##
    st.title("Predict")
    if st.button("Start", key="Start_btn"):
        with st.spinner("분석중....."):
            if use_full:
                prediction = model.predict(final_img, True)
            else:
                prediction = model.predict(final_img)
            st.latex(prediction)
            st.code(prediction, language="cmd")
            st.session_state.predict_latex = prediction
    else:
        # 'Start' 버튼을 누르지 않았을 때도 prediction 값을 유지
        if "predict_latex" in st.session_state:
            prediction = st.session_state.predict_latex

        if "predict_latex" in st.session_state:
            st.latex(st.session_state.predict_latex)
            st.code(st.session_state.predict_latex, language="cmd")

            # 사용자에게 수정된 LaTeX 문자열 입력을 받습니다.
            edited_prediction = st.text_input("수식 수정:", st.session_state.predict_latex)

            # 입력된 값이 변경되면 session_state를 업데이트합니다.
            if edited_prediction != st.session_state.predict_latex:
                st.session_state.predict_latex = edited_prediction
            ## 내보내기 기능 ##
            with st.expander("내보내기"):
                
                # 울프람알파 내보내기
                encoded_prediction = quote(prediction) # URL 또는 다른 web에 보내기위한 인코딩
                wolfram_url = f"https://www.wolframalpha.com/input/?i={encoded_prediction}"
                button_code = f"""
                <a href="{wolfram_url}" target="_blank" style="display: inline-block; text-decoration: none; background-color: #F96932; color: white; padding: 8px 16px; border-radius: 4px;">울프람알파로 이동</a>
                """
                st.markdown(button_code, unsafe_allow_html=True)
        
            # # 이미지 저장
            # latex_image = latex_to_image(prediction)
            # latex_image_bytes = latex_image.getvalue()
                    
            # st.download_button(
            #     label="다운로드",
            #     data=latex_image_bytes,
            #     file_name="latex_image.png",
            #     mime="image/png"
            # )
        
        # codecogs_url = f"https://latex.codecogs.com/png.latex?{encoded_prediction}"

        # # 사용자에게 URL 제공
        # button_code = f"""
        # <a href="{codecogs_url}" target="_blank" style="display: inline-block; text-decoration: none; background-color: #F96932; color: white; padding: 8px 16px; border-radius: 4px;">이미지로 변환하기</a>
        # """
        # st.markdown(button_code, unsafe_allow_html=True)
     
# References
# - streamlit-cropper https://github.com/turner-anderson/streamlit-cropper