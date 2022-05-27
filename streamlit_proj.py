import streamlit as st
import numpy as np
import cv2
import PIL.Image as Image
from EVFRTest import predict as predict_stantard
from EVFRTest_lite import predict as predict_lite
import time
import urllib
import torch
import sys

style = '''
        <style>
        footer {visibility: hidden;}
        </style>
        '''
st.markdown(style, unsafe_allow_html=True)

language_select = {'CN': '语言', 'EN': 'Language Select'}

LANG = st.sidebar.selectbox('Language', ['EN','CN'])
TITLE = {'CN': '电动汽车火灾图像痕迹识别', 'EN': 'Electric Vehicle Fire Trace Recognition'}

st.title(TITLE[LANG])
slot1 = st.empty()

select_upload_name = {'CN': '输入方式', 'EN': 'Input Method'}
select_upload_1 = {'CN': '本地上传', 'EN': 'Local Upload'}
select_upload_2 = {'CN': '网络图像', 'EN': 'Web URL'}
select_upload_3 = {'CN': '相机拍摄', 'EN': 'Camera Input'}
select_upload = st.sidebar.radio(select_upload_name[LANG],
                                 [select_upload_1[LANG], select_upload_2[LANG], select_upload_3[LANG]])

input_pass = False

col1, col2 = st.columns(2)
col1_caption = {'CN': '输入图像', 'EN': 'Input Images'}
col2_caption = {'CN': '输出图像', 'EN': 'Output Images'}
col1.caption(col1_caption[LANG])
col2.caption(col2_caption[LANG])

if select_upload == select_upload_2[LANG]:
    button_pressed = False
    url = st.sidebar.text_input('URL')
    try:
        url_open = urllib.request.urlopen(url)
        img_url_byte = np.asarray(bytearray(url_open.read()), dtype=np.uint8)
        img_arr = cv2.imdecode(img_url_byte, 1)
        img_arr = cv2.cvtColor(img_arr, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(img_arr)
        img_pil = img_pil.resize((560, 420))
        input_pass = True
    except:
        img_pil = None
    finally:
        img_pil_list = [img_pil]


elif select_upload == select_upload_1[LANG]:
    button_pressed = False
    img_upload = st.sidebar.file_uploader(label='input image', accept_multiple_files=True)
    if img_upload is not None:
        err_idx = []
        for ii, img_ in enumerate(img_upload):
            if 'image' not in img_.type:
                err_idx.append(ii)
        if len(err_idx) == 0:
            img_pil_list = []
            for img_up in img_upload:
                file_byte = np.asarray(bytearray(img_up.read()), dtype=np.uint8)
                img_arr = cv2.imdecode(file_byte, 1)
                img_arr = cv2.cvtColor(img_arr, cv2.COLOR_BGR2RGB)
                img_pil = Image.fromarray(img_arr)
                img_pil = img_pil.resize((560, 420))
                img_pil_list.append(img_pil)
            input_pass = True
        else:
            wrong_names = [img_upload[iid].name for iid in err_idx]
            if LANG == 'EN':
                slot1.error('Wrong input type of input: {}'.format(wrong_names))
            elif LANG == 'CN':
                slot1.error('错误输出：{}'.format(wrong_names))
            input_pass = False

elif select_upload == select_upload_3[LANG]:
    button_pressed = False
    photo_text = {'CN': '拍摄', 'EN': 'Take a Photo'}
    image = col1.camera_input(photo_text[LANG])
    if image is not None:
        input_pass = True

if input_pass:
    col1.image(img_pil_list)
    if any(img_pil_list):
        button_text = {'CN': '处理', 'EN': 'Process'}
        button_pressed = st.sidebar.button(button_text[LANG])
        model_select = st.sidebar.selectbox('Select Model', ['Standard', 'Lite'])
        device_select = st.sidebar.selectbox('Select Device', ['cpu', 'cuda:0', 'cuda:1'])

if button_pressed:
    start = time.time()
    predict = predict_lite if model_select == 'Lite' else predict_stantard
    pred_res = []
    with st.spinner('Processing...'):
        try:
            pred = predict(img_pil_list, device=device_select, RGB=True)
        except:
            s = sys.exc_info()
            st.write(s)
            pred = []
        finally:
            pred_res = pred
            torch.cuda.empty_cache()
    end = time.time()
    elapsed = (end-start)*1000
    finish_text_EN = 'Finished in {:.2f} ms, {:.2f} ms per image'.format(elapsed, elapsed/len(img_pil_list))
    finish_text_CN = ''
    if len(pred_res) > 0:
        slot1.success(finish_text_EN)
        col2.image(pred)
    else:
        slot1.error('Failed: CUDA error!')


