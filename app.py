import streamlit as st
from PIL import Image
import cv2
import numpy as np
import math

def pil2cv(image):
    # pillow型→opencv型
    new_image = np.array(image, dtype=np.uint8)
    if new_image.ndim == 2:  # モノクロ
        pass
    elif new_image.shape[2] == 3:  # カラー
        new_image = cv2.cvtColor(new_image, cv2.COLOR_RGB2BGR)
    elif new_image.shape[2] == 4:  # 透過
        new_image = cv2.cvtColor(new_image, cv2.COLOR_RGBA2BGRA)
    return new_image

def cv2pil(image):
    # opencv型→pillow型
    new_image = image.copy()
    if new_image.ndim == 2:  # モノクロ
        pass
    elif new_image.shape[2] == 3:  # カラー
        new_image = cv2.cvtColor(new_image, cv2.COLOR_BGR2RGB)
    elif new_image.shape[2] == 4:  # 透過
        new_image = cv2.cvtColor(new_image, cv2.COLOR_BGRA2RGBA)
    new_image = Image.fromarray(new_image)
    return new_image

def expa_shrink(img, x, y):
    # 拡大・縮小
    img_cv = pil2cv(img)
    h, w, c = img_cv.shape
    mat = np.array([[x, 0, 0], [0, y, 0]], dtype=np.float32)
    affine_img_scale_x = cv2.warpAffine(img_cv, mat, (int(w * 2), h))
    new_img = cv2pil(affine_img_scale_x)
    return new_img

def symmetry_onlyX(img):
    # # xy軸対称
    img_cv = pil2cv(img)
    h, w, c = img_cv.shape
    mat = np.array([[1, 0, 0], [0, -1, 0]], dtype=np.float32)
    affine_img_scale_xy = cv2.warpAffine(img_cv, mat, (int(w * 2), h))
    new_img = cv2pil(affine_img_scale_xy)
    return new_img

def symmetry_onlyY(img):
    # # xy軸対称
    img_cv = pil2cv(img)
    h, w, c = img_cv.shape
    mat = np.array([[-1, 0, 0], [0, 1, 0]], dtype=np.float32)
    affine_img_scale_xy = cv2.warpAffine(img_cv, mat, (int(w * 2), h))
    new_img = cv2pil(affine_img_scale_xy)
    return new_img

def symmetry_xy(img):
    # # xy軸対称
    img_cv = pil2cv(img)
    h, w, c = img_cv.shape
    mat = np.array([[0, 1, 0], [1, 0, 0]], dtype=np.float32)
    affine_img_scale_xy = cv2.warpAffine(img_cv, mat, (int(w * 2), h))
    new_img = cv2pil(affine_img_scale_xy)
    return new_img

def rotate(img, angle):
    img_cv = pil2cv(img)
    height, width = img_cv.shape[:2]
    center = (width // 2, height // 2)
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated_image = cv2.warpAffine(img_cv, rotation_matrix, (width, height))
    new_img = cv2pil(rotated_image)
    return new_img

def skew_x(img, x):
    #x軸スキュー変換
    img_cv=pil2cv(img)
    h, w, c = img_cv.shape
    tan = math.tan(math.radians(x))
    mat = np.array([[1, tan, 0], [0, 1, 0]], dtype=np.float32)
    affine_img_skew_x = cv2.warpAffine(img_cv, mat, (int(w + h * tan), h))
    new_img = cv2pil(affine_img_skew_x)
    return new_img

def skew_y(img, y):
    #y軸スキュー変換
    img_cv=pil2cv(img)
    h, w, c = img_cv.shape
    tan = math.tan(math.radians(y))
    mat = np.array([[1, 0, 0], [tan, 1, 0]], dtype=np.float32)
    affine_img_skew_y = cv2.warpAffine(img_cv, mat, (int(w + h * tan), h))
    new_img = cv2pil(affine_img_skew_y)
    return new_img

upload_file = st.file_uploader('画像をアップロード', ['png', 'jpg', 'jpeg'])

if upload_file:
    cmd = st.selectbox(label="変換方法を選んでください",options=[
        '拡大・縮小', 'X軸対称', 'Y軸対称', 'Y=X軸対称', '回転', 'X軸方向の歪み', 'Y軸方向の歪み'
        ])
    img_pil = Image.open(upload_file)
    
    if cmd == '拡大・縮小':
        x = st.sidebar.number_input('x軸倍率', 0, 10)
        y = st.sidebar.number_input('y軸倍率', 0, 10)
        exec_button = st.button('実行')
        if exec_button:
            image = expa_shrink(img_pil, x, y)
            st.image(image, use_column_width=True)
    elif cmd == 'X軸対称':
        exec_button = st.button('実行')
        if exec_button:
            image = symmetry_onlyX(img_pil)
            st.image(image, use_column_width=True)
    elif cmd == 'Y軸対称':
        exec_button = st.button('実行')
        if exec_button:
            image = symmetry_onlyY(img_pil)
            st.image(image, use_column_width=True)
    elif cmd == 'Y=X軸対称':
        exec_button = st.button('実行')
        if exec_button:
            image = symmetry_xy(img_pil)
            st.image(image, use_column_width=True)
    elif cmd == '回転':
        angle = st.sidebar.number_input('角度', 0, 360)
        exec_button = st.button('実行')
        if exec_button:
            image = rotate(img_pil, angle)
            st.image(image, use_column_width=True)
    elif cmd == 'X軸方向の歪み':
        x = st.sidebar.number_input('角度', 0, 360)
        exec_button = st.button('実行')
        if exec_button:
            image = skew_x(img_pil, x)
            st.image(image, use_column_width=True)
    elif cmd == 'Y軸方向の歪み':
        y = st.sidebar.number_input('角度', 0, 360)
        exec_button = st.button('実行')
        if exec_button:
            image = skew_y(img_pil, y)
            st.image(image, use_column_width=True)
    else:
        pass