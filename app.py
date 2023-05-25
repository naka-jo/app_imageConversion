import streamlit as st
from PIL import Image
import cv2
import numpy as np
import math

def pil2cv(image):
    #opencv型→pillow型
    new_image = np.array(image, dtype=np.uint8)
    if new_image.ndim == 2:  # モノクロ
        pass
    elif new_image.shape[2] == 3:  # カラー
        new_image = cv2.cvtColor(new_image, cv2.COLOR_RGB2BGR)
    elif new_image.shape[2] == 4:  # 透過
        new_image = cv2.cvtColor(new_image, cv2.COLOR_RGBA2BGRA)
    return new_image

def cv2pil(image):
    #opencv型→pillow型
    new_image = image.copy()
    if new_image.ndim == 2:  # モノクロ
        pass
    elif new_image.shape[2] == 3:  # カラー
        new_image = cv2.cvtColor(new_image, cv2.COLOR_BGR2RGB)
    elif new_image.shape[2] == 4:  # 透過
        new_image = cv2.cvtColor(new_image, cv2.COLOR_BGRA2RGBA)
    new_image = Image.fromarray(new_image)
    return new_image

def expa():
    # 画像の読み込み
    img = cv2.imread('.png')
    h, w, c = img.shape

    img_big_x = input("x軸に何倍拡大するか？：")
    img_big_y = input("y軸に何倍拡大するか？：")
    # アフィン変換行列の作成
    mat = np.array([[img_big_x, 0, 0], [0, img_big_y, 0]], dtype=np.float32)

    # アフィン変換の適用
    affine_img_scale_x = cv2.warpAffine(img, mat, (int(w * 2), h))

    # 変換後の画像の保存
    cv2.imwrite('/content/drive/MyDrive', affine_img_scale_x)
    
def shrink():
    # 画像の読み込み
    img = cv2.imread('.png')
    h, w, c = img.shape

    img_big_x = input("x軸に何倍拡大するか？：")
    img_big_y = input("y軸に何倍拡大するか？：")
    # アフィン変換行列の作成
    mat = np.array([[img_big_x, 0, 0], [0, img_big_y, 0]], dtype=np.float32)

    # アフィン変換の適用
    affine_img_scale_x = cv2.warpAffine(img, mat, (int(w * 2), h))

    # 変換後の画像の保存
    cv2.imwrite('/content/drive/MyDrive', affine_img_scale_x)

def skew_x(img, x):
    #x軸スキュー変換
    h, w, c = img.shape
    tan = math.tan(math.radians(15))
    mat = np.array([[1, tan, 0], [0, 1, 0]], dtype=np.float32)
    affine_img_skew_x = cv2.warpAffine(img, mat, (int(w + h * tan), h))
    new_img = cv2pil(affine_img_skew_x)
    return new_img

upload_file = st.file_uploader('画像をアップロード', ['png', 'jpg', 'jpeg'])

if upload_file:
    img_pil = Image.open(upload_file)
    exec_button = st.button('実行')
    #st.image(img_pil, caption='サンプル',use_column_width=True)
    img_cv=pil2cv(img_pil)
    if exec_button:
        skew_x = skew_x(img_cv, 15)
        st.image(skew_x, caption='サンプル',use_column_width=True)