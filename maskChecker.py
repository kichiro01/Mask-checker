import picamera
import picamera.array
import cv2 as cv
import io
import subprocess

import cognitive_face as CF
# faceAPIの設定
KEY = 'fd43800969d8471c99d6eeabbe571875'
BASE_URL = 'https://japaneast.api.cognitive.microsoft.com/face/v1.0'

CF.Key.set(KEY)
CF.BaseUrl.set(BASE_URL)

#音声出力のシェルコマンドを定義
cmdSuccess = "mplayer -ao alsa:device=bt-receiver /home/pi/shakin.mp3"
cmdFail = "mplayer -ao alsa:device=bt-receiver /home/pi/outSound.mp3"

# 目検出のための学習元データを読み込む
eye_cascade = cv.CascadeClassifier('/home/pi/opencv/data/haarcascades/haarcascade_eye.xml')
# カメラ初期化
with picamera.PiCamera() as camera:
    # カメラの画像をリアルタイムで取得するための処理 streamって名前でnumpy配列が格納される
    with picamera.array.PiRGBArray(camera) as stream:
        # 解像度の設定
        camera.resolution = (512, 384)
        failFlag = False
        successFlag = False
        
        def checkMask(image):    
            img_url = "greyImage.png"
            cv.imwrite(img_url, image)
            #faceApi叩く
            faces = CF.face.detect(img_url, attributes='accessories')
            if len(faces)==0:
                print('認識失敗')
            else:
                global failFlag #この関数内でグローバル変数に代入できるようにする
                global successFlag #この関数内でグローバル変数に代入できるようにする
                for face in faces:
                    accessories = face['faceAttributes']['accessories'] #アクセサリーーのリスト
                    accessory_types = [d.get('type') for d in accessories] #アクセサリータイプのみ抽出した配列
                    strs = "".join(accessory_types) #配列の中に含まれる文字列を羅列
                    #文字列にmaskという文字列が含まれているかどうか確認
                    if 'mask' in strs:
                        print('MaskOn')
                        if successFlag:
                            pass
                        else:
                            successFlag = True
                            failFlag = False
                            subprocess.call(cmdSuccess.split())
                        
                    else:
                        print('NoMask')
                        if failFlag:
                            pass
                        else:
                            failFlag = True
                            successFlag = False
                            subprocess.call(cmdFail.split())
                            

        while True:
            # カメラから映像を取得する（OpenCVへ渡すために、各ピクセルの色の並びをBGRの順番にする）
            camera.capture(stream, 'bgr', use_video_port=True)
            # 顔検出の処理効率化のために、写真の情報量を落とす（モノクロにする）
            grayimg = cv.cvtColor(stream.array, cv.COLOR_BGR2GRAY)
            # 目の検出を行う
            facerect = eye_cascade.detectMultiScale(grayimg)
            
            # 目が検出された場合
            if len(facerect) > 0:
                print('瞳が検出されました')
                checkMask(grayimg)
            else:
                print('瞳の検出に失敗しました')
            

            # 結果の画像を表示する
            cv.imshow('camera', stream.array)
            # カメラから読み込んだ映像を破棄する
            stream.seek(0)
            stream.truncate()
            # 何かキーが押されたかどうかを検出する（検出のため、1ミリ秒待つ）
            if cv.waitKey(1) > 0:
                break

        # 表示したウィンドウを閉じる
        cv.destroyAllWindows()              
    
