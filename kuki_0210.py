#茎とイチゴを一対一にするところまでのプログラム

import pyrealsense2 as rs
import numpy as np
import cv2
import time
from ultralytics import YOLO
import os
from sklearn.decomposition import PCA
from scipy.stats import linregress
import csv
from collections import defaultdict


# 画像ファイル場所とかの定義
script_dir = os.path.dirname(__file__)
model_path = os.path.join(script_dir, 'best.pt')
input_image_path = os.path.join(script_dir, 'ichigo2/ichigo3.jpg')
print(f"photo get")

#ichigo4.jpgは細線化処理が必要

# YOLOモデルをロード
model = YOLO(model_path)
# 画像読み込み
cv_image = cv2.imread(input_image_path)
print(f"load")
# YOLOで推論
results = model(cv_image, save=True,hide_conf=False,hide_labels=False) #Falseのとき尤度表示等off
print(f"YOLO start")


# バウンディングボックスの情報を取得
boxes = []
for result in results:
    for box in result.boxes:
        label = int(box.cls[0])
        cx, cy, w, h = box.xywhn[0]
        boxes.append((label, cx, cy, w, h))

#print(boxes)

# ラベルごとのバウンディングボックスの座標を格納するリストを定義
label_0_bboxes = [] #peduncle
label_1_bboxes = [] #ripe

print(f"BB get")
# label=0 のボックスを探して処理
# label=1 のボックスだけを格納
# label_0_boxesリストを作成: label=0 のボックスだけをフィルタリング
label_0_boxes = [b for b in boxes if b[0] == 0]

#print(label_0_boxes)

for b in label_0_boxes:  # boxesリストにある各要素を順番に取り出して、変数bに代入
    label, center_x, center_y, width, height = b
    if label != 0:  # labelが0でない場合はスキップ（念のための保険）
        print(f"Skipping label {label} (not 0)")
        continue
    
    # 画像の高さと幅を取得
    h, w = cv_image.shape[:2]
    # バウンディングボックスの座標を計算
    x_min = int((center_x - width / 2) * w)  # 左端
    y_min = int((center_y - height / 2) * h)  # 上端
    x_max = int((center_x + width / 2) * w)  # 右端
    y_max = int((center_y + height / 2) * h)  # 下端

     # バウンディングボックスをリストに格納
    label_0_bboxes.append((x_min, y_min, x_max, y_max))

    
    # 範囲外の座標を修正（画像の範囲内に収める）
    x_min = max(x_min, 0)
    y_min = max(y_min, 0)
    x_max = min(x_max, w)
    y_max = min(y_max, h)
     # バウンディングボックスをリストに格納
    label_0_bboxes.append((x_min, y_min, x_max, y_max))

    # バウンディングボックス内を切り取る
    cropped_image = cv_image[y_min:y_max, x_min:x_max]
    #print(f"cropped image size: {cropped_image.shape}")
    
    # 緑色のピクセルを抽出
    # 緑の閾値設定
    lower_green = np.array([10, 40, 40])  # 色相を広げて、より多くの緑色をキャッチ
    upper_green = np.array([80, 255, 255]) 
    
    #lower_greenのHを下げてより多くの緑色を含める
    #upper_greenのHを上げてより明るい緑や黄色っぽい色も含める
 
    
    # BGR -> HSVに変換
    hsv_image = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2HSV)
    
    # 緑色部分を抽出
    mask = cv2.inRange(hsv_image, lower_green, upper_green)
    green_pixels = cv2.bitwise_and(cropped_image, cropped_image, mask=mask)

    # 緑色ピクセルが抽出されていない場合のチェック
    green_indices = np.where((green_pixels[:,:,0] != 0) & (green_pixels[:,:,1] != 0) & (green_pixels[:,:,2] != 0))
    if len(green_indices[0]) == 0:
        print("PCAを計算できる緑色ピクセルがありませんでした。")
        continue

    # (x, y)座標に変換
    green_pixels_coordinates = np.column_stack((green_indices[1], green_indices[0]))
    
    # PCAの計算
    pca = PCA(n_components=2)
    pca.fit(green_pixels_coordinates)
    first_eigen_vector = pca.components_[0]
    #print(f"PCA done: First eigen vector {first_eigen_vector}")

    # バウンディングボックスを画像上に描画
    cv2.rectangle(cv_image, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

    # (x, y)座標に変換
    green_pixels_coordinates = np.column_stack((green_indices[1], green_indices[0]))
    
    # 緑色部分を保存・表示
    green_output_path = os.path.join(script_dir, "green_extracted.jpg")
    cv2.imwrite(green_output_path, green_pixels)  # 緑色部分を保存
    print(f"緑色の切り取り完了")


# label=1 のボックスを探してPCA計算まで
#ここでlabel = 0を含みたくない
# label=1 のボックスをフィルタリングしてリストに格納
label_1_boxes = [b for b in boxes if b[0] == 1]
#print(label_1_boxes)

# 各label=1 のボックスに対して処理を行う
for b in label_1_boxes:
    label, center_x, center_y, width, height = b
    #print(b)
    #ここまではok(Debug)

    # ここで再度 label が 1 であることを確認する
    if label != 1:
        continue
    #NumPy配列のスライシング操作でheightとwidthを抽出、h,wに格納
    h, w = cv_image.shape[:2] 
    #mask処理のために正規化してint型に格納
    x_min = int((center_x - width / 2) * w) #左端
    y_min = int((center_y - height / 2) * h) #左上
    x_max = int((center_x + width / 2) * w) #右端
    y_max = int((center_y + height / 2) * h) #右下

     # バウンディングボックスをリストに格納
    label_1_bboxes.append((x_min, y_min, x_max, y_max))

    cropped_image = cv_image[y_min:y_max, x_min:x_max]

    #赤色閾値
    lower_red=np.array([0,0,50])
    upper_red=np.array([255,255,255])
    mask = cv2.inRange(cropped_image, lower_red, upper_red)
    red_pixels = cv2.bitwise_and(cropped_image, cropped_image, mask=mask)
    
    red_indices = np.where((red_pixels[:,:,0] != 0) & (red_pixels[:,:,1] != 0) & (red_pixels[:,:,2] != 0))
    if len(red_indices[0]) == 0:
        continue
    
    red_pixels_coordinates = np.column_stack((red_indices[1], red_indices[0]))
    pca = PCA(n_components=2)
    pca.fit(red_pixels_coordinates)

    first_eigen_vector = pca.components_[0]  #PCAの第一主成分、赤ピクセルの分布方向ベクトル
    
    # BBの各点を定義
    BB_left_lower=(x_min, y_min)
    BB_left_higher=(x_min, y_max)
    BB_right_lower=(x_max, y_min)
    BB_right_higher=(x_max, y_max)

    # バウンディングボックスを画像上に描画
    #cv2.rectangle(cv_image, (x_min, y_min), (x_max, y_max), (0, 0, 255), 2)


#各labelの表示(Debug)
#print(label_0_bboxes) #x_min, y_min, x_max, y_max
#print(label_1_bboxes)

#BBの重心を求める関数
def moment(x_min,y_min, x_max, y_max):
    x_moment=(x_min+x_max)/2
    y_moment=(y_min+y_max)/2
    return x_moment, y_moment

# 重心のリストを作成
label_0_centers = [moment(x_min, y_min, x_max, y_max) for x_min, y_min, x_max, y_max in label_0_bboxes]
label_1_centers = [moment(x_min, y_min, x_max, y_max) for x_min, y_min, x_max, y_max in label_1_bboxes]

# 距離計算関数
def distance(center1, center2):
    return np.sqrt((center1[0] - center2[0]) ** 2 + (center1[1] - center2[1]) ** 2)

# 最も近いペアを探す
closest_pairs_distance = []

for i, center1 in enumerate(label_0_centers):
    min_dist = float('inf')
    closest_index = -1
    for j, center2 in enumerate(label_1_centers):
        dist = distance(center1, center2)
        if dist < min_dist:
            min_dist = dist
            closest_index = j
    # 最も近いペアを保存
    closest_pairs_distance.append(((label_0_bboxes[i], label_1_bboxes[closest_index]), min_dist))

# 最も近いペアを描画する色の生成
def generate_color(index):
    np.random.seed(index)  # 同じインデックスには同じ色を生成
    return (np.random.randint(0, 256), np.random.randint(0, 256), np.random.randint(0, 256))


#step2 共有面積を求める
# 共有面積を計算する関数
def intersection_area(label_0_bbox, label_1_bbox):
    x_min0, y_min0, x_max0, y_max0 = label_0_bbox
    x_min1, y_min1, x_max1, y_max1 = label_1_bbox
    
    # 交差範囲の座標を求める
    inter_left = max(x_min0, x_min1)
    inter_top = max(y_min0, y_min1)
    inter_right = min(x_max0, x_max1)
    inter_bottom = min(y_max0, y_max1)
    
    # 交差面積がある場合のみ計算
    if inter_right > inter_left and inter_bottom > inter_top:
        width = inter_right - inter_left
        height = inter_bottom - inter_top
        return width * height, (inter_left, inter_top, inter_right, inter_bottom)  # 面積と交差領域の座標
    return 0, None  # 面積0、交差領域なし

# 各label1のBBに対して最も共有面積の大きいlabel0のBBを見つける
def find_closest_label0_for_label1(label0_bboxes, label1_bboxes):
    closest_pairs_share_surface = []
    
    for label1_bbox in label1_bboxes:
        max_area = 0
        best_match = None
        
        for label0_bbox in label0_bboxes:
            area, _ = intersection_area(label1_bbox, label0_bbox)  # 面積と交差領域を取得
            # 最も共有面積が大きいlabel0のBBを選択
            if area > max_area:
                max_area = area
                best_match = label0_bbox
        
        closest_pairs_share_surface.append((label1_bbox, best_match, max_area))
    
    return closest_pairs_share_surface


# label_0_bboxes と label_1_bboxes が既に定義されている前提で、最も共有面積が大きいペアを探す
closest_pairs_share_surface = find_closest_label0_for_label1(label_0_bboxes, label_1_bboxes)

# 結果を表示
# for label1_bbox, label0_bbox, area in closest_pairs_share_surface:
#     print(f"label1 bbox: {label1_bbox}")
#     print(f"label0 bbox: {label0_bbox}")
#     print(f"共有面積: {area}")
#     print("-" * 30)

#print(closest_pairs_distance)


# closest_pairs_distanceの内容をCSVに保存
output_distance_csv = "closest_pairs_distance.csv"
with open(output_distance_csv, mode='w', newline='') as file:
    writer = csv.writer(file)
    
    # ヘッダーを書き込む
    writer.writerow(["label_0_bbox", "label_1_bbox", "distance"])
    
    # closest_pairs_distance内の各ペアをCSVに書き込む
    for (label_0_bbox, label_1_bbox), dist in closest_pairs_distance:
        writer.writerow([label_0_bbox, label_1_bbox, dist])

print(f"CSVファイル '{output_distance_csv}' に保存完了")


# 共有面積を計算する関数
def intersection_area(label_0_bbox, label_1_bbox):
    x_min0, y_min0, x_max0, y_max0 = label_0_bbox
    x_min1, y_min1, x_max1, y_max1 = label_1_bbox
    
    # 交差範囲の座標を求める
    inter_left = max(x_min0, x_min1)
    inter_top = max(y_min0, y_min1)
    inter_right = min(x_max0, x_max1)
    inter_bottom = min(y_max0, y_max1)
    
    # 交差面積がある場合のみ計算
    if inter_right > inter_left and inter_bottom > inter_top:
        width = inter_right - inter_left
        height = inter_bottom - inter_top
        return width * height, (inter_left, inter_top, inter_right, inter_bottom)  # 交差領域の座標も返す
    return 0, None


# cv_image = cv2.imread(input_image_path)

# 最も共有面積が大きいペアを取得
closest_pairs_share_surface = find_closest_label0_for_label1(label_0_bboxes, label_1_bboxes)

# CSV保存
output_share_surface_csv = "closest_pairs_share_surface.csv"
with open(output_share_surface_csv, mode='w', newline='') as file:
    writer = csv.writer(file)
    
    # ヘッダーを書き込む
    writer.writerow(["label_0_bbox", "label_1_bbox", "area"])
    
    # closest_pairs_share_surface内の各ペアをCSVに書き込む
    for label_1_bbox, label_0_bbox, area in closest_pairs_share_surface:
        writer.writerow([label_0_bbox, label_1_bbox, area])

print(f"CSVファイル '{output_share_surface_csv}' に保存完了")

# バウンディングボックスと交差領域を描画
# cv_image_with_bboxes = draw_bboxes_with_intersection(cv_image, label_0_bboxes, label_1_bboxes, closest_pairs_share_surface)



#一つの茎のBBに対して、イチゴのBBの共有面積が2つ以上共有されるものにduplicateラベルをふる


def find_matching_pairs_and_count(closest_pairs_distance, closest_pairs_share_surface):
    pair_count = defaultdict(int)  # 重複をカウントするための辞書
    pair = []  # 結果を格納するリスト
    
    # closest_pairs_distance 内のペアを取り出す
    for ((label_0_bbox_1, label_1_bbox_1), dist) in closest_pairs_distance:
        # closest_pairs_share_surface 内のペアと比較
        for label_1_bbox_2, label_0_bbox_2, area in closest_pairs_share_surface:
            # 同じ label_0_bbox と label_1_bbox のペアを見つける
            if label_0_bbox_1 == label_0_bbox_2 and label_1_bbox_1 == label_1_bbox_2:
                # 一致する場合、重複数を+1
                pair_count[(label_1_bbox_1, label_0_bbox_1)] += 1
    
    # 結果をリストに格納（pair_countから）
    for pair_key, count in pair_count.items():
        pair.append((pair_key, count))  # 結果を順次追加
    return pair

# 両方のリストを一致する組み合わせを探して統合
result = find_matching_pairs_and_count(closest_pairs_distance, closest_pairs_share_surface)


# CSV保存
output_share_surface_csv = "pair.csv"
with open(output_share_surface_csv, mode='w', newline='') as file:
    writer = csv.writer(file)
    
    # ヘッダーを書き込む
    writer.writerow(["pair", "count"])  # ヘッダー行を追加
    
    # result内の各ペアと重複回数をCSVに書き込む
    for pair, count in result:
        writer.writerow([pair, count])
print(f"CSVファイル '{output_share_surface_csv}' に保存完了")


#タプルを解消
List_pair = []  # 空のリストを作成

for i in range(len(result)):
    pair = list(result[i])  # result[i]をリストに変換(読み取り専用から編集可へ)
    List_pair.append(pair)   # pairをList_pairに追加
    #print(pair)  # pairを表示


#print((List_pair[0][0][0]))  




# List_pairのリストをループで回して茎の傾きを取得
#1が茎
for i in range(len(List_pair)):
    #print(List_pair[i])
    (x_min, y_min, x_max, y_max) = List_pair[i][0][1]
    #print(x_min, y_min, x_max, y_max)

    # 緑色のピクセルを抽出
    lower_green = np.array([10, 40, 40])  
    upper_green = np.array([80, 255, 255]) 

    cropped_image = cv_image[y_min:y_max, x_min:x_max]

    # BGR -> HSVに変換
    hsv_image = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2HSV)
    
    # 緑色部分を抽出
    mask = cv2.inRange(hsv_image, lower_green, upper_green)
    green_pixels = cv2.bitwise_and(cropped_image, cropped_image, mask=mask)


    # 緑色ピクセルが抽出されていない場合のチェック
    green_indices = np.where((green_pixels[:,:,0] != 0) & (green_pixels[:,:,1] != 0) & (green_pixels[:,:,2] != 0))
    if len(green_indices[0]) == 0:
        print("PCAを計算できる緑色ピクセルがありませんでした。")
    else:
        # (x, y)座標に変換
        green_pixels_coordinates = np.column_stack((green_indices[1], green_indices[0]))

        # 最小二乗法で直線近似（polyfitを使って1次関数をフィットさせる） 
        x_coords = green_pixels_coordinates[:, 0]
        y_coords = -green_pixels_coordinates[:, 1] #画像座標系ではy軸の正負が逆, 鉛直下向きが正
        #print(x_coords)

        # 最小二乗法で直線を近似 (1次関数)
        slope, intercept = np.polyfit(x_coords, y_coords, 1)

        # 直線の方程式: y = slope * x + intercept
        print(f"ペア {List_pair[i]} の最小二乗法による近似直線の傾き: {slope}, 切片: {intercept}")

        # 直線の描画範囲を決定（バウンディングボックス内のx座標に基づいて）
        x_line = np.array([x_min, x_max])  # バウンディングボックス内のx_minからx_maxまで
        y_line = slope * (x_line-x_min) + intercept - y_min  # オフセットとしてx_min, y_minを追記
        #この時点では、x,y座標は返還後の状態、imageには対応していない

        #y軸座標の向きを逆転させて、元に戻す
        y_line = -y_line

       # 近似直線を描画
        line_start = (int(x_line[0]), int(y_line[0]))  
        #print(line_start)
        line_end = (int(x_line[1]), int(y_line[1]))    
        #print(line_end)
        cv2.line(cv_image, line_start, line_end, (255, 0, 0), 2)

        # バウンディングボックスを画像上に描画
        cv2.rectangle(cv_image, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)  # 緑色で矩形を描画

        # 緑色ピクセルを画像に点として描画
        # BBの座標を足すために、範囲を(x + x_min, y + y_min)に
        # for (x, y) in green_pixels_coordinates:
        # cv2.circle(cv_image, (x + x_min, y + y_min), 1, (0, 255, 255), 1)  # 黄色の点で緑色ピクセルを描画


# 結果の保存
output_image_path = "output_with_fitted_lines.jpg"
cv2.imwrite(output_image_path, cv_image)

# 結果の画像を表示
cv2.imshow("Image with Bounding Box and Fitted Lines", cv_image)
cv2.waitKey(0)
cv2.destroyAllWindows()