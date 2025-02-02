#茎のBBを細線化処理
import cv2
import os
from ultralytics import YOLO
import numpy as np
import matplotlib.pyplot as plt

# 画像ファイル場所などの定義
script_dir = os.path.dirname(__file__)
model_path = os.path.join(script_dir, 'best.pt')
input_image_path = os.path.join(script_dir, 'ichigo2/ichigo4.jpg')

# 出力ディレクトリを作成
output_dir = os.path.join(script_dir, 'output_boxes')
output_thinned_dir = os.path.join(script_dir, 'output_thinned_boxes')
os.makedirs(output_dir, exist_ok=True)
os.makedirs(output_thinned_dir, exist_ok=True)

print(f"photo get")

# YOLOモデルをロード
model = YOLO(model_path)
# 画像読み込み
cv_image = cv2.imread(input_image_path)
print(f"load")
# YOLOで推論
results = model(cv_image, save=True, hide_conf=False, hide_labels=False)  # Falseのとき尤度表示等off
print(f"YOLO start")
# バウンディングボックスの情報を取得
boxes = []
for result in results:
    for box in result.boxes:
        label = int(box.cls[0])
        cx, cy, w, h = box.xywhn[0]
        boxes.append((label, cx, cy, w, h))

print(f"BB get")



# Zhang-Suenのアルゴリズムを用いて2値化画像を細線化します
def Zhang_Suen_thinning(binary_image):
    # オリジナルの画像をコピー
    image_thinned = binary_image.copy()
    # 初期化します。この値は次のwhile文の中で除かれます。
    changing_1 = changing_2 = [1]
    while changing_1 or changing_2:
        # ステップ1
        changing_1 = []
        rows, columns = image_thinned.shape
        for x in range(1, rows - 1):
            for y in range(1, columns -1):
                p2, p3, p4, p5, p6, p7, p8, p9 = neighbour_points = neighbours(x, y, image_thinned)
                if (image_thinned[x][y] == 1 and
                    2 <= sum(neighbour_points) <= 6 and # 条件2
                    count_transition(neighbour_points) == 1 and # 条件3
                    p2 * p4 * p6 == 0 and # 条件4
                    p4 * p6 * p8 == 0): # 条件5
                    changing_1.append((x,y))
        for x, y in changing_1:
            image_thinned[x][y] = 0
        # ステップ2
        changing_2 = []
        for x in range(1, rows - 1):
            for y in range(1, columns -1):
                p2, p3, p4, p5, p6, p7, p8, p9 = neighbour_points = neighbours(x, y, image_thinned)
                if (image_thinned[x][y] == 1 and
                    2 <= sum(neighbour_points) <= 6 and # 条件2
                    count_transition(neighbour_points) == 1 and # 条件3
                    p2 * p4 * p8 == 0 and # 条件4
                    p2 * p6 * p8 == 0): # 条件5
                    changing_2.append((x,y))
        for x, y in changing_2:
            image_thinned[x][y] = 0        
    
    return image_thinned


# 2値画像の黒を1、白を0とするように変換するメソッドです
def black_one(binary):
    bool_image = binary.astype(bool)
    inv_bool_image = ~bool_image
    return inv_bool_image.astype(int)


# 画像の外周を0で埋めるメソッドです
def padding_zeros(image):
    import numpy as np
    m, n = np.shape(image)
    padded_image = np.zeros((m + 2, n + 2))
    padded_image[1:-1, 1:-1] = image
    return padded_image


# 外周1行1列を除くメソッドです
def unpadding(image):
    return image[1:-1, 1:-1]


# 指定されたピクセルの周囲のピクセルを取得するメソッドです
def neighbours(x, y, image):
    return [image[x-1][y], image[x-1][y+1], image[x][y+1], image[x+1][y+1],  # 2, 3, 4, 5
            image[x+1][y], image[x+1][y-1], image[x][y-1], image[x-1][y-1]]  # 6, 7, 8, 9


# 0→1の変化の回数を数えるメソッドです
def count_transition(neighbours):
    neighbours += neighbours[:1]
    return sum((n1, n2) == (0, 1) for n1, n2 in zip(neighbours, neighbours[1:]))


# 黒を1、白を0とする画像を、2値画像に戻すメソッドです
def inv_black_one(inv_bool_image):
    bool_image = ~inv_bool_image.astype(bool)
    return bool_image.astype(int)


# label=0 のボックスを探して処理
for idx, b in enumerate(boxes):  # boxesリストにある各要素を順番に取り出して、変数bに代入
    label, center_x, center_y, width, height = b
    if label != 0:  # labelが0でない場合はスキップ
        print(f"Skipping label {label} (not 0)")
        continue
    
    # 画像の高さと幅を取得
    h, w = cv_image.shape[:2]
    # バウンディングボックスの座標を計算
    x_min = int((center_x - width / 2) * w)  # 左端
    y_min = int((center_y - height / 2) * h)  # 上端
    x_max = int((center_x + width / 2) * w)  # 右端
    y_max = int((center_y + height / 2) * h)  # 下端
    
    # 範囲外の座標を修正（画像の範囲内に収める）
    x_min = max(x_min, 0)
    y_min = max(y_min, 0)
    x_max = min(x_max, w)
    y_max = min(y_max, h)
    
    # バウンディングボックス内を切り取る
    cropped_image = cv_image[y_min:y_max, x_min:x_max]
    print(f"cropped image size: {cropped_image.shape}")

    # バウンディングボックス画像を保存
    output_path = os.path.join(output_dir, f"box_{idx}.jpg")
    cv2.imwrite(output_path, cropped_image)  # バウンディングボックス内の画像を保存
    print(f"バウンディングボックス画像保存完了: {output_path}")

    # バウンディングボックス画像をグレースケールに変換
    cropped_image_gray = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2GRAY)
    
    # Zhang-Suenアルゴリズムを実行する
    thinned_image = Zhang_Suen_thinning(cropped_image_gray)

    # 細線化した画像を保存
    thinned_output_path = os.path.join(output_thinned_dir, f"thinned_box_{idx}.jpg")
    cv2.imwrite(thinned_output_path, thinned_image)  # 細線化画像を保存
    print(f"細線化画像保存完了: {thinned_output_path}")

print("すべてのバウンディングボックス画像と細線化画像を保存しました。")

