import matplotlib.pyplot as plt
import cv2
import numpy as np
from collections import deque
from typing import List, Tuple


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
    m,n = np.shape(image)
    padded_image = np.zeros((m+2,n+2))
    padded_image[1:-1,1:-1] = image
    return padded_image

# 外周1行1列を除くメソッドです。
def unpadding(image):
    return image[1:-1, 1:-1]

# 指定されたピクセルの周囲のピクセルを取得するメソッドです
def neighbours(x, y, image):
    return [image[x-1][y], image[x-1][y+1], image[x][y+1], image[x+1][y+1], # 2, 3, 4, 5
             image[x+1][y], image[x+1][y-1], image[x][y-1], image[x-1][y-1]] # 6, 7, 8, 9

# 0→1の変化の回数を数えるメソッドです
def count_transition(neighbours):
    neighbours += neighbours[:1]
    return sum( (n1, n2) == (0, 1) for n1, n2 in zip(neighbours, neighbours[1:]) )

# 黒を1、白を0とする画像を、2値画像に戻すメソッドです
def inv_black_one(inv_bool_image):
    bool_image = ~inv_bool_image.astype(bool)
    return bool_image.astype(int) * 255


Point2dArray = List[Tuple[int, int]]

def i2p(i: int, width: int) -> Tuple[int, int]:
    """ Convert pixel index to (x, y) coordinates """
    x = i % width
    y = i // width
    return x, y


def find_neighbors(focus_idx: int, ignore_idx: int, start_dir: int, pix: np.ndarray, width: int, height: int,
                   searched: np.ndarray, neighbor_type4: bool) -> List[Tuple[int, int, int]]:
    """ Find neighboring pixels of the given focus index """
    offsets = [(1, 0, 0), (1, -1, 1), (0, -1, 2), (-1, -1, 3),
               (-1, 0, 4), (-1, 1, 5), (0, 1, 6), (1, 1, 7)]
    neighbors = []

    for i in range(8):
        ti = (i + start_dir) % 8
        if neighbor_type4 and ti % 2 != 0:
            continue
        
        x = offsets[ti][0] + (focus_idx % width)
        y = offsets[ti][1] + (focus_idx // width)
        test_idx = y * width + x
        
        if (test_idx != ignore_idx and 0 <= x < width and 0 <= y < height
                and searched[test_idx] == 0 and pix[test_idx] > 0):
            neighbors.append((focus_idx, test_idx, offsets[ti][2]))
    return neighbors


def get_direction(focus_idx: int, target_idx: int, width: int) -> int:
    """ Get direction from focus index to target index """
    offsets = [(1, 0, 0), (1, -1, 1), (0, -1, 2), (-1, -1, 3),
               (-1, 0, 4), (-1, 1, 5), (0, 1, 6), (1, 1, 7)]
    
    for dx, dy, d in offsets:
        x = dx + (focus_idx % width)
        y = dy + (focus_idx // width)
        if y * width + x == target_idx:
            return d
    raise Exception("Can't find direction.")


def divide_branched_chain(pix: np.ndarray, width: int, height: int, neighbor_type4: bool) -> List[Point2dArray]:
    """ 高速化した枝分かれチェーンの分割 """
    searched = np.zeros_like(pix, dtype=np.uint8)

    # OpenCVの輪郭検出を利用
    contours, _ = cv2.findContours(pix, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    dst_buf = []
    for contour in contours:
        segment = []
        for point in contour:
            x, y = point[0]
            if searched[y, x] == 0:
                segment.append((x, y))
                searched[y, x] = 1

        if len(segment) > 1:
            dst_buf.append(segment)

    return dst_buf



# 画像を読み込みます
image = cv2.imread('output_thinned_boxes/thinned_box_6.jpg')
# グレースケールに変換します
image_gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
# ガウシアンフィルタをかけます
blur = cv2.GaussianBlur(image_gray,(5,5), 3)
# 大津のアルゴリズムで2値化します
ret,th2 = cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
# 2値化画像の黒を1、白を0に変換します。外周を0で埋めておきます。
th2 = padding_zeros(th2)
new_image = black_one(th2)
# Zhang Suenアルゴリズムによる細線化を行います
result_image = inv_black_one(Zhang_Suen_thinning(new_image))
cv2.imwrite(f'result_image.jpg', result_image)
#new_image = inv_black_one(unpadding(new_image))
#cv2.imwrite(f'new_image.jpg', new_image)
thinned_image = result_image.astype(np.uint8)

cv2.imwrite(f'thinned_image.jpg', thinned_image)
#ここまでok


segments = divide_branched_chain(thinned_image, thinned_image.shape[1], thinned_image.shape[0], False)
print('Segmentation Done')

print(f"Total segments: {len(segments)}")
for i, segment in enumerate(segments):
    mask = np.zeros_like(thinned_image)  # 外枠削除後のサイズ
    for j in range(len(segment) - 1):
        cv2.line(mask, segment[j], segment[j+1], 255, 1)  # 各セグメントを描画
    # 外周の1行1列分のピクセルを消す
    mask = mask[1:-1, 1:-1]
    
    cv2.imwrite(f'segment_{i}.png', mask)
    cv2.imshow(f'Segment {i}', mask)


cv2.waitKey(0)
cv2.destroyAllWindows()

