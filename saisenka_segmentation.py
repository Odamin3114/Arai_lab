#細かいセグメント化はできた
#細かすぎ

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


def find_endpoints_and_junctions(image):
    endpoints = []
    junctions = []
    rows, cols = image.shape
    
    for x in range(1, rows - 1):
        for y in range(1, cols - 1):
            if image[x, y] == 255:
                neighbors = [image[x-1, y-1], image[x-1, y], image[x-1, y+1],
                             image[x, y-1],                 image[x, y+1],
                             image[x+1, y-1], image[x+1, y], image[x+1, y+1]]
                count = np.count_nonzero(neighbors)
                if count == 1:
                    endpoints.append((x, y))
                elif count >= 3:
                    junctions.append((x, y))
    return endpoints, junctions

def trace_segments(image, endpoints, junctions):
    visited = np.zeros_like(image, dtype=bool)
    segments = []
    
    def trace_path(x, y):
        path = [(x, y)]
        visited[x, y] = True
        
        while True:
            neighbors = [(x-1, y-1), (x-1, y), (x-1, y+1),
                         (x, y-1),           (x, y+1),
                         (x+1, y-1), (x+1, y), (x+1, y+1)]
            next_points = [(nx, ny) for nx, ny in neighbors 
                           if image[nx, ny] == 255 and not visited[nx, ny]]
            
            if not next_points or (x, y) in junctions:
                break
            
            x, y = next_points[0]
            visited[x, y] = True
            path.append((x, y))
        
        return path
    
    for x, y in endpoints:
        if not visited[x, y]:
            segment = trace_path(x, y)
            if len(segment) > 1:
                segments.append(segment)
    
    return segments

def save_segments(image, segments):
    for i, segment in enumerate(segments):
        mask = np.zeros_like(image)
        for j in range(len(segment) - 1):
            cv2.line(mask, segment[j][::-1], segment[j+1][::-1], 255, 1)
        cv2.imwrite(f'segment_{i}.png', mask)

def main():
    thinned_image = cv2.imread('thinned_image.jpg', cv2.IMREAD_GRAYSCALE)
    endpoints, junctions = find_endpoints_and_junctions(thinned_image)
    segments = trace_segments(thinned_image, endpoints, junctions)
    save_segments(thinned_image, segments)



# # 画像を読み込みます
# image = cv2.imread('output_thinned_boxes/thinned_box_6.jpg')
# # グレースケールに変換します
# image_gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
# # ガウシアンフィルタをかけます
# blur = cv2.GaussianBlur(image_gray,(5,5), 3)
# # 大津のアルゴリズムで2値化します
# ret,th2 = cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
# # 2値化画像の黒を1、白を0に変換します。外周を0で埋めておきます。
# th2 = padding_zeros(th2)
# new_image = black_one(th2)
# # Zhang Suenアルゴリズムによる細線化を行います
# result_image = inv_black_one(Zhang_Suen_thinning(new_image))
# cv2.imwrite(f'result_image.jpg', result_image)
# thinned_image = result_image.astype(np.uint8)

# cv2.imwrite(f'thinned_image.jpg', thinned_image)
# #ここまでok


# # segments = divide_branched_chain(thinned_image, thinned_image.shape[1], thinned_image.shape[0], False)
# # print('Segmentation Done')

# # print(f"Total segments: {len(segments)}")
# # for i, segment in enumerate(segments):
# #     mask = np.zeros_like(thinned_image)  # 外枠削除後のサイズ
# #     for j in range(len(segment) - 1):
# #         cv2.line(mask, segment[j], segment[j+1], 255, 1)  # 各セグメントを描画
# #     # 外周の1行1列分のピクセルを消す
# #     mask = mask[1:-1, 1:-1]
    
# #     cv2.imwrite(f'segment_{i}.png', mask)
# #     cv2.imshow(f'Segment {i}', mask)


# cv2.waitKey(0)
# cv2.destroyAllWindows()

