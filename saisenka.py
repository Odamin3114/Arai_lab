import numpy as np
import cv2
from collections import deque
from typing import List, Tuple

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

def Zhang_Suen_thinning(binary_image):
    image_thinned = binary_image.copy()
    changing = True
    
    while changing:
        changing = False
        to_remove = []
        rows, cols = image_thinned.shape

        # 近傍8ピクセルの取得
        p2 = np.roll(image_thinned, -1, axis=0)  # 上
        p6 = np.roll(image_thinned, 1, axis=0)   # 下
        p4 = np.roll(image_thinned, -1, axis=1)  # 右
        p8 = np.roll(image_thinned, 1, axis=1)   # 左
        p3 = np.roll(p2, -1, axis=1)  # 右上
        p9 = np.roll(p2, 1, axis=1)   # 左上
        p5 = np.roll(p6, -1, axis=1)  # 右下
        p7 = np.roll(p6, 1, axis=1)   # 左下

        # 2ステップ処理
        for step in range(2):
            neighbors = p2 + p3 + p4 + p5 + p6 + p7 + p8 + p9
            transitions = ((p2 == 0) & (p3 == 1)).astype(int) + \
                          ((p3 == 0) & (p4 == 1)).astype(int) + \
                          ((p4 == 0) & (p5 == 1)).astype(int) + \
                          ((p5 == 0) & (p6 == 1)).astype(int) + \
                          ((p6 == 0) & (p7 == 1)).astype(int) + \
                          ((p7 == 0) & (p8 == 1)).astype(int) + \
                          ((p8 == 0) & (p9 == 1)).astype(int) + \
                          ((p9 == 0) & (p2 == 1)).astype(int)

            # 条件を満たすピクセルの取得
            condition = (image_thinned == 1) & \
                        (neighbors >= 2) & (neighbors <= 6) & \
                        (transitions == 1) & \
                        ((p2 * p4 * p6 == 0) if step == 0 else (p2 * p4 * p8 == 0)) & \
                        ((p4 * p6 * p8 == 0) if step == 0 else (p2 * p6 * p8 == 0))

            to_remove.append(np.where(condition))

            # 削除
            image_thinned[to_remove[step]] = 0
            if len(to_remove[step][0]) > 0:
                changing = True

    return image_thinned



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


image = cv2.imread('output_thinned_boxes/thinned_box_9.jpg', cv2.IMREAD_GRAYSCALE)
print('load image')

blur = cv2.GaussianBlur(image, (5, 5), 3)
_, th2 = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
th2 = th2[1:-1, 1:-1]  # 外枠を削除
print('Binarization Done')

thinned_image = Zhang_Suen_thinning(th2)
thinned_image = thinned_image[1:-1, 1:-1]  # 外枠を再度削除
print('Saisenka Done')

segments = divide_branched_chain(thinned_image, thinned_image.shape[1], thinned_image.shape[0], False)
print('Segmentation Done')

print(f"Total segments: {len(segments)}")
for i, segment in enumerate(segments):
    mask = np.zeros_like(thinned_image)  # 外枠削除後のサイズ
    for j in range(len(segment) - 1):
        cv2.line(mask, segment[j], segment[j+1], 255, 1)
    cv2.imwrite(f'segment_{i}.png', mask)
    cv2.imshow(f'Segment {i}', mask)

cv2.imshow('Thinned Image', thinned_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
