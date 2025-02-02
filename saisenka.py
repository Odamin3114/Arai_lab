import matplotlib.pyplot as plt
import cv2
import numpy as np
#https://qiita.com/hrs1985/items/7751d4b5241d5c314a6d


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


# 画像を読み込みます
image = cv2.imread('ichigo2/ichigo7.jpg')
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
new_image = inv_black_one(unpadding(new_image))
# 結果を出力します
plt.subplot(121), plt.imshow(image_gray, cmap='gray')
plt.title("input image")
plt.subplot(122), plt.imshow(result_image, cmap='gray')
plt.title("thinned image")
plt.show()
