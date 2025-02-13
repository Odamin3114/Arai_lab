#List_pairの構造

#data[0] はリストの最初の要素 [ ((1205, 473, 1304, 569), (1211, 370, 1257, 495)), 2 ] 
#data[0][0] はその中の最初のタプル ((1205, 473, 1304, 569), (1211, 370, 1257, 495))
#data[0][0][0] で、最初のタプル (1205, 473, 1304, 569)


#タプルを解消
List_pair = []  # 空のリストを作成

for i in range(len(result)):
    pair = list(result[i])  # result[i]をリストに変換(読み取り専用から編集可へ)
    List_pair.append(pair)   # pairをList_pairに追加
    #print(pair)  # pairを表示

print((List_pair[0][0][0]))  
