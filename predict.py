from ultralytics import YOLO
import os

# 標籤轉換字典
MAHJONG_DICT = {
    'Character 1': '一萬', 'Character 2': '二萬', 'Character 3': '三萬',
    'Character 4': '四萬', 'Character 5': '五萬', 'Character 6': '六萬',
    'Character 7': '七萬', 'Character 8': '八萬', 'Character 9': '九萬',
    'Bamboo 1': '一條', 'Bamboo 2': '二條', 'Bamboo 3': '三條',
    'Bamboo 4': '四條', 'Bamboo 5': '五條', 'Bamboo 6': '六條',
    'Bamboo 7': '七條', 'Bamboo 8': '八條', 'Bamboo 9': '九條',
    'Circle 1': '一筒', 'Circle 2': '二筒', 'Circle 3': '三筒',
    'Circle 4': '四筒', 'Circle 5': '五筒', 'Circle 6': '六筒',
    'Circle 7': '七筒', 'Circle 8': '八筒', 'Circle 9': '九筒',
    'East': '東風', 'South': '南風', 'West': '西風', 'North': '北風',
    'Red': '紅中', 'Green': '發財', 'White': '白板'
}

# 全域設定
SUITS = ['萬', '條', '筒']
NUMBERS = ['一', '二', '三', '四', '五', '六', '七', '八', '九']

# 🏆 牌型台數計算引擎 
def calculate_tai(sequences, triplets, quads, pairs, final_hand, is_men_qing, concealed_triplets_count):
    total_tai = 0
    details = []

    dragons = ['紅中', '發財', '白板']
    winds = ['東風', '南風', '西風', '北風']
    has_honor = any(tile in dragons or tile in winds for tile in final_hand)

    suits_present = set()
    for tile in final_hand:
        if '萬' in tile: suits_present.add('萬')
        elif '條' in tile: suits_present.add('條')
        elif '筒' in tile: suits_present.add('筒')

    is_all_honors = (len(suits_present) == 0 and has_honor)

    # 1. 四槓牌 (8台) -> 新增！
    if len(quads) == 4:
        total_tai += 8
        details.append("四槓牌 (8台)")

    # 2. 碰碰胡 (4台)
    if len(sequences) == 0 and not is_all_honors:
        total_tai += 4
        details.append("碰碰胡 (4台)")

    # 3. 三元台
    dragon_triplets = [d for d in dragons if d in triplets or d in quads]
    dragon_pair = [d for d in dragons if d in pairs]
    if len(dragon_triplets) == 3:
        total_tai += 8
        details.append("大三元 (8台)")
    elif len(dragon_triplets) == 2 and len(dragon_pair) == 1:
        total_tai += 4
        details.append("小三元 (4台)")
    else:
        for d in dragon_triplets:
            total_tai += 1
            details.append(f"三元刻-{d} (1台)")

    # 4. 四喜台 
    wind_triplets = [w for w in winds if w in triplets or w in quads]
    wind_pair = [w for w in winds if w in pairs]
    if len(wind_triplets) == 4:
        total_tai += 16
        details.append("大四喜 (16台)")
    elif len(wind_triplets) == 3 and len(wind_pair) == 1:
        total_tai += 8
        details.append("小四喜 (8台)")

    # 5. 一色台
    if len(suits_present) == 0 and has_honor:
        total_tai += 16
        details.append("字一色 (16台)")
    elif len(suits_present) == 1:
        if not has_honor:
            total_tai += 8
            details.append("清一色 (8台)")
        else:
            total_tai += 4
            details.append("混一色 (4台)")

    # 6. 平胡 (2台)
    if len(triplets) == 0 and len(quads) == 0 and not has_honor:
        total_tai += 2
        details.append("平胡 (2台)")

    # 7. 門清 (1台)
    if is_men_qing:
        total_tai += 1
        details.append("門清 (1台)")

    # 8. 暗刻
    if concealed_triplets_count == 5:
        total_tai += 8
        details.append("五暗刻 (8台) *若胡別人且單吊刻子需扣台")
    elif concealed_triplets_count == 4:
        total_tai += 5
        details.append("四暗刻 (5台) *若胡別人且單吊刻子需扣台")
    elif concealed_triplets_count == 3:
        total_tai += 2
        details.append("三暗刻 (2台) *若胡別人且單吊刻子需扣台")


    return total_tai, details


# 📷 影像辨識工具
def detect_tiles_from_image(model, image_path):
    if not os.path.exists(image_path):
        return []
    results = model.predict(source=image_path, show=False, save=True, conf=0.4, agnostic_nms=True)
    detected_tiles = []
    for r in results:
        for box in r.boxes:
            x1 = float(box.xyxy[0][0]) 
            cls_id = int(box.cls[0])
            class_name = model.names[cls_id]
            detected_tiles.append({"x_pos": x1, "name": class_name})
    detected_tiles = sorted(detected_tiles, key=lambda tile: tile["x_pos"])
    return [MAHJONG_DICT.get(tile["name"], tile["name"]) for tile in detected_tiles]


# 🧠 明牌解析器
def analyze_exposed_melds(hand_tiles):
    temp_hand = list(hand_tiles)
    quads, triplets, sequences = [], [], []

    for t in set(temp_hand):
        if temp_hand.count(t) == 4:
            quads.append(t)
            for _ in range(4): temp_hand.remove(t)
        elif temp_hand.count(t) == 3:
            triplets.append(t)
            for _ in range(3): temp_hand.remove(t)

    for suit in SUITS:
        for i in range(7):
            t1, t2, t3 = NUMBERS[i]+suit, NUMBERS[i+1]+suit, NUMBERS[i+2]+suit
            while t1 in temp_hand and t2 in temp_hand and t3 in temp_hand:
                sequences.append(f"{t1}-{t2}-{t3}")
                temp_hand.remove(t1)
                temp_hand.remove(t2)
                temp_hand.remove(t3)
                
    return sequences, triplets, quads, temp_hand


# 🧠 暗牌解析器
def analyze_concealed_hand(hand_tiles):
    unique_tiles = set(hand_tiles)
    
    for potential_pair in unique_tiles:
        if hand_tiles.count(potential_pair) >= 2:
            temp_hand = list(hand_tiles)
            temp_hand.remove(potential_pair)
            temp_hand.remove(potential_pair)

            triplets = []
            for t in set(temp_hand):
                while temp_hand.count(t) >= 3:
                    triplets.append(t)
                    temp_hand.remove(t)
                    temp_hand.remove(t)
                    temp_hand.remove(t)

            sequences = []
            for suit in SUITS:
                for i in range(7):
                    t1, t2, t3 = NUMBERS[i]+suit, NUMBERS[i+1]+suit, NUMBERS[i+2]+suit
                    while t1 in temp_hand and t2 in temp_hand and t3 in temp_hand:
                        sequences.append(f"{t1}-{t2}-{t3}")
                        temp_hand.remove(t1)
                        temp_hand.remove(t2)
                        temp_hand.remove(t3)

            if len(temp_hand) == 0:
                return sequences, triplets, [potential_pair], temp_hand

    temp_hand = list(hand_tiles)
    pairs, triplets = [], []
    for t in set(temp_hand):
        if temp_hand.count(t) >= 3:
            triplets.append(t)
            for _ in range(3): temp_hand.remove(t)
        elif temp_hand.count(t) == 2:
            pairs.append(t)
            for _ in range(2): temp_hand.remove(t)
            
    sequences = []
    for suit in SUITS:
        for i in range(7):
            t1, t2, t3 = NUMBERS[i]+suit, NUMBERS[i+1]+suit, NUMBERS[i+2]+suit
            while t1 in temp_hand and t2 in temp_hand and t3 in temp_hand:
                sequences.append(f"{t1}-{t2}-{t3}")
                temp_hand.remove(t1)
                temp_hand.remove(t2)
                temp_hand.remove(t3)
                
    return sequences, triplets, pairs, temp_hand


def main():
    model = YOLO("best.pt")


    concealed_hand = detect_tiles_from_image(model, "front.jpg")  
    exposed_melds = detect_tiles_from_image(model, "top.jpg")     
    is_men_qing = len(exposed_melds) == 0

    print("辨識結果：")
    print(f"站立手牌 (暗牌): {concealed_hand}")
    print(f"攤平牌組 (明牌): {'無 (門清)' if is_men_qing else str(exposed_melds) + ' (無門清)'}")
    
    final_hand_zh = concealed_hand + exposed_melds
    print(f"共偵測到: {len(final_hand_zh)} 張牌")

    # 2. 獨立解析明牌與暗牌
    exp_seq, exp_tri, exp_quad, exp_rem = analyze_exposed_melds(exposed_melds)
    con_seq, con_tri, con_pair, con_rem = analyze_concealed_hand(concealed_hand)

    # 計算暗牌處理器抓到了幾個刻子
    concealed_triplets_count = len(con_tri)

    final_sequences = exp_seq + con_seq
    final_triplets = exp_tri + con_tri
    final_quads = exp_quad
    final_pairs = con_pair
    final_remaining = exp_rem + con_rem

    print("牌型歸類:")
    if final_quads: print(f"槓子: {', '.join(final_quads)}")
    if final_triplets: print(f"刻子: {', '.join(final_triplets)}")
    if final_sequences: print(f"順子: {', '.join(final_sequences)}")
    if final_pairs: print(f"將眼: {', '.join(final_pairs)}")

    if final_remaining:
        print(f"\n 剩下的孤張/廢牌: {', '.join(final_remaining)}")
        print(" 牌型尚未完整，無法計算台數。")
    else:
        tai, details = calculate_tai(final_sequences, final_triplets, final_quads, final_pairs, final_hand_zh, is_men_qing, concealed_triplets_count)
        
        # 粗體與重置的 ANSI 代碼
        BOLD = '\033[1m'
        RESET = '\033[0m'

        if tai > 0:
            # 在字串最前面加上 BOLD，最後面加上 RESET
            print(f"\n{BOLD}符合牌型: {', '.join(details)}{RESET}")
            print(f"{BOLD}基礎台數共獲得: {tai}台{RESET}")
        else:
            print("平胡或無特殊牌型 (0台)。")

    # 系統動態台數備註區 ( ANSI 色碼變更為灰色) 
    GRAY = '\033[90m'
    RESET = '\033[0m'

    print("\n" + GRAY + "="*50)
    print("[貼心提醒] 系統無法判斷遊戲過程，請依實際情況額外加計以下「動態台數」：")
    print("莊家身分：莊家 (1台) / 連莊拉莊 (2N+1 台)")
    print("摸牌狀態：自摸 (1台) / 河底撈魚 (1台) / 海底撈月 (1台) / 槓上開花 (1台) / 搶槓 (1台)")
    print("聽牌型態：獨聽 [邊張/中洞/單吊] (1台)")
    print("天地人胡：天胡 (24/32台) / 地胡 (16台) / 人胡 (16台) / 天聽 (16台) / 地聽 (8台)")
    print("="*50 + RESET + "\n")

if __name__ == '__main__':
    main()

# 狀態與身分： (一) 莊家台、(二) 自摸 / 門清 / 海底撈月、(四) 圈風 / 門風、(十三) 天地人胡。

# 動作過程： (八) 搶槓 / 槓上開花、(十四) 全求人 / 天地聽。
