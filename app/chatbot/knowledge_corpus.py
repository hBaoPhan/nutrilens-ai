# ai-service/app/chatbot/knowledge_corpus.py
"""
Clinical Nutrition Knowledge Base for AR-NutriLens RAG System.
Contains comprehensive structured knowledge across:
1. 30 Canonical Vietnamese foods & nutritional profiles.
2. Clinical dietary guidelines (WHO, National Institute of Nutrition - NIN Vietnam).
3. Medical conditions & chronic disease dietary management.
4. 12 Evidence-based nutrition tips.
5. AR-NutriLens system capabilities & user guidance.
"""
from typing import List, Dict, Any

CANONICAL_FOODS_DATA: Dict[str, Dict[str, Any]] = {
    "BANH_CUON": {
        "displayName": "Bánh Cuốn Nóng Thịt Mộc Nhĩ",
        "category": "Breakfast",
        "servingWeightG": 250.0,
        "caloriesPer100g": 140.0,
        "proteinPer100g": 4.5,
        "carbsPer100g": 24.0,
        "fatPer100g": 3.0,
        "glycemicIndex": "Medium",
        "description": "Bánh tráng bột gạo hấp mỏng cuộn nhân thịt băm, mộc nhĩ, ăn kèm chả lụa và nước mắm chua ngọt.",
        "pros": "Dễ tiêu hóa, lượng dầu mỡ tương đối thấp so với các món chiên.",
        "cons": "Chứa nhiều tinh bột tinh chế từ bột gạo, nước chấm có đường.",
        "suitability": "Thích hợp cho bữa sáng duy trì cân nặng hoặc người cần năng lượng hoạt động buổi sáng. Người tiểu đường nên ăn kèm nhiều rau thơm, giá đỗ và hạn chế chấm ngập nước mắm."
    },
    "BANH_GIO": {
        "displayName": "Bánh Giò Nóng Thịt Nấm",
        "category": "Breakfast",
        "servingWeightG": 200.0,
        "caloriesPer100g": 175.0,
        "proteinPer100g": 5.2,
        "carbsPer100g": 22.0,
        "fatPer100g": 7.5,
        "glycemicIndex": "Medium-High",
        "description": "Bánh bột gạo và bột năng nhân thịt nạc băm, mộc nhĩ, nấm hương, gói lá chuối hấp chín.",
        "pros": "Hương vị thơm ngon, giàu năng lượng cho buổi sáng năng động.",
        "cons": "Lượng chất béo từ mỡ thịt và gia vị nêm khá đậm đà.",
        "suitability": "Phù hợp bữa sáng nhanh hoặc người cần tăng cân. Người mỡ máu cao hoặc giảm cân nên ăn điều độ 1 chiếc/tuần."
    },
    "BANH_MI": {
        "displayName": "Bánh Mì Kẹp Thịt Chả",
        "category": "Breakfast",
        "servingWeightG": 180.0,
        "caloriesPer100g": 260.0,
        "proteinPer100g": 9.5,
        "carbsPer100g": 36.0,
        "fatPer100g": 8.5,
        "glycemicIndex": "High",
        "description": "Bánh mì giòn rụm kẹp pa-tê, bơ trứng, thịt xíu, chả lụa, dưa leo và ngò rí.",
        "pros": "Cân bằng nhanh các nhóm chất, tiện lợi, cung cấp năng lượng tức thì.",
        "cons": "Vỏ bánh mì có chỉ số GI cao, pa-tê và bơ tăng hàm lượng chất béo bão hòa.",
        "suitability": "Nên yêu cầu giảm sốt bơ pa-tê và tăng dưa leo, rau thơm nếu đang trong chế độ giảm mỡ."
    },
    "BUN_BO_HUE": {
        "displayName": "Bún Bò Huế",
        "category": "Lunch",
        "servingWeightG": 550.0,
        "caloriesPer100g": 105.0,
        "proteinPer100g": 6.8,
        "carbsPer100g": 13.5,
        "fatPer100g": 3.2,
        "glycemicIndex": "Medium",
        "description": "Món bún sợi to đậm đà với nước dùng xương hầm, sả ớt, bắp bò, giò heo và chả cua.",
        "pros": "Giàu đạm chất lượng cao từ thịt bò nạc, nước dùng nhiều vi khoáng.",
        "cons": "Hàm lượng natri và mỡ nổi (váng dầu) cao. Người huyết áp cao cần tránh húp cạn nước dùng.",
        "suitability": "Tốt cho người tập gym tăng cơ (chọn nạm bò, bỏ bớt da giò heo). Người cao huyết áp hoặc suy thận cần hạn chế nước dùng."
    },
    "BUN_CHA_HA_NOI": {
        "displayName": "Bún Chả Hà Nội",
        "category": "Lunch",
        "servingWeightG": 450.0,
        "caloriesPer100g": 145.0,
        "proteinPer100g": 8.2,
        "carbsPer100g": 16.5,
        "fatPer100g": 5.2,
        "glycemicIndex": "Medium",
        "description": "Thịt ba chỉ nướng và chả viên nướng than hoa, ăn kèm bún tươi, nước mắm dưa góp và rau sống ngập tràn.",
        "pros": "Ăn kèm lượng rau sống rất dồi dào, cung cấp chất xơ và vitamin C.",
        "cons": "Thịt nướng than có thể tạo hợp chất cháy khét; nước chấm có tỷ lệ đường tương đối.",
        "suitability": "Ưu tiên chả viên nạc thay vì thịt mỡ; ăn hết phần rau sống để làm chậm hấp thu đường huyết."
    },
    "BUN_DAU_MAM_TOM": {
        "displayName": "Bún Đậu Mắm Tôm",
        "category": "Lunch",
        "servingWeightG": 400.0,
        "caloriesPer100g": 165.0,
        "proteinPer100g": 7.5,
        "carbsPer100g": 18.0,
        "fatPer100g": 7.0,
        "glycemicIndex": "Medium",
        "description": "Mẹt bún lá, đậu phụ rán giòn, chả cốm, thịt chân giò luộc, dưa leo, tía tô chấm mắm tôm pha chanh ớt.",
        "pros": "Đậu phụ cung cấp isoflavone thực vật, thịt luộc thanh đạm.",
        "cons": "Đậu rán ngập dầu tích nhiều calo; mắm tôm chứa lượng muối rất cao.",
        "suitability": "Người cao huyết áp, bệnh thận hoặc gout nên thay mắm tôm bằng nước mắm nhạt pha loãng hoặc hạn chế chấm."
    },
    "BUN_THIT_NUONG": {
        "displayName": "Bún Thịt Nướng Chả Giò",
        "category": "Lunch",
        "servingWeightG": 420.0,
        "caloriesPer100g": 145.0,
        "proteinPer100g": 8.2,
        "carbsPer100g": 20.5,
        "fatPer100g": 3.8,
        "glycemicIndex": "Medium",
        "description": "Bún tươi trộn thịt heo nướng sả, chả giò rán, đậu phộng rang, mỡ hành, đồ chua và rau thơm.",
        "pros": "Hương vị hấp dẫn, giàu chất xơ từ đồ chua (cà rốt, củ cải) và rau sống.",
        "cons": "Chả giò chiên và mỡ hành làm tăng calo đáng kể nếu không kiểm soát.",
        "suitability": "Để giảm cân: giảm nửa phần chả giò, không chan mỡ hành, chan nước mắm vừa phải."
    },
    "CHAO_LONG": {
        "displayName": "Cháo Lòng Hà Nội",
        "category": "Dinner",
        "servingWeightG": 400.0,
        "caloriesPer100g": 95.0,
        "proteinPer100g": 6.0,
        "carbsPer100g": 12.0,
        "fatPer100g": 2.5,
        "glycemicIndex": "High",
        "description": "Cháo gạo ninh nhừ với nước hầm xương và tiết heo, ăn kèm dồi, dạ dày, gan, tim và lòng non luộc.",
        "pros": "Dễ tiêu hóa khi ốm mệt, gan và tiết bổ sung lượng sắt dồi dào.",
        "cons": "Nội tạng động vật chứa cực kỳ nhiều cholesterol và purin.",
        "suitability": "Chống chỉ định cho người mắc bệnh Gout (Axit Uric cao), mỡ máu cao và xơ vữa động mạch."
    },
    "COM_DEP": {
        "displayName": "Cốm Dẹp Trộn Dừa",
        "category": "Breakfast",
        "servingWeightG": 150.0,
        "caloriesPer100g": 285.0,
        "proteinPer100g": 4.5,
        "carbsPer100g": 52.0,
        "fatPer100g": 6.8,
        "glycemicIndex": "High",
        "description": "Món ăn truyền thống từ hạt nếp nương giã dẹp trộn dừa nạo sợi tươi và đường thốt nốt.",
        "pros": "Giàu năng lượng, thơm mùi gạo mới, chứa chất béo chuỗi trung bình (MCT) từ dừa tươi.",
        "cons": "Hàm lượng carbohydrate và đường cao.",
        "suitability": "Thích hợp làm bữa phụ trước tập thể thao cường độ cao (Pre-workout) để nạp năng lượng nhanh."
    },
    "COM_TAM": {
        "displayName": "Cơm Tấm Sườn Bì Chả",
        "category": "Lunch",
        "servingWeightG": 380.0,
        "caloriesPer100g": 171.0,
        "proteinPer100g": 10.0,
        "carbsPer100g": 18.9,
        "fatPer100g": 6.3,
        "glycemicIndex": "Medium-High",
        "description": "Gạo tấm thơm ăn kèm sườn cốt lết nướng mật ong, bì heo trộn thính, chả trứng hấp, mỡ hành và cà chua dưa leo.",
        "pros": "Lượng protein dồi dào, no lâu, hương vị đặc trưng Nam Bộ.",
        "cons": "Calo một đĩa đầy đủ có thể vượt 650 - 800 kcal do mỡ hành và sườn ướp ngọt.",
        "suitability": "Khi giảm cân: chọn sườn nạc, bỏ mỡ hành, tăng gấp đôi dưa leo và cà chua."
    },
    "COM_RANG": {
        "displayName": "Cơm Rang Dưa Bò",
        "category": "Lunch",
        "servingWeightG": 300.0,
        "caloriesPer100g": 195.0,
        "proteinPer100g": 8.5,
        "carbsPer100g": 26.0,
        "fatPer100g": 6.8,
        "glycemicIndex": "Medium-High",
        "description": "Cơm nguội đảo săn giòn trên chảo lửa lớn với trứng, ăn kèm thịt bò xào dưa cải chua giòn đậm vị.",
        "pros": "Thịt bò bổ sung kẽm và sắt, dưa chua kích thích vị giác.",
        "cons": "Hạt cơm ngậm dầu khi rang, dưa muối chua chứa nhiều muối natri.",
        "suitability": "Người cao huyết áp nên hạn chế ăn dưa muối; dùng giấy thấm dầu hoặc xào với ít dầu ăn."
    },
    "PHO_BO": {
        "displayName": "Phở Bò Tái Nạm",
        "category": "Breakfast",
        "servingWeightG": 550.0,
        "caloriesPer100g": 88.0,
        "proteinPer100g": 6.2,
        "carbsPer100g": 11.5,
        "fatPer100g": 2.2,
        "glycemicIndex": "Medium",
        "description": "Bánh phở mềm mượt chan nước dùng xương bò hầm cùng thảo quả, quế, hồi, phủ thịt bò tái nạc và hành hoa.",
        "pros": "Protein chất lượng cao, nước dùng giàu collagen thảo mộc tự nhiên, ít chất béo bão hòa nếu vớt váng mỡ.",
        "cons": "Natri trong nước phở thường dao động từ 1500 - 2000mg.",
        "suitability": "Rất phù hợp cho bữa sáng dinh dưỡng; người cần siết natri chỉ nên ăn cái, không húp cạn nước dùng."
    },
    "GOI_CUON": {
        "displayName": "Gỏi Cuốn Tôm Thịt Rau Sống",
        "category": "Lunch",
        "servingWeightG": 200.0,
        "caloriesPer100g": 130.0,
        "proteinPer100g": 7.0,
        "carbsPer100g": 18.0,
        "fatPer100g": 3.2,
        "glycemicIndex": "Low-Medium",
        "description": "Bánh tráng cuốn tôm luộc, thịt nạc luộc, bún tươi, hẹ lá, xà lách, rau thơm chấm tương đen hoặc nước mắm chua ngọt.",
        "pros": "Một trong những món ăn lành mạnh nhất thế giới: ít dầu mỡ, nhiều rau xanh tươi sống, đạm nạc.",
        "cons": "Nước chấm tương bơ đậu phộng có thể chứa nhiều đường và calo.",
        "suitability": "Tuyệt vời cho người giảm cân, tiểu đường, ăn kiêng lành mạnh."
    },
    "XOI": {
        "displayName": "Xôi Thập Cẩm Ruốc Thịt",
        "category": "Breakfast",
        "servingWeightG": 220.0,
        "caloriesPer100g": 235.0,
        "proteinPer100g": 6.8,
        "carbsPer100g": 42.0,
        "fatPer100g": 4.5,
        "glycemicIndex": "High",
        "description": "Gạo nếp đồ dẻo thơm ăn kèm thịt kho, chả lụa, ruốc thịt heo sợi, lạp xưởng và hành phi thơm lừng.",
        "pros": "No rất lâu, cung cấp năng lượng bền bỉ cho người lao động thể lực nặng.",
        "cons": "Gạo nếp chứa hàm lượng amylopectin cao làm đường huyết tăng nhanh sau ăn; dễ gây đầy bụng ợ chua.",
        "suitability": "Người tiểu đường hoặc trào ngược dạ dày nên hạn chế tối đa."
    },
    "BOILED_EGG": {
        "displayName": "Trứng Gà Luộc",
        "category": "Breakfast",
        "servingWeightG": 50.0,
        "caloriesPer100g": 155.0,
        "proteinPer100g": 13.0,
        "carbsPer100g": 1.1,
        "fatPer100g": 11.0,
        "glycemicIndex": "Low (0)",
        "description": "Trứng gà tươi luộc chín lòng đào hoặc chín kỹ, giữ trọn vẹn dinh dưỡng nguyên bản.",
        "pros": "Nguồn protein tham chiếu sinh học hoàn hảo (BV=100), giàu Choline cho não bộ và Lutein cho mắt.",
        "cons": "Lòng đỏ chứa khoảng 180-200mg cholesterol (tuy nhiên ít ảnh hưởng đến cholesterol máu ở người khỏe mạnh).",
        "suitability": "Lý tưởng cho mọi đối tượng: giảm béo, tăng cơ, tiểu đường, người tập thể hình."
    },
    "CARROT": {
        "displayName": "Cà Rốt Tươi Cắt Lát",
        "category": "Lunch",
        "servingWeightG": 100.0,
        "caloriesPer100g": 41.0,
        "proteinPer100g": 0.9,
        "carbsPer100g": 9.6,
        "fatPer100g": 0.2,
        "glycemicIndex": "Low",
        "description": "Cà rốt tươi giòn ngọt, giàu sắc tố tự nhiên Beta-carotene.",
        "pros": "Chuyển hóa thành Vitamin A tăng cường thị lực, chống oxy hóa, giàu chất xơ hòa tan Pectin.",
        "cons": "Ăn quá nhiều liên tục có thể gây vàng da lành tính (Carotenemia).",
        "suitability": "Tuyệt vời cho món ăn kèm, salad, hỗ trợ kiểm soát đường huyết và làm đẹp da."
    },
    "CHICKEN_BREAST": {
        "displayName": "Ức Gà Áp Chảo",
        "category": "Dinner",
        "servingWeightG": 150.0,
        "caloriesPer100g": 165.0,
        "proteinPer100g": 31.0,
        "carbsPer100g": 0.0,
        "fatPer100g": 3.6,
        "glycemicIndex": "Low (0)",
        "description": "Thịt lườn ức gà lọc bỏ da, áp chảo với chút dầu ô-liu, tiêu đen và lá hương thảo.",
        "pros": "Tỷ lệ protein trên calo cao nhất trong các loại thịt, không carbohydrate, cực thấp chất béo bão hòa.",
        "cons": "Nếu chế biến kỹ dễ bị khô bã.",
        "suitability": "Tiêu chuẩn vàng cho người tập thể hình tăng cơ nạc, vận động viên và chế độ giảm mỡ cấp tốc."
    },
    "BROCCOLI": {
        "displayName": "Bông Cải Xanh Hấp",
        "category": "Dinner",
        "servingWeightG": 150.0,
        "caloriesPer100g": 34.0,
        "proteinPer100g": 2.8,
        "carbsPer100g": 7.0,
        "fatPer100g": 0.4,
        "glycemicIndex": "Low",
        "description": "Súp lơ xanh tươi thái miếng vừa ăn, hấp cách thủy giữ màu xanh ngọc và dưỡng chất.",
        "pros": "Chứa hợp chất chống ung thư Sulforaphane, hàm lượng Vitamin C cao gấp đôi cam, giàu Canxi và Vitamin K.",
        "cons": "Có thể gây sinh hơi nhẹ ở người có hội chứng ruột kích thích nếu ăn sống.",
        "suitability": "Khuyên dùng hàng ngày cho mọi lứa tuổi, đặc biệt hỗ trợ thải độc gan và giảm mỡ máu."
    },
    "CUCUMBER": {
        "displayName": "Dưa Chuột Thái Lát",
        "category": "Lunch",
        "servingWeightG": 120.0,
        "caloriesPer100g": 15.0,
        "proteinPer100g": 0.7,
        "carbsPer100g": 3.6,
        "fatPer100g": 0.1,
        "glycemicIndex": "Low",
        "description": "Dưa chuột tươi rửa sạch thái lát mỏng giòn mát.",
        "pros": "Chứa 95% nước, bù điện giải tự nhiên, calo cực thấp giúp lấp đầy dạ dày mà không lo dư calo.",
        "cons": "Ít chất đạm và chất béo, cần kết hợp nhóm thức ăn khác.",
        "suitability": "Món ăn kèm hoàn hảo cho người ăn kiêng giảm cân cấp tốc, cấp nước ngày hè."
    },
    "SWEET_POTATO": {
        "displayName": "Khoai Lang Vàng Luộc",
        "category": "Breakfast",
        "servingWeightG": 180.0,
        "caloriesPer100g": 86.0,
        "proteinPer100g": 1.6,
        "carbsPer100g": 20.1,
        "fatPer100g": 0.1,
        "glycemicIndex": "Low-Medium (khi luộc)",
        "description": "Củ khoai lang ruột vàng mật luộc chín mềm, dẻo ngọt tự nhiên.",
        "pros": "Tinh bột chuyển hóa chậm (Complex Carbs), giàu Kali điều hòa huyết áp, chống táo bón hiệu quả.",
        "cons": "Khoai lang nướng có GI cao hơn khoai luộc.",
        "suitability": "Lựa chọn thay thế cơm trắng tuyệt vời cho người giảm mỡ, kiểm soát đường huyết và vận động viên."
    },
    "TOFU_STEAMED": {
        "displayName": "Đậu Phụ Trắng Luộc",
        "category": "Lunch",
        "servingWeightG": 150.0,
        "caloriesPer100g": 76.0,
        "proteinPer100g": 8.0,
        "carbsPer100g": 1.9,
        "fatPer100g": 4.8,
        "glycemicIndex": "Low",
        "description": "Đậu nành nguyên chất nấu đông tự nhiên, luộc nước nóng thơm bùi.",
        "pros": "Protein thực vật lành mạnh, không cholesterol, dồi dào canxi sinh học.",
        "cons": "Hàm lượng purin ở mức trung bình.",
        "suitability": "Tuyệt vời cho người mỡ máu cao, tim mạch, người ăn chay hoặc muốn giảm tải gánh nặng tiêu hóa thịt đỏ."
    },
    "SHRIMP_BOILED": {
        "displayName": "Tôm Luộc Bóc Vỏ",
        "category": "Dinner",
        "servingWeightG": 100.0,
        "caloriesPer100g": 99.0,
        "proteinPer100g": 24.0,
        "carbsPer100g": 0.2,
        "fatPer100g": 0.3,
        "glycemicIndex": "Low (0)",
        "description": "Tôm biển tươi luộc nước gừng sả, bóc nõn thịt ngọt săn chắc.",
        "pros": "Hàm lượng protein cực cao (24g/100g), giàu Astaxanthin chống oxy hóa mạnh, ít calo và chất béo.",
        "cons": "Người có cơ địa dị ứng hải sản cần lưu ý.",
        "suitability": "Lý tưởng cho bữa tối nhẹ bụng, tăng cơ giảm mỡ và bổ sung kẽm cho hệ miễn dịch."
    },
    "RICE_WHITE": {
        "displayName": "Cơm Trắng Nấu Chín",
        "category": "Lunch",
        "servingWeightG": 180.0,
        "caloriesPer100g": 130.0,
        "proteinPer100g": 2.7,
        "carbsPer100g": 28.2,
        "fatPer100g": 0.3,
        "glycemicIndex": "High (73)",
        "description": "Gạo tẻ trắng xới xốp dẻo mềm, linh hồn của mâm cơm gia đình Việt.",
        "pros": "Cung cấp năng lượng nền tảng, dễ hấp thu, không gây kích ứng đường ruột.",
        "cons": "Chỉ số GI cao, nếu ăn nhiều dễ chuyển hóa thành mỡ thừa khi ít vận động.",
        "suitability": "Người bình thường: 1 - 1.5 bát/bữa. Người giảm cân hoặc tiểu đường: nên kiểm soát 1/2 đến 2/3 bát và ăn rau/thịt trước khi ăn cơm."
    },
    "PORK_GRILLED": {
        "displayName": "Thịt Heo Nướng Vỉ",
        "category": "Dinner",
        "servingWeightG": 120.0,
        "caloriesPer100g": 240.0,
        "proteinPer100g": 22.0,
        "carbsPer100g": 2.5,
        "fatPer100g": 16.0,
        "glycemicIndex": "Low",
        "description": "Thịt nạc vai heo ướp hành tỏi, mật ong nướng chín vàng trên vỉ than hoa.",
        "pros": "Giàu kẽm, sắt, vitamin nhóm B (B1, B6, B12) giúp tạo máu và phục hồi cơ bắp.",
        "cons": "Chất béo bão hòa từ thịt mỡ có thể làm tăng LDL-cholesterol nếu ăn liên tục.",
        "suitability": "Khuyên dùng phần nạc thăn hoặc nạc vai lọc mỡ; ăn kèm nhiều rau xanh để bài tiết chất độc."
    },
    "ORANGE": {
        "displayName": "Cam Tươi Mọng Nước",
        "category": "Breakfast",
        "servingWeightG": 150.0,
        "caloriesPer100g": 47.0,
        "proteinPer100g": 0.9,
        "carbsPer100g": 11.8,
        "fatPer100g": 0.1,
        "glycemicIndex": "Low-Medium",
        "description": "Quả cam sành hoặc cam vàng mọng nước, múi vàng ươm tươi ngon.",
        "pros": "Giàu Vitamin C củng cố sức đề kháng, Kali, Folate và chất xơ Hesperidin tốt cho thành mạch tim.",
        "cons": "Không nên uống nước cam lúc đói cồn cào nếu bị viêm loét dạ dày.",
        "suitability": "Nên ăn nguyên múi cả tép thay vì vắt lấy nước bỏ bã để giữ nguyên chất xơ hòa tan."
    },
    "BANANA": {
        "displayName": "Chuối Tiêu Chín",
        "category": "Breakfast",
        "servingWeightG": 120.0,
        "caloriesPer100g": 89.0,
        "proteinPer100g": 1.1,
        "carbsPer100g": 22.8,
        "fatPer100g": 0.3,
        "glycemicIndex": "Medium",
        "description": "Chuối tiêu chín tự nhiên thơm dẻo, ngọt thanh.",
        "pros": "Giàu Kali ngăn ngừa chuột rút cơ bắp, Magie hỗ trợ dẫn truyền thần kinh, Tryptophan giúp ngủ ngon.",
        "cons": "Chuối chín kỹ có lượng đường tự nhiên (Fructose, Glucose) khá cao.",
        "suitability": "Bữa phụ lý tưởng trước hoặc sau buổi tập thể thao 30 phút."
    },
    "APPLE": {
        "displayName": "Táo Đỏ Tươi Giòn",
        "category": "Lunch",
        "servingWeightG": 150.0,
        "caloriesPer100g": 52.0,
        "proteinPer100g": 0.3,
        "carbsPer100g": 13.8,
        "fatPer100g": 0.2,
        "glycemicIndex": "Low (36)",
        "description": "Quả táo tươi giòn ngọt tự nhiên, nhiều nước.",
        "pros": "Chứa chất xơ hòa tan Pectin nuôi lợi khuẩn đường ruột, Quercetin kháng viêm, giúp no lâu và giảm thèm ngọt.",
        "cons": "Cần rửa sạch vỏ dưới vòi nước chảy để loại bỏ tồn dư sáp bảo quản.",
        "suitability": "Trái cây tuyệt vời cho mọi thực đơn giảm cân và ăn kiêng tiểu đường."
    },
    "COCONUT": {
        "displayName": "Cơm Dừa Tươi Nạo",
        "category": "Breakfast",
        "servingWeightG": 80.0,
        "caloriesPer100g": 354.0,
        "proteinPer100g": 3.3,
        "carbsPer100g": 15.2,
        "fatPer100g": 33.5,
        "glycemicIndex": "Low",
        "description": "Cơm dừa rám nạo sợi tươi béo ngậy đặc trưng miền Tây.",
        "pros": "Chứa axit béo chuỗi trung bình (MCT - Lauric acid) giúp cung cấp năng lượng nhanh cho não bộ và kháng khuẩn.",
        "cons": "Mật độ năng lượng và chất béo bão hòa cực kỳ cao.",
        "suitability": "Chỉ nên dùng một lượng nhỏ (20-30g) làm gia vị hoặc điểm xuyết; người mỡ máu cao nên hạn chế."
    },
    "SUA_CHUA": {
        "displayName": "Sữa Chua Tự Nhiên",
        "category": "Breakfast",
        "servingWeightG": 100.0,
        "caloriesPer100g": 61.0,
        "proteinPer100g": 3.5,
        "carbsPer100g": 4.7,
        "fatPer100g": 3.3,
        "glycemicIndex": "Low",
        "description": "Sữa chua lên men tự nhiên từ sữa tươi tiệt trùng, không thêm đường hóa học.",
        "pros": "Chứa hàng tỷ lợi khuẩn Probiotics (Lactobacillus, Bifidobacterium) cân bằng hệ vi sinh đường ruột, giàu Canxi sinh học.",
        "cons": "Tránh các loại sữa chua hương trái cây công nghiệp chứa nhiều siro đường fructose.",
        "suitability": "Khuyên dùng 1 hộp mỗi ngày sau bữa ăn 30-60 phút để hỗ trợ tiêu hóa khỏe mạnh."
    },
    "WATERMELON": {
        "displayName": "Dưa Hấu Đỏ Tươi",
        "category": "Breakfast",
        "servingWeightG": 250.0,
        "caloriesPer100g": 30.0,
        "proteinPer100g": 0.6,
        "carbsPer100g": 7.6,
        "fatPer100g": 0.2,
        "glycemicIndex": "High (nhưng Tải lượng đường GL thấp)",
        "description": "Dưa hấu chín đỏ ngọt mát, mọng nước giải khát mùa hè.",
        "pros": "Chứa 92% nước, giàu sắc tố Lycopene chống oxy hóa bảo vệ tim mạch, Citrulline giảm đau mỏi cơ.",
        "cons": "Chỉ số GI cao nên không nên ăn nửa quả một lúc đối với người tiểu đường.",
        "suitability": "Ăn 1-2 miếng (150-200g) giải khát mùa nóng rất an toàn và đã khát."
    }
}

CLINICAL_GUIDELINES: Dict[str, Dict[str, Any]] = {
    "LOSE_WEIGHT": {
        "title": "Nguyên Tắc Dinh Dưỡng Giảm Mỡ & Thâm Hụt Calo Khoa Học",
        "keyRules": [
            "Tạo thâm hụt calo an toàn: Giảm 300 - 500 kcal/ngày so với TDEE (giảm 0.5 - 0.75kg/tuần mà không mất cơ).",
            "Tỷ lệ đĩa ăn lý tưởng: 1/2 đĩa là rau củ quả giàu chất xơ, 1/4 đĩa protein nạc (ức gà, tôm, trứng, đậu phụ), 1/4 đĩa tinh bột chuyển hóa chậm (khoai lang, gạo lứt).",
            "Tăng cường Protein: Nạp tối thiểu 1.2 - 1.6g protein/kg cân nặng để bảo toàn khối cơ nạc và tăng độ no.",
            "Loại bỏ calo lỏng: Cắt bỏ hoàn toàn nước ngọt có ga, trà sữa trân châu, nước ép đóng chai thêm đường; thay bằng nước lọc, trà xanh.",
            "Ăn ngược: Ăn đĩa rau trước -> ăn thức ăn giàu đạm -> ăn cơm cuối cùng để làm phẳng đỉnh đường huyết."
        ]
    },
    "GAIN_MUSCLE": {
        "title": "Nguyên Tắc Dinh Dưỡng Tăng Cơ Nạc & Phục Hồi Thể Lực",
        "keyRules": [
            "Tạo thặng dư calo nhẹ: Nạp vượt TDEE từ 250 - 400 kcal/ngày kèm tập kháng lực (Gym/Calisthenics).",
            "Định mức Protein: Nạp 1.6 - 2.2g protein/kg cân nặng mỗi ngày, chia đều 4 - 5 cữ ăn trong ngày (25-35g protein/cữ).",
            "Nạp Carbs thông minh: Ưu tiên carb phức hợp trước tập 2 tiếng và chuối/khoai ngay sau tập để bù glycogen.",
            "Chất béo tốt: Bổ sung Omega-3 từ cá, dầu ô-liu, các loại hạt để kháng viêm khớp và hỗ trợ hormone sinh trưởng.",
            "Hydrat hóa: Uống 2.5 - 3.5L nước mỗi ngày để tế bào cơ ngậm nước và tổng hợp protein tối đa."
        ]
    },
    "MAINTAIN": {
        "title": "Nguyên Tắc Dinh Dưỡng Cân Bằng & Duy Trì Thể Trạng Khỏe Mạnh",
        "keyRules": [
            "Cân bằng năng lượng nạp vào bằng đúng mức TDEE tiêu hao mỗi ngày.",
            "Đa dạng hóa màu sắc mâm cơm: Đảm bảo tối thiểu 3 màu rau củ mỗi ngày (xanh lá, đỏ, cam/vàng) để hấp thu đủ vitamin và chất chống oxy hóa.",
            "Giảm muối và gia vị công nghiệp: Giữ lượng muối dưới 5g/ngày (khoảng 1 thìa cà phê gạt ngang).",
            "Duy trì thói quen ăn uống đúng giờ, không bỏ bữa sáng và hạn chế ăn đêm sau 20h."
        ]
    },
    "DIABETES": {
        "title": "Hướng Dẫn Dinh Dưỡng Lâm Sàng Cho Bệnh Đái Tháo Đường Type 2",
        "keyRules": [
            "Kiểm soát Chỉ số đường huyết (GI) và Tải lượng đường (GL): Hạn chế xôi nếp, bánh mì trắng, cơm trắng số lượng lớn.",
            "Ưu tiên tinh bột nguyên cám, khoai lang luộc, yến mạch; luôn kết hợp tinh bột cùng đạm và chất xơ để làm chậm tốc độ rỗng dạ dày.",
            "Tuyệt đối không uống nước trái cây vắt bỏ bã; nên ăn trái cây nguyên múi có GI thấp như táo, ổi, bưởi, cam sành.",
            "Tránh xa đường tinh luyện, siro ngô giàu fructose (HFCS), đồ tráng miệng ngọt sau bữa ăn."
        ]
    },
    "GOUT": {
        "title": "Hướng Dẫn Dinh Dưỡng Cho Người Bệnh Gout & Tăng Acid Uric Máu",
        "keyRules": [
            "Cấm hoặc hạn chế tối đa thực phẩm giàu purin (>150mg/100g): Cháo lòng, gan heo, thận, dồi, tôm biển số lượng lớn, thịt bê non.",
            "Tuyệt đối kiêng rượu, bia (kể cả bia không cồn) vì cồn làm ức chế bài tiết acid uric qua ống thận.",
            "Khuyến khích: Sữa chua tự nhiên, sữa ít béo, đậu phụ vừa phải, các loại rau xanh (bông cải xanh, dưa chuột, cà rốt).",
            "Uống tối thiểu 2.5 - 3L nước khoáng kiềm hoặc nước lọc mỗi ngày để hỗ trợ đào thải tinh thể urat qua thận."
        ]
    },
    "HYPERTENSION": {
        "title": "Chế Độ Ăn DASH Kiểm Soát Tăng Huyết Áp & Tim Mạch",
        "keyRules": [
            "Giảm muối tối đa: Dưới 1500 - 2000mg natri/ngày. Không chan ngập nước dùng phở bò, bún bò, tránh mắm tôm, dưa cải muối chua mặn.",
            "Tăng cường Kali & Magie tự nhiên: Bổ sung chuối tiêu chín, khoai lang luộc, rau bina, bông cải xanh để làm giãn thành mạch.",
            "Hạn chế mỡ động vật và thịt đỏ chế biến sẵn (lạp xưởng, giò thủ, chả mỡ); ưu tiên đạm nạc từ cá biển và ức gà."
        ]
    }
}

EVIDENCE_BASED_TIPS: List[Dict[str, str]] = [
    {
        "id": "TIP_WATER",
        "title": "Uống đủ nước mỗi ngày",
        "content": "Uống 2L - 2.5L nước mỗi ngày, đặc biệt uống một ly nước ấm 250ml ngay khi thức dậy để kích hoạt nhu động ruột và trao đổi chất."
    },
    {
        "id": "TIP_PROTEIN",
        "title": "Ưu tiên Protein nạc",
        "content": "Bổ sung ức gà, cá, tôm, trứng hoặc đậu phụ trong các bữa ăn giúp tạo cảm giác no lâu, bảo toàn khối cơ và ổn định đường huyết."
    },
    {
        "id": "TIP_LIQUID_CAL",
        "title": "Cắt giảm calo lỏng",
        "content": "Hạn chế trà sữa, nước ngọt có ga và cà phê nhiều đường sữa. Thay bằng trà xanh không đường hoặc nước lọc ngâm chanh dưa leo."
    },
    {
        "id": "TIP_HALF_PLATE",
        "title": "Quy tắc đĩa ăn 1/2 rau củ",
        "content": "Xếp 1/2 đĩa là rau củ quả tươi, 1/4 đĩa là protein nạc và 1/4 đĩa là tinh bột chuyển hóa chậm (khoai lang, cơm gạo lứt)."
    },
    {
        "id": "TIP_MINDFUL_EATING",
        "title": "Nhai kỹ và ăn chậm",
        "content": "Não bộ mất 15-20 phút để nhận tín hiệu no từ dạ dày. Nhai kỹ ít nhất 20 lần mỗi miếng giúp tiêu hóa êm ái và ngăn ngừa ăn quá mức."
    },
    {
        "id": "TIP_SLEEP",
        "title": "Giấc ngủ sâu hỗ trợ đốt mỡ",
        "content": "Ngủ đủ 7-8 tiếng mỗi đêm giúp cân bằng hormone Leptin (tạo cảm giác no) và Ghrelin (kích thích đói), triệt tiêu cơn thèm ăn vặt đêm."
    },
    {
        "id": "TIP_LIGHT_DINNER",
        "title": "Bữa tối thanh nhẹ trước 19h",
        "content": "Hạn chế các món chiên xào nhiều dầu mỡ vào bữa tối để hệ tiêu hóa được nghỉ ngơi, gan thải độc tốt và giấc ngủ sâu trọn vẹn."
    },
    {
        "id": "TIP_SMART_SNACK",
        "title": "Ăn vặt thông minh với hạt",
        "content": "Thay bánh ngọt bằng một nắm nhỏ (25g) hạt hạnh nhân, óc chó hoặc một quả táo nhỏ giúp bổ sung chất béo tốt không bão hòa."
    },
    {
        "id": "TIP_SALT_REDUCTION",
        "title": "Kiểm soát muối natri",
        "content": "Hạn chế chấm ngập chén nước mắm hoặc húp cạn tô nước dùng bún phở quán ăn vì một tô nước dùng có thể chứa toàn bộ lượng muối cho phép cả ngày."
    },
    {
        "id": "TIP_WALK_AFTER_MEAL",
        "title": "Đi bộ 10 phút sau bữa ăn",
        "content": "Vận động nhẹ nhàng 10-15 phút sau bữa ăn giúp cơ bắp thu nạp glucose trực tiếp từ máu, làm giảm đến 30% đỉnh tăng đường huyết."
    },
    {
        "id": "TIP_FIBER_FIRST",
        "title": "Thứ tự ăn uống thông minh",
        "content": "Hãy ăn rau củ và canh trước, sau đó đến thịt cá, và cuối cùng mới ăn cơm. Cách này giúp chất xơ tráng dạ dày, làm chậm hấp thu đường và chất béo."
    },
    {
        "id": "TIP_PROBIOTICS",
        "title": "Bổ sung men vi sinh đường ruột",
        "content": "Một hộp sữa chua không đường mỗi ngày cung cấp hàng tỷ lợi khuẩn giúp cải thiện hấp thu, tăng cường miễn dịch và giảm đầy hơi khó tiêu."
    }
]

SYSTEM_FAQ: List[Dict[str, str]] = [
    {
        "question": "AR-NutriLens quét và nhận diện món ăn như thế nào?",
        "answer": "AR-NutriLens sử dụng mô hình AI thị giác máy tính YOLOv8 Segmentation được huấn luyện trên 30 món ăn Việt Nam chuẩn hóa. Khi bạn đưa camera qua đĩa thức ăn, hệ thống nhận diện từng thành phần, vẽ đa giác bao quanh (polygon mask) và ước tính thể tích đĩa 3D bằng công nghệ AR để tính ra khối lượng gram thực tế."
    },
    {
        "question": "Chỉ số TDEE và BMR được tính toán ra sao?",
        "answer": "BMR (Tỷ lệ trao đổi chất cơ bản) được tính theo công thức Mifflin-St Jeor chuẩn quốc tế dựa trên giới tính, cân nặng, chiều cao và độ tuổi của bạn. TDEE (Tổng năng lượng tiêu hao hàng ngày) bằng BMR nhân với hệ số vận động thể chất (PAL: từ 1.2 cho ít vận động đến 1.9 cho vận động viên)."
    },
    {
        "question": "Làm thế nào để chatbot gợi ý món ăn tối ưu cho tôi?",
        "answer": "Chatbot AR-NutriLens hoạt động theo cơ chế RAG Hybrid: Hệ thống trích xuất hồ sơ cá nhân và lượng calo bạn đã nạp hôm nay từ meal-service, sau đó kết hợp với cơ sở tri thức dinh dưỡng lâm sàng để đưa ra lời khuyên chính xác, phù hợp với mục tiêu tăng cơ, giảm mỡ hoặc tình trạng bệnh lý của bạn."
    }
]
