# ai-service/app/chatbot/guardrails.py
"""
Defense-in-Depth Guardrails for AR-NutriLens RAG Chatbot.
Enforces strict domain scope (nutrition, food, health, AR-NutriLens),
blocks prompt injection / secret extraction, handles medical safety boundaries,
and validates LLM output before returning to clients.
"""
import re
import logging
from dataclasses import dataclass
from typing import List, Dict, Any, Optional

logger = logging.getLogger("nutrilens-ai")


@dataclass(frozen=True)
class GuardrailDecision:
    allowed: bool
    reason: Optional[str] = None
    model_source: Optional[str] = None
    refusal_reply: Optional[str] = None


# 1. Prompt Injection / Jailbreak / Secret Extraction Patterns
_INJECTION_PATTERNS = [
    re.compile(r"(ignore|disregard|forget|override)\s+(all\s+|the\s+|previous\s+|above\s+|prior\s+)*(instructions?|rules?|prompts?|constraints?)", re.IGNORECASE),
    re.compile(r"(bỏ\s+qua|quên\s+hết|phớt\s+lờ|vượt\s+qua)\s+(mọi\s+|tất\s+cả\s+|các\s+)?(hướng\s+dẫn|quy\s+tắc|lệnh|ràng\s+buộc|chỉ\s+dẫn)", re.IGNORECASE),
    re.compile(r"(reveal|show|print|dump|leak|tiết\s+lộ|in\s+ra|cho\s+xem|hiển\s+thị)\s+.*?(system\s*prompt|prompt\s*hệ\s*thống|api[_\s-]?key|secret|password|mật\s+khẩu|token)", re.IGNORECASE),
    re.compile(r"\b(gemini_api_key|google_api_key|jwt_secret|internal_service_secret|mysql_root_password)\b", re.IGNORECASE),
    re.compile(r"\b(jailbreak|dan\s+mode|developer\s+mode|do\s+anything\s+now)\b", re.IGNORECASE),
    re.compile(r"(đóng\s+vai|hãy\s+trở\s+thành|act\s+as|pretend\s+to\s+be)\s+(một\s+)?(hacker|lập\s+trình\s+viên|chính\s+trị\s+gia|nhà\s+thơ|luật\s+sư|bác\s+sĩ\s+kê\s+đơn)", re.IGNORECASE),
]

# 2. High-Risk Medical Emergency or Eating-Disorder / Self-Harm Patterns
_MEDICAL_EMERGENCY_PATTERNS = [
    re.compile(r"(đau\s+ngực\s+dữ\s+dội|khó\s+thở\s+cấp|ngất\s+xỉu|co\s+giật|đột\s+quỵ|uống\s+quá\s+liều|ngộ\s+độc\s+nặng)", re.IGNORECASE),
    re.compile(r"(cách\s+móc\s+họng|nôn\s+sau\s+khi\s+ăn|nhịn\s+ăn\s+hoàn\s+toàn\s+\d+\s+ngày|tuyệt\s+thực\s+giảm\s+cân)", re.IGNORECASE),
]

# 3. In-Scope Nutrition / Health / Fitness / AR-NutriLens Domain Signals
_NUTRITION_DOMAIN_KEYWORDS = (
    "ăn", "món", "bữa", "thực đơn", "calo", "kcal", "calorie", "protein", "đạm", "carb",
    "tinh bột", "fat", "chất béo", "chất xơ", "vitamin", "khoáng", "nước", "uống",
    "dinh dưỡng", "giảm cân", "giảm mỡ", "tăng cân", "tăng cơ", "duy trì", "vóc dáng",
    "tdee", "bmr", "bmi", "cân nặng", "chiều cao", "thể hình", "tập gym", "tập luyện",
    "chạy bộ", "vận động", "mets", "eat clean", "keto", "low carb", "nhịn ăn gián đoạn",
    "đường huyết", "tiểu đường", "đái tháo đường", "gút", "gout", "acid uric", "huyết áp",
    "mỡ máu", "cholesterol", "tim mạch", "dạ dày", "gan nhiễm mỡ", "thận", "dị ứng",
    "phở", "bún", "cơm", "bánh mì", "bánh cuốn", "bánh giò", "cháo", "xôi", "gỏi cuốn",
    "ức gà", "trứng", "cá hồi", "tôm", "đậu phụ", "thịt", "rau", "cà rốt", "súp lơ",
    "bông cải", "dưa chuột", "dưa leo", "khoai lang", "cam", "chuối", "táo", "dừa",
    "sữa chua", "dưa hấu", "trái cây", "hoa quả", "sữa", "cà phê", "trà", "nhật ký",
    "quét ar", "nutrilens", "thể tích", "khẩu phần", "béo phì", "thừa cân", "thiếu cân",
    "sức khỏe", "ngủ", "tiêu hóa", "no lâu", "đói", "thèm ăn", "sáng", "trưa", "tối", "phụ"
)

# 4. Explicit Off-Topic Domain Patterns (blocked when no nutrition signals are present)
_OFF_TOPIC_PATTERNS = [
    # Programming / Tech / Coding
    re.compile(r"\b(viết\s+code|lập\s+trình|debug|thuật\s+toán|python|javascript|typescript|java|c\+\+|c#|golang|rust|html|css|sql|react|spring\s*boot|docker|kubernetes|git\s+commit|leetcode|regex)\b", re.IGNORECASE),
    # Finance / Crypto / Gambling
    re.compile(r"\b(bitcoin|crypto|tiền\s+ảo|chứng\s+khoán|cổ\s+phiếu|forex|lô\s+đề|cá\s+độ|xổ\s+số|soi\s+kèo|tài\s+xỉu)\b", re.IGNORECASE),
    # Math / School Homework / Translation
    re.compile(r"\b(giải\s+phương\s+trình|tích\s+phân|đạo\s+hàm|hình\s+học\s+không\s+gian|vật\s+lý\s+lượng\s+tử|lịch\s+sử\s+chiến\s+tranh|dịch\s+đoạn\s+văn|viết\s+luận\s+văn)\b", re.IGNORECASE),
    # Entertainment / Creative Writing / Politics / Legal
    re.compile(r"\b(làm\s+thơ|viết\s+thơ|viết\s+truyện|kể\s+chuyện\s+cười|chính\s+trị|bầu\s+cử|đảng\s+phái|tư\s+vấn\s+pháp\s+lý|luật\s+sư|ly\s+hôn|liên\s+quân|valorant|genshin|gta\s*v|minecraft|bóng\s+đá|world\s+cup)\b", re.IGNORECASE),
    # Weather / Horoscope
    re.compile(r"\b(tử\s+vi|xem\s+bói|cung\s+hoàng\s+đạo|thời\s+tiết\s+hôm\s+nay|dự\s+báo\s+thời\s+tiết)\b", re.IGNORECASE),
]

# 5. Output Leak Detection Patterns (Post-LLM check)
_OUTPUT_LEAK_PATTERNS = [
    re.compile(r"AQ\.Ab8[A-Za-z0-9_\-]{10,}"),
    re.compile(r"AIza[0-9A-Za-z_\-]{20,}"),
    re.compile(r"===\s*(THÔNG TIN NGƯỜI DÙNG HIỆN TẠI|NGUYÊN TẮC CỐT LÕI|DỮ LIỆU TRI THỨC ĐÃ TRUY XUẤT)"),
    re.compile(r"STRICT\s+GUARDRAILS|REAL-TIME\s+CONTEXT|GROUNDED\s+KNOWLEDGE\s+BASE"),
]

# Official Google Gemini Safety Settings
GEMINI_SAFETY_SETTINGS: List[Dict[str, str]] = [
    {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
    {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
    {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
    {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
]


def _has_nutrition_signal(text_lower: str) -> bool:
    for kw in _NUTRITION_DOMAIN_KEYWORDS:
        if len(kw) <= 3:
            if re.search(rf"(?<!\w){re.escape(kw)}(?!\w)", text_lower):
                return True
        elif kw in text_lower:
            return True
    return False


def evaluate_input_guardrails(
    query: str,
    retrieved_chunks: Optional[List[Dict[str, Any]]] = None
) -> GuardrailDecision:
    """
    Fast deterministic pre-LLM guardrail check (<1ms).
    Blocks prompt injection, dangerous medical emergencies, and clearly off-topic queries
    while preserving natural greetings and multi-turn nutrition follow-ups.
    """
    text = (query or "").strip()
    if not text:
        return GuardrailDecision(
            allowed=False,
            reason="empty_query",
            model_source="GUARDRAIL_INPUT_VALIDATION",
            refusal_reply="Bạn hãy nhập câu hỏi về dinh dưỡng, món ăn hoặc thực đơn để mình hỗ trợ nhé!"
        )

    # 1. Block prompt injection & secret extraction immediately
    for pattern in _INJECTION_PATTERNS:
        if pattern.search(text):
            logger.warning("🛡️ Guardrail blocked potential prompt injection / secret extraction attempt.")
            return GuardrailDecision(
                allowed=False,
                reason="prompt_injection",
                model_source="GUARDRAIL_SECURITY_BLOCK",
                refusal_reply=(
                    "Mình chỉ có thể hỗ trợ các câu hỏi về dinh dưỡng, món ăn và theo dõi sức khỏe trên AR-NutriLens, "
                    "không thể thay đổi vai trò hay tiết lộ cấu hình hệ thống. Bạn cần mình tư vấn thực đơn hoặc tính calo món nào không?"
                )
            )

    # 2. Intercept acute medical emergencies or unsafe eating behaviors
    for pattern in _MEDICAL_EMERGENCY_PATTERNS:
        if pattern.search(text):
            logger.warning("🛡️ Guardrail intercepted high-risk medical / unsafe dieting query.")
            return GuardrailDecision(
                allowed=False,
                reason="medical_safety",
                model_source="GUARDRAIL_MEDICAL_SAFETY",
                refusal_reply=(
                    "Tình huống bạn vừa đề cập nằm ngoài phạm vi tư vấn dinh dưỡng thông thường và có thể ảnh hưởng nghiêm trọng đến sức khỏe. "
                    "Nếu bạn đang gặp triệu chứng cấp tính, hãy liên hệ ngay cơ sở y tế gần nhất hoặc gọi cấp cứu (115). "
                    "Mình không thể hướng dẫn các biện pháp nhịn ăn cực đoan hay gây hại cho cơ thể — nếu bạn muốn giảm cân an toàn và bền vững, mình luôn sẵn sàng lên thực đơn khoa học cùng bạn."
                )
            )

    # 3. Block clearly off-topic queries immediately when no nutrition/health keywords are present
    text_lower = text.lower()
    if not _has_nutrition_signal(text_lower):
        for pattern in _OFF_TOPIC_PATTERNS:
            if pattern.search(text):
                logger.info("🛡️ Guardrail blocked explicit off-topic query before calling LLM.")
                return GuardrailDecision(
                    allowed=False,
                    reason="off_topic",
                    model_source="GUARDRAIL_OFF_TOPIC_BLOCK",
                    refusal_reply=(
                        "Mình là trợ lý dinh dưỡng của AR-NutriLens nên chỉ hỗ trợ các chủ đề về **dinh dưỡng, thực đơn, tính calo món ăn, "
                        "mục tiêu cân nặng và sức khỏe ăn uống** thôi nhé. Bạn muốn mình gợi ý món ăn phù hợp với mức calo hôm nay không?"
                    )
                )

    return GuardrailDecision(allowed=True)


def sanitize_output_guardrails(reply: str) -> str:
    """
    Post-LLM output verification. Prevents accidental leakage of API keys or internal system prompt sections.
    """
    if not reply:
        return reply

    for pattern in _OUTPUT_LEAK_PATTERNS:
        if pattern.search(reply):
            logger.error("🛡️ Output guardrail detected potential secret/prompt leak and redacted response!")
            return (
                "Xin lỗi, phản hồi vừa tạo không đạt chuẩn an toàn nội dung. "
                "Bạn hãy đặt lại câu hỏi về dinh dưỡng hoặc thực đơn để mình hỗ trợ chính xác hơn nhé!"
            )

    # Cap overly long walls of text on mobile screens
    max_chars = 2400
    if len(reply) > max_chars:
        truncated = reply[:max_chars].rsplit("\n", 1)[0].strip()
        return truncated + "\n\nBạn có muốn mình giải thích kỹ hơn phần nào không?"

    return reply
