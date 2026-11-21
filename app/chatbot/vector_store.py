# ai-service/app/chatbot/vector_store.py
"""
Vector Store & Knowledge Retriever Engine for AR-NutriLens RAG Chatbot.
Provides semantic search over clinical nutrition corpus, 30 Vietnamese foods,
dietary guidelines, and wellness tips.
"""
import re
import math
import logging
from typing import List, Dict, Any, Optional
from app.chatbot.knowledge_corpus import (
    CANONICAL_FOODS_DATA,
    CLINICAL_GUIDELINES,
    EVIDENCE_BASED_TIPS,
    SYSTEM_FAQ
)

logger = logging.getLogger("nutrilens-ai")


class KnowledgeChunk:
    def __init__(self, chunk_id: str, title: str, category: str, content: str, metadata: Optional[Dict[str, Any]] = None):
        self.chunk_id = chunk_id
        self.title = title
        self.category = category
        self.content = content
        self.metadata = metadata or {}

    def to_dict(self) -> Dict[str, Any]:
        return {
            "chunk_id": self.chunk_id,
            "title": self.title,
            "category": self.category,
            "content": self.content,
            "metadata": self.metadata
        }


class NutritionVectorRetriever:
    """
    In-memory semantic vector and keyword retriever.
    Uses TF-IDF and N-gram cosine similarity with BM25-style term saturation.
    Zero external vector DB dependencies required; lightning fast (<5ms response).
    """

    def __init__(self):
        self.chunks: List[KnowledgeChunk] = []
        self.vocab: Dict[str, int] = {}
        self.idf: Dict[str, float] = {}
        self.doc_vectors: List[Dict[str, float]] = []
        self._build_corpus_chunks()
        self._index_chunks()
        logger.info(f"📚 NutritionVectorRetriever initialized with {len(self.chunks)} knowledge chunks.")

    def _build_corpus_chunks(self):
        """Compiles knowledge corpus into indexable semantic chunks."""
        self.chunks = []

        # 1. 30 Canonical Food Items
        for code, data in CANONICAL_FOODS_DATA.items():
            serving_kcal = (data["caloriesPer100g"] * data["servingWeightG"]) / 100.0
            serving_p = (data["proteinPer100g"] * data["servingWeightG"]) / 100.0
            serving_c = (data["carbsPer100g"] * data["servingWeightG"]) / 100.0
            serving_f = (data["fatPer100g"] * data["servingWeightG"]) / 100.0

            content = (
                f"Món ăn: {data['displayName']} (Mã hệ thống: {code}).\n"
                f"Phân loại bữa: {data['category']}. Khối lượng khẩu phần chuẩn: {data['servingWeightG']:.0f}g.\n"
                f"Chỉ số dinh dưỡng trên 100g: {data['caloriesPer100g']} kcal, {data['proteinPer100g']}g protein, "
                f"{data['carbsPer100g']}g carbs, {data['fatPer100g']}g fat. Chỉ số đường huyết (GI): {data['glycemicIndex']}.\n"
                f"Giá trị dinh dưỡng cả khẩu phần ({data['servingWeightG']:.0f}g): ~{serving_kcal:.0f} kcal, "
                f"{serving_p:.1f}g protein, {serving_c:.1f}g carbs, {serving_f:.1f}g chất béo.\n"
                f"Mô tả: {data['description']}\n"
                f"Ưu điểm sức khỏe: {data['pros']}\n"
                f"Nhược điểm cần lưu ý: {data['cons']}\n"
                f"Khuyến nghị lâm sàng: {data['suitability']}"
            )
            self.chunks.append(
                KnowledgeChunk(
                    chunk_id=f"FOOD_{code}",
                    title=data["displayName"],
                    category="FOOD_PROFILE",
                    content=content,
                    metadata={"foodCode": code, "category": data["category"], "calories": serving_kcal}
                )
            )

        # 2. Clinical Dietary Guidelines
        for guide_key, guide_data in CLINICAL_GUIDELINES.items():
            rules_str = "\n- " + "\n- ".join(guide_data["keyRules"])
            content = f"Hướng dẫn lâm sàng: {guide_data['title']}.\nCác nguyên tắc cốt lõi:{rules_str}"
            self.chunks.append(
                KnowledgeChunk(
                    chunk_id=f"GUIDE_{guide_key}",
                    title=guide_data["title"],
                    category="CLINICAL_GUIDELINE",
                    content=content,
                    metadata={"condition": guide_key}
                )
            )

        # 3. Evidence-based Nutrition Tips
        for tip in EVIDENCE_BASED_TIPS:
            content = f"Mẹo dinh dưỡng khoa học: {tip['title']}.\nNội dung: {tip['content']}"
            self.chunks.append(
                KnowledgeChunk(
                    chunk_id=tip["id"],
                    title=tip["title"],
                    category="NUTRITION_TIP",
                    content=content,
                    metadata={"tipId": tip["id"]}
                )
            )

        # 4. AR-NutriLens FAQ
        for idx, faq in enumerate(SYSTEM_FAQ, start=1):
            content = f"Câu hỏi hệ thống AR-NutriLens: {faq['question']}\nTrả lời chuyên môn: {faq['answer']}"
            self.chunks.append(
                KnowledgeChunk(
                    chunk_id=f"FAQ_{idx}",
                    title=faq["question"],
                    category="SYSTEM_FAQ",
                    content=content,
                    metadata={"faqIndex": idx}
                )
            )

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        """Lowers and tokenizes text including Vietnamese word segments."""
        text_clean = text.lower()
        # Keep unicode letters, numbers, and common food terms
        words = re.findall(r"[\w\u00C0-\u1EF9]+", text_clean)
        # Generate unigrams and bigrams for phrase matching
        tokens = list(words)
        for i in range(len(words) - 1):
            tokens.append(f"{words[i]}_{words[i+1]}")
        return tokens

    def _index_chunks(self):
        """Builds TF-IDF inverted index across all chunks."""
        doc_count = len(self.chunks)
        df: Dict[str, int] = {}
        tf_list: List[Dict[str, int]] = []

        for chunk in self.chunks:
            full_text = f"{chunk.title} {chunk.title} {chunk.content}"
            tokens = self._tokenize(full_text)
            counts: Dict[str, int] = {}
            for t in tokens:
                counts[t] = counts.get(t, 0) + 1
            tf_list.append(counts)

            for unique_t in counts.keys():
                df[unique_t] = df.get(unique_t, 0) + 1

        # Calculate IDF with smoothing
        self.idf = {t: math.log((doc_count + 1) / (freq + 1)) + 1.0 for t, freq in df.items()}

        # Build normalized TF-IDF vector per chunk
        self.doc_vectors = []
        for counts in tf_list:
            vec: Dict[str, float] = {}
            norm_sq = 0.0
            for t, count in counts.items():
                w = (1.0 + math.log(count)) * self.idf.get(t, 1.0)
                vec[t] = w
                norm_sq += w * w
            norm = math.sqrt(norm_sq) if norm_sq > 0 else 1.0
            normalized_vec = {t: w / norm for t, w in vec.items()}
            self.doc_vectors.append(normalized_vec)

    def retrieve(self, query: str, top_k: int = 3, category_filter: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Retrieves top-K most semantically relevant knowledge chunks for a given query.
        """
        q_tokens = self._tokenize(query)
        if not q_tokens:
            return []

        # Build query vector
        q_counts: Dict[str, int] = {}
        for t in q_tokens:
            q_counts[t] = q_counts.get(t, 0) + 1

        q_vec: Dict[str, float] = {}
        q_norm_sq = 0.0
        for t, count in q_counts.items():
            if t in self.idf:
                w = (1.0 + math.log(count)) * self.idf[t]
                q_vec[t] = w
                q_norm_sq += w * w

        q_norm = math.sqrt(q_norm_sq) if q_norm_sq > 0 else 1.0
        normalized_q_vec = {t: w / q_norm for t, w in q_vec.items()}

        # Keyword-based category boosting
        q_lower = query.lower()
        condition_map = {
            "gút": "GOUT",
            "gout": "GOUT",
            "acid uric": "GOUT",
            "tiểu đường": "DIABETES",
            "đái tháo đường": "DIABETES",
            "đường huyết": "DIABETES",
            "huyết áp": "HYPERTENSION",
            "tim mạch": "HYPERTENSION",
            "giảm cân": "LOSE_WEIGHT",
            "giảm mỡ": "LOSE_WEIGHT",
            "thâm hụt": "LOSE_WEIGHT",
            "tăng cơ": "GAIN_MUSCLE",
            "thể hình": "GAIN_MUSCLE",
            "duy trì": "MAINTAIN"
        }

        # Compute Cosine Similarity
        scored_results = []
        for idx, doc_vec in enumerate(self.doc_vectors):
            chunk = self.chunks[idx]
            if category_filter and chunk.category != category_filter:
                continue

            # Dot product
            score = 0.0
            for t, q_w in normalized_q_vec.items():
                if t in doc_vec:
                    score += q_w * doc_vec[t]

            # Boost if query words match in chunk title
            title_lower = chunk.title.lower()
            for w in q_lower.split():
                if len(w) > 2 and w in title_lower:
                    score += 0.20

            # Condition guideline boost
            for kw, cond in condition_map.items():
                if kw in q_lower:
                    if chunk.category == "CLINICAL_GUIDELINE" and chunk.metadata.get("condition") == cond:
                        score += 0.45

            if score > 0.02:
                scored_results.append({
                    "score": round(score, 4),
                    "chunk": chunk.to_dict()
                })

        scored_results.sort(key=lambda x: x["score"], reverse=True)
        return scored_results[:top_k]


# Singleton instance
retriever = NutritionVectorRetriever()
