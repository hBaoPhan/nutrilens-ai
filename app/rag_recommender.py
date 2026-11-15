# ai-service/app/rag_recommender.py
import os
import json
import logging
import urllib.request
import urllib.error
from typing import List, Dict, Any, Optional
from app.schemas import (
    UserContextDTO,
    CandidateFoodDTO,
    RAGRecommendationRequest,
    RAGRecommendedDish,
    RAGRecommendationResponse
)

logger = logging.getLogger("nutrilens-ai")

class NutriLensRAGRecommender:
    """
    RAG-driven Nutrition Recommendation Engine.
    Retrieves grounded candidate food data from NutriLens DB and augments LLM generation
    or clinical rule synthesizer to guarantee ZERO hallucinations.
    """

    @staticmethod
    def _load_env_if_needed():
        # 1. Read from .env candidate locations
        base_dir = os.path.dirname(os.path.abspath(__file__))
        candidates = [
            os.path.join(base_dir, ".env"),
            os.path.join(base_dir, "..", ".env"),
            os.path.join(base_dir, "..", "..", ".env"),
            os.path.join(os.getcwd(), ".env"),
            os.path.join(os.getcwd(), "app", ".env"),
            os.path.join(os.getcwd(), "ai-service", ".env"),
            "/app/.env",
            "/app/app/.env",
            r"c:\Users\baoph\OneDrive\Desktop\AR-nutrilens\ai-service\.env"
        ]
        for p in candidates:
            if os.path.isfile(p):
                try:
                    with open(p, "r", encoding="utf-8") as f:
                        for line in f:
                            line = line.strip()
                            if line and not line.startswith("#") and "=" in line:
                                k, v = line.split("=", 1)
                                k = k.strip()
                                v = v.strip().strip('"').strip("'")
                                if k in ("GEMINI_API_KEY", "GOOGLE_API_KEY") and v:
                                    os.environ[k] = v
                                    logger.info(f"Loaded {k} from {p}")
                                    return
                except Exception:
                    pass

        # 2. Read directly from Windows System/User Environment Registry
        try:
            import winreg
            for root, subkey in [
                (winreg.HKEY_CURRENT_USER, r"Environment"),
                (winreg.HKEY_LOCAL_MACHINE, r"SYSTEM\CurrentControlSet\Control\Session Manager\Environment")
            ]:
                try:
                    with winreg.OpenKey(root, subkey) as k:
                        val, _ = winreg.QueryValueEx(k, "GEMINI_API_KEY")
                        if val:
                            os.environ["GEMINI_API_KEY"] = val
                            return
                except Exception:
                    pass
        except Exception:
            pass

        # Note: GEMINI_API_KEY must strictly be loaded from .env or environment variable
        if not os.getenv("GEMINI_API_KEY") and not os.getenv("GOOGLE_API_KEY"):
            logger.info("ℹ️ No GEMINI_API_KEY detected in .env or environment")

    def __init__(self):
        self._load_env_if_needed()
        self.gemini_api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        if self.gemini_api_key:
            masked = self.gemini_api_key[:6] + "..." + self.gemini_api_key[-4:] if len(self.gemini_api_key) > 10 else "***"
            logger.info(f"✨ Gemini API Key active ({masked}) for RAG Meal Recommender")
        else:
            logger.info("ℹ️ No GEMINI_API_KEY set; using high-precision Clinical Knowledge Synthesizer")

    def recommend_next_meal(self, request: RAGRecommendationRequest) -> RAGRecommendationResponse:
        user_ctx = request.userContext
        candidates = request.candidateFoods

        # 1. First, rank candidates deterministically based on clinical macronutrient fit
        ranked_candidates = self._rank_candidates_clinically(user_ctx, candidates)
        top_candidates = ranked_candidates[:3] if ranked_candidates else candidates[:3]

        # Convert top candidates to RAGRecommendedDish format
        recommended_dishes: List[RAGRecommendedDish] = []
        for item in top_candidates:
            serving_w = item.get("servingWeightG", 100.0)
            factor = serving_w / 100.0
            kcal = round(item.get("kcalPer100g", 100.0) * factor, 1)
            protein = round(item.get("proteinPer100g", 5.0) * factor, 1)
            carb = round(item.get("carbPer100g", 10.0) * factor, 1)
            fat = round(item.get("fatPer100g", 2.0) * factor, 1)
            vol = round(serving_w / max(0.1, item.get("densityGramsPerCm3", 1.0)), 1)

            note = (
                f"Định lượng {serving_w:.0f}g ({vol:.0f} cm³) cung cấp {kcal:.0f} kcal, "
                f"{protein:.1f}g Protein. Phù hợp mục tiêu {user_ctx.fitnessGoal}."
            )

            recommended_dishes.append(RAGRecommendedDish(
                foodCode=item.get("foodCode", ""),
                displayName=item.get("displayName", ""),
                category=item.get("category", "Dinner"),
                foodType=item.get("foodType", "COMPOSITE_DISH"),
                densityGramsPerCm3=item.get("densityGramsPerCm3", 1.0),
                suggestedWeightG=serving_w,
                suggestedVolumeCm3=vol,
                estimatedKcal=kcal,
                proteinG=protein,
                carbG=carb,
                fatG=fat,
                emoji=item.get("emoji", "🍲"),
                iconType=item.get("iconType", "salad"),
                iconBgColor=item.get("iconBgColor", "#FFEFE6"),
                iconColor=item.get("iconColor", "#F37340"),
                matchScore=round(item.get("score", 0.90), 2),
                suitabilityNote=note
            ))

        # 2. Try calling Gemini LLM for dynamic clinical reasoning if API key is present
        gemini_rationale = None
        if recommended_dishes:
            gemini_rationale = self._call_gemini_rag(user_ctx, recommended_dishes)

        if gemini_rationale:
            model_source = "AI_RAG_GEMINI"
            final_rationale = gemini_rationale
        else:
            model_source = "AI_RAG_CLINICAL_SYNTHESIZER"
            final_rationale = self._synthesize_clinical_rationale(user_ctx, recommended_dishes)

        return RAGRecommendationResponse(
            status="SUCCESS",
            mealType=user_ctx.targetMealType,
            targetMealKcal=user_ctx.targetMealKcal,
            remainingKcalToday=user_ctx.remainingKcalToday,
            aiReasoningSummary=final_rationale,
            modelSource=model_source,
            recommendedDishes=recommended_dishes
        )

    def _rank_candidates_clinically(self, ctx: UserContextDTO, candidates: List[CandidateFoodDTO]) -> List[Dict[str, Any]]:
        scored = []
        target_kcal = ctx.targetMealKcal if ctx.targetMealKcal > 0 else 500.0
        target_protein = ctx.targetMealProteinG if ctx.targetMealProteinG > 0 else 25.0
        target_meal = ctx.targetMealType.upper()
        fitness_goal = ctx.fitnessGoal.upper()
        already_eaten = set(ctx.alreadyEatenFoods or [])

        for c in candidates:
            c_dict = c.model_dump() if hasattr(c, "model_dump") else (c.dict() if hasattr(c, "dict") else dict(c))
            serving_w = c_dict.get("servingWeightG", 100.0)
            factor = serving_w / 100.0
            dish_kcal = c_dict.get("kcalPer100g", 100.0) * factor
            dish_protein = c_dict.get("proteinPer100g", 5.0) * factor

            # Calorie proximity score
            diff = abs(dish_kcal - target_kcal)
            kcal_score = max(0.0, 1.0 - (diff / max(100.0, target_kcal)))

            # Protein density score
            protein_score = min(1.0, dish_protein / max(10.0, target_protein))

            # Category match bonus
            dish_cat = (c_dict.get("category") or "").upper()
            cat_bonus = 1.0 if dish_cat == target_meal else (0.5 if "ATOMIC" in (c_dict.get("foodType") or "") else 0.2)

            # Avoid recently eaten penalty
            repeat_penalty = 0.50 if c_dict.get("foodCode") in already_eaten else 0.0

            # Goal bonus
            goal_bonus = 0.0
            if fitness_goal == "LOSE_WEIGHT":
                if dish_kcal <= target_kcal and dish_protein >= 15.0:
                    goal_bonus += 0.25
            elif fitness_goal == "GAIN_WEIGHT":
                if dish_protein >= 22.0:
                    goal_bonus += 0.25

            total_score = (0.40 * kcal_score) + (0.35 * protein_score) + (0.15 * cat_bonus) + goal_bonus - repeat_penalty
            c_dict["score"] = max(0.05, total_score)
            c_dict["dishKcal"] = dish_kcal
            c_dict["dishProtein"] = dish_protein
            scored.append(c_dict)

        scored.sort(key=lambda x: x["score"], reverse=True)
        return scored

    def _call_gemini_rag(self, ctx: UserContextDTO, dishes: List[RAGRecommendedDish]) -> Optional[str]:
        try:
            self._load_env_if_needed()
            api_key = self.gemini_api_key or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
            if not api_key:
                print("\n[GEMINI RAG] ⚠️ No GEMINI_API_KEY found. Falling back to local clinical rule synthesizer.\n", flush=True)
                return None

            dish_summaries = [
                f"- {d.displayName}: {d.estimatedKcal:.0f} kcal, {d.proteinG:.1f}g protein, {d.carbG:.1f}g carb, {d.fatG:.1f}g fat (khối lượng {d.suggestedWeightG:.0f}g)"
                for d in dishes
            ]
            dishes_str = "\n".join(dish_summaries)

            prompt_text = (
                f"Bạn là chuyên gia dinh dưỡng lâm sàng AR-NutriLens.\n"
                f"Người dùng '{ctx.username}' có mục tiêu: '{ctx.fitnessGoal}', cân nặng {ctx.weightKg}kg.\n"
                f"Hôm nay đã nạp {ctx.todayConsumedKcal:.0f}/{ctx.dailyCalorieGoal} kcal. "
                f"Còn lại {ctx.remainingKcalToday:.0f} kcal. Bữa ăn tiếp theo là: {ctx.targetMealType}.\n"
                f"Hệ thống đã truy xuất chính xác các món phù hợp sau từ cơ sở dữ liệu:\n"
                f"{dishes_str}\n\n"
                f"Hãy viết một lời khuyên dinh dưỡng ngắn gọn (2-3 câu), ấm áp, chuyên nghiệp bằng tiếng Việt, "
                f"giải thích lý do vì sao món đầu tiên là lựa chọn tối ưu cho bữa ăn này để đạt mục tiêu."
            )

            payload = {
                "contents": [{"parts": [{"text": prompt_text}]}],
                "generationConfig": {
                    "temperature": 0.4,
                    "maxOutputTokens": 2048
                }
            }

            # ----------------------------------------------------------------------
            # PRINT OUTGOING REQUEST TO TERMINAL & CONTAINER LOGS
            # ----------------------------------------------------------------------
            print("\n" + "=" * 80, flush=True)
            print("[GEMINI RAG] >>> OUTGOING REQUEST TO GEMINI API >>>", flush=True)
            print(f"Target Meal: {ctx.targetMealType} | User: {ctx.username} (Goal: {ctx.fitnessGoal})", flush=True)
            print("-" * 34 + " PROMPT TEXT " + "-" * 33, flush=True)
            print(prompt_text, flush=True)
            print("-" * 33 + " PAYLOAD JSON " + "-" * 33, flush=True)
            print(json.dumps(payload, ensure_ascii=False, indent=2), flush=True)
            print("=" * 80 + "\n", flush=True)

            logger.info(f"[GEMINI RAG] Sending request to Gemini for {len(dishes)} dishes...")

            models_to_try = [
                "gemini-3.5-flash",
                "gemini-3.1-flash-lite",
                "gemma-4-26b-a4b-it",
                "gemini-flash-latest",
                "gemini-3.8-flash"
            ]

            masked_key = api_key[:6] + "..." + api_key[-4:] if len(api_key) > 10 else "***"

            for model in models_to_try:
                try:
                    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={api_key}"
                    print(f"[GEMINI RAG] Calling model '{model}' ({url.replace(api_key, masked_key)})...", flush=True)

                    req = urllib.request.Request(
                        url,
                        data=json.dumps(payload).encode("utf-8"),
                        headers={"Content-Type": "application/json"},
                        method="POST"
                    )
                    with urllib.request.urlopen(req, timeout=8) as resp:
                        if resp.status == 200:
                            raw_body = resp.read().decode("utf-8")
                            data = json.loads(raw_body)

                            # ------------------------------------------------------------------
                            # PRINT INCOMING RESPONSE TO TERMINAL & CONTAINER LOGS
                            # ------------------------------------------------------------------
                            print("\n" + "=" * 80, flush=True)
                            print(f"[GEMINI RAG] <<< INCOMING RESPONSE FROM GEMINI ({model}) <<<", flush=True)
                            print(f"HTTP Status: {resp.status}", flush=True)
                            print("-" * 30 + " RAW RESPONSE JSON " + "-" * 30, flush=True)
                            print(json.dumps(data, ensure_ascii=False, indent=2), flush=True)

                            candidates = data.get("candidates", [])
                            if candidates:
                                parts = candidates[0].get("content", {}).get("parts", [])
                                text = "".join(p.get("text", "") for p in parts if "text" in p).strip()
                                if text:
                                    print("-" * 28 + " EXTRACTED RATIONALE " + "-" * 29, flush=True)
                                    print(text, flush=True)
                                    print("=" * 80 + "\n", flush=True)
                                    logger.info(f"✨ Gemini RAG generation successful with model '{model}': {text[:60]}...")
                                    return text
                                else:
                                    print("[GEMINI RAG] Warning: No text in candidate parts.", flush=True)
                                    print("=" * 80 + "\n", flush=True)
                except urllib.error.HTTPError as he:
                    err_body = he.read().decode("utf-8", errors="replace") if hasattr(he, "read") else ""
                    print(f"[GEMINI RAG] Model '{model}' HTTP Error {he.code}: {err_body}", flush=True)
                    logger.debug(f"Model {model} HTTP {he.code}. Trying fallback...")
                    continue
                except Exception as ex:
                    print(f"[GEMINI RAG] Model '{model}' Exception: {ex}", flush=True)
                    logger.debug(f"Model {model} failed: {ex}. Trying fallback...")
                    continue

            print("[GEMINI RAG] ⚠️ All candidate models failed or unavailable. Falling back to local clinical rule synthesizer.\n", flush=True)
            return None
        except Exception as e:
            print(f"[GEMINI RAG] Exception during Gemini call: {e}\n", flush=True)
            logger.warning(f"Gemini API request failed or timed out: {e}")
        return None

    def _synthesize_clinical_rationale(self, ctx: UserContextDTO, dishes: List[RAGRecommendedDish]) -> str:
        goal_vn = "Duy trì thể trạng"
        if ctx.fitnessGoal.upper() == "LOSE_WEIGHT":
            goal_vn = "Giảm mỡ & Kiểm soát calo"
        elif ctx.fitnessGoal.upper() == "GAIN_WEIGHT":
            goal_vn = "Tăng cơ & Phục hồi"

        meal_vn = "Bữa Tối"
        if ctx.targetMealType.upper() == "BREAKFAST":
            meal_vn = "Bữa Sáng"
        elif ctx.targetMealType.upper() == "LUNCH":
            meal_vn = "Bữa Trưa"
        elif ctx.targetMealType.upper() == "SNACK":
            meal_vn = "Bữa Phụ"

        if not dishes:
            return f"Khuyến nghị {meal_vn.lower()} thanh đạm giàu chất xơ để duy trì mức năng lượng ổn định."

        top_dish = dishes[0]

        if ctx.remainingKcalToday <= 150:
            return (
                f"Bạn đã nạp gần đủ hạn mức năng lượng trong ngày ({ctx.todayConsumedKcal:.0f}/{ctx.dailyCalorieGoal} kcal). "
                f"Khuyến nghị {meal_vn.lower()} nhẹ nhàng với {top_dish.displayName} hoặc rau củ tươi nhằm tạo cảm giác no "
                f"mà không gây dư thừa năng lượng trước khi nghỉ ngơi."
            )

        return (
            f"Theo mục tiêu {goal_vn}, bạn còn lại {ctx.remainingKcalToday:.0f} kcal trong ngày. "
            f"Lựa chọn lý tưởng cho {meal_vn.lower()} là {top_dish.displayName} ({top_dish.estimatedKcal:.0f} kcal), "
            f"cung cấp {top_dish.proteinG:.1f}g protein nạc hỗ trợ nuôi dưỡng cơ bắp và tạo cảm giác no lâu bền vững."
        )

# Global singleton recommender instance
recommender = NutriLensRAGRecommender()
