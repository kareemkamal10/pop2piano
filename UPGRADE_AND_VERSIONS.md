# التحديثات والإصدارات / Upgrade & Versions

المشروع الأصلي (Pop2Piano) توقف تحديثه منذ سنوات. هذا المستند يوضح **ما تم تحديثه** ليعمل على آخر إصدارات مستقرة (2025–2026).

---

## 1. المكتبات المحدّثة

| المكتبة | الإصدار المستهدف | ملاحظات |
|--------|-------------------|---------|
| **PyTorch** | 2.1+ | دعم mixed precision و gradient checkpointing |
| **Lightning** | 2.2+ | المشروع يدعم Lightning 2.x مع fallback لـ pytorch-lightning 1.x |
| **Transformers** | 4.40+ | متوافق مع Pop2Piano على HuggingFace و T5 |
| **librosa** | 0.10+ | تحليل الصوت والـ chroma |
| **omegaconf** | 2.3+ | إعدادات المشروع (يفضّل Python 3.10 أو 3.11 مع antlr4) |
| **yt-dlp** | 2024+ | بديل youtube-dl لتحميل فيديوهات YouTube |
| **essentia** | 2.1b6+ | لاستخراج الإيقاع (اختياري في الاستنتاج إذا استخدمت HuggingFace فقط) |

---

## 2. التعديلات التي تمت في الكود

- **Lightning:** استيراد `lightning as pl` مع fallback لـ `pytorch_lightning`؛ استخدام `lightning.pytorch.callbacks` و `lightning.pytorch.loggers` عند التوفر.
- **التحميل:** استخدام `yt-dlp` بدل `youtube-dl`، مع دعم Windows (مسارات وامتداد `.exe`).
- **المسارات:** استخدام `subprocess` مع تمرير القوائم لتجنب مشاكل المسارات التي تحتوي على فراغات.
- **التحقق من المدة:** السماح بالكفرات الجزئية (نسبة 15%–120% من طول الأغنية).
- **المقامات:** إضافة نظام tokens للمقامات العربية وربطه بالـ config والـ dataset والـ inference.

---

## 3. بيئة التشغيل الموصى بها

- **Python:** 3.10 أو 3.11 (مستقر مع omegaconf و antlr4).
- **التدريب:** GPU مع CUDA 12.1 مفضّل؛ الـ notebooks جاهزة لـ Colab.
- **الاستنتاج:** يعمل على CPU أو GPU؛ استنتاج HuggingFace لا يلزم Essentia إذا استخدمت المعالج الجاهز.

---

## 4. التحقق من أن كل شيء يعمل

```bash
# إنشاء بيئة وتثبيت المتطلبات
python -m venv .venv
.venv\Scripts\activate   # Windows
pip install -r requirements.txt

# (اختياري) تشغيل اختبار استيراد
python -c "from transformer_wrapper import TransformerWrapper; from dataset import Pop2PianoDataset; print('OK')"
```

للـ inference فقط (بدون تدريب):

```bash
pip install transformers librosa pretty_midi soundfile
# ثم استخدم inference.ipynb أو inference_colab.ipynb
```

---

## 5. ملخص الحالة

| المكوّن | الحالة |
|---------|--------|
| التدريب (train.py + train.ipynb) | محدّث (Lightning 2.x، mixed precision، maqam-aware) |
| الاستنتاج (inference + smart_inference) | محدّث (HuggingFace، اكتشاف مقام، post-processing) |
| التحميل (download.py) | محدّث (yt-dlp، Windows، حد الحجم) |
| الداتا (dataset + preprocess_maqam) | محدّث (مقامات، كفرات جزئية) |
| المتطلبات (requirements.txt) | محدّث لإصدارات 2025–2026 |

إذا واجهت خطأ متعلقاً بإصدار مكتبة معيّن، راجع `.copilot/TECHNICAL_DEBT_AND_FIXES.md` لتفاصيل إضافية.
