# دليل اختبار التدريب المحلي (Local Training Test Guide)

هذا الملف يمثل "دليلاً" أو Prompt لكيفية اختبار الكود الذي قمنا بإنشائه (`train.py` و `dataset.py`) للتأكد من فاعليته محلياً قبل رفعه على Colab.

## الهدف
إثبات أن دورة التدريب (Training Loop) تعمل، وأن النموذج يمكنه استدعاء دالة `training_step`، حتى لو كانت البيانات وهمية أو قليلة جداً.

---

## 1. تجهيز بيئة العمل (Setup)

تأكد من أنك قمت بتثبيت المكتبات (يفترض أنك قمت بذلك مسبقاً):
```powershell
pip install -r requirements.txt
pip install pytorch-lightning omegaconf
```

## 2. إنشاء بيانات "وهمية" للتجربة (Dummy Data)

بدلاً من تحميل 50GB، سنقوم يدوياً بإنشاء بنية مجلدات تخدع كود الـ Dataset ليعتقد أن هناك بيانات حقيقية.

قم بتشغيل هذا السكربت البسيط في التيرمنال لإنشاء ملفات وهمية:

```python
import os
import numpy as np

# إنشاء مجلدات وهمية
base_dir = "output_dir/test_song_01"
os.makedirs(base_dir, exist_ok=True)

# إنشاء ملفات وهمية يتوقعها الكود
# 1. beatstep (مهم)
np.save(f"{base_dir}/test.beatstep.npy", np.array([0.5, 1.0, 1.5]))
# 2. midi (مهم)
open(f"{base_dir}/test.mid", 'w').close()
# 3. wav (مهم)
open(f"{base_dir}/test.wav", 'w').close()

print("✅ Created dummy data at output_dir/test_song_01")
```

## 3. تشغيل التدريب (Dry Run)

الآن، قم بتشغيل كود التدريب الجديد.

```powershell
python train.py
```

### النتائج المتوقعة:
1. سيقوم الكود بقراءة ملف الكونفيج `config.yaml`.
2. سيجد ملف البيانات الوهمي الذي أنشأناه.
3. سيبدأ الـ `TransformerWrapper` ويطبع رسائل الإعداد.
4. **الأهم:** ستبدأ دورة التدريب (Epoch 0) وسترى شريط التقدم يتحرك (Progress Bar).
5. لن تكون النتائج حقيقية (لأن الـ Loss وهمي حالياً)، لكن هذا **يثبت أن البنية التحتية للتدريب تعمل**.

---

## 4. الخطوات التالية (للتدريب الحقيقي)

إذا نجح الاختبار أعلاه، فإن الخطوات لجعل الـ `training_step` حقيقي هي:
1. تعديل `dataset.py` ليقرأ الـ wav ويحوله لـ LogMelSpectrogram فعلياً.
2. تعديل `dataset.py` ليقرأ الـ midi ويحوله لـ Tokens (باستخدام `midi_tokenizer.py`).
3. إزالة الـ Dummy Loss من `transformer_wrapper.py` ووضع `loss = outputs.loss`.

---
**ملاحظة:** تم كتابة كود `train.py` ليكون آمناً (Safe Mode)، فهو لا يمسح أي ملفات، ويتوقف بأمان إذا وُجد خطأ.
