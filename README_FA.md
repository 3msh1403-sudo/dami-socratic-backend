# بک‌اند سقراطی دامی — انتقال Amas به Gemini و Supabase

این نسخه بر اساس ساختار و جریان اصلی پروژه Amas بازنویسی شده است. هدف، حفظ pipeline آموزشی و سقراطی Amas و تعویض لایه‌های وابسته به مدل و Vector Database است.

## حفظ سورس اصلی Amas

نسخه دست‌نخورده ماژول‌های اصلی Amas در `reference/amas_original_src/` قرار دارد تا بتوان منطق و پیاده‌سازی اولیه را با نسخه دامی مقایسه کرد. این فایل‌ها آرشیوی‌اند و در اجرای دامی import نمی‌شوند. نسخه اجرایی سازگارشده در `src/` است.

## چه چیزهایی حفظ شده‌اند؟

- جداسازی ماژول‌ها و pipeline چندمرحله‌ای
- Scope Detection پیش از ورود به RAG
- دسته‌بندی سؤال و مدیریت درخواست‌های خارج از محدوده
- بازیابی جداگانه متن کتاب و نمونه‌های سؤال/پاسخ
- قالب‌بندی Context و Few-shot
- System Prompt سقراطی
- هماهنگ‌کننده اصلی orchestration
- تست تعاملی سؤال تکی

## چه چیزهایی عوض شده‌اند؟

| بخش Amas | نسخه دامی |
|---|---|
| Qwen local embeddings | Google `gemini-embedding-001` |
| Chroma collections | Supabase RPC `match_dami_documents` |
| OpenAI-compatible Scope LLM | Gemini API |
| OpenAI-compatible Response LLM | Gemini API |
| اجرای محلی CLI | FastAPI برای Render |

هیچ Qwen، Chroma، مدل محلی یا Vector DB جدیدی در مسیر اجرای این نسخه وجود ندارد.

## نکته مهم درباره بردارهای فعلی

کد Embedding پرسش را با `outputDimensionality=768` تولید می‌کند تا با بردارهای 768بعدی ثبت‌شده در طرح فعلی دیتابیس دامی سازگار باشد. مدل و تنظیمات Embedding پرسش باید با مدل و تنظیمات تولید بردارهای ذخیره‌شده یکسان باشند. این نسخه بردارهای ذخیره‌شده را بازتولید یا تغییر نمی‌دهد.

## RPC مورد استفاده

کد مستقیماً تابع موجود زیر را صدا می‌زند:

`match_dami_documents(query_embedding, match_count, filter_metadata, match_threshold)`

پارامترهای ارسالی:
- `query_embedding`: آرایه عددی بردار پرسش
- `match_count`: تعداد نتایج
- `filter_metadata`: فیلتر JSONB
- `match_threshold`: حداقل شباهت

خروجی مورد انتظار:
`id, document_key, content, source, page, chunk_index, metadata, similarity`

هیچ SQL برای ساخت مجدد جدول یا ایندکس در این پروژه اجرا نمی‌شود.

## دو مسیر Retrieval در Amas

Amas دو مسیر جدا داشت:
1. متن کتاب برای دانش factual
2. نمونه‌های سؤال و پاسخ برای سبک راهنمایی (few-shot)، نه به‌عنوان منبع factual مستقل

در دامی هر دو مسیر (در صورت فعال‌سازی نمونه‌ها) از همان RPC استفاده می‌کنند و با `filter_metadata` جدا می‌شوند. بردار سؤال یک بار تولید و بین دو مسیر مشترک استفاده می‌شود. فیلتر کتاب به‌صورت پیش‌فرض `{}` است تا اسناد فعلی به‌اشتباه حذف نشوند. بازیابی نمونه‌ها تا زمان تأیید وجود داده‌های Q&A در Supabase به‌صورت پیش‌فرض خاموش است. برای فعال‌سازی، ابتدا وجود رکوردهای نمونه و ساختار metadata آن‌ها را بررسی کنید؛ سپس `QA_RETRIEVAL_ENABLED=true` و فیلتر درست را تنظیم کنید. فیلتر پیش‌فرض نمونه‌ها `{"type":"qa_example"}` است، اما وجود این metadata در دیتابیس فعلی تأیید نشده است. اگر مقدار metadata واقعی متفاوت است، فقط مقدار env مربوطه را مطابق metadata واقعی تغییر دهید.

برای خاموش کردن مسیر Q&A:
`QA_RETRIEVAL_ENABLED=false` (مقدار پیش‌فرض)

**توجه:** ساختار دقیق metadata ردیف‌های زنده Supabase از امضای RPC مشخص نمی‌شود. پس وجود `metadata.type=qa_example` در دیتابیس فعلی تأیید نشده است؛ این فیلتر قابل تنظیم است.

## راه‌اندازی محلی

1. Python 3.11 یا جدیدتر نصب باشد.
2. نصب وابستگی‌ها:

```bash
pip install -r requirements.txt
```

3. از `.env.example` یک کپی با نام `.env` بسازید و مقادیر واقعی را وارد کنید.
4. اجرا:

```bash
uvicorn main:app --host 0.0.0.0 --port 8000
```

## متغیرهای ضروری

```env
GEMINI_API_KEY=
GEMINI_MODEL_NAME=gemini-2.5-flash
SCOPE_MODEL_NAME=gemini-2.5-flash

GOOGLE_EMBEDDING_MODEL=gemini-embedding-001
EMBEDDING_DIMENSION=768

SUPABASE_URL=
SUPABASE_SERVICE_ROLE_KEY=
SUPABASE_RPC_NAME=match_dami_documents
```

کلید `SUPABASE_SERVICE_ROLE_KEY` و کلید Gemini فقط در Environment Variables سرور Render قرار بگیرند؛ هرگز آن‌ها را در Frontend، فایل‌های عمومی یا GitHub قرار ندهید.

## استقرار روی Render

- Build Command: `pip install -r requirements.txt`
- Start Command: `uvicorn main:app --host 0.0.0.0 --port $PORT`
- Health Check: `/health`

مقادیر محرمانه را در بخش Environment سرویس Render تنظیم کنید.

## API

### GET `/`
وضعیت ساده سرویس را برمی‌گرداند.

### GET `/health`
وجود تنظیمات ضروری را بررسی می‌کند. این endpoint صحت دسترسی واقعی به APIها را تضمین نمی‌کند.

### POST `/chat`

درخواست:

```json
{
  "question": "عوامل موثر در اعتیاد چیست؟"
}
```

پاسخ:

```json
{
  "answer": "..."
}
```

## تست اتصال Embedding و Supabase

```bash
python build_vector_db.py
```

با وجود نام سازگاری فایل، این اسکریپت هیچ دیتابیسی نمی‌سازد یا پاک نمی‌کند؛ فقط یک Embedding موقت تولید می‌کند و RPC موجود را با `match_count=1` فراخوانی می‌کند.

## تست تعاملی

```bash
python test_system.py
```

## محدودیت‌های تأیید

این بسته از روی سورس Amas و امضای RPC ارائه‌شده برای دامی ساخته شده است. در این محیط امکان اتصال به پروژه زنده Supabase یا فراخوانی کلیدهای واقعی Gemini وجود نداشت؛ بنابراین تست زنده end-to-end انجام نشده است. قبل از بهره‌برداری، تست RPC و تطبیق metadata واقعی باید انجام شود.

## گزارش بررسی فنی نسخه فعلی

- کامپایل تمام فایل‌های Python با موفقیت انجام شده است.
- مسیر اصلی pipeline با Mock برای Scope، Embedding، Retrieval و Gemini آزمایش شده است.
- مسیر توقف Prompt Injection پیش از فراخوانی Embedding و Retrieval آزمایش شده است.
- payload ارسالی به RPC با نام و آرگومان‌های اعلام‌شده برای `match_dami_documents` بررسی شده است.
- درخواست Embedding با خروجی 768بعدی به‌صورت Mock بررسی شده است.
- endpoint ریشه و اعتبارسنجی سؤال خالی با FastAPI TestClient آزمایش شده‌اند.
- این تست‌ها جای تست زنده با کلیدهای واقعی Gemini و پروژه واقعی Supabase را نمی‌گیرند.

### مواردی که پیش از انتشار عمومی باید تعیین تکلیف شوند

1. باید تأیید شود بردارهای ذخیره‌شده واقعاً با `gemini-embedding-001` و ابعاد 768 تولید شده‌اند. نوع task هنگام ساخت بردارهای سند نیز باید با تنظیمات مورد انتظار مدل سازگار باشد؛ کد فعلی برای پرسش از `RETRIEVAL_QUERY` استفاده می‌کند.
2. منطق واقعی `filter_metadata` داخل تابع SQL و شکل metadata رکوردهای دامی از امضای RPC قابل استنتاج نیست. فیلتر کتاب فعلاً `{}` است و مسیر Q&A پیش‌فرض خاموش است.
3. API در این نسخه احراز هویت کاربر و rate limit ندارد. CORS به‌تنهایی محافظت امنیتی یا جلوگیری از سوءاستفاده و مصرف سهمیه Gemini نیست. پیش از انتشار عمومی باید احراز هویت/محدودسازی درخواست و محدودیت مصرف اضافه شود.
4. API فعلی history مکالمه را دریافت یا نگهداری نمی‌کند؛ هر درخواست یک اجرای مستقل از pipeline است، مانند نسخه CLI اصلی Amas.
5. Prompt-based scope detection یک لایه دفاعی است، نه تضمین کامل در برابر prompt injection. برای انتشار حساس باید تست‌های حمله و دفاع تکمیلی انجام شوند.
