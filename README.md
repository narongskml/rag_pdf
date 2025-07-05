## PDF RAG ##

    Project นี้สำหรับการ ศึกษา RAG ด้วยไฟล์ knowledg base จาก PDF
    เพื่อเรียนรู้ จาก ช่อง T-LIVE-CODE   ต่อเนื่องจากการเรียนรู้เรื่อง RAG ด้วย Text
    ใน project นี้ยังไม่ได้ ทำ multimodal  RAG  แต่เป็นการทำ RAG จาก PDF


## โครงสร้าง ของ Project ##

    rag_pdf.py          # โปรแกรม 
    requirements.txt    # requirement file

## สิ่งที่ต้องเตรียม
    1. ติดตั้ง ollama
    - pull ollama image
      > ollama pull qwen2.5:1.5b
      > ollama pull gemma3:1b
      > ollama pull llama3.2:latest

    2. ติดตั้ง python
    3. ติดตั้ง git
   

## โมเดล ที่ใช้
  # Embedding Model
    - intfloat/multilingual-e5-base
  
  # Text Summarize Model
    - StelleX/mt5-base-thaisum-text-summarization
  
  # Chat Model
    - qwen2.5:1.5b
    - gemma3:1b
    - llama3.2:latest
  

## Library ที่ใช้งาน
    - PyMuPDF สำหรับ ประมวลผล PDF
    - sentence_transformers สำหรับทำ Text Embeding โมเดลจาก Hugging Face
    - pythainlp ตัดคำ ภาษาไทย
    - transformers สำหรับ ดึง Model Text Summary จาก Hugging Face
    - Chroma สำหรับเก็บข้อมูล
    - Ollama (qwen2.5vl:3b) สำหรับสร้างคำตอบ

## สร้าง Model เพื่อใช้งานก่อน
    > model-create.bat

## วิธีการ Run
    > git clone https://github.com/narongskml/rag_pdf.git
    > cd rag_pdf
    > python -m venv .venv
    > .venv\Script\activate
    > pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
    > pip install -r requirements.txt
    > python rag_pdf.py

    เปิด browser  ไปที่ http://127.0.0.1:7860

## Screen Shot
![admin](image.png)
![chat](image-1.png)


## LICENSE
    GNU GENERAL PUBLIC LICENSE

## Remark
    ไม่สามารถใช้งาน จริงบน Production ได้ เฉพาะการเรียนรู้ เพื่อการศึกษาเท่านั้น