import base64
import gradio as gr
import os
import shutil
import pymupdf as fitz  # PyMuPDF
import numpy as np
from PIL import Image
import io
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
from pythainlp.tokenize import word_tokenize
from transformers import MT5Tokenizer, MT5ForConditionalGeneration

import torch
import ollama
import shortuuid
import logging
import re

from typing import List, Dict, Tuple

# Image folder
TEMP_IMG="./data/images"
TEMP_VECTOR="./data/chromadb"
# รายชื่อ Model ที่คุณมีบน Ollama
AVAILABLE_MODELS = ["pdf-gemma", "pdf-qwen","pdf-llama"]

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Initialize Chroma client Disable telemetry 
os.environ["CHROMA_TELEMETRY_ENABLED"] = "false"
chroma_client = chromadb.PersistentClient(path=TEMP_VECTOR, settings=Settings(anonymized_telemetry=False))
collection = chroma_client.get_or_create_collection(name="pdf_data")

# ตั้งค่า device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
logging.info(f"Using device: {device}")

# โหลดโมเดล embedding
# SentenceTransformer สำหรับข้อความหลายภาษา (เน้นภาษาไทย)
sentence_model = SentenceTransformer('intfloat/multilingual-e5-base', device=device)

# Create directory for storing images
os.makedirs(TEMP_IMG, exist_ok=True)

sum_tokenizer = MT5Tokenizer.from_pretrained('StelleX/mt5-base-thaisum-text-summarization')
sum_model = MT5ForConditionalGeneration.from_pretrained('StelleX/mt5-base-thaisum-text-summarization')

def summarize_content(content: str) -> str:
    """
        สรุปเนื้อหา 
    """
    logging.info("%%%%%%%%%%%%%% SUMMARY %%%%%%%%%%%%%%%%%%%%%")    
       
    input_ = sum_tokenizer(content, truncation=True, max_length=1024, return_tensors="pt")
    with torch.no_grad():
        preds = sum_model.generate(
            input_['input_ids'].to('cpu'),
            num_beams=15,
            num_return_sequences=1,
            no_repeat_ngram_size=1,
            remove_invalid_values=True,
            max_length=250
        )

    summary = sum_tokenizer.decode(preds[0], skip_special_tokens=True)

    logging.info(f" summary: {summary}.")
    logging.info("%%%%%%%%%%%%%% SUMMARY %%%%%%%%%%%%%%%%%%%%%")
    return summary

# แยกเนื้อหา, รูป ออกจาก PDF
def extract_pdf_content(pdf_path: str) -> List[Dict]:
    """
    แยกข้อความและรูปภาพจาก PDF โดยใช้ PyMuPDF
    """
    try:
        doc = fitz.open(pdf_path)
        content_chunks = []
        all_text=[]

        for page_num in range(len(doc)):
            page = doc[page_num]
            # Extract text
            text = page.get_text("text").strip()
            all_text.append(f"{text} \n\n\n")
            if not text:
                text = f"ไม่มีข้อความในหน้า {page_num + 1}"
            
            logging.info("################# Text data ##################")
            chunk_data = {"text": f"ข้อมูลจากหน้า {page_num + 1} : {text}" , "images": []}
            
            # Extract images
            image_list = page.get_images(full=True)
            logging.info("################# images list ##################")
            for img_index, img in enumerate(image_list):
                xref = img[0]
                base_image = doc.extract_image(xref)
                image_bytes = base_image["image"]
                image_ext = base_image["ext"]
                
                # Convert to PIL Image
                try:
                    image = Image.open(io.BytesIO(image_bytes))
                    if image.mode != "RGB":
                        image = image.convert("RGB")
                    
                    img_id = f"pic_{str(page_num+1)}_{str(img_index+1)}"
                    img_path = f"{TEMP_IMG}/{img_id}.{image_ext}"
                    image.save(img_path, format=image_ext.upper())

                    img_desc = f"รูปภาพ จากหน้า {str(page_num+1)} ของ รูปที่ {str(img_index+1)}, บริบทข้อความ: {text[:80]}..."  
                    chunk_data["text"] += f"\n[ภาพ: {img_id}.{image_ext}]"                    
                    chunk_data["images"].append({
                        "data": image,
                        "path": img_path,
                        "description": img_desc
                    })
                except Exception as e:
                    logger.warning(f"ไม่สามารถประมวลผลรูปภาพที่หน้า {str(page_num+1)}, รูปที่ {str(img_index+1)}: {str(e)}")
            
            if chunk_data["text"]:
                content_chunks.append(chunk_data)
        
        if not any(chunk["images"] for chunk in content_chunks):
            logger.warning("ไม่พบรูปภาพใน PDF: %s", pdf_path)
        
        doc.close()
        content_text= "".join(all_text)
        # ตัดคำภาษาไทย
        thaitoken_text = preprocess_thai_text(content_text) if any(ord(c) >= 0x0E00 and ord(c) <= 0x0E7F for c in text) else text
        print("################################")
        print(f"{ thaitoken_text }")
        print("################################")
        global summarize
        summarize = summarize_content(thaitoken_text)
        return content_chunks
    except Exception as e:
        logger.error("เกิดข้อผิดพลาดในการแยก PDF: %s", str(e))
        raise

# ตัดคำภาษาไทย 
def preprocess_thai_text(text: str) -> str:
    """
    ตัดคำภาษาไทยด้วย pythainlp เพื่อเตรียมข้อความ

    Args:
        text (str): ข้อความภาษาไทย

    Returns:
        str: ข้อความที่ตัดคำแล้ว
    """
    return " ".join(word_tokenize(text, engine="newmm"))


def embed_text(text: str) -> np.ndarray:
    """
    สร้าง embedding สำหรับข้อความโดยใช้ SentenceTransformer 

    Args:
        text (str): ข้อความที่ต้องการสร้าง embedding        

    Returns:
        np.ndarray: Embedding vector ที่รวมจากหลายโมเดล
    """
    logging.info("-------------- start embed text  -------------------")
    
    # ตัดคำภาษาไทย
    processed_text = preprocess_thai_text(text) if any(ord(c) >= 0x0E00 and ord(c) <= 0x0E7F for c in text) else text
    
    # สร้าง embedding ด้วย SentenceTransformer
    sentence_embedding = sentence_model.encode(processed_text, normalize_embeddings=True, device=device)    
        
    return sentence_embedding

def store_in_chroma(content_chunks: List[Dict], pdf_name: str):
    """
    เก็บข้อมูลข้อความและรูปภาพใน Chroma พร้อม embedding
    """
    logging.info("##### Start store in chroma #########")
    for chunk in content_chunks:
        text = chunk["text"]
        images = chunk["images"]
        logging.info("################# Text embeding store ##################")
        text_embedding = embed_text(text)
        text_id = shortuuid.uuid()[:8]

        logging.info(f"text: {text} ")
        #logging.info(f"text_embedding: {text_embedding} ")
        
        collection.add(
            documents=[text],
            metadatas=[{"type": "text", "source": pdf_name}],
            embeddings=[text_embedding.tolist()],
            ids=[text_id]
        )
        logging.info("################# images embeding store ##################")
        logging.info(f"images: {images} ")
        for img in images:
            logging.info(f"images desc: {img["description"]} ")
            logging.info(f"images path: {img["path"]} ")            
            img_id = shortuuid.uuid()[:8]
            img_path = img["path"]
            logging.info(f"img_path: {text} ")
            collection.add(
                documents=[text],
                metadatas=[{"type": "image", "source": pdf_name, "image_path": img_path}],
                embeddings=[text_embedding.tolist()],
                ids=[img_id]
            )

def process_pdf_upload(pdf_file):
    """
    จัดการการอัปโหลดและประมวลผล PDF
    """
    try:
        clear_vector_db()
        pdf_path = pdf_file.name
        pdf_name = os.path.basename(pdf_path)
        logging.info("#### start process pdf ####")
        
        content_chunks = extract_pdf_content(pdf_path)
        logging.info("#### start store in vector db ####")
        logging.info(f"content: {content_chunks}")

        store_in_chroma(content_chunks, pdf_name)
        
        return f"ประมวลผลและจัดเก็บ {pdf_name} สำเร็จ"
    except Exception as e:
        return f"เกิดข้อผิดพลาดในการประมวลผล PDF: {str(e)}"

def clear_vector_db():
    try:
        
       # Clear existing collection to avoid duplicates
        chroma_client.delete_collection(name="pdf_data")
        global collection
        collection = chroma_client.create_collection(name="pdf_data")

    except Exception as e:
        return f"เกิดข้อผิดพลาดในการล้างข้อมูล: {str(e)}"
    
def clear_vector_db_and_images():
    """
    ล้างข้อมูลใน Chroma vector database และไฟล์ในโฟลเดอร์ images
    """
    
    try:
        clear_vector_db()
        
        pdf_input.clear()
        if os.path.exists(TEMP_IMG):
            shutil.rmtree(TEMP_IMG)
            os.makedirs(TEMP_IMG, exist_ok=True)
        
        return "ล้างข้อมูลใน vector database และโฟลเดอร์ images สำเร็จ"
    except Exception as e:
        return f"เกิดข้อผิดพลาดในการล้างข้อมูล: {str(e)}"


def query_rag(question: str,  chat_llm: str = "pdf-qwen"):
    """
    ค้นหาในระบบ RAG และสร้างคำตอบแบบ streaming โดยใช้ Ollama
    """
    logging.info(f"####  RAG get Question #### ")
    question_embedding = embed_text(question)
    
    results=[]
    # เช็คข้อมูลจาก เอกสาร ดึงมา 3 รายการ   ##  Retrival
    max_result = 3
    if "กี่" in question:
        max_result = 5
    
    if "ทั้งหมด" in question:
        max_result = 10

    if "บ้าง" in question:
        max_result = 5

    results = collection.query(
        query_embeddings=[question_embedding.tolist()],       
        n_results=max_result
    )
    logging.info(f"##### results from vector: { results }")
    context_texts = []
    image_paths = []

    for doc, metadata in zip(results["documents"][0], results["metadatas"][0]):
        context_texts.append(doc)
        logging.info(doc)
        logging.info(f"metadata: {metadata}")
        # Regex pattern สำหรับค้นหา [img: ชื่อไฟล์.jpeg]
        pattern = r"pic_(\d+)_(\d+)\.jpeg"

        # ค้นหาทุกรูป แบบที่ตรงกับ ส่งเข้ามา
        imgs = re.findall(pattern, doc)
        print("----------IIIII------------") 
        print(imgs)
        print("----------IIIII------------") 
        if imgs:
            image_paths.append(imgs)
            logging.info(f"img: {imgs}")

        print("---------------------------")
        if metadata:
            if metadata["type"] == "image":
                logging.info(f"image_path : { metadata["image_path"]}")
                image_paths.append(metadata['image_path'])
    


    context = "\n".join(context_texts)
    ##  Augmented
    logging.info("############## Begin Augmented prompt #################")
    prompt = f"""จากบริบทต่อไปนี้ ตอบคำถาม: {question}

    บริบท: 
        {summarize}

        {context}

    ให้คำตอบที่ชัดเจนและกระชับเป็นภาษาไทย หากบริบทมีชื่อไฟล์รูปภาพ ให้แสดงรูปภาพประกอบด้วย """ 
    
    logging.info(f"promt: {prompt}")
    logging.info("##############  End Augmented prompt #################")

    logging.info("+++++++++++++  Send prompt To LLM  ++++++++++++++++++")
    ## Generation  เพื่อการตอบ chat
    stream = ollama.chat(
        model=chat_llm,
        messages=[{"role": "user", "content": prompt}],      
        stream=True
    )
    
    return stream

def user(user_message: str, history: List[Dict]) -> Tuple[str, List[Dict]]:
    """
    จัดการ input ของผู้ใช้และเพิ่มลงในประวัติการแชท
    """
    return "", history + [{"role": "user", "content": user_message}]

def chatbot_interface(history: List[Dict], llm_model: str):
    """
    อินเทอร์เฟซแชทบอทแบบ streaming
    """
    user_message = history[-1]["content"]
    
    stream= query_rag(user_message, chat_llm=llm_model)

    history.append({"role": "assistant", "content": ""})
    full_answer=""
    """
    ส่วนของการ ตอบคำถาม
    """
    for chunk in stream:
        content = chunk["message"]["content"]
        full_answer += content 
        history[-1]["content"] += content
        #logging.info(f"content: {content}")
        yield history
    

    """
    ส่วนของการดึงรูปภาพ ที่เกี่ยวข้องมาแสดง โดยดึงจาก คำตอบด้านบน 
    """

    # ใช้ regex เพื่อดึงชื่อไฟล์ที่อยู่ใน [ภาพ: ...] 
    print(full_answer)
    pattern1 = r"\[(?:ภาพ:\s*)?(pic_\w+[-_]?\w*\.jpeg)\]"
    pattern2 = r"(pic_\w+[-_]?\w*\.jpeg)"
    # ค้นหาทุกรูป แบบที่ตรงกับ ส่งเข้ามา
    
    print("----------PPPP------------")       
    image_list = re.findall(pattern1, full_answer)
    print(image_list)
    if (len(image_list)==0):
        image_list = re.findall(pattern2, full_answer)
    print("----------xxxx------------")  
    # ดึงเฉพาะรูปที่ไม่ซ้ำกัน
    image_list_uniq = list(dict.fromkeys(image_list))  
    if image_list_uniq:
        history[-1]["content"] += "\n\nรูปภาพที่เกี่ยวข้อง:"
        yield history    
        # ดึงรูปมาแสดง 
        for img in image_list_uniq:
            img_path = f"{TEMP_IMG}/{img}"
            logger.info(f"img_path: {img_path}")      
            if os.path.exists(img_path):
                    image = Image.open(img_path)
                    buffered = io.BytesIO()
                    image.save(buffered, format="PNG")
                    image_response = f"{img} ![{img}](data:image/png;base64,{base64.b64encode(buffered.getvalue()).decode()})"
                    #ส่งรูปไปที่ Chat
                    history.append({"role": "assistant", "content": image_response })
                    yield history



# Gradio interface

with gr.Blocks() as demo:
    logo="https://camo.githubusercontent.com/9433204b08afdc976c2e4f5a4ba0d81f8877b585cc11206e2969326d25c41657/68747470733a2f2f63646e2e6a7364656c6976722e6e65742f67682f6e61726f6e67736b6d6c2f68746d6c352d6c6561726e406c61746573742f6173736574732f696d67732f546c697665636f64654c6f676f2d3435302e77656270"
    gr.Markdown(f"""<h3 style='display: flex; align-items: center; gap: 15px; padding: 10px; margin: 0;'>
        <img alt='T-LIVE-CODE' src='{logo}' style='height: 100px;' >
        <span style='font-size: 1.5em;'>แชทบอท PDF: RAG</span></h3>""")

    with gr.Tab("แอดมิน - อัปโหลด PDF"):
        pdf_input = gr.File(label="อัปโหลดไฟล์ PDF")
        upload_button = gr.Button("ประมวลผล PDF")
        clear_button = gr.Button("ล้างข้อมูล")
        upload_output = gr.Textbox(label="สถานะการอัปโหลด")
        upload_button.click(
            fn=process_pdf_upload,
            inputs=pdf_input,
            outputs=upload_output
        )
        clear_button.click(
            fn=clear_vector_db_and_images,
            inputs=None,
            outputs=upload_output,
            queue=False
        )
    
    with gr.Tab("แชท"):
        # Choice เลือก Model
        model_selector = gr.Dropdown(
            choices=AVAILABLE_MODELS,
            value="pdf-gemma",
            label="เลือก LLM Model"
        )
        selected_model = gr.State(value="pdf-gemma")  # เก็บไว้ใน state
        model_selector.change(fn=lambda x: x, inputs=model_selector, outputs=selected_model)
        # Chat Bot
        chatbot = gr.Chatbot(type="messages")
        msg = gr.Textbox(label="ถามคำถามเกี่ยวกับ PDF")
        # Clear button 
        clear_chat = gr.Button("ล้าง")
        # Submit function 
        msg.submit(
            fn=user,
            inputs=[msg, chatbot],
            outputs=[msg, chatbot],
            queue=False
        ).then(
            fn=chatbot_interface,
            inputs=[chatbot, selected_model],
            outputs=chatbot
        )
        clear_chat.click(lambda: [], None, chatbot, queue=False)

if __name__ == "__main__":
    # ล้างข้อมูล ออกจากระบบ ก่อน เริ่ม Start Web
    clear_vector_db_and_images()
    demo.launch()